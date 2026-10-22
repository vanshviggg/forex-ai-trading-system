"""Research API. No route submits an order or runs a shell command."""

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from forex_ai import RESEARCH_DISCLAIMER, __version__
from forex_ai.backtesting.engine import BacktestConfig, run_research_backtest
from forex_ai.config import Settings, configure_logging, get_settings
from forex_ai.data.market_data import (
    ALLOWED_INTERVALS,
    ALLOWED_PERIODS,
    DATA_SOURCE_LIMITATIONS,
    SUPPORTED_SYMBOLS,
    MarketDataError,
    MarketDataProvider,
)
from forex_ai.indicators.technical import add_indicators
from forex_ai.ml.features import FEATURE_COLUMNS, build_feature_frame
from forex_ai.ml.predict import load_model, predict_probability
from forex_ai.risk.risk_manager import RiskManager
from forex_ai.strategies.signal_engine import latest_signal

logger = logging.getLogger(__name__)

INDICATOR_FIELDS = (
    "timestamp",
    "close",
    "sma_20",
    "sma_50",
    "ema_20",
    "ema_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "bb_upper",
    "bb_lower",
    "volatility_20",
    "pct_return",
    "log_return",
    "ema20_distance",
    "volume_change",
    "support_20",
    "resistance_20",
)


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str
    research_only: bool = True


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketResponse(BaseModel):
    symbol: str
    period: str
    interval: str
    source: str
    rows: int
    limitations: list[str]
    candles: list[Candle]


class IndicatorPoint(BaseModel):
    timestamp: datetime
    close: float
    sma_20: float | None = None
    sma_50: float | None = None
    ema_20: float | None = None
    ema_50: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    atr_14: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    volatility_20: float | None = None
    pct_return: float | None = None
    log_return: float | None = None
    ema20_distance: float | None = None
    volume_change: float | None = None
    support_20: float | None = None
    resistance_20: float | None = None


class IndicatorResponse(BaseModel):
    symbol: str
    rows: int
    latest: IndicatorPoint
    history: list[IndicatorPoint]


class SignalResponse(BaseModel):
    symbol: str
    timestamp: datetime
    signal: str
    technical_score: float = Field(ge=0, le=1)
    ml_probability: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    reasons: list[str]
    disclaimer: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = "EURUSD"
    period: str = "2y"
    interval: str = "1d"


class PredictResponse(BaseModel):
    symbol: str
    timestamp: datetime
    probability_up: float
    probability_down: float
    predicted_direction: str
    model_name: str
    observation_was_in_fit: bool
    disclaimer: str


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = "EURUSD"
    period: str = "2y"
    interval: str = "1d"
    initial_capital: float = Field(default=10_000, gt=0)
    risk_per_trade: float = Field(default=0.01, gt=0, le=0.02)
    use_ml: bool = False
    compare_random_forest: bool = False
    transaction_cost_pips: float = Field(default=1.0, ge=0)
    slippage_pips: float = Field(default=0.5, ge=0)


class BacktestResponse(BaseModel):
    symbol: str
    use_ml: bool
    metrics: dict[str, Any]
    assumptions: dict[str, str]
    trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    model_report: dict[str, Any] | None = None
    disclaimer: str


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API. Tests pass an explicit ``Settings`` instance."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="AI-Powered Forex Market Analysis & Trading System",
        version=__version__,
        description=(
            "Research, analysis, backtesting, and paper-trading signals only. "
            "No endpoint places a live broker order.\n\n" + RESEARCH_DISCLAIMER
        ),
    )
    app.state.settings = settings

    @app.exception_handler(MarketDataError)
    async def market_data_error(request: Request, exc: MarketDataError) -> JSONResponse:
        logger.error("Market data error on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Request validation failed on %s.", request.url.path)
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", environment=settings.app_env, version=__version__)

    @app.get("/market/{symbol}", response_model=MarketResponse, tags=["market"])
    def market(
        symbol: str,
        period: str | None = None,
        interval: str | None = None,
        limit: int = Query(default=200, ge=1, le=2000),
        provider: MarketDataProvider = Depends(get_provider),
    ) -> MarketResponse:
        normalized = _symbol(symbol)
        chosen_period = _choice(period or settings.default_period, ALLOWED_PERIODS, "period")
        chosen_interval = _choice(interval or settings.default_interval, ALLOWED_INTERVALS, "interval")
        frame = provider.get_history(normalized, chosen_period, chosen_interval)
        tail = frame.tail(limit)
        candles = [
            Candle(
                timestamp=_time(row.timestamp),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for row in tail.itertuples(index=False)
        ]
        return MarketResponse(
            symbol=normalized,
            period=chosen_period,
            interval=chosen_interval,
            source="yahoo_finance",
            rows=len(frame),
            limitations=list(DATA_SOURCE_LIMITATIONS),
            candles=candles,
        )

    @app.get("/indicators/{symbol}", response_model=IndicatorResponse, tags=["market"])
    def indicators(
        symbol: str,
        period: str | None = None,
        interval: str | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        provider: MarketDataProvider = Depends(get_provider),
    ) -> IndicatorResponse:
        normalized = _symbol(symbol)
        frame = provider.get_history(
            normalized,
            _choice(period or settings.default_period, ALLOWED_PERIODS, "period"),
            _choice(interval or settings.default_interval, ALLOWED_INTERVALS, "interval"),
        )
        indicated = add_indicators(frame)
        ready = indicated.dropna(subset=["rsi_14", "macd", "atr_14", "sma_50"])
        if ready.empty:
            raise HTTPException(status_code=422, detail="Not enough history to compute indicators.")
        history = [_indicator_point(row) for _, row in ready.tail(limit).iterrows()]
        return IndicatorResponse(symbol=normalized, rows=len(ready), latest=history[-1], history=history)

    @app.get("/signal/{symbol}", response_model=SignalResponse, tags=["signal"])
    def signal(
        symbol: str,
        period: str | None = None,
        interval: str | None = None,
        provider: MarketDataProvider = Depends(get_provider),
    ) -> SignalResponse:
        normalized = _symbol(symbol)
        frame = provider.get_history(
            normalized,
            _choice(period or settings.default_period, ALLOWED_PERIODS, "period"),
            _choice(interval or settings.default_interval, ALLOWED_INTERVALS, "interval"),
        )
        try:
            scored = latest_signal(frame, normalized)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SignalResponse(
            symbol=scored.symbol,
            timestamp=_time(scored.timestamp),
            signal=scored.signal,
            technical_score=scored.technical_score,
            ml_probability=scored.ml_probability,
            confidence=scored.confidence,
            reasons=scored.reasons,
            disclaimer=RESEARCH_DISCLAIMER,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["model"])
    def predict(
        body: PredictRequest,
        provider: MarketDataProvider = Depends(get_provider),
    ) -> PredictResponse:
        symbol = _symbol(body.symbol)
        period = _choice(body.period, ALLOWED_PERIODS, "period")
        interval = _choice(body.interval, ALLOWED_INTERVALS, "interval")
        if not settings.model_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No trained model found. Run python scripts/train_model.py.",
            )
        try:
            bundle = load_model(settings.model_path)
        except ValueError as exc:
            logger.error("Model load failed: %s", exc)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        features = build_feature_frame(provider.get_history(symbol, period, interval))
        ready = features.dropna(subset=FEATURE_COLUMNS)
        if ready.empty:
            raise HTTPException(status_code=422, detail="Not enough history to build features.")
        row = ready.iloc[[-1]]
        scored = predict_probability(bundle["model"], bundle["scaler"], row).iloc[0]
        trained_through = bundle.get("trained_through")
        observed_at = pd.Timestamp(row["timestamp"].iloc[0])
        in_fit = bool(trained_through) and observed_at <= pd.Timestamp(trained_through)
        return PredictResponse(
            symbol=symbol,
            timestamp=_time(observed_at),
            probability_up=float(scored["probability_up"]),
            probability_down=float(scored["probability_down"]),
            predicted_direction=str(scored["predicted_direction"]),
            model_name=str(bundle.get("model_name", "unknown")),
            observation_was_in_fit=in_fit,
            disclaimer=RESEARCH_DISCLAIMER,
        )

    @app.post("/backtest", response_model=BacktestResponse, tags=["backtest"])
    def backtest(
        body: BacktestRequest,
        provider: MarketDataProvider = Depends(get_provider),
    ) -> BacktestResponse:
        symbol = _symbol(body.symbol)
        period = _choice(body.period, ALLOWED_PERIODS, "period")
        interval = _choice(body.interval, ALLOWED_INTERVALS, "interval")
        if body.risk_per_trade > settings.max_risk_per_trade:
            raise HTTPException(status_code=422, detail="risk_per_trade exceeds MAX_RISK_PER_TRADE.")
        frame = provider.get_history(symbol, period, interval)
        config = BacktestConfig(
            initial_capital=body.initial_capital,
            risk_per_trade=body.risk_per_trade,
            transaction_cost_pips=body.transaction_cost_pips,
            slippage_pips=body.slippage_pips,
        )
        manager = RiskManager(
            risk_per_trade=body.risk_per_trade,
            min_risk_reward=settings.min_risk_reward,
            max_risk_per_trade=settings.max_risk_per_trade,
            max_drawdown=settings.max_drawdown,
            max_leverage=settings.max_leverage,
        )
        try:
            result = run_research_backtest(
                frame,
                symbol=symbol,
                config=config,
                risk_manager=manager,
                use_ml=body.use_ml,
                compare_random_forest=body.compare_random_forest,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return BacktestResponse(
            symbol=symbol,
            use_ml=body.use_ml,
            metrics=result.metrics,
            assumptions=result.assumptions,
            trades=result.trades,
            equity_curve=result.equity_curve,
            model_report=result.model_report,
            disclaimer=RESEARCH_DISCLAIMER,
        )

    return app


def get_provider(request: Request) -> MarketDataProvider:
    """Build a provider rooted at the configured data directory."""
    return MarketDataProvider(cache_dir=request.app.state.settings.data_dir)


def _symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if normalized not in SUPPORTED_SYMBOLS:
        choices = ", ".join(SUPPORTED_SYMBOLS)
        raise HTTPException(status_code=422, detail=f"Unsupported symbol '{symbol}'. Choose one of: {choices}.")
    return normalized


def _choice(value: str, allowed: frozenset[str], name: str) -> str:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise HTTPException(status_code=422, detail=f"Unsupported {name} '{value}'. Choose one of: {choices}.")
    return value


def _time(value: object) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def _indicator_point(row: pd.Series) -> IndicatorPoint:
    payload: dict[str, Any] = {"timestamp": _time(row["timestamp"]), "close": float(row["close"])}
    for column in INDICATOR_FIELDS:
        if column in {"timestamp", "close"}:
            continue
        value = row[column]
        payload[column] = None if pd.isna(value) else float(value)
    return IndicatorPoint(**payload)


app = create_app()
