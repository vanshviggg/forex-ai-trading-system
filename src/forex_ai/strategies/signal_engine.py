"""Rules that turn the latest known indicators into BUY, SELL, or HOLD.

``technical_score`` and ``confidence`` live in ``[0, 1]``. A value near 1 is a
strong bullish score and a value near 0 is a strong bearish score. Neither
number is a probability that a trade will make money. When an ML probability
is supplied, confidence is ``0.6 * technical_score + 0.4 * ml_probability``.
Disagreement between the two keeps the signal at HOLD.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forex_ai.indicators.technical import add_indicators

WEIGHTS = {
    "ema_trend": 0.35,
    "price_vs_ma": 0.25,
    "rsi": 0.20,
    "macd": 0.20,
}
BUY_THRESHOLD = 0.62
SELL_THRESHOLD = 0.38
BLENDED_BUY_THRESHOLD = 0.60
BLENDED_SELL_THRESHOLD = 0.40
TECHNICAL_AGREE_BUY = 0.55
TECHNICAL_AGREE_SELL = 0.45
ML_AGREE_BUY = 0.55
ML_AGREE_SELL = 0.45
VOLATILITY_DAMPEN_ABOVE = 0.01
TECHNICAL_BLEND = 0.60
ML_BLEND = 0.40

_READY_COLUMNS = (
    "close",
    "ema_20",
    "ema_50",
    "sma_20",
    "sma_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
)


@dataclass(frozen=True)
class Signal:
    """One research signal. ``confidence`` is a model score, not expected profit."""

    symbol: str
    timestamp: pd.Timestamp
    signal: str
    technical_score: float
    ml_probability: float | None
    confidence: float
    reasons: list[str]


def score_observation(
    *,
    close: float,
    ema_20: float,
    ema_50: float,
    sma_20: float,
    sma_50: float,
    rsi_14: float,
    macd: float,
    macd_signal: float,
    macd_hist: float,
    atr_14: float,
    ml_probability: float | None = None,
) -> dict[str, object]:
    """Score one bar from values already known at that bar's close."""
    reasons: list[str] = []
    ema_component = _sign(ema_20 - ema_50)
    if ema_component > 0:
        reasons.append("EMA20 above EMA50")
    elif ema_component < 0:
        reasons.append("EMA20 below EMA50")

    if close > ema_20 and close > sma_20 and close > sma_50:
        price_component = 1.0
        reasons.append("Price above EMA20, SMA20, and SMA50")
    elif close < ema_20 and close < sma_20 and close < sma_50:
        price_component = -1.0
        reasons.append("Price below EMA20, SMA20, and SMA50")
    else:
        price_component = 0.0

    if rsi_14 >= 70:
        rsi_component = -0.5
        reasons.append("RSI overbought")
    elif rsi_14 <= 30:
        rsi_component = 0.5
        reasons.append("RSI oversold")
    elif rsi_14 > 50:
        rsi_component = 1.0
        reasons.append("RSI above 50")
    elif rsi_14 < 50:
        rsi_component = -1.0
        reasons.append("RSI below 50")
    else:
        rsi_component = 0.0

    if macd > macd_signal and macd_hist > 0:
        macd_component = 1.0
        reasons.append("MACD bullish")
    elif macd < macd_signal and macd_hist < 0:
        macd_component = -1.0
        reasons.append("MACD bearish")
    else:
        macd_component = 0.0

    raw = (
        WEIGHTS["ema_trend"] * ema_component
        + WEIGHTS["price_vs_ma"] * price_component
        + WEIGHTS["rsi"] * rsi_component
        + WEIGHTS["macd"] * macd_component
    )
    if close > 0 and atr_14 / close > VOLATILITY_DAMPEN_ABOVE:
        raw *= 0.5
        reasons.append("Elevated volatility reduces conviction")

    technical_score = _clip01((raw + 1.0) / 2.0)
    probability = _clean_probability(ml_probability)
    if probability is None:
        confidence = technical_score
        if technical_score >= BUY_THRESHOLD:
            signal = "BUY"
            reasons.append("Bullish score cleared the buy threshold")
        elif technical_score <= SELL_THRESHOLD:
            signal = "SELL"
            reasons.append("Bearish score cleared the sell threshold")
        else:
            signal = "HOLD"
            reasons.append("Score is inside the hold band")
    else:
        confidence = _clip01(TECHNICAL_BLEND * technical_score + ML_BLEND * probability)
        reasons.append(
            f"ML upward probability is {probability:.2f} (classification score, not profit probability)"
        )
        bullish = (
            technical_score >= TECHNICAL_AGREE_BUY
            and probability >= ML_AGREE_BUY
            and confidence >= BLENDED_BUY_THRESHOLD
        )
        bearish = (
            technical_score <= TECHNICAL_AGREE_SELL
            and probability <= ML_AGREE_SELL
            and confidence <= BLENDED_SELL_THRESHOLD
        )
        if bullish:
            signal = "BUY"
            reasons.append("Technical score and ML probability agree on a bullish score")
        elif bearish:
            signal = "SELL"
            reasons.append("Technical score and ML probability agree on a bearish score")
        else:
            signal = "HOLD"
            reasons.append("Technical score and ML probability disagree or sit near 0.5")

    return {
        "signal": signal,
        "technical_score": round(technical_score, 6),
        "ml_probability": None if probability is None else round(probability, 6),
        "confidence": round(confidence, 6),
        "reasons": reasons,
    }


def signals_for_frame(
    frame: pd.DataFrame,
    ml_probabilities: pd.Series | None = None,
) -> pd.DataFrame:
    """Attach a causal signal to every row.

    Row ``t`` uses indicators at ``t`` only. Early rows without a full warmup
    are HOLD. Changing a future close does not change earlier signals.
    """
    indicated = add_indicators(frame)
    signals: list[str] = []
    scores: list[float] = []
    probabilities: list[float | None] = []
    confidences: list[float] = []
    reasons: list[str] = []
    for index, row in indicated.iterrows():
        probability = _lookup_probability(ml_probabilities, index)
        if not _row_ready(row):
            signals.append("HOLD")
            scores.append(0.5)
            probabilities.append(probability)
            confidences.append(0.5)
            reasons.append("Insufficient indicator history")
            continue
        scored = score_observation(
            close=float(row["close"]),
            ema_20=float(row["ema_20"]),
            ema_50=float(row["ema_50"]),
            sma_20=float(row["sma_20"]),
            sma_50=float(row["sma_50"]),
            rsi_14=float(row["rsi_14"]),
            macd=float(row["macd"]),
            macd_signal=float(row["macd_signal"]),
            macd_hist=float(row["macd_hist"]),
            atr_14=float(row["atr_14"]),
            ml_probability=probability,
        )
        signals.append(str(scored["signal"]))
        scores.append(float(scored["technical_score"]))
        probabilities.append(scored["ml_probability"])  # type: ignore[arg-type]
        confidences.append(float(scored["confidence"]))
        reasons.append(" | ".join(scored["reasons"]))  # type: ignore[arg-type]
    out = indicated.copy()
    out["signal"] = signals
    out["technical_score"] = scores
    out["ml_probability"] = probabilities
    out["confidence"] = confidences
    out["reasons"] = reasons
    return out


def latest_signal(
    frame: pd.DataFrame,
    symbol: str,
    ml_probability: float | None = None,
) -> Signal:
    """Score the latest row that has a complete indicator set."""
    scored = signals_for_frame(frame)
    ready = scored.dropna(subset=list(_READY_COLUMNS))
    if ready.empty:
        raise ValueError("Not enough history to compute a signal.")
    row = ready.iloc[-1]
    # Re-score so a caller-supplied probability is applied to the latest bar only.
    result = score_observation(
        close=float(row["close"]),
        ema_20=float(row["ema_20"]),
        ema_50=float(row["ema_50"]),
        sma_20=float(row["sma_20"]),
        sma_50=float(row["sma_50"]),
        rsi_14=float(row["rsi_14"]),
        macd=float(row["macd"]),
        macd_signal=float(row["macd_signal"]),
        macd_hist=float(row["macd_hist"]),
        atr_14=float(row["atr_14"]),
        ml_probability=ml_probability,
    )
    return Signal(
        symbol=symbol.upper(),
        timestamp=pd.Timestamp(row["timestamp"]),
        signal=str(result["signal"]),
        technical_score=float(result["technical_score"]),
        ml_probability=result["ml_probability"],  # type: ignore[arg-type]
        confidence=float(result["confidence"]),
        reasons=list(result["reasons"]),  # type: ignore[arg-type]
    )


def _row_ready(row: pd.Series) -> bool:
    for column in _READY_COLUMNS:
        value = row[column]
        if pd.isna(value) or not np.isfinite(float(value)):
            return False
    return float(row["ema_20"]) != 0 and float(row["ema_50"]) != 0


def _lookup_probability(probabilities: pd.Series | None, index: object) -> float | None:
    if probabilities is None or index not in probabilities.index:
        return None
    value = probabilities.loc[index]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _clean_probability(value: float | None) -> float | None:
    if value is None or not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
        return None
    return float(value)


def _sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
