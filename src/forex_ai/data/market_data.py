"""Historical FX candles from Yahoo Finance via yfinance.

Limitations of this free source (also returned by the API):

* Unofficial FX history. Bars can be revised, delayed, or missing.
* Spot FX volume is often zero or not meaningful.
* There is no bid/ask, so backtest spread and slippage are assumptions.
* Intraday history is much shorter than daily history.
* This is not a broker feed and cannot be used for live execution.
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from forex_ai.data.preprocessing import clean_ohlcv

logger = logging.getLogger(__name__)

SUPPORTED_SYMBOLS: tuple[str, ...] = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
)
ALLOWED_PERIODS: frozenset[str] = frozenset({"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"})
ALLOWED_INTERVALS: frozenset[str] = frozenset({"1d", "1h", "1wk"})

DATA_SOURCE_LIMITATIONS: tuple[str, ...] = (
    "Yahoo Finance FX candles are an unofficial free source and can contain gaps or revisions.",
    "Spot FX volume from this source is often missing or not meaningful.",
    "The feed has no bid/ask, so transaction costs in the backtest are assumptions.",
    "Intraday history is limited; prefer 1d for a more complete sample.",
    "This data is for research only and is not a broker execution feed.",
)


class MarketDataError(Exception):
    """Raised when candles cannot be downloaded or fail validation."""


def yahoo_ticker(symbol: str) -> str:
    """Map a project symbol such as ``EURUSD`` to Yahoo's ``EURUSD=X`` ticker."""
    return f"{symbol.upper()}=X"


def pip_size(symbol: str) -> float:
    """Return the standard pip size. JPY pairs use 0.01; other majors use 0.0001."""
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def quote_to_account(symbol: str, price: float) -> float:
    """Convert one unit of quote-currency PnL into USD at ``price``.

    XXXUSD pairs are already quoted in USD. USDXXX pairs (USDJPY, USDCAD, USDCHF)
    are converted with ``1 / price``, fixed by the caller for the life of a trade.
    """
    normalized = symbol.upper()
    if normalized.endswith("USD"):
        return 1.0
    if normalized.startswith("USD"):
        if price <= 0:
            raise MarketDataError("Price must be positive to convert a USD-base pair.")
        return 1.0 / price
    raise MarketDataError(f"No USD conversion is defined for {symbol}.")


def _download_yahoo(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download one ticker. Isolated so tests can replace it."""
    kwargs = {
        "tickers": ticker,
        "period": period,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
        "threads": False,
    }
    try:
        try:
            frame = yf.download(**kwargs, multi_level_index=False)
        except TypeError:
            frame = yf.download(**kwargs)
    except Exception as exc:
        raise MarketDataError(
            f"Yahoo Finance download failed for {ticker} ({period}, {interval}): {exc}"
        ) from exc
    if frame is None or frame.empty:
        raise MarketDataError(
            f"Yahoo Finance returned no rows for {ticker} ({period}, {interval}). "
            "Intraday intervals have a short history; try interval=1d."
        )
    return frame


class MarketDataProvider:
    """Download, cache, and normalize historical OHLCV candles."""

    def __init__(self, cache_dir: Path | None = Path("data")) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_history(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return clean candles for a supported symbol.

        Args:
            symbol: One of the seven configured majors, case-insensitive.
            period: Yahoo period such as ``2y``.
            interval: ``1d``, ``1h``, or ``1wk``.
            refresh: When False, a valid cache file is used before downloading.

        Raises:
            MarketDataError: The symbol, period, or interval is unsupported, or
                the download and the cache both fail.
        """
        normalized = self._validate(symbol, period, interval)
        cache_path = self._cache_path(normalized, period, interval)
        if cache_path is not None and cache_path.exists() and not refresh:
            logger.info("Loading cached market data for %s from %s.", normalized, cache_path.name)
            try:
                return clean_ohlcv(pd.read_csv(cache_path))
            except (OSError, ValueError) as exc:
                logger.warning("Cache for %s was unreadable (%s). Downloading again.", normalized, exc)

        logger.info(
            "Downloading %s (%s, %s) from Yahoo Finance. Volume may be unusable and there is no bid/ask.",
            normalized,
            period,
            interval,
        )
        try:
            raw = _download_yahoo(yahoo_ticker(normalized), period, interval)
            clean = clean_ohlcv(raw)
        except MarketDataError:
            return self._fallback_cache(cache_path, normalized)
        except ValueError as exc:
            if cache_path is not None and cache_path.exists():
                logger.warning("Download for %s failed validation. Using cache.", normalized)
                return self._fallback_cache(cache_path, normalized)
            raise MarketDataError(str(exc)) from exc

        if cache_path is not None:
            clean.to_csv(cache_path, index=False)
        logger.info("Loaded %s rows for %s %s %s.", len(clean), normalized, period, interval)
        return clean

    def _validate(self, symbol: str, period: str, interval: str) -> str:
        normalized = symbol.strip().upper()
        if normalized not in SUPPORTED_SYMBOLS:
            choices = ", ".join(SUPPORTED_SYMBOLS)
            raise MarketDataError(f"Unsupported symbol '{symbol}'. Choose one of: {choices}.")
        if period not in ALLOWED_PERIODS:
            choices = ", ".join(sorted(ALLOWED_PERIODS))
            raise MarketDataError(f"Unsupported period '{period}'. Choose one of: {choices}.")
        if interval not in ALLOWED_INTERVALS:
            choices = ", ".join(sorted(ALLOWED_INTERVALS))
            raise MarketDataError(f"Unsupported interval '{interval}'. Choose one of: {choices}.")
        return normalized

    def _cache_path(self, symbol: str, period: str, interval: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{symbol}_{period}_{interval}.csv"

    def _fallback_cache(self, cache_path: Path | None, symbol: str) -> pd.DataFrame:
        if cache_path is not None and cache_path.exists():
            logger.warning("Using cached candles for %s after a download failure.", symbol)
            try:
                return clean_ohlcv(pd.read_csv(cache_path))
            except (OSError, ValueError) as exc:
                raise MarketDataError(f"Download failed and the cache for {symbol} is unusable.") from exc
        raise MarketDataError(f"No market data is available for {symbol}.")
