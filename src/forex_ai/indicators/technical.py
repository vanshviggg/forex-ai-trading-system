"""Pandas implementations of common indicators.

Every window ends at the current bar. Nothing here is centered, and nothing
reads a future row. Leading NaNs are left in place so later stages can drop
incomplete warmup rows instead of inventing zeros.
"""

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average. The first ``window - 1`` values are NaN."""
    _require_window(window)
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average seeded with the SMA of the first ``span`` points.

    ``alpha = 2 / (span + 1)``. Leading values before the seed are NaN.
    """
    _require_window(span)
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    start = _first_finite(values)
    if start is None or len(values) - start < span:
        return pd.Series(out, index=series.index)
    segment = values[start:]
    if not np.isfinite(segment).all():
        # Internal gaps stay NaN from the first gap onward rather than being filled.
        gap = int(np.argmax(~np.isfinite(segment)))
        segment = segment[:gap]
    if len(segment) < span:
        return pd.Series(out, index=series.index)
    alpha = 2.0 / (span + 1.0)
    seeded = np.full(len(segment), np.nan)
    seeded[span - 1] = float(np.mean(segment[:span]))
    for index in range(span, len(segment)):
        seeded[index] = alpha * segment[index] + (1.0 - alpha) * seeded[index - 1]
    out[start : start + len(segment)] = seeded
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI.

    The first average is the mean of the first ``period`` changes. Later values
    use Wilder smoothing. A flat window is 50, a pure-gain window is 100, and a
    pure-loss window is 0.
    """
    _require_window(period)
    values = close.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) <= period:
        return pd.Series(result, index=close.index)
    delta = np.diff(values, prepend=np.nan)
    gain = np.where(np.isnan(delta), np.nan, np.clip(delta, 0, None))
    loss = np.where(np.isnan(delta), np.nan, np.clip(-delta, 0, None))
    avg_gain = np.full(len(values), np.nan)
    avg_loss = np.full(len(values), np.nan)
    avg_gain[period] = np.mean(gain[1 : period + 1])
    avg_loss[period] = np.mean(loss[1 : period + 1])
    for index in range(period + 1, len(values)):
        avg_gain[index] = (avg_gain[index - 1] * (period - 1) + gain[index]) / period
        avg_loss[index] = (avg_loss[index - 1] * (period - 1) + loss[index]) / period

    valid = np.isfinite(avg_gain) & np.isfinite(avg_loss)
    both_zero = valid & (avg_gain == 0) & (avg_loss == 0)
    loss_zero = valid & (avg_loss == 0) & (avg_gain > 0)
    gain_zero = valid & (avg_gain == 0) & (avg_loss > 0)
    normal = valid & (avg_gain > 0) & (avg_loss > 0)
    result[both_zero] = 50.0
    result[loss_zero] = 100.0
    result[gain_zero] = 0.0
    result[normal] = 100.0 - (100.0 / (1.0 + avg_gain[normal] / avg_loss[normal]))
    return pd.Series(result, index=close.index)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram (MACD minus signal)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal_span)
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        },
        index=close.index,
    )


def average_true_range(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR.

    True range is the greatest of the bar range, the gap from the previous
    close to the high, and the gap from the previous close to the low.
    The seed is the mean of the first ``period`` true ranges that have a previous close.
    """
    _require_window(period)
    ranges = true_range(frame).to_numpy(dtype=float)
    atr = np.full(len(frame), np.nan)
    if len(frame) <= period:
        return pd.Series(atr, index=frame.index)
    window = ranges[1 : period + 1]
    if len(window) < period or not np.isfinite(window).all():
        return pd.Series(atr, index=frame.index)
    atr[period] = float(np.mean(window))
    for index in range(period + 1, len(frame)):
        atr[index] = (atr[index - 1] * (period - 1) + ranges[index]) / period
    return pd.Series(atr, index=frame.index)


def true_range(frame: pd.DataFrame) -> pd.Series:
    """True range of each bar, including the gap from the previous close."""
    prev_close = frame["close"].shift(1)
    parts = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands using the population standard deviation (``ddof=0``)."""
    _require_window(window)
    mid = sma(close, window)
    deviation = close.rolling(window=window, min_periods=window).std(ddof=0)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": mid + num_std * deviation,
            "bb_lower": mid - num_std * deviation,
        },
        index=close.index,
    )


def rolling_volatility(log_return: pd.Series, window: int = 20) -> pd.Series:
    """Rolling sample standard deviation of log returns. Not annualized."""
    _require_window(window)
    return log_return.rolling(window=window, min_periods=window).std(ddof=1)


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with indicator columns appended.

    Raises:
        ValueError: OHLC columns are missing.
    """
    missing = [column for column in ("open", "high", "low", "close") if column not in frame.columns]
    if missing:
        raise ValueError(f"Indicator input is missing columns: {missing}.")
    out = frame.copy()
    close = out["close"].astype(float)
    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["ema_20"] = ema(close, 20)
    out["ema_50"] = ema(close, 50)
    out["rsi_14"] = rsi(close, 14)
    out = pd.concat([out, macd(close)], axis=1)
    out["atr_14"] = average_true_range(out, 14)
    out = pd.concat([out, bollinger_bands(close)], axis=1)
    out["pct_return"] = close.pct_change()
    out["log_return"] = np.log(close / close.shift(1))
    out["volatility_20"] = rolling_volatility(out["log_return"], 20)
    out["ema20_distance"] = (close - out["ema_20"]) / out["ema_20"]
    out["ema50_distance"] = (close - out["ema_50"]) / out["ema_50"]
    out["support_20"] = out["low"].rolling(window=20, min_periods=20).min()
    out["resistance_20"] = out["high"].rolling(window=20, min_periods=20).max()
    if "volume" in out.columns:
        volume = pd.to_numeric(out["volume"], errors="coerce")
        change = volume.pct_change().replace([np.inf, -np.inf], np.nan)
        # Yahoo FX volume is often entirely zero. Keep the column, but do not
        # force downstream dropna logic to discard every row because of it.
        out["volume_change"] = change.fillna(0.0)
    else:
        out["volume_change"] = 0.0
    return out


def _require_window(window: int) -> None:
    if window <= 0:
        raise ValueError("Indicator window must be positive.")


def _first_finite(values: np.ndarray) -> int | None:
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) == 0:
        return None
    return int(finite[0])
