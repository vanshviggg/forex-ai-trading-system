"""Shared deterministic candles for unit tests."""

import numpy as np
import pandas as pd


def make_ohlcv(n: int = 160, start: float = 1.10, seed: int = 7) -> pd.DataFrame:
    """Build a reproducible OHLCV frame. This is not a market sample."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.002, n)
    close = start * np.exp(np.cumsum(returns))
    open_ = np.empty(n)
    open_[0] = start
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = rng.integers(100, 1000, n).astype(float)
    timestamps = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
