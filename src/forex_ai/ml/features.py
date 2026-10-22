"""Feature matrix and next-bar direction labels.

Features at row ``t`` are computed only from prices at ``t`` and earlier.
``target`` is the only column that looks one bar ahead, and it is never a
model input. The final row has no label because the next close is unknown.
"""

import numpy as np
import pandas as pd

from forex_ai.indicators.technical import add_indicators

FEATURE_COLUMNS: list[str] = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema_20",
    "ema_50",
    "sma_20",
    "sma_50",
    "atr_14",
    "bb_upper",
    "bb_lower",
    "volatility_20",
    "pct_return",
    "log_return",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "ema20_distance",
    "price_ema20_ratio",
    "price_ema50_ratio",
]


def assert_chronological(frame: pd.DataFrame) -> None:
    """Refuse unsorted or duplicate timestamps.

    Time-series rows are never shuffled back into order here. Callers must
    pass a clean ascending series so a leaked future row cannot be hidden.
    """
    if "timestamp" not in frame.columns:
        raise ValueError("timestamp column is required.")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    if timestamps.duplicated().any():
        raise ValueError("Duplicate timestamps are not allowed in model training.")
    if not bool(timestamps.is_monotonic_increasing):
        raise ValueError(
            "Time series must already be sorted ascending. Unsorted rows are rejected."
        )


def make_target(close: pd.Series) -> pd.Series:
    """Label 1 when the next close is higher, else 0. The last label is NaN.

    Equal closes are labeled 0 (not an upward move). This series is a label,
    not a feature.
    """
    future = close.shift(-1)
    target = pd.Series(np.nan, index=close.index, dtype=float)
    comparable = future.notna()
    target.loc[comparable] = (future.loc[comparable] > close.loc[comparable]).astype(float)
    return target


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add indicators, lagged returns, price/EMA ratios, and the next-bar target."""
    if "timestamp" in frame.columns:
        assert_chronological(frame)
    out = add_indicators(frame)
    out["return_lag_1"] = out["pct_return"].shift(1)
    out["return_lag_2"] = out["pct_return"].shift(2)
    out["return_lag_3"] = out["pct_return"].shift(3)
    out["price_ema20_ratio"] = out["close"] / out["ema_20"]
    out["price_ema50_ratio"] = out["close"] / out["ema_50"]
    out["target"] = make_target(out["close"])
    return out


def supervised_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose features and next-bar label are all known.

    Warmup rows and the final unlabeled row are dropped. Nothing is imputed.
    """
    built = build_feature_frame(frame)
    subset = FEATURE_COLUMNS + ["target"]
    clean = built.dropna(subset=subset).copy()
    clean["target"] = clean["target"].astype(int)
    if clean.empty:
        raise ValueError("No supervised rows remained after dropping indicator warmup.")
    return clean
