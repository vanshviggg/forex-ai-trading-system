"""Indicator formulas and the rule that bar t ignores future prices."""

import numpy as np
import pandas as pd
import pytest

from forex_ai.indicators.technical import (
    add_indicators,
    average_true_range,
    bollinger_bands,
    ema,
    rsi,
    sma,
    true_range,
)
from tests.conftest import make_ohlcv


def test_sma_and_ema_match_hand_calculation() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    averages = sma(close, 3)
    assert np.isnan(averages.iloc[0]) and np.isnan(averages.iloc[1])
    assert averages.iloc[2:].tolist() == [2.0, 3.0, 4.0]

    smoothed = ema(close, 3)
    assert smoothed.iloc[2] == 2.0
    assert smoothed.iloc[3] == 3.0
    assert smoothed.iloc[4] == 4.0


def test_rsi_extremes_and_flat_market() -> None:
    rising = rsi(pd.Series(np.arange(1, 21, dtype=float)), 14)
    falling = rsi(pd.Series(np.arange(20, 0, -1, dtype=float)), 14)
    flat = rsi(pd.Series(np.full(20, 5.0)), 14)
    assert rising.iloc[:14].isna().all()
    assert rising.iloc[14] == 100.0
    assert falling.iloc[14] == 0.0
    assert flat.iloc[14] == 50.0


def test_true_range_uses_previous_close() -> None:
    frame = pd.DataFrame({"high": [12.0], "low": [9.0], "close": [11.0]})
    # No previous close, so true range is the bar range.
    assert true_range(frame).iloc[0] == 3.0
    frame = pd.DataFrame(
        {
            "high": [10.0, 12.0],
            "low": [9.0, 9.0],
            "close": [10.0, 11.0],
        }
    )
    assert true_range(frame).iloc[1] == 3.0


def test_atr_seed_is_the_mean_of_the_first_complete_true_ranges() -> None:
    frame = make_ohlcv(30, seed=2)
    period = 14
    ranges = true_range(frame)
    atr = average_true_range(frame, period)
    assert np.isnan(atr.iloc[period - 1])
    assert atr.iloc[period] == ranges.iloc[1 : period + 1].mean()


def test_bollinger_bands_use_population_std() -> None:
    close = pd.Series([1.0, 2.0, 3.0])
    bands = bollinger_bands(close, window=2)
    assert bands["bb_mid"].iloc[1] == 1.5
    # Population std of [1, 2] is 0.5, so the 2-std band is 1.5 ± 1.0.
    assert bands["bb_upper"].iloc[1] == 2.5
    assert bands["bb_lower"].iloc[1] == 0.5


def test_support_resistance_returns_and_volume_change() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
            "open": [10.0, 10.0, 11.0, 12.0],
            "high": [10.5, 11.0, 12.0, 12.5],
            "low": [9.5, 9.0, 10.5, 11.0],
            "close": [10.0, 11.0, 12.0, 12.0],
            "volume": [10.0, 20.0, 0.0, 0.0],
        }
    )
    out = add_indicators(frame)
    assert out["pct_return"].iloc[1] == pytest.approx(0.1)
    assert out["log_return"].iloc[1] == pytest.approx(np.log(1.1))
    assert out["volume_change"].iloc[1] == 1.0
    assert np.isfinite(out["volume_change"]).all()


def test_support_and_resistance_are_trailing_windows() -> None:
    frame = make_ohlcv(30, seed=8)
    out = add_indicators(frame)
    assert out["support_20"].iloc[19] == frame["low"].iloc[:20].min()
    assert out["resistance_20"].iloc[19] == frame["high"].iloc[:20].max()


def test_indicators_do_not_use_the_next_close() -> None:
    frame = make_ohlcv(80, seed=4)
    original = add_indicators(frame)
    changed = frame.copy()
    changed.loc[changed.index[-1], "close"] = changed.loc[changed.index[-1], "close"] * 1.5
    changed.loc[changed.index[-1], "high"] = max(
        changed.loc[changed.index[-1], "high"],
        changed.loc[changed.index[-1], "close"],
    )
    updated = add_indicators(changed)
    columns = ["sma_20", "ema_20", "rsi_14", "macd", "atr_14", "bb_upper"]
    pd.testing.assert_frame_equal(original.iloc[:-1][columns], updated.iloc[:-1][columns])


def test_warmup_values_stay_missing_instead_of_zero() -> None:
    out = add_indicators(make_ohlcv(80, seed=5))
    assert out["sma_20"].iloc[:19].isna().all()
    assert pd.notna(out["ema_50"].iloc[49])
    assert out.loc[60:, ["rsi_14", "macd_hist", "atr_14", "volatility_20"]].notna().all().all()
