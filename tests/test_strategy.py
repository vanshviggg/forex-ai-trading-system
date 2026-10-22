"""Deterministic signal rules."""

import pandas as pd
from tests.conftest import make_ohlcv

from forex_ai.strategies.signal_engine import score_observation, signals_for_frame

BULLISH = {
    "close": 1.2,
    "ema_20": 1.1,
    "ema_50": 1.0,
    "sma_20": 1.1,
    "sma_50": 1.0,
    "rsi_14": 60.0,
    "macd": 0.01,
    "macd_signal": 0.005,
    "macd_hist": 0.005,
    "atr_14": 0.001,
}


def test_aligned_trend_is_buy_and_its_mirror_is_sell() -> None:
    bullish = score_observation(**BULLISH)
    assert bullish["signal"] == "BUY"
    assert bullish["technical_score"] == 1.0
    assert "EMA20 above EMA50" in bullish["reasons"]
    assert "MACD bullish" in bullish["reasons"]
    assert "RSI above 50" in bullish["reasons"]

    bearish = score_observation(
        close=1.0,
        ema_20=1.1,
        ema_50=1.2,
        sma_20=1.1,
        sma_50=1.2,
        rsi_14=40,
        macd=-0.01,
        macd_signal=0.0,
        macd_hist=-0.01,
        atr_14=0.001,
    )
    assert bearish["signal"] == "SELL"
    assert bearish["technical_score"] == 0.0


def test_mixed_evidence_holds_until_ml_agrees() -> None:
    mixed = score_observation(
        close=1.2,
        ema_20=1.1,
        ema_50=1.0,
        sma_20=1.1,
        sma_50=1.0,
        rsi_14=40,
        macd=-0.01,
        macd_signal=0.0,
        macd_hist=-0.01,
        atr_14=0.001,
    )
    assert mixed["signal"] == "HOLD"
    assert mixed["technical_score"] == 0.6

    agreed = score_observation(**{**mixed_inputs(), "ml_probability": 0.9})
    assert agreed["signal"] == "BUY"
    assert agreed["confidence"] == 0.72
    assert "not profit probability" in " ".join(agreed["reasons"])

    conflict = score_observation(**{**mixed_inputs(), "ml_probability": 0.4})
    assert conflict["signal"] == "HOLD"


def test_elevated_volatility_reduces_the_score() -> None:
    scored = score_observation(**{**BULLISH, "atr_14": 0.03})
    assert scored["technical_score"] == 0.75
    assert any("volatility" in reason.lower() for reason in scored["reasons"])


def test_earlier_signals_ignore_a_later_close() -> None:
    frame = make_ohlcv(90, seed=9)
    original = signals_for_frame(frame)
    changed = frame.copy()
    changed.loc[changed.index[-1], "close"] *= 1.4
    changed.loc[changed.index[-1], "high"] = max(
        changed.loc[changed.index[-1], "high"],
        changed.loc[changed.index[-1], "close"],
    )
    updated = signals_for_frame(changed)
    assert original["signal"].iloc[:-1].tolist() == updated["signal"].iloc[:-1].tolist()


def mixed_inputs() -> dict[str, float]:
    return {
        "close": 1.2,
        "ema_20": 1.1,
        "ema_50": 1.0,
        "sma_20": 1.1,
        "sma_50": 1.0,
        "rsi_14": 40.0,
        "macd": -0.01,
        "macd_signal": 0.0,
        "macd_hist": -0.01,
        "atr_14": 0.001,
    }


def test_signal_frame_uses_only_known_labels() -> None:
    frame = make_ohlcv(30, seed=1)
    signals = signals_for_frame(frame)
    assert set(signals["signal"]).issubset({"BUY", "SELL", "HOLD"})
    assert signals["signal"].iloc[0] == "HOLD"
    assert pd.notna(signals["technical_score"]).all()
