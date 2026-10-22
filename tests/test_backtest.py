"""Execution assumptions and performance math."""

import numpy as np
import pandas as pd
import pytest

from forex_ai.backtesting.engine import BacktestConfig, run_backtest
from forex_ai.backtesting.metrics import compute_metrics
from tests.conftest import make_ohlcv


def _frame(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "signal"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["volume"] = 1_000.0
    return frame


def _config(**overrides) -> BacktestConfig:
    values = {
        "initial_capital": 10_000.0,
        "risk_per_trade": 0.01,
        "transaction_cost_pips": 0.0,
        "slippage_pips": 0.0,
        "stop_pips": 50.0,
        "reward_risk_ratio": 2.0,
    }
    values.update(overrides)
    return BacktestConfig(**values)


def test_signal_fills_on_the_next_open_not_the_signal_close() -> None:
    frame = _frame(
        [
            ("2024-01-01", 1.49, 1.50, 1.48, 1.50, "BUY"),
            ("2024-01-02", 1.10, 1.101, 1.099, 1.1005, "HOLD"),
            ("2024-01-03", 1.101, 1.102, 1.100, 1.1015, "HOLD"),
            ("2024-01-04", 1.102, 1.103, 1.101, 1.1018, "SELL"),
            ("2024-01-05", 1.103, 1.104, 1.102, 1.1035, "HOLD"),
        ]
    )
    result = run_backtest(frame, symbol="EURUSD", config=_config())
    first = result.trades[0]
    assert first["entry_time"].startswith("2024-01-02")
    assert first["exit_time"].startswith("2024-01-05")
    assert first["entry_price"] == pytest.approx(1.10)
    assert first["exit_price"] == pytest.approx(1.103)
    assert first["exit_reason"] == "signal"
    assert first["entry_price"] != frame["close"].iloc[0]
    # 50-pip stop, 1% of 10,000, EURUSD pip value 0.0001 -> 20,000 units.
    assert first["units"] == pytest.approx(20_000)
    assert first["pnl"] == pytest.approx(20_000 * (1.103 - 1.10))


def test_take_profit_and_same_bar_stop_priority() -> None:
    winner = _frame(
        [
            ("2024-01-01", 1.10, 1.10, 1.10, 1.10, "BUY"),
            ("2024-01-02", 1.10, 1.101, 1.099, 1.1005, "HOLD"),
            ("2024-01-03", 1.101, 1.112, 1.100, 1.111, "HOLD"),
        ]
    )
    won = run_backtest(winner, symbol="EURUSD", config=_config())
    assert won.trades[0]["exit_reason"] == "take_profit"
    assert won.trades[0]["exit_price"] == pytest.approx(1.11)
    assert won.metrics["final_capital"] == pytest.approx(10_200)
    assert won.metrics["total_return"] == pytest.approx(0.02)
    assert won.metrics["profit_factor"] is None

    both = _frame(
        [
            ("2024-01-01", 1.10, 1.10, 1.10, 1.10, "BUY"),
            ("2024-01-02", 1.10, 1.12, 1.09, 1.11, "HOLD"),
        ]
    )
    stopped = run_backtest(both, symbol="EURUSD", config=_config())
    assert stopped.trades[0]["exit_reason"] == "stop"
    assert stopped.trades[0]["pnl"] == pytest.approx(-100)
    assert stopped.metrics["final_capital"] == pytest.approx(9_900)


def test_cost_worsens_the_entry_fill() -> None:
    frame = _frame(
        [
            ("2024-01-01", 1.10, 1.10, 1.10, 1.10, "BUY"),
            ("2024-01-02", 1.10, 1.101, 1.099, 1.1005, "HOLD"),
            ("2024-01-03", 1.101, 1.102, 1.100, 1.101, "SELL"),
            ("2024-01-04", 1.102, 1.103, 1.101, 1.102, "HOLD"),
        ]
    )
    result = run_backtest(
        frame,
        symbol="EURUSD",
        config=_config(transaction_cost_pips=1.0, slippage_pips=0.0),
    )
    assert result.trades[0]["entry_price"] == pytest.approx(1.10 + 0.0001)


def test_hold_only_frame_does_not_invent_trades() -> None:
    frame = make_ohlcv(5, seed=1)
    frame["signal"] = "HOLD"
    result = run_backtest(frame, symbol="EURUSD", config=_config())
    assert result.metrics["number_of_trades"] == 0
    assert result.metrics["final_capital"] == pytest.approx(10_000)
    assert result.metrics["sharpe_ratio"] is None
    assert result.metrics["win_rate"] == 0.0


def test_metric_drawdown_profit_factor_and_sharpe() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    equity = pd.Series([10_000.0, 11_000.0, 9_000.0], index=index)
    trades = pd.DataFrame({"pnl": [1_000.0, -2_000.0]})
    metrics = compute_metrics(equity, trades, 10_000, periods_per_year=252)
    assert metrics["maximum_drawdown"] == pytest.approx((11_000 - 9_000) / 11_000)
    assert metrics["profit_factor"] == pytest.approx(0.5)
    assert metrics["average_loss"] == pytest.approx(2_000)
    assert metrics["win_rate"] == pytest.approx(0.5)
    returns = equity.pct_change().dropna()
    expected = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    assert metrics["sharpe_ratio"] == pytest.approx(expected)

    flat = pd.Series([100.0, 100.0, 100.0], index=index)
    assert compute_metrics(flat, pd.DataFrame({"pnl": []}), 100, 252)["sharpe_ratio"] is None
