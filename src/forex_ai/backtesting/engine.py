"""Event-driven backtest.

Assumptions, also returned with every result:

* A signal on bar ``t`` is filled at the open of bar ``t+1``. The signal bar's
  own high and low are not used to fill that signal.
* Stop and take-profit are checked against the high and low after the fill.
  If both are touched in one bar, the stop is filled. A gap through the stop
  fills at the open, not at the stop.
* Transaction cost and slippage are added in the trader's disfavor on entry
  and exit. Position size uses that worsened stop, so the risk budget includes
  the round trip to the stop.
* An opposite signal closes the position and may open the other side at the
  same next open. HOLD keeps the position.
* A position still open on the last bar is closed at that bar's close.
* When ``use_ml`` is true, a model is fit only on the earlier slice and trades
  are allowed only on the later test slice.

The engine does not look at a future signal to decide the fill of the current bar.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forex_ai.backtesting.metrics import compute_metrics, infer_periods_per_year
from forex_ai.data.market_data import pip_size, quote_to_account
from forex_ai.ml.features import assert_chronological
from forex_ai.risk.risk_manager import RiskManager
from forex_ai.strategies.signal_engine import signals_for_frame

logger = logging.getLogger(__name__)

ASSUMPTIONS: dict[str, str] = {
    "execution": "A signal at the close of bar t is filled at the open of bar t+1.",
    "stop_and_target": (
        "Intrabar stop and take-profit use high/low. If both are touched, the stop fills. "
        "A gap through the target still fills at the target."
    ),
    "gaps": "If the open gaps through the stop, the fill is the open, worsened by cost.",
    "costs": "Transaction cost and slippage, in pips, worsen every entry and exit.",
    "sizing": "Position size risks a fixed fraction of cash, using the cost-adjusted stop distance.",
    "conversion": "USD-base pairs convert quote PnL at the entry price, held constant for that trade.",
    "liquidation": "A position still open on the last bar is closed at that bar's close.",
    "ml_window": "With use_ml, trades are opened only on the held-out chronological test slice.",
    "sharpe": "Sharpe uses a zero risk-free rate and an inferred periods-per-year factor.",
    "confidence": "Signal confidence is a model score, not a probability of profit.",
}


@dataclass
class BacktestConfig:
    """Cost, sizing, and stop assumptions for one simulation."""

    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    transaction_cost_pips: float = 1.0
    slippage_pips: float = 0.5
    atr_stop_multiple: float = 1.5
    reward_risk_ratio: float = 2.0
    stop_pips: float | None = None


@dataclass
class BacktestResult:
    """Closed trades, equity, and the metrics derived from them."""

    symbol: str
    metrics: dict[str, Any]
    trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    assumptions: dict[str, str] = field(default_factory=lambda: dict(ASSUMPTIONS))
    model_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "metrics": self.metrics,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "assumptions": self.assumptions,
            "model_report": self.model_report,
        }


def run_backtest(
    frame: pd.DataFrame,
    *,
    symbol: str,
    config: BacktestConfig | None = None,
    risk_manager: RiskManager | None = None,
) -> BacktestResult:
    """Simulate ``frame`` in order. ``frame`` must already contain ``signal``."""
    config = config or BacktestConfig()
    if config.initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    required = {"timestamp", "open", "high", "low", "close", "signal"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Backtest frame is missing columns: {sorted(missing)}.")
    data = frame.reset_index(drop=True)
    assert_chronological(data)
    manager = risk_manager or RiskManager(risk_per_trade=config.risk_per_trade)
    pip = pip_size(symbol)
    cost = (config.transaction_cost_pips + config.slippage_pips) * pip

    cash = float(config.initial_capital)
    peak = cash
    current_drawdown = 0.0
    position: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    rejected = 0
    logger.info(
        "Backtest start symbol=%s bars=%s initial_capital=%s.",
        symbol,
        len(data),
        config.initial_capital,
    )

    for index in range(len(data)):
        row = data.iloc[index]
        prior_signal = "HOLD" if index == 0 else _normalize_signal(data.iloc[index - 1]["signal"])
        if position is not None and _is_opposite(position["direction"], prior_signal):
            cash += _close(position, float(row["open"]), row["timestamp"], "signal", cost, trades)

            position = None
        if position is None and prior_signal in {"BUY", "SELL"}:
            opened, was_rejected = _try_open(
                row=row,
                prior=data.iloc[index - 1],
                signal=prior_signal,
                symbol=symbol,
                cash=cash,
                current_drawdown=current_drawdown,
                config=config,
                risk_manager=manager,
                pip=pip,
                cost=cost,
            )
            rejected += int(was_rejected)
            position = opened
        if position is not None:
            intrabar = _intrabar_exit(position, row, cost)
            if intrabar is not None:
                exit_price, reason = intrabar
                cash += _append_trade(position, exit_price, row["timestamp"], reason, trades)
                position = None
        if position is not None and index == len(data) - 1:
            cash += _close(position, float(row["close"]), row["timestamp"], "end_of_data", cost, trades)
            position = None

        marked = cash if position is None else cash + _unrealized(position, float(row["close"]))
        peak = max(peak, marked)
        current_drawdown = 0.0 if peak <= 0 else (peak - marked) / peak
        equity_rows.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "equity": marked,
                "cash": cash,
                "drawdown": current_drawdown,
            }
        )

    equity = pd.Series(
        [row["equity"] for row in equity_rows],
        index=pd.to_datetime([row["timestamp"] for row in equity_rows], utc=True),
        dtype=float,
    )
    trade_frame = pd.DataFrame(trades)
    metrics = compute_metrics(
        equity,
        trade_frame,
        config.initial_capital,
        infer_periods_per_year(equity.index),
    )
    logger.info(
        "Backtest end symbol=%s trades=%s final_capital=%.2f rejected_entries=%s.",
        symbol,
        metrics["number_of_trades"],
        metrics["final_capital"],
        rejected,
    )
    return BacktestResult(symbol=symbol.upper(), metrics=metrics, trades=trades, equity_curve=equity_rows)


def run_research_backtest(
    frame: pd.DataFrame,
    *,
    symbol: str,
    config: BacktestConfig | None = None,
    risk_manager: RiskManager | None = None,
    use_ml: bool = False,
    compare_random_forest: bool = False,
) -> BacktestResult:
    """Build causal signals, then simulate them.

    Technical mode trades the full sample after indicator warmup. ML mode fits
    on the earlier slice only and forces every pre-test signal to HOLD.
    """
    if use_ml:
        from forex_ai.ml.train import train_model

        trained = train_model(
            frame,
            compare_random_forest=compare_random_forest,
            model_path=None,
        )
        probabilities = trained.test_predictions["probability_up"]
        signal_frame = signals_for_frame(frame, probabilities)
        in_test = signal_frame.index.isin(trained.test_predictions.index)
        signal_frame.loc[~in_test, "signal"] = "HOLD"
        result = run_backtest(
            signal_frame,
            symbol=symbol,
            config=config,
            risk_manager=risk_manager,
        )
        result.model_report = {
            "selected_model": trained.selected_model,
            "classification_metrics": trained.test_metrics,
            "validation_metrics": trained.validation_metrics,
            "comparison": trained.comparison,
            "split": trained.split_info,
            "note": (
                "Classification metrics and trades both refer to the held-out test slice. "
                "They are not a forecast of live profit."
            ),
        }
        return result

    signal_frame = signals_for_frame(frame)
    return run_backtest(signal_frame, symbol=symbol, config=config, risk_manager=risk_manager)


def _try_open(
    *,
    row: pd.Series,
    prior: pd.Series,
    signal: str,
    symbol: str,
    cash: float,
    current_drawdown: float,
    config: BacktestConfig,
    risk_manager: RiskManager,
    pip: float,
    cost: float,
) -> tuple[dict[str, Any] | None, bool]:
    direction = "LONG" if signal == "BUY" else "SHORT"
    raw_entry = float(row["open"])
    if config.stop_pips is not None:
        distance = config.stop_pips * pip
    else:
        if "atr_14" not in prior.index:
            raise ValueError("atr_14 is required when stop_pips is not set.")
        atr_value = prior["atr_14"]
        if not np.isfinite(atr_value) or float(atr_value) <= 0:
            return None, True
        distance = float(atr_value) * config.atr_stop_multiple
    if distance <= 0 or raw_entry <= 0:
        return None, True

    if direction == "LONG":
        trigger_stop = raw_entry - distance
        trigger_tp = raw_entry + distance * config.reward_risk_ratio
        entry_fill = raw_entry + cost
    else:
        trigger_stop = raw_entry + distance
        trigger_tp = raw_entry - distance * config.reward_risk_ratio
        entry_fill = raw_entry - cost
    stop_fill = _worsen(direction, trigger_stop, cost)
    take_profit_fill = _worsen(direction, trigger_tp, cost)
    rate = quote_to_account(symbol, raw_entry)
    decision = risk_manager.assess(
        account_balance=cash,
        risk_per_trade=config.risk_per_trade,
        entry_price=entry_fill,
        stop_loss_price=stop_fill,
        take_profit_price=take_profit_fill,
        pip_value=pip * rate,
        pip_size=pip,
        quote_to_account=rate,
        current_drawdown=current_drawdown,
    )
    if not decision.approved or decision.position_size is None:
        logger.debug("Skipped entry at %s: %s", row["timestamp"], decision.reasons)
        return None, True
    return (
        {
            "direction": direction,
            "entry_time": pd.Timestamp(row["timestamp"]).isoformat(),
            "entry_fill": entry_fill,
            "trigger_stop": trigger_stop,
            "trigger_tp": trigger_tp,
            "units": float(decision.position_size),
            "rate": rate,
        },
        False,
    )


def _intrabar_exit(
    position: dict[str, Any],
    row: pd.Series,
    cost: float,
) -> tuple[float, str] | None:
    bar_open = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    stop = float(position["trigger_stop"])
    take_profit = float(position["trigger_tp"])
    direction = position["direction"]
    if direction == "LONG":
        if bar_open <= stop:
            return _worsen(direction, bar_open, cost), "gap_stop"
        hit_stop = low <= stop
        hit_target = high >= take_profit
        if hit_stop:
            return _worsen(direction, stop, cost), "stop"
        if hit_target:
            return _worsen(direction, take_profit, cost), "take_profit"
        return None
    if bar_open >= stop:
        return _worsen(direction, bar_open, cost), "gap_stop"
    hit_stop = high >= stop
    hit_target = low <= take_profit
    if hit_stop:
        return _worsen(direction, stop, cost), "stop"
    if hit_target:
        return _worsen(direction, take_profit, cost), "take_profit"
    return None


def _close(
    position: dict[str, Any],
    raw_price: float,
    timestamp: object,
    reason: str,
    cost: float,
    trades: list[dict[str, Any]],
) -> float:
    return _append_trade(position, _worsen(position["direction"], raw_price, cost), timestamp, reason, trades)


def _append_trade(
    position: dict[str, Any],
    exit_fill: float,
    timestamp: object,
    reason: str,
    trades: list[dict[str, Any]],
) -> float:
    pnl = _pnl(position, exit_fill)
    trades.append(
        {
            "entry_time": position["entry_time"],
            "exit_time": pd.Timestamp(timestamp).isoformat(),
            "direction": position["direction"],
            "entry_price": float(position["entry_fill"]),
            "exit_price": float(exit_fill),
            "units": float(position["units"]),
            "pnl": float(pnl),
            "exit_reason": reason,
            "stop_loss": float(position["trigger_stop"]),
            "take_profit": float(position["trigger_tp"]),
        }
    )
    return pnl


def _pnl(position: dict[str, Any], exit_fill: float) -> float:
    if position["direction"] == "LONG":
        return position["units"] * (exit_fill - position["entry_fill"]) * position["rate"]
    return position["units"] * (position["entry_fill"] - exit_fill) * position["rate"]


def _unrealized(position: dict[str, Any], price: float) -> float:
    return _pnl(position, price)


def _worsen(direction: str, price: float, cost: float) -> float:
    if direction == "LONG":
        return price - cost
    return price + cost


def _is_opposite(direction: str, signal: str) -> bool:
    return (direction == "LONG" and signal == "SELL") or (direction == "SHORT" and signal == "BUY")


def _normalize_signal(value: object) -> str:
    signal = str(value).upper()
    if signal not in {"BUY", "SELL", "HOLD"}:
        return "HOLD"
    return signal
