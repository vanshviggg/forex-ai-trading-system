"""Performance statistics computed from an equity curve and closed trades.

Sharpe uses a zero risk-free rate. It is undefined when the equity curve is
flat, non-positive, or too short to have a sample standard deviation. Profit
factor is undefined when there are no losing trades, so it is returned as
``None`` rather than an invented large number.
"""

from typing import Any

import numpy as np
import pandas as pd


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer an annualization factor from the median spacing of ``index``.

    Daily bars use 252, weekly bars use 52, and shorter bars use ``24 / hours * 252``.
    Weekend gaps in daily FX data usually leave the median at one day. Hourly
    annualization treats the market as 24-hour, which overstates a five-day week.
    """
    if len(index) < 3:
        return 252
    seconds = pd.Series(index).diff().dt.total_seconds().median()
    if seconds is None or not np.isfinite(seconds) or seconds <= 0:
        return 252
    days = float(seconds) / 86_400
    if days >= 5:
        return 52
    if days >= 0.9:
        return 252
    hours = float(seconds) / 3_600
    return max(int(round((24.0 / hours) * 252)), 1)


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Summarize a finished backtest. Losses are reported as positive amounts."""
    if equity.empty:
        raise ValueError("Equity curve is empty.")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")

    final_capital = float(equity.iloc[-1])
    pnls = trades["pnl"].to_numpy(dtype=float) if not trades.empty and "pnl" in trades else np.array([])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    breakeven = int(np.sum(pnls == 0)) if len(pnls) else 0
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    profit_factor = None if gross_loss == 0 else gross_profit / abs(gross_loss)

    peak = equity.cummax()
    drawdown = (peak - equity) / peak.where(peak > 0)
    maximum_drawdown = float(drawdown.max()) if drawdown.notna().any() else 0.0

    annualization = periods_per_year
    if annualization is None:
        if isinstance(equity.index, pd.DatetimeIndex):
            annualization = infer_periods_per_year(equity.index)
        else:
            annualization = 252
    sharpe = _sharpe(equity, annualization)

    trade_count = int(len(pnls))
    return {
        "initial_capital": float(initial_capital),
        "final_capital": final_capital,
        "total_return": (final_capital - float(initial_capital)) / float(initial_capital),
        "number_of_trades": trade_count,
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "breakeven_trades": breakeven,
        "win_rate": 0.0 if trade_count == 0 else float(len(wins) / trade_count),
        "average_win": None if len(wins) == 0 else float(wins.mean()),
        "average_loss": None if len(losses) == 0 else float(abs(losses.mean())),
        "profit_factor": None if profit_factor is None else float(profit_factor),
        "maximum_drawdown": maximum_drawdown,
        "sharpe_ratio": sharpe,
        "periods_per_year": int(annualization),
    }


def _sharpe(equity: pd.Series, periods_per_year: int) -> float | None:
    if (equity <= 0).any() or len(equity) < 3:
        return None
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return None
    deviation = float(returns.std(ddof=1))
    if not np.isfinite(deviation) or deviation == 0:
        return None
    return float(returns.mean() / deviation * np.sqrt(periods_per_year))
