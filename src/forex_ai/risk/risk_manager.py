"""Size a research position and reject unsafe inputs.

This class does not recommend a live trade. It answers a narrower question:
given an account, a stop, and a target, how large can a paper position be
before it breaks the configured limits?

``pip_value`` is the account-currency value of a one-pip move per one unit of
the base currency. ``pip_size`` is the price distance of one pip (0.0001 for
most majors, 0.01 for JPY pairs).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskDecision:
    """Sizing result. ``approved`` is false when any safeguard fails."""

    approved: bool
    monetary_risk: float | None
    position_size: float | None
    risk_reward_ratio: float | None
    max_position_size: float | None
    reasons: list[str]


class RiskManager:
    """Rejects invalid balances, stops, reward ratios, drawdowns, and leverage."""

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        min_risk_reward: float = 1.5,
        max_risk_per_trade: float = 0.02,
        max_drawdown: float = 0.20,
        max_leverage: float = 30.0,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.min_risk_reward = min_risk_reward
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage

    def assess(
        self,
        *,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        pip_value: float,
        pip_size: float,
        risk_per_trade: float | None = None,
        quote_to_account: float = 1.0,
        current_drawdown: float = 0.0,
    ) -> RiskDecision:
        """Return position size in base-currency units, or a rejection.

        The side is inferred from the stop: a stop below entry is treated as a
        long, and a stop above entry is treated as a short. Take-profit must
        sit on the other side of entry.
        """
        requested_risk = self.risk_per_trade if risk_per_trade is None else risk_per_trade
        reasons: list[str] = []

        if not _positive(account_balance):
            reasons.append("Account balance must be positive.")
        if not _positive(requested_risk):
            reasons.append("Risk per trade must be positive.")
        if _positive(requested_risk) and requested_risk > self.max_risk_per_trade:
            reasons.append("Risk per trade exceeds the configured maximum.")
        if not _finite(current_drawdown) or current_drawdown < 0:
            reasons.append("Current drawdown must be zero or a positive fraction.")
        elif current_drawdown >= self.max_drawdown:
            reasons.append("Current drawdown exceeds the maximum drawdown threshold.")
        if not _positive(entry_price):
            reasons.append("Entry price must be positive.")
        if not _positive(pip_value) or not _positive(pip_size):
            reasons.append("Pip value and pip size must be positive.")
        if not _positive(quote_to_account):
            reasons.append("Quote-to-account conversion must be positive.")

        geometry = _geometry(entry_price, stop_loss_price, take_profit_price)
        reasons.extend(geometry["reasons"])
        ratio = geometry["risk_reward_ratio"]
        if ratio is not None and ratio < self.min_risk_reward:
            reasons.append(
                f"Risk/reward {ratio:.2f} is below the minimum of {self.min_risk_reward:.2f}."
            )

        if reasons or ratio is None:
            return RiskDecision(False, None, None, ratio, None, reasons)

        stop_distance = abs(entry_price - stop_loss_price)
        pips = stop_distance / pip_size
        loss_per_unit = pips * pip_value
        if not _positive(loss_per_unit):
            return RiskDecision(
                False,
                None,
                None,
                ratio,
                None,
                ["Stop distance must produce a positive loss per unit."],
            )

        monetary_risk = account_balance * requested_risk
        position_size = monetary_risk / loss_per_unit
        max_units_by_risk = (account_balance * self.max_risk_per_trade) / loss_per_unit
        notional_per_unit = entry_price * quote_to_account
        max_units_by_leverage = (account_balance * self.max_leverage) / notional_per_unit
        max_position_size = min(max_units_by_risk, max_units_by_leverage)
        if position_size > max_position_size + 1e-9:
            reasons.append("Position size exceeds maximum leverage.")
            return RiskDecision(
                False,
                monetary_risk,
                None,
                ratio,
                max_position_size,
                reasons,
            )

        reasons.append("Within risk, reward, drawdown, and leverage limits.")
        return RiskDecision(
            True,
            monetary_risk,
            position_size,
            ratio,
            max_position_size,
            reasons,
        )


def _geometry(entry: float, stop: float, take_profit: float) -> dict[str, object]:
    reasons: list[str] = []
    if not (_finite(entry) and _finite(stop) and _finite(take_profit)):
        return {"risk_reward_ratio": None, "reasons": ["Entry, stop, and take-profit must be finite."]}
    stop_distance = abs(entry - stop)
    if stop_distance == 0:
        return {"risk_reward_ratio": None, "reasons": ["Stop distance must be greater than zero."]}
    if stop < entry and take_profit <= entry:
        reasons.append("Take-profit must be above entry when the stop is below entry.")
    elif stop > entry and take_profit >= entry:
        reasons.append("Take-profit must be below entry when the stop is above entry.")
    ratio = abs(take_profit - entry) / stop_distance
    return {"risk_reward_ratio": float(ratio), "reasons": reasons}


def _positive(value: float) -> bool:
    return _finite(value) and float(value) > 0


def _finite(value: float) -> bool:
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return False
