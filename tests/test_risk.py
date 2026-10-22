"""Position sizing safeguards."""

import pytest

from forex_ai.risk.risk_manager import RiskManager


def manager() -> RiskManager:
    return RiskManager(
        risk_per_trade=0.01,
        min_risk_reward=1.5,
        max_risk_per_trade=0.02,
        max_drawdown=0.20,
        max_leverage=30,
    )


def test_sizes_a_long_with_room_under_the_leverage_cap() -> None:
    decision = manager().assess(
        account_balance=10_000,
        entry_price=1.10,
        stop_loss_price=1.09,
        take_profit_price=1.12,
        pip_value=0.0001,
        pip_size=0.0001,
    )
    assert decision.approved
    assert decision.monetary_risk == pytest.approx(100)
    assert decision.position_size == pytest.approx(10_000)
    assert decision.risk_reward_ratio == pytest.approx(2.0)
    assert decision.max_position_size == pytest.approx(20_000)


def test_rejects_invalid_balance_stop_reward_drawdown_and_leverage() -> None:
    engine = manager()
    balance = engine.assess(
        account_balance=0,
        entry_price=1.1,
        stop_loss_price=1.09,
        take_profit_price=1.12,
        pip_value=0.0001,
        pip_size=0.0001,
    )
    assert not balance.approved
    assert any("balance" in reason.lower() for reason in balance.reasons)

    stop = engine.assess(
        account_balance=10_000,
        entry_price=1.1,
        stop_loss_price=1.1,
        take_profit_price=1.12,
        pip_value=0.0001,
        pip_size=0.0001,
    )
    assert any("stop distance" in reason.lower() for reason in stop.reasons)

    reward = engine.assess(
        account_balance=10_000,
        entry_price=1.1,
        stop_loss_price=1.09,
        take_profit_price=1.105,
        pip_value=0.0001,
        pip_size=0.0001,
    )
    assert any("risk/reward" in reason.lower() for reason in reward.reasons)

    risk = engine.assess(
        account_balance=10_000,
        risk_per_trade=0.05,
        entry_price=1.1,
        stop_loss_price=1.09,
        take_profit_price=1.12,
        pip_value=0.0001,
        pip_size=0.0001,
    )
    assert any("maximum" in reason.lower() for reason in risk.reasons)

    drawdown = engine.assess(
        account_balance=10_000,
        entry_price=1.1,
        stop_loss_price=1.09,
        take_profit_price=1.12,
        pip_value=0.0001,
        pip_size=0.0001,
        current_drawdown=0.25,
    )
    assert any("drawdown" in reason.lower() for reason in drawdown.reasons)

    wrong_side = engine.assess(
        account_balance=10_000,
        entry_price=1.1,
        stop_loss_price=1.09,
        take_profit_price=1.05,
        pip_value=0.0001,
        pip_size=0.0001,
    )
    assert any("take-profit" in reason.lower() for reason in wrong_side.reasons)

    tight_stop = engine.assess(
        account_balance=10_000,
        entry_price=1.10,
        stop_loss_price=1.0999,
        take_profit_price=1.102,
        pip_value=0.0001,
        pip_size=0.0001,
    )
    assert any("leverage" in reason.lower() for reason in tight_stop.reasons)
