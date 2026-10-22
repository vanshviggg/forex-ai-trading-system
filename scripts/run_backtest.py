"""Run a historical backtest and write the measured result to data/."""

import argparse
import json
import logging

from forex_ai.backtesting.engine import BacktestConfig, run_research_backtest
from forex_ai.config import configure_logging, get_settings
from forex_ai.data.market_data import MarketDataProvider
from forex_ai.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Backtest a symbol on cached or downloaded history.")
    parser.add_argument("--symbol", default=settings.default_symbol)
    parser.add_argument("--period", default=settings.default_period)
    parser.add_argument("--interval", default=settings.default_interval)
    parser.add_argument("--initial-capital", type=float, default=settings.initial_capital)
    parser.add_argument("--use-ml", action="store_true")
    parser.add_argument("--compare-random-forest", action="store_true")
    args = parser.parse_args()
    provider = MarketDataProvider(cache_dir=settings.data_dir)
    frame = provider.get_history(args.symbol, args.period, args.interval)
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        risk_per_trade=settings.risk_per_trade,
        transaction_cost_pips=settings.transaction_cost_pips,
        slippage_pips=settings.slippage_pips,
    )
    manager = RiskManager(
        risk_per_trade=settings.risk_per_trade,
        min_risk_reward=settings.min_risk_reward,
        max_risk_per_trade=settings.max_risk_per_trade,
        max_drawdown=settings.max_drawdown,
        max_leverage=settings.max_leverage,
    )
    result = run_research_backtest(
        frame,
        symbol=args.symbol,
        config=config,
        risk_manager=manager,
        use_ml=args.use_ml,
        compare_random_forest=args.compare_random_forest,
    )
    payload = result.to_dict()
    payload["period"] = args.period
    payload["interval"] = args.interval
    payload["use_ml"] = args.use_ml
    output = settings.data_dir / f"backtest_{args.symbol.upper()}_{args.period}_{args.interval}.json"
    if args.use_ml:
        output = settings.data_dir / f"backtest_{args.symbol.upper()}_{args.period}_{args.interval}_ml.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote backtest result to %s.", output)
    logger.info(
        "Measured total_return=%s trades=%s final_capital=%s.",
        result.metrics["total_return"],
        result.metrics["number_of_trades"],
        result.metrics["final_capital"],
    )


if __name__ == "__main__":
    main()
