"""Train the directional baseline and save the joblib bundle."""

import argparse
import json
import logging

from forex_ai.config import configure_logging, get_settings
from forex_ai.data.market_data import MarketDataProvider
from forex_ai.ml.train import train_model

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Train the next-bar direction model.")
    parser.add_argument("--symbol", default=settings.default_symbol)
    parser.add_argument("--period", default=settings.default_period)
    parser.add_argument("--interval", default=settings.default_interval)
    parser.add_argument(
        "--no-random-forest",
        action="store_true",
        help="Skip the optional random-forest comparison and train logistic regression only.",
    )
    args = parser.parse_args()
    provider = MarketDataProvider(cache_dir=settings.data_dir)
    frame = provider.get_history(args.symbol, args.period, args.interval)
    result = train_model(
        frame,
        compare_random_forest=not args.no_random_forest,
        model_path=settings.model_path,
    )
    metrics_path = settings.model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
    logger.info("Wrote evaluation summary to %s.", metrics_path)
    logger.info("Test metrics: %s", result.test_metrics)


if __name__ == "__main__":
    main()
