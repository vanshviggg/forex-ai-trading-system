"""Download historical candles into the local cache."""

import argparse
import logging

from forex_ai.config import configure_logging, get_settings
from forex_ai.data.market_data import DATA_SOURCE_LIMITATIONS, MarketDataProvider

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Download historical FX candles from Yahoo Finance.")
    parser.add_argument("--symbol", default=settings.default_symbol)
    parser.add_argument("--period", default=settings.default_period)
    parser.add_argument("--interval", default=settings.default_interval)
    args = parser.parse_args()
    for limitation in DATA_SOURCE_LIMITATIONS:
        logger.info("%s", limitation)
    provider = MarketDataProvider(cache_dir=settings.data_dir)
    frame = provider.get_history(args.symbol, args.period, args.interval, refresh=True)
    logger.info(
        "Downloaded %s rows for %s. Cache directory: %s.",
        len(frame),
        args.symbol.upper(),
        settings.data_dir,
    )


if __name__ == "__main__":
    main()
