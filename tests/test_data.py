"""Candle cleaning and the market-data provider."""

import pandas as pd
import pytest

from forex_ai.data.market_data import MarketDataError, MarketDataProvider, pip_size, quote_to_account
from forex_ai.data.preprocessing import clean_ohlcv


def test_clean_ohlcv_sorts_dedupes_and_drops_bad_rows() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-01", "2024-01-01", "2024-01-02"],
            "Open": [1.2, 1.0, 1.05, 1.1],
            "High": [1.3, 1.1, 1.2, 1.0],
            "Low": [1.1, 0.9, 1.0, 0.8],
            "Close": [1.25, 1.05, 1.15, 0.9],
            "Volume": [5, 1, 2, None],
        }
    )
    clean = clean_ohlcv(raw)
    # 2024-01-02 has high < close-side geometry (high 1.0, open 1.1) and is dropped.
    # Duplicate 2024-01-01 keeps the later input row.
    assert clean["timestamp"].is_monotonic_increasing
    assert len(clean) == 2
    assert clean["close"].iloc[0] == 1.15
    assert clean["volume"].iloc[0] == 2
    assert list(clean.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_clean_ohlcv_explains_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        clean_ohlcv(pd.DataFrame({"open": [1.0]}))


def test_provider_normalizes_a_yahoo_frame(tmp_path, monkeypatch) -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.1, 1.0],
            "High": [1.2, 1.2],
            "Low": [1.0, 0.9],
            "Close": [1.15, 1.1],
            "Volume": [2, 1],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-01"]),
    )
    raw.index.name = "Date"
    monkeypatch.setattr("forex_ai.data.market_data._download_yahoo", lambda *args, **kwargs: raw)
    provider = MarketDataProvider(cache_dir=tmp_path)
    frame = provider.get_history("eurusd", period="1mo", interval="1d", refresh=True)
    assert list(frame["timestamp"]) == sorted(frame["timestamp"])
    assert frame["close"].iloc[0] == 1.1
    cached = provider.get_history("EURUSD", period="1mo", interval="1d", refresh=False)
    assert len(cached) == 2


def test_provider_flattens_multiindex_columns(tmp_path, monkeypatch) -> None:
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["EURUSD=X"]]
    )
    raw = pd.DataFrame(
        [[1.0, 1.2, 0.9, 1.1, 10], [1.1, 1.3, 1.0, 1.2, 12]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        columns=columns,
    )
    monkeypatch.setattr("forex_ai.data.market_data._download_yahoo", lambda *args, **kwargs: raw)
    frame = MarketDataProvider(cache_dir=tmp_path).get_history("EURUSD", "1mo", "1d", refresh=True)
    assert frame["close"].tolist() == [1.1, 1.2]


def test_provider_rejects_unknown_symbols() -> None:
    with pytest.raises(MarketDataError, match="Unsupported symbol"):
        MarketDataProvider(cache_dir=None).get_history("BTCUSD")


def test_pip_and_quote_conversion() -> None:
    assert pip_size("EURUSD") == 0.0001
    assert pip_size("USDJPY") == 0.01
    assert quote_to_account("EURUSD", 1.1) == 1.0
    assert quote_to_account("USDJPY", 150.0) == pytest.approx(1 / 150)
