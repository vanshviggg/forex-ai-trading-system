"""API health, validation, and offline research routes."""

from fastapi.testclient import TestClient
from tests.conftest import make_ohlcv

from forex_ai.api.main import create_app, get_provider
from forex_ai.config import Settings
from forex_ai.data.market_data import MarketDataProvider
from forex_ai.ml.train import train_model


def _client(tmp_path, frame, with_model: bool = False):
    settings = Settings(
        app_env="test",
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
    )
    settings.model_dir.mkdir()
    settings.data_dir.mkdir()
    if with_model:
        train_model(frame, compare_random_forest=False, model_path=settings.model_path)
    app = create_app(settings)
    provider = MarketDataProvider(cache_dir=None)

    def history(symbol: str, period: str = "2y", interval: str = "1d", *, refresh: bool = False):
        return frame.copy()

    provider.get_history = history  # type: ignore[method-assign]
    app.dependency_overrides[get_provider] = lambda: provider
    return TestClient(app)


def test_health_and_docs(tmp_path) -> None:
    client = _client(tmp_path, make_ohlcv(80, seed=1))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["research_only"] is True
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_validation_errors(tmp_path) -> None:
    client = _client(tmp_path, make_ohlcv(80, seed=1))
    assert client.get("/market/BTCUSD").status_code == 422
    assert client.get("/market/EURUSD", params={"period": "2minutes"}).status_code == 422
    assert client.post("/backtest", json={"symbol": "EURUSD", "initial_capital": -1}).status_code == 422
    rejected = client.post("/predict", json={"symbol": "EURUSD", "code": "import os"})
    assert rejected.status_code == 422
    missing = client.post("/predict", json={"symbol": "EURUSD", "period": "1y", "interval": "1d"})
    assert missing.status_code == 404


def test_market_indicators_signal_and_backtest(tmp_path) -> None:
    client = _client(tmp_path, make_ohlcv(180, seed=6))
    market = client.get("/market/eurusd", params={"period": "1y", "interval": "1d", "limit": 5})
    assert market.status_code == 200
    body = market.json()
    assert body["symbol"] == "EURUSD"
    assert body["source"] == "yahoo_finance"
    assert len(body["candles"]) == 5
    assert body["limitations"]

    indicators = client.get("/indicators/EURUSD", params={"period": "1y", "interval": "1d"})
    assert indicators.status_code == 200
    assert "rsi_14" in indicators.json()["latest"]

    signal = client.get("/signal/EURUSD", params={"period": "1y", "interval": "1d"})
    assert signal.status_code == 200
    payload = signal.json()
    assert payload["signal"] in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= payload["confidence"] <= 1.0
    assert "profit" in payload["disclaimer"]

    backtest = client.post(
        "/backtest",
        json={
            "symbol": "EURUSD",
            "period": "1y",
            "interval": "1d",
            "initial_capital": 10000,
            "use_ml": False,
        },
    )
    assert backtest.status_code == 200
    metrics = backtest.json()["metrics"]
    assert metrics["initial_capital"] == 10000
    assert "maximum_drawdown" in metrics
    assert "sharpe_ratio" in metrics
    assert backtest.json()["model_report"] is None


def test_predict_and_ml_backtest_do_not_require_network(tmp_path) -> None:
    frame = make_ohlcv(180, seed=8)
    client = _client(tmp_path, frame, with_model=True)
    predicted = client.post("/predict", json={"symbol": "EURUSD", "period": "1y", "interval": "1d"})
    assert predicted.status_code == 200
    body = predicted.json()
    assert body["probability_up"] + body["probability_down"] == 1
    assert body["predicted_direction"] in {"UP", "DOWN"}
    assert body["observation_was_in_fit"] is False

    simulated = client.post(
        "/backtest",
        json={"symbol": "EURUSD", "period": "1y", "interval": "1d", "use_ml": True},
    )
    assert simulated.status_code == 200
    report = simulated.json()["model_report"]
    assert report["selected_model"] == "logistic_regression"
    assert report["classification_metrics"]["sample_count"] > 0
