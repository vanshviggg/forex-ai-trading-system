# Final checklist

## What was implemented

Research-only FX analysis stack. No broker, no live orders, no hardcoded API keys.

- Project layout, `pyproject.toml`, environment settings, logging
- Yahoo Finance provider for EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD
- Causal indicators: SMA, EMA, RSI, MACD, ATR, Bollinger Bands, volatility, support, resistance, returns
- Next-bar up/down features with a chronological 70/15/15 split and a one-row embargo
- Logistic regression baseline, optional random-forest comparison, joblib bundle
- BUY / SELL / HOLD signal. Confidence is a model score, not a probability of profit
- Risk manager: 1% default risk, 1.5 minimum reward/risk, drawdown and leverage rejections
- Event-driven backtest: signal at bar close, fill at the next open, costs, pessimistic same-bar stops
- FastAPI: `/health`, `/market/{symbol}`, `/indicators/{symbol}`, `/signal/{symbol}`, `/predict`, `/backtest`
- pytest, ruff, Dockerfile (non-root), Compose, GitHub Actions

## How to run it

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py --symbol EURUSD --period 2y --interval 1d
python scripts/train_model.py --symbol EURUSD --period 2y --interval 1d
python scripts/run_backtest.py --symbol EURUSD --period 2y --interval 1d
uvicorn forex_ai.api.main:app --reload
```

Docs: http://127.0.0.1:8000/docs

Docker:

```bash
docker compose up --build
```

## Test status

- `ruff check src tests scripts`: passed
- `pytest`: 37 passed
- `GET /health`, `/docs`, and `/openapi.json`: 200
- Live Yahoo calls for `/market`, `/indicators`, and `/signal` on EURUSD daily data: 200
- `python scripts/run_backtest.py --symbol EURUSD --period 1y --interval 1d` completed and wrote `data/backtest_EURUSD_1y_1d.json`
- Docker image `forex-ai-trading-system:mvp` built and answered `/health` as user `appuser`

That backtest file is gitignored. Read it locally. It is one sample under the documented cost assumptions, not a performance claim. The README does not copy the numbers.

## Known limitations

- Yahoo Finance FX data can gap, revise, and has no usable bid/ask. Volume is not a model feature.
- One chronological split is not walk-forward validation.
- The label is the next close. The backtest can hold until a stop, target, or opposite signal.
- Raw moving-average levels are non-stationary. Scaling does not fix that.
- Sharpe uses a zero risk-free rate and is undefined on a flat equity curve.
- pytest prints a Starlette deprecation warning about `httpx` versus `httpx2`. The suite still passes with the requested `httpx` dependency.

## Interview talking points

- Separation of concerns: data, indicators, features, model, signals, risk, backtest, and API do not share hidden state.
- Leakage: features at t use data through t. Only the label reads t+1. Unsorted timestamps are rejected instead of shuffled.
- Model selection uses validation ROC-AUC. The test slice is scored once, after an embargo row so the last training label cannot see the first test close.
- Logistic regression is the interpretable baseline. The forest is a comparison, not the default story.
- Backtest assumptions are returned with every result: next-open fills, stop-first when both levels trade, and cost-adjusted sizing.
- The API cannot execute code or load a user-supplied pickle. `/predict` reads only the configured model path.
- The container runs as a non-root user. CI installs the package, runs ruff, then pytest.

## Future improvements

- Walk-forward retraining with a longer purge gap
- Drop raw price levels and keep stationary features
- Probability calibration and an always-flat trading baseline
- Bid/ask data if a documented free source is available
