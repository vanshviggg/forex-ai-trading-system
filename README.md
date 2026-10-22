# AI-Powered Forex Market Analysis & Trading System

Research system for historical FX candles, technical indicators, a leakage-aware directional classifier, deterministic signals, position-sizing checks, and an event-driven backtest. The HTTP API exposes those pieces for inspection. It does not place live orders.

## 1. Overview

The project is an interview-sized MVP of a market-analysis stack. Market data is normalized to OHLCV, indicators are causal, the model is trained on a chronological split, and the backtest fills a signal from bar `t` at the open of bar `t+1`. Every number the backtest prints is computed from the candles you download. This README does not publish performance results.

## 2. Problem Statement

Interview projects often mix three different questions:

- Did the next candle close higher?
- Did a trading rule make money after costs?
- Should a real account take this trade?

This repository keeps those questions in separate modules. The classifier answers only the first. The backtest answers the second under stated assumptions. The risk manager answers a narrower sizing question and rejects unsafe inputs. None of them is a broker.

## 3. Features

- Seven major pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD
- Yahoo Finance history via `yfinance`, cached under `data/`
- SMA, EMA, RSI, MACD, ATR, Bollinger Bands, rolling volatility, support, and resistance
- Next-bar up/down label with an explicit feature allow-list
- Logistic regression baseline, with an optional shallow random forest comparison
- BUY / SELL / HOLD signal with a documented score, not a profit probability
- Risk checks: balance, stop distance, reward ratio, drawdown, and leverage
- Event-driven backtest with spread, slippage, and pessimistic same-bar stops
- FastAPI routes for health, candles, indicators, signals, prediction, and backtest
- pytest, ruff, Docker, and GitHub Actions

## 4. Architecture

```
                +------------------+
                |  Yahoo Finance   |
                +--------+---------+
                         |
                         v
                +------------------+
                | MarketDataProvider|
                | clean OHLCV      |
                +--------+---------+
                         |
          +--------------+--------------+
          |                             |
          v                             v
 +----------------+            +------------------+
 | Indicators     |            | Feature pipeline |
 | causal windows |            | label = next bar |
 +--------+-------+            +--------+---------+
          |                             |
          v                             v
 +----------------+            +------------------+
 | Signal engine  |<-----------| Logistic model   |
 | BUY/SELL/HOLD  |            | time-ordered split|
 +--------+-------+            +------------------+
          |
          v
 +----------------+     +------------------+
 | Risk manager   |---->| Backtest engine  |
 | size or reject |     | next-bar fills   |
 +----------------+     +--------+---------+
                                 |
                                 v
                        +------------------+
                        | FastAPI          |
                        | research only    |
                        +------------------+
```

Each box is a package under `src/forex_ai/`. The API calls the library. It does not own the formulas.

## 5. Technology Stack

Python 3.12, FastAPI, Pydantic Settings, pandas, numpy, scikit-learn, yfinance, joblib, pytest, httpx, uvicorn, python-dotenv, Docker, and GitHub Actions. Indicators are implemented in pandas/numpy so the formulas in `indicators/technical.py` are the ones under test. There is no paid data vendor.

## 6. Data Pipeline

`MarketDataProvider.get_history(symbol, period, interval)` downloads `SYMBOL=X` from Yahoo Finance, then `clean_ohlcv` does the following:

1. Flattens MultiIndex columns and lower-cases names.
2. Produces `timestamp, open, high, low, close, volume`.
3. Parses timestamps as UTC.
4. Drops missing or non-positive prices and impossible OHLC rows. Prices are not forward-filled.
5. Sorts ascending and keeps the last row when timestamps collide.
6. Writes a CSV cache on success. A later call can reuse it if the download fails.

Allowed periods: `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `max`. Allowed intervals: `1d`, `1h`, `1wk`.

Yahoo Finance limitations:

- The FX history is unofficial and can contain gaps or revisions.
- Spot volume is often missing or meaningless. It is not a model feature.
- There is no bid/ask. Backtest costs are assumptions.
- Intraday history is short. Daily bars are the intended MVP sample.
- This is not a broker feed.

## 7. Technical Indicators

| Group | Columns | Formula used here |
| --- | --- | --- |
| Trend | `sma_20`, `sma_50`, `ema_20`, `ema_50` | Rolling mean. EMA is seeded with the SMA, then `alpha = 2 / (span + 1)`. |
| Momentum | `rsi_14`, `macd`, `macd_signal`, `macd_hist` | Wilder RSI. MACD is EMA(12) - EMA(26). Signal is EMA(9) of the MACD line. |
| Volatility | `atr_14`, `bb_upper`, `bb_lower`, `volatility_20` | Wilder ATR. Bollinger uses a 20-period SMA and 2 population standard deviations. Volatility is the 20-period sample std of log returns, not annualized. |
| Structure | `support_20`, `resistance_20` | Trailing 20-bar low and high. |
| Returns | `pct_return`, `log_return`, `ema20_distance`, `volume_change` | Close-to-close returns and `(close - EMA20) / EMA20`. |

Windows end at the current bar. Leading values stay NaN until the window is full. They are dropped later, not replaced with zero.

## 8. ML Pipeline

The label at time `t` is 1 when `close[t+1] > close[t]`, else 0. A flat next close is 0. The last row has no label.

`FEATURE_COLUMNS` is the only matrix the scaler and the model see. It includes RSI, MACD and its signal and histogram, both EMAs, both SMAs, ATR, Bollinger upper and lower, rolling volatility, percentage and log returns, three lagged percentage returns, EMA distance, and close/EMA ratios. `target` is not in that list.

Leakage controls:

- Indicator windows are trailing.
- Lags use `shift(1)` and further positive shifts.
- Only the label uses `shift(-1)`.
- Rows must already be sorted. Unsorted or duplicate timestamps raise. The trainer does not shuffle them back into order.
- Split is 70% train, 15% validation, 15% test by row order. `sklearn.model_selection.train_test_split` is not used.
- The row immediately before the test set is an embargo. Its label would read the first test close, so it is excluded from fitting and from validation.
- `StandardScaler` is fit on the training rows only while models are compared. After a winner is chosen, the scaler and that winner are refit on train+validation. The test slice is scored once.
- Logistic regression is the default interpretable model. A depth-capped random forest is compared on validation ROC-AUC. Ties stay with logistic regression.
- Raw moving-average levels are non-stationary. Scaling does not fix that. Ratios and returns are the more stable inputs; the level features are included because they are part of the requested baseline, and L2 logistic regression can shrink them.

The saved bundle is `models/direction_model.joblib`. A readable copy of the metrics is `models/metrics.json`. Test metrics describe the up/down classification on that split. They are not a trading track record, and the model does not predict prices exactly.

`/predict` loads only the configured path. It does not accept an uploaded pickle. Unpickling can execute code, so the path is not a request field.

## 9. Signal Generation

The technical score is a weighted vote in `[-1, 1]`, mapped to `[0, 1]`:

- 0.35 EMA20 versus EMA50
- 0.25 price versus EMA20, SMA20, and SMA50
- 0.20 RSI (above/below 50, with a smaller weight at 70/30)
- 0.20 MACD line versus its signal

If ATR/close is above 1%, the raw vote is halved. Without a model, BUY requires a score of at least 0.62 and SELL requires 0.38 or below. Otherwise the signal is HOLD.

With a model, confidence is `0.6 * technical_score + 0.4 * ml_probability`. A BUY also requires the technical score and the upward probability to agree. Disagreement is HOLD.

Confidence near 1 is a bullish score and confidence near 0 is a bearish score. Distance from 0.5 is how decisive the score is. It is not the probability of making money.

## 10. Risk Management

`RiskManager.assess` takes balance, entry, stop, take-profit, pip value, and pip size.

- Monetary risk is `balance * risk_per_trade` (default 1%).
- Position size in base units is monetary risk divided by the loss of one unit at the stop.
- Risk/reward is reward distance divided by stop distance. The default minimum is 1.5.
- Maximum position size is the tighter of the 2% risk cap and a 30x leverage cap.
- Requests are rejected when the balance is not positive, the stop distance is zero, the stop and target are on the same side, risk exceeds the cap, drawdown is already at the 20% threshold, or the size would exceed leverage.

The class returns a decision. It does not recommend a live order. JPY pairs use a pip size of 0.01. USD-base pairs convert quote-currency PnL with `1 / price` at entry, and that rate stays fixed for the trade.

## 11. Backtesting

`run_backtest` walks candles from oldest to newest.

- The signal on bar `t` is read at the open of bar `t+1`. The fill is not the signal bar's close.
- A long stop is `entry - 1.5 * ATR` from the signal bar, and the target is twice that distance, unless a test supplies a fixed stop.
- High/low after the fill can exit the trade. If both the stop and the target are touched, the stop fills. A gap through the stop fills at the open. A gap through the target still fills at the target.
- Default costs are 1.0 pip of transaction cost plus 0.5 pip of slippage, applied against the trader on entry and on exit. Sizing uses the worsened stop, so the risk budget includes that round trip.
- HOLD keeps the position. An opposite signal closes it and can open the other side at the same next open.
- The last open position is closed at the last close.
- New entries stop after the drawdown threshold. An open trade can still lose more than that.
- Sharpe uses a zero risk-free rate. Daily bars are annualized with 252. The ratio is `None` when equity is flat or non-positive, rather than a fake zero. Profit factor is `None` when there are no losing trades.

`use_ml` fits the model on the earlier slice only and forces every pre-test signal to HOLD, so the trading result is out of sample. The classifier's horizon is the next close. The backtest can hold longer, until a stop, a target, or an opposite signal. Those are related but different questions.

Run the backtest command to generate current results. Do not copy an old JSON file into a write-up without the period, costs, and pair beside it.

## 12. API

Interactive docs: `http://127.0.0.1:8000/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Process is up. |
| GET | `/market/{symbol}` | Recent candles and the data-source limitations. |
| GET | `/indicators/{symbol}` | Latest causal indicators. |
| GET | `/signal/{symbol}` | Technical BUY/SELL/HOLD. |
| POST | `/predict` | Up/down probabilities from the saved model. |
| POST | `/backtest` | Historical simulation. `use_ml` adds the out-of-sample model path. |

Unknown symbols, bad periods, and non-positive capital return 422. A missing model returns 404. A Yahoo failure returns 502. Request bodies reject unknown fields, so the API cannot be handed a pickle path or a snippet of code.

## 13. Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

`.env` is optional. The example file is the template. Do not commit a real `.env`. No API key is required.

## 14. Usage

```bash
python scripts/download_data.py --symbol EURUSD --period 2y --interval 1d
python scripts/train_model.py --symbol EURUSD --period 2y --interval 1d
python scripts/run_backtest.py --symbol EURUSD --period 2y --interval 1d
python scripts/run_backtest.py --symbol EURUSD --period 2y --interval 1d --use-ml
uvicorn forex_ai.api.main:app --reload
```

Training compares logistic regression with a random forest unless you pass `--no-random-forest`. The ML backtest compares the forest only when you pass `--compare-random-forest`.

## 15. Running Tests

```bash
ruff check src tests scripts
pytest
```

The tests use synthetic candles. They do not download a market and they do not assert that the model is profitable.

## 16. Docker

```bash
docker compose up --build
```

The image installs the package, listens on port 8000, and runs as `appuser`. Mounted `data/` and `models/` directories persist the cache and a model you trained on the host. Build the image after a model exists if you want `/predict` to succeed inside the container. `/health` does not need one.

## 17. Example Output

Signal responses have this shape. The values come from the latest candle, not from this README:

```json
{
  "symbol": "EURUSD",
  "signal": "HOLD",
  "technical_score": 0.5,
  "ml_probability": null,
  "confidence": 0.5,
  "reasons": ["Score is inside the hold band"],
  "disclaimer": "Research and paper-trading analysis only..."
}
```

A backtest writes `data/backtest_EURUSD_2y_1d.json` (or the `_ml` variant). The file contains `metrics`, `trades`, `equity_curve`, and `assumptions`. Run the backtest command to generate current results.

`metrics` includes initial capital, final capital, total return as a fraction, trade counts, win rate, average win, average loss as a positive amount, profit factor, maximum drawdown as a positive fraction, and Sharpe. `None` means the statistic is undefined for that run.

## 18. Model Evaluation

`scripts/train_model.py` logs and saves:

- accuracy, precision, recall, and F1 for the upward class
- ROC-AUC when both classes are present
- a 2x2 confusion matrix ordered `[actual 0/1][predicted 0/1]`
- the validation comparison that selected the model
- the train, validation, embargo, and test sizes

Precision is "of the bars predicted up, how many closed up." It is not a hit rate of a trading account. Read `models/metrics.json` after training rather than memorizing a number from documentation.

`observation_was_in_fit` on `/predict` is true when the scored bar is on or before the last row the saved model was fit on. A bar inside the fit window is not a fresh test.

## 19. Limitations

- Yahoo Finance is not a dealing price. Costs are a fixed pip assumption.
- One chronological split is not walk-forward cross-validation.
- The label is next-bar direction. The backtest can hold for many bars.
- Moving-average levels can dominate a linear model across a drifting price.
- RSI and ATR use the common Wilder seed. They will not match every charting package to the last decimal.
- USD-base PnL is converted at the entry price and held constant.
- Sharpe on a short daily sample is unstable, and the hourly annualization assumes a 24-hour year.
- There is no portfolio, no rollover, no news filter, and no execution venue.
- Joblib files should be ones this project wrote. Do not load a pickle you did not train.

## 20. Future Improvements

- Walk-forward retraining with a purged gap longer than one bar
- Drop raw price levels and keep stationary features only
- Calibrate probabilities and report reliability, not just ROC-AUC
- A baseline that is always HOLD, so a backtest has to beat doing nothing after costs
- Bid/ask data if a free, documented source is available
- Separate read-only model artifacts from the API process and load them by version

## 21. Disclaimer

This software is for research, education, and paper analysis. It is not financial advice and it does not place trades. Confidence is a model score, not a probability of profit. Past classification metrics and one historical simulation are not evidence that a rule will make money.
