"""Normalize raw candles into the project's OHLCV contract."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

_COLUMN_ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "volume": "volume",
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns and rename common vendor aliases.

    Yahoo frames sometimes use a MultiIndex of ``(field, ticker)``. The level
    that contains OHLC names is kept. A nameless index is treated as the timestamp.
    """
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        chosen: list[str] | None = None
        for level in range(out.columns.nlevels):
            values = [str(value).lower().strip() for value in out.columns.get_level_values(level)]
            if any(value in {"close", "adj close", "open", "high"} for value in values):
                chosen = values
                break
        if chosen is None:
            chosen = [str(value).lower().strip() for value in out.columns.get_level_values(0)]
        out.columns = chosen
    else:
        out.columns = [str(column).lower().strip() for column in out.columns]

    if not {"timestamp", "date", "datetime", "time"} & set(out.columns):
        out = out.reset_index()
        out.columns = [str(column).lower().strip() for column in out.columns]
        if "index" in out.columns and "timestamp" not in out.columns:
            out = out.rename(columns={"index": "timestamp"})

    out = out.rename(columns=_COLUMN_ALIASES)
    out = out.loc[:, ~out.columns.duplicated()]

    if "close" not in out.columns and "adj_close" in out.columns:
        out["close"] = out["adj_close"]
    if "volume" not in out.columns:
        logger.warning("Volume column missing; filling with zeros.")
        out["volume"] = 0.0
    return out


def clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, sort, and de-duplicate candles.

    Rows with missing prices, non-positive prices, or impossible OHLC geometry
    are dropped. Prices are never forward-filled. Naive timestamps are treated
    as UTC.

    Raises:
        ValueError: Required price columns are absent, or no valid rows remain.
    """
    out = normalize_columns(frame)
    missing = [column for column in ("timestamp", "open", "high", "low", "close") if column not in out.columns]
    if missing:
        found = ", ".join(str(column) for column in out.columns)
        raise ValueError(f"Market data is missing {missing}. Found columns: {found}.")

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    starting_rows = len(out)
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    out["volume"] = out["volume"].fillna(0.0)
    out.loc[out["volume"] < 0, "volume"] = np.nan
    out["volume"] = out["volume"].fillna(0.0)

    positive = (out[["open", "high", "low", "close"]] > 0).all(axis=1)
    geometry = (out["high"] >= out[["open", "close", "low"]].max(axis=1) - 1e-12) & (
        out["low"] <= out[["open", "close", "high"]].min(axis=1) + 1e-12
    )
    out = out.loc[positive & geometry].copy()
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    out = out.loc[:, list(REQUIRED_COLUMNS)].reset_index(drop=True)

    removed = starting_rows - len(out)
    if removed > 0:
        logger.warning("Dropped %s invalid, incomplete, or duplicate rows.", removed)
    if out.empty:
        raise ValueError("No valid OHLCV rows remained after cleaning.")
    logger.info(
        "Clean candles: %s rows from %s to %s.",
        len(out),
        out["timestamp"].iloc[0].isoformat(),
        out["timestamp"].iloc[-1].isoformat(),
    )
    return out
