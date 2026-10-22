"""Load a saved bundle and score directional probabilities."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forex_ai.ml.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def predict_probability(model: Any, scaler: Any, features: pd.DataFrame) -> pd.DataFrame:
    """Score rows that already contain ``FEATURE_COLUMNS``.

    Returns ``probability_up``, ``probability_down``, and ``predicted_direction``.
    Direction is ``UP`` when the upward probability is at least 0.5.
    These are classification scores, not probabilities of profit.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Feature frame is missing columns: {missing}.")
    matrix = scaler.transform(features.loc[:, FEATURE_COLUMNS])
    probabilities = model.predict_proba(matrix)
    classes = [int(label) for label in model.classes_]
    if 1 in classes:
        probability_up = probabilities[:, classes.index(1)]
    else:
        probability_up = np.zeros(len(features))
    return pd.DataFrame(
        {
            "probability_up": probability_up.astype(float),
            "probability_down": (1.0 - probability_up).astype(float),
            "predicted_direction": np.where(probability_up >= 0.5, "UP", "DOWN"),
        },
        index=features.index,
    )


def load_model(path: Path) -> dict[str, Any]:
    """Load a joblib bundle produced by this project.

    Only the configured path is loaded. Callers must not pass a user-supplied
    path: unpickling can execute code.
    """
    import joblib

    logger.info("Loading model bundle from %s.", path)
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Saved model at {path} could not be loaded.") from exc
    if not isinstance(bundle, dict):
        raise ValueError("Saved model bundle has an unexpected format.")
    required = {"model", "scaler", "feature_columns"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Saved model bundle is missing keys: {sorted(missing)}.")
    if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Saved features do not match the current pipeline. Retrain the model.")
    return bundle
