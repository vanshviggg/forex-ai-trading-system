"""Classification metrics for the directional baseline.

Precision, recall, and F1 use label 1 (the next close is higher). They measure
classification quality on a chosen split. They are not trading results.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from forex_ai.ml.predict import predict_probability


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_probability_up: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score predicted upward probabilities against binary labels."""
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_probability_up, dtype=float)
    predictions = (scores >= threshold).astype(int)
    roc_auc: float | None
    if len(np.unique(labels)) > 1:
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        roc_auc = None
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": matrix.astype(int).tolist(),
        "positive_label": 1,
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "sample_count": int(len(labels)),
        "note": (
            "Metrics describe the up/down classification on this split. "
            "They are not trading profits."
        ),
    }


def evaluate_model(
    model: Any,
    scaler: Any,
    features: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Transform ``features``, score them, and return classification metrics."""
    predicted = predict_probability(model, scaler, features)
    return classification_metrics(y_true, predicted["probability_up"])
