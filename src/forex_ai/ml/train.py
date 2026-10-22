"""Chronological training for the directional baseline.

The sample is split 70% train, 15% validation, and 15% test by row order.
Rows are never shuffled. One embargo row sits between validation and test so
no training label is allowed to use the first test close.

Logistic regression is the interpretable baseline. A shallow random forest is
an optional comparison on the validation slice only. The winner is refit on
train+validation and scored once on the untouched test slice.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from forex_ai.ml.evaluation import evaluate_model
from forex_ai.ml.features import FEATURE_COLUMNS, assert_chronological, supervised_dataset
from forex_ai.ml.predict import predict_probability

logger = logging.getLogger(__name__)

MIN_SUPERVISED_ROWS = 40


@dataclass
class ChronologicalSplit:
    """Ordered partitions of a supervised frame."""

    train: pd.DataFrame
    validation: pd.DataFrame
    embargo: pd.DataFrame
    test: pd.DataFrame


@dataclass
class TrainResult:
    """Fitted model plus the metrics used to select and report it."""

    selected_model: str
    model: Any
    scaler: StandardScaler
    feature_columns: list[str]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    comparison: dict[str, dict[str, Any]]
    split_info: dict[str, Any]
    test_predictions: pd.DataFrame
    trained_through: str
    test_start: str

    def summary(self) -> dict[str, Any]:
        """JSON-friendly view with the estimator objects removed."""
        return {
            "selected_model": self.selected_model,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "comparison": self.comparison,
            "split": self.split_info,
            "trained_through": self.trained_through,
            "test_start": self.test_start,
            "feature_columns": self.feature_columns,
        }


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    """Split ``frame`` in time order, with a one-row embargo before the test set."""
    if "timestamp" in frame.columns:
        assert_chronological(frame)
    row_count = len(frame)
    if row_count < MIN_SUPERVISED_ROWS:
        raise ValueError(
            f"Need at least {MIN_SUPERVISED_ROWS} supervised rows for a chronological split, "
            f"found {row_count}."
        )
    train_end = int(row_count * train_fraction)
    test_start = int(row_count * (train_fraction + validation_fraction))
    embargo_at = test_start - 1
    validation_length = embargo_at - train_end
    test_length = row_count - test_start
    if train_end < 5 or validation_length < 5 or test_length < 5:
        raise ValueError("Chronological split produced a partition that is too small.")
    return ChronologicalSplit(
        train=frame.iloc[:train_end],
        validation=frame.iloc[train_end:embargo_at],
        embargo=frame.iloc[embargo_at:test_start],
        test=frame.iloc[test_start:],
    )


def train_model(
    frame: pd.DataFrame,
    *,
    compare_random_forest: bool = True,
    model_path: Path | None = None,
) -> TrainResult:
    """Train the directional baseline and optionally save the bundle.

    ``frame`` is raw OHLCV. Features and the next-bar label are built inside
    this function. The saved model is the refit winner. It has not been fit
    on the test labels.
    """
    logger.info("Starting model training. compare_random_forest=%s.", compare_random_forest)
    supervised = supervised_dataset(frame)
    split = chronological_split(supervised)
    selection_scaler = StandardScaler()
    x_train = selection_scaler.fit_transform(split.train.loc[:, FEATURE_COLUMNS])
    y_train = split.train["target"].to_numpy()

    candidates: dict[str, Any] = {"logistic_regression": _logistic_regression()}
    if compare_random_forest:
        candidates["random_forest"] = _random_forest()

    comparison: dict[str, dict[str, Any]] = {}
    for name, candidate in candidates.items():
        logger.info("Fitting %s on %s training rows.", name, len(split.train))
        candidate.fit(x_train, y_train)
        comparison[name] = evaluate_model(
            candidate,
            selection_scaler,
            split.validation,
            split.validation["target"],
        )

    selected = _select_model(comparison)
    logger.info("Validation selected %s.", selected)
    fit_frame = pd.concat([split.train, split.validation], axis=0)
    final_scaler = StandardScaler()
    x_fit = final_scaler.fit_transform(fit_frame.loc[:, FEATURE_COLUMNS])
    final_model = _logistic_regression() if selected == "logistic_regression" else _random_forest()
    final_model.fit(x_fit, fit_frame["target"].to_numpy())
    test_metrics = evaluate_model(final_model, final_scaler, split.test, split.test["target"])
    test_predictions = predict_probability(final_model, final_scaler, split.test)
    trained_through = pd.Timestamp(fit_frame["timestamp"].iloc[-1]).isoformat()
    test_start = pd.Timestamp(split.test["timestamp"].iloc[0]).isoformat()
    split_info = {
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "embargo_rows": int(len(split.embargo)),
        "test_rows": int(len(split.test)),
        "trained_through": trained_through,
        "test_start": test_start,
        "embargo": (
            "The row before the test set is excluded so its label cannot use the first test close."
        ),
    }
    result = TrainResult(
        selected_model=selected,
        model=final_model,
        scaler=final_scaler,
        feature_columns=list(FEATURE_COLUMNS),
        validation_metrics=comparison[selected],
        test_metrics=test_metrics,
        comparison=comparison,
        split_info=split_info,
        test_predictions=test_predictions,
        trained_through=trained_through,
        test_start=test_start,
    )
    if model_path is not None:
        _save_bundle(result, model_path)
    logger.info(
        "Training finished. Selected=%s test_rows=%s roc_auc=%s.",
        selected,
        test_metrics["sample_count"],
        test_metrics["roc_auc"],
    )
    return result


def _logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)


def _random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )


def _select_model(comparison: dict[str, dict[str, Any]]) -> str:
    """Pick the higher validation ROC-AUC. Ties, and missing AUC, prefer logistic regression."""

    def rank(metrics: dict[str, Any]) -> tuple[int, float]:
        roc_auc = metrics.get("roc_auc")
        if roc_auc is None:
            return (0, float(metrics.get("f1") or 0.0))
        return (1, float(roc_auc))

    selected = "logistic_regression"
    best = rank(comparison[selected])
    for name, metrics in comparison.items():
        if name == selected:
            continue
        candidate_rank = rank(metrics)
        if candidate_rank > best:
            selected = name
            best = candidate_rank
    return selected


def _save_bundle(result: TrainResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": result.model,
        "scaler": result.scaler,
        "feature_columns": result.feature_columns,
        "model_name": result.selected_model,
        "trained_through": result.trained_through,
        "test_start": result.test_start,
        "test_metrics": result.test_metrics,
        "validation_metrics": result.validation_metrics,
        "comparison": result.comparison,
        "split": result.split_info,
        "saved_at": datetime.now(UTC).isoformat(),
        "disclaimer": (
            "Directional classification baseline. Test metrics are not a promise of profit."
        ),
    }
    joblib.dump(bundle, path)
    logger.info("Saved model bundle to %s.", path)
