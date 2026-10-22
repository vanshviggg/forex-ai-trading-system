"""Feature leakage checks and the chronological split."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_ohlcv

from forex_ai.ml.evaluation import evaluate_model
from forex_ai.ml.features import FEATURE_COLUMNS, make_target, supervised_dataset
from forex_ai.ml.train import chronological_split, train_model


def test_target_uses_only_the_next_close() -> None:
    close = pd.Series([1.0, 1.2, 1.2, 1.0])
    target = make_target(close)
    assert target.iloc[:3].tolist() == [1.0, 0.0, 0.0]
    assert np.isnan(target.iloc[-1])
    assert "target" not in FEATURE_COLUMNS


def test_training_rejects_unsorted_and_duplicate_timestamps() -> None:
    frame = make_ohlcv(90, seed=1)
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        supervised_dataset(reversed_frame)
    duplicated = frame.copy()
    duplicated.loc[1, "timestamp"] = duplicated.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="Duplicate"):
        supervised_dataset(duplicated)


def test_split_is_a_prefix_with_an_embargo_before_test() -> None:
    frame = make_ohlcv(160, seed=2)
    supervised = supervised_dataset(frame)
    split = chronological_split(supervised)
    assert split.train.index.max() < split.validation.index.min()
    assert split.validation.index.max() < split.embargo.index.min()
    assert split.embargo.index.max() < split.test.index.min()
    assert len(split.embargo) == 1
    ordered = list(split.train.index) + list(split.validation.index)
    ordered += list(split.embargo.index) + list(split.test.index)
    assert ordered == list(supervised.index)


def test_changing_test_closes_does_not_change_earlier_features() -> None:
    frame = make_ohlcv(160, seed=3)
    supervised = supervised_dataset(frame)
    split = chronological_split(supervised)
    changed = frame.copy()
    for index in split.test.index:
        changed.loc[index, "close"] = changed.loc[index, "close"] * 1.25
        changed.loc[index, "high"] = max(
            changed.loc[index, ["open", "high", "close"]].astype(float)
        )
        changed.loc[index, "low"] = min(changed.loc[index, ["open", "low", "close"]].astype(float))
    rebuilt = supervised_dataset(changed)
    pd.testing.assert_frame_equal(
        supervised.loc[split.train.index, FEATURE_COLUMNS],
        rebuilt.loc[split.train.index, FEATURE_COLUMNS],
    )
    pd.testing.assert_frame_equal(
        supervised.loc[split.validation.index, FEATURE_COLUMNS],
        rebuilt.loc[split.validation.index, FEATURE_COLUMNS],
    )
    assert not np.allclose(
        supervised.loc[split.test.index, "close"],
        rebuilt.loc[split.test.index, "close"],
    )


def test_train_model_reports_real_split_metrics(tmp_path) -> None:
    frame = make_ohlcv(180, seed=4)
    model_path = tmp_path / "direction_model.joblib"
    result = train_model(frame, compare_random_forest=True, model_path=model_path)
    assert result.selected_model in {"logistic_regression", "random_forest"}
    assert result.test_metrics["sample_count"] == len(result.test_predictions)
    assert set(result.comparison) == {"logistic_regression", "random_forest"}
    roc_auc = result.test_metrics["roc_auc"]
    assert roc_auc is None or 0.0 <= roc_auc <= 1.0
    assert "train_test_split" not in Path("src/forex_ai/ml/train.py").read_text(encoding="utf-8")
    assert model_path.exists()
    assert result.test_predictions["probability_up"].between(0, 1).all()
    sums = result.test_predictions["probability_up"] + result.test_predictions["probability_down"]
    assert np.allclose(sums, 1.0)


def test_evaluate_model_confusion_matrix_is_label_ordered() -> None:
    class DummyModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, matrix):
            up = matrix[:, 0]
            return np.column_stack([1 - up, up])

    class IdentityScaler:
        def transform(self, matrix):
            return np.asarray(matrix, dtype=float)

    features = pd.DataFrame({"probability": [0.2, 0.8, 0.9, 0.1]})
    for column in FEATURE_COLUMNS:
        features[column] = features["probability"]
    labels = pd.Series([0, 1, 1, 0])
    metrics = evaluate_model(DummyModel(), IdentityScaler(), features, labels)
    # Rows are actual 0, 1 and columns are predicted 0, 1.
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["precision"] == 1.0
    assert metrics["roc_auc"] == 1.0
