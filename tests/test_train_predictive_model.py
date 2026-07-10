"""Tests for train_predictive_model.py helper functions."""
import json
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from train_predictive_model import (
    _build_feature_matrix,
    _build_pipeline,
    _evaluate_model,
    _extract_feature_importance,
    _set_class_weight,
)


class TestBuildFeatureMatrix:

    def test_separates_X_and_y(self, sample_features_df):
        X, y, cat_cols, num_cols = _build_feature_matrix(sample_features_df)
        assert "in_study_repair" not in X.columns
        assert len(y) == len(sample_features_df)

    def test_drops_correct_columns(self, sample_features_df):
        X, y, cat_cols, num_cols = _build_feature_matrix(sample_features_df)
        for col in ["vehicle_id", "in_study_repair", "length_of_study_time_step"]:
            assert col not in X.columns

    def test_identifies_categorical_columns(self, sample_features_df):
        X, y, cat_cols, num_cols = _build_feature_matrix(sample_features_df)
        assert len(cat_cols) > 0
        for c in cat_cols:
            assert sample_features_df[c].dtype == "object"

    def test_identifies_numeric_columns(self, sample_features_df):
        X, y, cat_cols, num_cols = _build_feature_matrix(sample_features_df)
        assert len(num_cols) > 0

    def test_y_values_correct(self, sample_features_df):
        X, y, cat_cols, num_cols = _build_feature_matrix(sample_features_df)
        expected = sample_features_df["in_study_repair"].values
        np.testing.assert_array_equal(y.values, expected)


class TestBuildPipeline:

    def test_returns_pipeline(self):
        pipe = _build_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_preprocessor(self):
        pipe = _build_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        assert "preprocessor" in pipe.named_steps

    def test_pipeline_has_classifier(self):
        pipe = _build_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        assert "classifier" in pipe.named_steps

    def test_pipeline_transforms_data(self):
        pipe = _build_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        X = pd.DataFrame({
            "cat1": ["A", "B", "A"],
            "num1": [1.0, 2.0, 3.0],
        })
        y = np.array([0, 1, 0])
        pipe.fit(X, y)
        result = pipe.predict_proba(X)
        assert result.shape[0] == 3

    def test_classifier_is_xgboost(self):
        pipe = _build_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        clf = pipe.named_steps["classifier"]
        assert clf.__class__.__name__ == "XGBClassifier"


class TestSetClassWeight:

    def test_updates_scale_pos_weight(self):
        pipe = _build_pipeline(cat_cols=[], num_cols=["num1"])
        y = pd.Series([0, 0, 0, 0, 1])
        _set_class_weight(pipe, y)
        spw = pipe.named_steps["classifier"].scale_pos_weight
        assert spw == pytest.approx(4.0)

    def test_handles_all_negative(self):
        pipe = _build_pipeline(cat_cols=[], num_cols=["num1"])
        y = pd.Series([0, 0, 0])
        _set_class_weight(pipe, y)
        spw = pipe.named_steps["classifier"].scale_pos_weight
        assert spw == pytest.approx(15.0)

    def test_balanced_classes(self):
        pipe = _build_pipeline(cat_cols=[], num_cols=["num1"])
        y = pd.Series([0, 0, 1, 1])
        _set_class_weight(pipe, y)
        spw = pipe.named_steps["classifier"].scale_pos_weight
        assert spw == pytest.approx(1.0)


class TestEvaluateModel:

    def test_returns_all_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = _evaluate_model(y_true, y_pred, y_proba)
        expected_keys = {
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "confusion_matrix", "classification_report",
        }
        assert set(metrics.keys()) == expected_keys

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = _evaluate_model(y_true, y_pred, y_proba)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_confusion_matrix_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = _evaluate_model(y_true, y_pred, y_proba)
        cm = metrics["confusion_matrix"]
        assert set(cm.keys()) == {"true_negative", "false_positive", "false_negative", "true_positive"}

    def test_cm_values_correct(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.4, 0.2, 0.8, 0.4, 0.9])
        metrics = _evaluate_model(y_true, y_pred, y_proba)
        cm = metrics["confusion_matrix"]
        assert cm["true_negative"] == 2
        assert cm["false_positive"] == 1
        assert cm["false_negative"] == 1
        assert cm["true_positive"] == 2

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1])
        metrics = _evaluate_model(y_true, y_pred, y_proba)
        assert metrics["accuracy"] == 0.0

    def test_metrics_are_floats(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = _evaluate_model(y_true, y_pred, y_proba)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            assert isinstance(metrics[key], float)


class TestExtractFeatureImportance:

    def test_returns_dataframe(self):
        pipe = _build_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        X = pd.DataFrame({
            "cat1": ["A", "B", "A", "B"] * 25,
            "num1": np.random.randn(100),
        })
        y = np.random.randint(0, 2, 100)
        pipe.fit(X, y)
        df = _extract_feature_importance(pipe)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        pipe = _build_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        X = pd.DataFrame({
            "cat1": ["A", "B", "A", "B"] * 25,
            "num1": np.random.randn(100),
        })
        y = np.random.randint(0, 2, 100)
        pipe.fit(X, y)
        df = _extract_feature_importance(pipe)
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert "rank" in df.columns

    def test_sorted_by_importance_desc(self):
        pipe = _build_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        X = pd.DataFrame({
            "cat1": ["A", "B", "A", "B"] * 25,
            "num1": np.random.randn(100),
        })
        y = np.random.randint(0, 2, 100)
        pipe.fit(X, y)
        df = _extract_feature_importance(pipe)
        assert (df["importance"].diff().dropna() <= 1e-12).all() or len(df) <= 1

    def test_ranks_are_sequential(self):
        pipe = _build_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        X = pd.DataFrame({
            "cat1": ["A", "B", "A", "B"] * 25,
            "num1": np.random.randn(100),
        })
        y = np.random.randint(0, 2, 100)
        pipe.fit(X, y)
        df = _extract_feature_importance(pipe)
        expected_ranks = list(range(1, len(df) + 1))
        assert list(df["rank"]) == expected_ranks
