"""Tests for train_xgboost_compare.py functions."""
import numpy as np
import pandas as pd
import pytest

from train_xgboost_compare import (
    _build_feature_matrix,
    _build_rf_pipeline,
    _build_xgb_pipeline,
    _evaluate_model,
    _extract_feature_importance,
)


class TestBuildFeatureMatrix:

    def test_separates_X_and_y(self, sample_features_df):
        X, y, cat_cols, num_cols = _build_feature_matrix(sample_features_df)
        assert "in_study_repair" not in X.columns
        assert len(y) == len(sample_features_df)

    def test_drops_required_columns(self, sample_features_df):
        X, y, cat_cols, num_cols = _build_feature_matrix(sample_features_df)
        for col in ["vehicle_id", "in_study_repair", "length_of_study_time_step"]:
            assert col not in X.columns

    def test_identifies_categorical(self, sample_features_df):
        X, y, cat_cols, num_cols = _build_feature_matrix(sample_features_df)
        assert len(cat_cols) > 0

    def test_identifies_numeric(self, sample_features_df):
        X, y, cat_cols, num_cols = _build_feature_matrix(sample_features_df)
        assert len(num_cols) > 0

    def test_returns_four_values(self, sample_features_df):
        result = _build_feature_matrix(sample_features_df)
        assert len(result) == 4


class TestBuildXGBPipeline:

    def test_returns_pipeline(self):
        pipe = _build_xgb_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        assert pipe is not None

    def test_has_preprocessor(self):
        pipe = _build_xgb_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        from sklearn.pipeline import Pipeline
        assert isinstance(pipe, Pipeline)

    def test_transforms_data(self):
        pipe = _build_xgb_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        X = pd.DataFrame({
            "cat1": ["A", "B", "A"],
            "num1": [1.0, 2.0, 3.0],
        })
        y = np.array([0, 1, 0])
        pipe.fit(X, y)
        result = pipe.predict_proba(X)
        assert result.shape[0] == 3

    def test_classifier_is_xgb(self):
        pipe = _build_xgb_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        clf = pipe.named_steps["classifier"]
        assert clf.__class__.__name__ == "XGBClassifier"


class TestBuildRFPipeline:

    def test_returns_pipeline(self):
        pipe = _build_rf_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        assert pipe is not None

    def test_has_preprocessor(self):
        pipe = _build_rf_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        from sklearn.pipeline import Pipeline
        assert isinstance(pipe, Pipeline)

    def test_transforms_data(self):
        pipe = _build_rf_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        X = pd.DataFrame({
            "cat1": ["A", "B", "A"],
            "num1": [1.0, 2.0, 3.0],
        })
        y = np.array([0, 1, 0])
        pipe.fit(X, y)
        result = pipe.predict_proba(X)
        assert result.shape[0] == 3

    def test_classifier_is_rf(self):
        pipe = _build_rf_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        clf = pipe.named_steps["classifier"]
        assert clf.__class__.__name__ == "RandomForestClassifier"


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

    def test_perfect_score(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = _evaluate_model(y_true, y_pred, y_proba)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_confusion_matrix_sum(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.4, 0.2, 0.8, 0.4, 0.9])
        metrics = _evaluate_model(y_true, y_pred, y_proba)
        cm = metrics["confusion_matrix"]
        total = cm["true_negative"] + cm["false_positive"] + cm["false_negative"] + cm["true_positive"]
        assert total == 6

    def test_metrics_are_floats(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = _evaluate_model(y_true, y_pred, y_proba)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            assert isinstance(metrics[key], float)

    def test_zero_division_handled(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        metrics = _evaluate_model(y_true, y_pred, y_proba)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


class TestExtractFeatureImportance:

    def test_returns_dataframe(self):
        pipe = _build_xgb_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        X = pd.DataFrame({
            "cat1": ["A", "B", "A", "B"] * 25,
            "num1": np.random.randn(100),
        })
        y = np.random.randint(0, 2, 100)
        pipe.fit(X, y)
        df = _extract_feature_importance(pipe)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        pipe = _build_xgb_pipeline(cat_cols=["cat1"], num_cols=["num1"])
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

    def test_sorted_descending(self):
        pipe = _build_xgb_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        X = pd.DataFrame({
            "cat1": ["A", "B", "A", "B"] * 25,
            "num1": np.random.randn(100),
        })
        y = np.random.randint(0, 2, 100)
        pipe.fit(X, y)
        df = _extract_feature_importance(pipe)
        if len(df) > 1:
            vals = df["importance"].values
            assert (vals[:-1] >= vals[1:]).all()

    def test_ranks_sequential(self):
        pipe = _build_xgb_pipeline(cat_cols=["cat1"], num_cols=["num1"])
        X = pd.DataFrame({
            "cat1": ["A", "B", "A", "B"] * 25,
            "num1": np.random.randn(100),
        })
        y = np.random.randint(0, 2, 100)
        pipe.fit(X, y)
        df = _extract_feature_importance(pipe)
        expected = list(range(1, len(df) + 1))
        assert list(df["rank"]) == expected
