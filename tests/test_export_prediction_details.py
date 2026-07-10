"""Tests for export_prediction_details.py functions."""
import numpy as np
import pandas as pd
import pytest

from export_prediction_details import (
    _classify_priority,
    _export_predictions,
    _export_risk_tier_summary,
    _fill_missing_columns,
    _get_model_columns,
)
from config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD


class TestClassifyPriority:

    def test_critical_at_high_threshold(self):
        assert _classify_priority(HIGH_RISK_THRESHOLD) == "CRITICAL DISPATCH"

    def test_preventative_at_medium(self):
        assert _classify_priority(MEDIUM_RISK_THRESHOLD) == "PREVENTATIVE MONITOR"

    def test_stable_below_medium(self):
        assert _classify_priority(0.0) == "STABLE OPERATION"

    def test_boundary_just_below_high(self):
        val = HIGH_RISK_THRESHOLD - 0.0001
        result = _classify_priority(val)
        assert result != "CRITICAL DISPATCH"

    def test_boundary_exactly_medium(self):
        assert _classify_priority(MEDIUM_RISK_THRESHOLD) == "PREVENTATIVE MONITOR"


class MockPipeline:
    """Minimal mock of a fitted sklearn Pipeline for testing."""

    def __init__(self, cat_cols, num_cols):
        self.named_steps = {
            "preprocessor": MockPreprocessor(cat_cols, num_cols),
            "classifier": MockClassifier(),
        }

    def predict_proba(self, X):
        return MockClassifier().predict_proba(X)

    def predict(self, X):
        return MockClassifier().predict(X)


class MockPreprocessor:
    def __init__(self, cat_cols, num_cols):
        self.transformers_ = [
            ("num", "passthrough", num_cols),
            ("cat", "passthrough", cat_cols),
        ]


class MockClassifier:
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TestGetModelColumns:

    def test_returns_combined_columns(self):
        model = MockPipeline(cat_cols=["cat1", "cat2"], num_cols=["num1", "num2"])
        cols = _get_model_columns(model)
        assert set(cols) == {"cat1", "cat2", "num1", "num2"}

    def test_deduplicates(self):
        model = MockPipeline(cat_cols=["x"], num_cols=["x"])
        cols = _get_model_columns(model)
        assert cols.count("x") == 1


class TestFillMissingColumns:

    def test_adds_missing_columns(self):
        X = pd.DataFrame({"num1": [1.0, 2.0]})
        model = MockPipeline(cat_cols=["cat1"], num_cols=["num1", "num2"])
        result = _fill_missing_columns(X, model)
        assert "cat1" in result.columns
        assert "num2" in result.columns
        assert result["cat1"].tolist() == [0, 0]
        assert result["num2"].tolist() == [0, 0]

    def test_no_change_when_all_present(self):
        X = pd.DataFrame({"cat1": ["A", "B"], "num1": [1.0, 2.0]})
        model = MockPipeline(cat_cols=["cat1"], num_cols=["num1"])
        result = _fill_missing_columns(X, model)
        assert len(result.columns) == len(X.columns)

    def test_preserves_existing_values(self):
        X = pd.DataFrame({"num1": [5.0, 10.0]})
        model = MockPipeline(cat_cols=["cat1"], num_cols=["num1"])
        result = _fill_missing_columns(X, model)
        assert result["num1"].tolist() == [5.0, 10.0]


class TestExportPredictions:
    """Test _export_predictions writes valid CSV."""

    def test_writes_csv(self, tmp_path, monkeypatch):
        output_path = str(tmp_path / "predictions.csv")
        monkeypatch.setattr("export_prediction_details.PREDICTIONS_PATH", output_path)

        df = pd.DataFrame({
            "vehicle_id": [1, 2, 3],
            "in_study_repair": [1, 0, 0],
            "length_of_study_time_step": [100, 100, 100],
            "num1": [1.0, 2.0, 3.0],
        })
        model = MockPipeline(cat_cols=[], num_cols=["num1"])
        _export_predictions(df, model)

        assert pd.io.common.file_exists(output_path)
        result = pd.read_csv(output_path)
        assert "Vehicle Id" in result.columns
        assert "Actual" in result.columns
        assert "Predicted" in result.columns
        assert "Failure Probability" in result.columns
        assert "Maintenance Priority" in result.columns

    def test_correct_row_count(self, tmp_path, monkeypatch):
        output_path = str(tmp_path / "predictions.csv")
        monkeypatch.setattr("export_prediction_details.PREDICTIONS_PATH", output_path)

        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "in_study_repair": [1, 0],
            "length_of_study_time_step": [100, 100],
            "num1": [1.0, 2.0],
        })
        model = MockPipeline(cat_cols=[], num_cols=["num1"])
        _export_predictions(df, model)

        result = pd.read_csv(output_path)
        assert len(result) == 2

    def test_handles_nan_actual(self, tmp_path, monkeypatch):
        output_path = str(tmp_path / "predictions.csv")
        monkeypatch.setattr("export_prediction_details.PREDICTIONS_PATH", output_path)

        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "in_study_repair": [np.nan, 0],
            "length_of_study_time_step": [100, 100],
            "num1": [1.0, 2.0],
        })
        model = MockPipeline(cat_cols=[], num_cols=["num1"])
        _export_predictions(df, model)

        result = pd.read_csv(output_path)
        assert result["Actual"].iloc[0] == -1


class TestExportRiskTierSummary:
    """Test _export_risk_tier_summary writes valid CSV."""

    def test_writes_csv(self, tmp_path, monkeypatch):
        output_path = str(tmp_path / "risk_tier_summary.csv")
        monkeypatch.setattr("export_prediction_details.RISK_TIER_SUMMARY_PATH", output_path)

        df = pd.DataFrame({
            "vehicle_id": [1, 2, 3],
            "in_study_repair": [1, 0, 0],
            "length_of_study_time_step": [100, 100, 100],
            "num1": [1.0, 2.0, 3.0],
        })
        model = MockPipeline(cat_cols=[], num_cols=["num1"])
        _export_risk_tier_summary(df, model)

        result = pd.read_csv(output_path)
        assert "Priority" in result.columns
        assert "Vehicle Count" in result.columns
        assert "Average Probability" in result.columns

    def test_has_expected_tiers(self, tmp_path, monkeypatch):
        output_path = str(tmp_path / "risk_tier_summary.csv")
        monkeypatch.setattr("export_prediction_details.RISK_TIER_SUMMARY_PATH", output_path)

        df = pd.DataFrame({
            "vehicle_id": [1, 2, 3],
            "in_study_repair": [1, 0, 0],
            "length_of_study_time_step": [100, 100, 100],
            "num1": [1.0, 2.0, 3.0],
        })
        model = MockPipeline(cat_cols=[], num_cols=["num1"])
        _export_risk_tier_summary(df, model)

        result = pd.read_csv(output_path)
        tiers = result["Priority"].tolist()
        # Since mock model returns 0.2 probability, all should be CRITICAL DISPATCH
        assert "CRITICAL DISPATCH" in tiers

    def test_sorted_by_probability_desc(self, tmp_path, monkeypatch):
        output_path = str(tmp_path / "risk_tier_summary.csv")
        monkeypatch.setattr("export_prediction_details.RISK_TIER_SUMMARY_PATH", output_path)

        df = pd.DataFrame({
            "vehicle_id": list(range(10)),
            "in_study_repair": [0] * 10,
            "length_of_study_time_step": [100] * 10,
            "num1": [1.0] * 10,
        })
        model = MockPipeline(cat_cols=[], num_cols=["num1"])
        _export_risk_tier_summary(df, model)

        result = pd.read_csv(output_path)
        probs = result["Average Probability"].tolist()
        assert probs == sorted(probs, reverse=True)
