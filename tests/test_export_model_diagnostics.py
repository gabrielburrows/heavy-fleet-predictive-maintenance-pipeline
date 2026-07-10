"""Tests for export_model_diagnostics.py functions."""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from export_model_diagnostics import (
    _categorize_feature,
    _export_feature_importance,
    _export_model_metrics,
)
from config import (
    EVALUATION_METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_IMPORTANCE_TABLEAU_PATH,
    MODEL_METRICS_PATH,
)


class TestCategorizeFeature:

    def test_trend_slope_397_is_exhaust(self):
        assert _categorize_feature("s_397_0_trend_slope") == "Exhaust Temperature"

    def test_trend_slope_459_is_dpf(self):
        assert _categorize_feature("s_459_0_trend_slope") == "DPF Pressure"

    def test_trend_slope_291_is_soot(self):
        assert _categorize_feature("s_291_0_trend_slope") == "Soot"

    def test_trend_slope_unknown_is_trend(self):
        assert _categorize_feature("s_100_0_trend_slope") == "Trend"

    def test_mean_397_is_exhaust(self):
        assert _categorize_feature("s_397_0_mean") == "Exhaust Temperature"

    def test_mean_459_is_dpf(self):
        assert _categorize_feature("s_459_5_mean") == "DPF Pressure"

    def test_mean_291_is_soot(self):
        assert _categorize_feature("s_291_0_mean") == "Soot"

    def test_mean_unknown_is_other(self):
        assert _categorize_feature("s_158_0_mean") == "Other"

    def test_max_397_is_exhaust(self):
        assert _categorize_feature("s_397_0_max") == "Exhaust Temperature"

    def test_std_459_is_dpf(self):
        assert _categorize_feature("s_459_0_std") == "DPF Pressure"

    def test_std_291_is_soot(self):
        assert _categorize_feature("s_291_3_std") == "Soot"

    def test_unknown_metric_is_other(self):
        assert _categorize_feature("vehicle_age_years") == "Other"

    def test_case_insensitive(self):
        assert _categorize_feature("S_397_0_MEAN") == "Exhaust Temperature"

    def test_empty_string_is_other(self):
        assert _categorize_feature("") == "Other"


class TestExportModelMetrics:
    """Test _export_model_metrics writes valid CSV."""

    def test_writes_csv_file(self, tmp_path, monkeypatch):
        metrics_path = str(tmp_path / "metrics.json")
        output_path = str(tmp_path / "model_metrics.csv")

        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy": 0.85,
                "precision": 0.80,
                "recall": 0.75,
                "f1": 0.77,
                "roc_auc": 0.90,
                "pr_auc": 0.60,
                "confusion_matrix": {
                    "true_negative": 100,
                    "false_positive": 10,
                    "false_negative": 20,
                    "true_positive": 80,
                },
            }, f)

        monkeypatch.setattr("export_model_diagnostics.EVALUATION_METRICS_PATH", metrics_path)
        monkeypatch.setattr("export_model_diagnostics.MODEL_METRICS_PATH", output_path)

        _export_model_metrics()

        assert os.path.exists(output_path)
        df = pd.read_csv(output_path)
        assert len(df) > 0
        assert "Metric" in df.columns
        assert "Value" in df.columns

    def test_includes_all_metrics(self, tmp_path, monkeypatch):
        metrics_path = str(tmp_path / "metrics.json")
        output_path = str(tmp_path / "model_metrics.csv")

        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy": 0.90,
                "precision": 0.85,
                "recall": 0.80,
                "f1": 0.82,
                "roc_auc": 0.92,
                "pr_auc": 0.70,
                "confusion_matrix": {
                    "true_negative": 200,
                    "false_positive": 10,
                    "false_negative": 15,
                    "true_positive": 100,
                },
            }, f)

        monkeypatch.setattr("export_model_diagnostics.EVALUATION_METRICS_PATH", metrics_path)
        monkeypatch.setattr("export_model_diagnostics.MODEL_METRICS_PATH", output_path)

        _export_model_metrics()

        df = pd.read_csv(output_path)
        metric_names = df["Metric"].tolist()
        for name in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"]:
            assert name in metric_names

    def test_includes_cm_values(self, tmp_path, monkeypatch):
        metrics_path = str(tmp_path / "metrics.json")
        output_path = str(tmp_path / "model_metrics.csv")

        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy": 0.90,
                "precision": 0.85,
                "recall": 0.80,
                "f1": 0.82,
                "roc_auc": 0.92,
                "pr_auc": 0.70,
                "confusion_matrix": {
                    "true_negative": 500,
                    "false_positive": 50,
                    "false_negative": 30,
                    "true_positive": 200,
                },
            }, f)

        monkeypatch.setattr("export_model_diagnostics.EVALUATION_METRICS_PATH", metrics_path)
        monkeypatch.setattr("export_model_diagnostics.MODEL_METRICS_PATH", output_path)

        _export_model_metrics()

        df = pd.read_csv(output_path)
        metric_names = df["Metric"].tolist()
        assert "True Negative" in metric_names
        assert "False Positive" in metric_names
        assert "False Negative" in metric_names
        assert "True Positive" in metric_names

    def test_includes_algorithm_metadata(self, tmp_path, monkeypatch):
        metrics_path = str(tmp_path / "metrics.json")
        output_path = str(tmp_path / "model_metrics.csv")

        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy": 0.90,
                "precision": 0.85,
                "recall": 0.80,
                "f1": 0.82,
                "roc_auc": 0.92,
                "pr_auc": 0.70,
                "confusion_matrix": {
                    "true_negative": 100,
                    "false_positive": 10,
                    "false_negative": 20,
                    "true_positive": 80,
                },
            }, f)

        monkeypatch.setattr("export_model_diagnostics.EVALUATION_METRICS_PATH", metrics_path)
        monkeypatch.setattr("export_model_diagnostics.MODEL_METRICS_PATH", output_path)

        _export_model_metrics()

        df = pd.read_csv(output_path)
        assert "Algorithm" in df.columns
        assert "Model Version" in df.columns


class TestExportFeatureImportance:
    """Test _export_feature_importance writes valid CSV."""

    def test_writes_csv_file(self, tmp_path, monkeypatch):
        fi_path = str(tmp_path / "fi_output.csv")
        output_path = str(tmp_path / "fi_tableau.csv")

        fi_df = pd.DataFrame({
            "feature": ["s_397_0_mean", "s_459_0_mean", "vehicle_age_years"],
            "importance": [0.5, 0.3, 0.2],
            "rank": [1, 2, 3],
        })
        fi_df.to_csv(fi_path, index=False)

        monkeypatch.setattr("export_model_diagnostics.FEATURE_IMPORTANCE_PATH", fi_path)
        monkeypatch.setattr("export_model_diagnostics.FEATURE_IMPORTANCE_TABLEAU_PATH", output_path)

        _export_feature_importance()

        assert os.path.exists(output_path)
        df = pd.read_csv(output_path)
        assert "Category" in df.columns

    def test_categorizes_all_features(self, tmp_path, monkeypatch):
        fi_path = str(tmp_path / "fi_output.csv")
        output_path = str(tmp_path / "fi_tableau.csv")

        fi_df = pd.DataFrame({
            "feature": ["s_397_0_mean", "s_459_1_trend_slope", "s_291_0_std", "Spec_0"],
            "importance": [0.4, 0.3, 0.2, 0.1],
            "rank": [1, 2, 3, 4],
        })
        fi_df.to_csv(fi_path, index=False)

        monkeypatch.setattr("export_model_diagnostics.FEATURE_IMPORTANCE_PATH", fi_path)
        monkeypatch.setattr("export_model_diagnostics.FEATURE_IMPORTANCE_TABLEAU_PATH", output_path)

        _export_feature_importance()

        df = pd.read_csv(output_path)
        categories = df["Category"].tolist()
        assert "Exhaust Temperature" in categories
        assert "DPF Pressure" in categories
        assert "Soot" in categories
        assert "Other" in categories

    def test_no_crash_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "export_model_diagnostics.FEATURE_IMPORTANCE_PATH",
            str(tmp_path / "nonexistent.csv"),
        )
        _export_feature_importance()

    def test_ranks_reordered(self, tmp_path, monkeypatch):
        fi_path = str(tmp_path / "fi_output.csv")
        output_path = str(tmp_path / "fi_tableau.csv")

        fi_df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "importance": [0.1, 0.3, 0.2],
            "rank": [1, 2, 3],
        })
        fi_df.to_csv(fi_path, index=False)

        monkeypatch.setattr("export_model_diagnostics.FEATURE_IMPORTANCE_PATH", fi_path)
        monkeypatch.setattr("export_model_diagnostics.FEATURE_IMPORTANCE_TABLEAU_PATH", output_path)

        _export_feature_importance()

        df = pd.read_csv(output_path)
        assert list(df["Rank"]) == [1, 2, 3]
        assert df["Importance"].iloc[0] == 0.3
