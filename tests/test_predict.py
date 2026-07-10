"""Tests for predict.py helper functions."""
import numpy as np
import pandas as pd
import pytest

from predict import (
    _classify_priority,
    _compute_fleet_scores,
    _compute_operational_metrics,
)
from config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD


class TestClassifyPriority:

    def test_critical_dispatch_at_high_threshold(self):
        assert _classify_priority(HIGH_RISK_THRESHOLD) == "CRITICAL DISPATCH"

    def test_critical_dispatch_above_high_threshold(self):
        assert _classify_priority(0.5) == "CRITICAL DISPATCH"

    def test_preventative_monitor_at_medium_threshold(self):
        assert _classify_priority(MEDIUM_RISK_THRESHOLD) == "PREVENTATIVE MONITOR"

    def test_preventative_monitor_between_thresholds(self):
        mid = (HIGH_RISK_THRESHOLD + MEDIUM_RISK_THRESHOLD) / 2
        assert _classify_priority(mid) == "PREVENTATIVE MONITOR"

    def test_stable_operation_below_medium(self):
        assert _classify_priority(MEDIUM_RISK_THRESHOLD - 0.01) == "STABLE OPERATION"

    def test_stable_operation_at_zero(self):
        assert _classify_priority(0.0) == "STABLE OPERATION"

    def test_string_return_type(self):
        result = _classify_priority(0.1)
        assert isinstance(result, str)

    def test_boundary_high_risk_just_below(self):
        val = HIGH_RISK_THRESHOLD - 0.0001
        result = _classify_priority(val)
        assert result != "CRITICAL DISPATCH"


class TestComputeOperationalMetrics:

    def test_creates_exhaust_temperature_columns(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "s_397_0_mean": [100.0, 200.0],
            "s_397_1_mean": [50.0, 60.0],
            "s_397_0_max": [120.0, 220.0],
            "s_397_1_max": [70.0, 80.0],
        })
        _compute_operational_metrics(df)
        assert "avg_exhaust_temperature_c" in df.columns
        assert "peak_exhaust_temperature_c" in df.columns

    def test_creates_dpf_pressure_columns(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "s_459_0_mean": [10.0, 20.0],
            "s_459_0_max": [15.0, 25.0],
        })
        _compute_operational_metrics(df)
        assert "avg_dpf_differential_pressure_kpa" in df.columns
        assert "peak_dpf_differential_pressure_kpa" in df.columns

    def test_creates_soot_mass_columns(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "s_291_0_mean": [5.0, 6.0],
            "s_291_0_max": [7.0, 8.0],
        })
        _compute_operational_metrics(df)
        assert "avg_engine_soot_mass_index" in df.columns
        assert "peak_engine_soot_mass_index" in df.columns

    def test_no_error_without_sensor_columns(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
        })
        _compute_operational_metrics(df)
        assert "avg_exhaust_temperature_c" not in df.columns

    def test_avg_exhaust_temperature_value(self):
        df = pd.DataFrame({
            "vehicle_id": [1],
            "s_397_0_mean": [100.0],
            "s_397_1_mean": [200.0],
        })
        _compute_operational_metrics(df)
        assert df["avg_exhaust_temperature_c"].iloc[0] == pytest.approx(150.0)

    def test_peak_exhaust_temperature_value(self):
        df = pd.DataFrame({
            "vehicle_id": [1],
            "s_397_0_max": [120.0],
            "s_397_1_max": [220.0],
        })
        _compute_operational_metrics(df)
        assert df["peak_exhaust_temperature_c"].iloc[0] == pytest.approx(220.0)

    def test_modifies_in_place(self):
        df = pd.DataFrame({
            "vehicle_id": [1],
            "s_397_0_mean": [100.0],
        })
        _compute_operational_metrics(df)
        assert "avg_exhaust_temperature_c" in df.columns


class TestComputeFleetScores:
    """Test _compute_fleet_scores with a mock model."""

    def _mock_model(self):
        """Return a dict that mimics a fitted sklearn pipeline for predict_proba/predict."""
        class MockModel:
            def predict_proba(self, X):
                return np.column_stack([
                    1 - np.zeros(len(X)),
                    np.zeros(len(X)),
                ])

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        return MockModel()

    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "s_397_0_mean": [10.0, 20.0],
        })
        result = _compute_fleet_scores(df, self._mock_model())
        assert isinstance(result, pd.DataFrame)

    def test_adds_failure_probability(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "s_397_0_mean": [10.0, 20.0],
        })
        result = _compute_fleet_scores(df, self._mock_model())
        assert "failure_probability" in result.columns

    def test_adds_risk_percentage(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "s_397_0_mean": [10.0, 20.0],
        })
        result = _compute_fleet_scores(df, self._mock_model())
        assert "risk_percentage" in result.columns

    def test_adds_maintenance_priority(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "s_397_0_mean": [10.0, 20.0],
        })
        result = _compute_fleet_scores(df, self._mock_model())
        assert "maintenance_priority" in result.columns

    def test_adds_financial_metrics(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "s_397_0_mean": [10.0, 20.0],
        })
        result = _compute_fleet_scores(df, self._mock_model())
        for col in ["expected_breakdown_cost_jpy", "preventative_maintenance_cost_jpy",
                     "maintenance_roi_jpy", "net_savings_jpy"]:
            assert col in result.columns

    def test_adds_sustainability_metrics(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "s_397_0_mean": [10.0, 20.0],
        })
        result = _compute_fleet_scores(df, self._mock_model())
        for col in ["expected_co2_saved_kg", "expected_materials_saved_kg", "expected_repair_avoided"]:
            assert col in result.columns

    def test_adds_model_metadata(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "s_397_0_mean": [10.0, 20.0],
        })
        result = _compute_fleet_scores(df, self._mock_model())
        assert "model_version" in result.columns
        assert "prediction_timestamp" in result.columns

    def test_drops_target_columns_before_inference(self):
        df = pd.DataFrame({
            "vehicle_id": [1, 2],
            "in_study_repair": [0, 1],
            "length_of_study_time_step": [100, 100],
            "s_397_0_mean": [10.0, 20.0],
        })
        result = _compute_fleet_scores(df, self._mock_model())
        assert "failure_probability" in result.columns
