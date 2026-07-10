"""Tests for eda_component_x_structure.py functions."""
import pytest

from eda_component_x_structure import infer_sensor_role


class TestInferSensorRole:

    def test_cumulative_counter_high_monotonicity(self):
        role = infer_sensor_role("100", 1, 0.99)
        assert "CUMULATIVE COUNTER" in role

    def test_fine_grain_histogram_16_bins(self):
        role = infer_sensor_role("397", 36, 0.5)
        assert "FINE-GRAIN HISTOGRAM" in role

    def test_fine_grain_histogram_exact_16(self):
        role = infer_sensor_role("397", 16, 0.5)
        assert "FINE-GRAIN HISTOGRAM" in role

    def test_med_grain_histogram_10_bins(self):
        role = infer_sensor_role("459", 10, 0.5)
        assert "MED-GRAIN HISTOGRAM" in role

    def test_med_grain_histogram_15_bins(self):
        role = infer_sensor_role("291", 15, 0.5)
        assert "MED-GRAIN HISTOGRAM" in role

    def test_coarse_histogram_5_bins(self):
        role = infer_sensor_role("158", 5, 0.5)
        assert "COARSE HISTOGRAM" in role

    def test_coarse_histogram_9_bins(self):
        role = infer_sensor_role("158", 9, 0.5)
        assert "COARSE HISTOGRAM" in role

    def test_single_value_gauge_1_bin(self):
        role = infer_sensor_role("100", 1, 0.5)
        assert "SINGLE-VALUE GAUGE" in role

    def test_single_value_gauge_4_bins(self):
        role = infer_sensor_role("100", 4, 0.5)
        assert "SINGLE-VALUE GAUGE" in role

    def test_monotonicity_threshold_0_96(self):
        role = infer_sensor_role("100", 1, 0.96)
        assert "CUMULATIVE COUNTER" in role

    def test_monotonicity_threshold_0_95_exactly(self):
        """At exactly 0.95, should NOT be cumulative counter (strictly greater)."""
        role = infer_sensor_role("100", 1, 0.95)
        assert "SINGLE-VALUE GAUGE" in role

    def test_monotonicity_0_94(self):
        role = infer_sensor_role("100", 1, 0.94)
        assert "SINGLE-VALUE GAUGE" in role

    def test_family_name_ignored(self):
        """Family name string does not affect output."""
        role1 = infer_sensor_role("AAA", 5, 0.5)
        role2 = infer_sensor_role("BBB", 5, 0.5)
        assert role1 == role2

    def test_returns_string(self):
        result = infer_sensor_role("100", 1, 0.5)
        assert isinstance(result, str)

    def test_no_empty_return(self):
        result = infer_sensor_role("100", 1, 0.5)
        assert len(result) > 0
