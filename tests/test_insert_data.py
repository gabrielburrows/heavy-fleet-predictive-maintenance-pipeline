"""Tests for insert_data.py helper functions."""
import numpy as np
import pandas as pd
import pytest

from insert_data import (
    _aggregate_sensor_family,
    _assign_vehicles_to_accounts,
    _compute_slope_from_stats,
    _detect_sensor_families,
    _generate_company_pool,
    _init_sensor_stats,
    _update_sensor_stats,
)


class TestInitSensorStats:

    def test_returns_correct_keys(self):
        ss = _init_sensor_stats()
        expected_keys = {
            "count", "sum", "sum_sq", "min", "max",
            "missing", "sum_x", "sum_xy", "last_val",
        }
        assert set(ss.keys()) == expected_keys

    def test_count_is_zero(self):
        assert _init_sensor_stats()["count"] == 0

    def test_sum_is_zero(self):
        assert _init_sensor_stats()["sum"] == 0.0

    def test_min_is_inf(self):
        assert _init_sensor_stats()["min"] == float("inf")

    def test_max_is_neg_inf(self):
        assert _init_sensor_stats()["max"] == float("-inf")

    def test_missing_is_zero(self):
        assert _init_sensor_stats()["missing"] == 0

    def test_last_val_is_nan(self):
        assert np.isnan(_init_sensor_stats()["last_val"])


class TestUpdateSensorStats:

    def test_updates_count(self):
        ss = _init_sensor_stats()
        vals = np.array([1.0, 2.0, 3.0])
        times = np.array([0, 1, 2])
        _update_sensor_stats(ss, vals, times)
        assert ss["count"] == 3

    def test_updates_sum(self):
        ss = _init_sensor_stats()
        vals = np.array([1.0, 2.0, 3.0])
        times = np.array([0, 1, 2])
        _update_sensor_stats(ss, vals, times)
        assert ss["sum"] == pytest.approx(6.0)

    def test_updates_sum_sq(self):
        ss = _init_sensor_stats()
        vals = np.array([1.0, 2.0, 3.0])
        times = np.array([0, 1, 2])
        _update_sensor_stats(ss, vals, times)
        assert ss["sum_sq"] == pytest.approx(14.0)

    def test_updates_min_max(self):
        ss = _init_sensor_stats()
        vals = np.array([5.0, 10.0, 3.0])
        times = np.array([0, 1, 2])
        _update_sensor_stats(ss, vals, times)
        assert ss["min"] == 3.0
        assert ss["max"] == 10.0

    def test_updates_last_val(self):
        ss = _init_sensor_stats()
        vals = np.array([1.0, 2.0, 3.0])
        times = np.array([0, 1, 2])
        _update_sensor_stats(ss, vals, times)
        assert ss["last_val"] == pytest.approx(3.0)

    def test_handles_nan_values(self):
        ss = _init_sensor_stats()
        vals = np.array([1.0, np.nan, 3.0])
        times = np.array([0, 1, 2])
        _update_sensor_stats(ss, vals, times)
        assert ss["count"] == 2

    def test_handles_all_nan(self):
        ss = _init_sensor_stats()
        vals = np.array([np.nan, np.nan])
        times = np.array([0, 1])
        _update_sensor_stats(ss, vals, times)
        assert ss["count"] == 0
        assert ss["min"] == float("inf")
        assert ss["max"] == float("-inf")

    def test_accumulates_across_calls(self):
        ss = _init_sensor_stats()
        _update_sensor_stats(ss, np.array([1.0, 2.0]), np.array([0, 1]))
        _update_sensor_stats(ss, np.array([3.0, 4.0]), np.array([2, 3]))
        assert ss["count"] == 4
        assert ss["sum"] == pytest.approx(10.0)

    def test_updates_sum_x_and_sum_xy(self):
        ss = _init_sensor_stats()
        vals = np.array([2.0, 4.0])
        times = np.array([1, 2])
        _update_sensor_stats(ss, vals, times)
        assert ss["sum_x"] == pytest.approx(3.0)
        assert ss["sum_xy"] == pytest.approx(10.0)


class TestComputeSlopeFromStats:

    def test_zero_slope_perfect_linear(self):
        """y = 2x + 1 => slope = 2."""
        ss = {
            "count": 3,
            "sum": 10.0,
            "sum_x": 3.0,
            "sum_xy": 14.0,
            "sum_x_sq": 5.0,
        }
        slope = _compute_slope_from_stats(ss)
        expected = (3 * 14.0 - 3.0 * 10.0) / (3 * 5.0 - 3.0 ** 2)
        assert slope == pytest.approx(expected)

    def test_returns_zero_for_less_than_2_points(self):
        ss = {"count": 1, "sum": 5.0, "sum_x": 1.0, "sum_xy": 5.0, "sum_x_sq": 1.0}
        assert _compute_slope_from_stats(ss) == 0.0

    def test_returns_zero_for_zero_denominator(self):
        ss = {"count": 3, "sum": 9.0, "sum_x": 3.0, "sum_xy": 9.0, "sum_x_sq": 3.0}
        assert _compute_slope_from_stats(ss) == 0.0

    def test_returns_nonzero_when_sum_x_sq_is_zero(self):
        ss = {"count": 3, "sum": 10.0, "sum_x": 3.0, "sum_xy": 14.0}
        slope = _compute_slope_from_stats(ss)
        denom = 3 * 0 - 3 ** 2
        expected = (3 * 14.0 - 3.0 * 10.0) / denom
        assert slope == pytest.approx(expected)

    def test_returns_float(self):
        ss = {
            "count": 5,
            "sum": 15.0,
            "sum_x": 10.0,
            "sum_xy": 50.0,
            "sum_x_sq": 30.0,
        }
        result = _compute_slope_from_stats(ss)
        assert isinstance(result, float)


class TestAggregateSensorFamily:

    def test_returns_dataframe_with_vehicle_id(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        assert "vehicle_id" in df.columns

    def test_returns_correct_number_of_rows(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        assert len(df) == 2

    def test_computed_mean(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        row1 = df[df["vehicle_id"] == 1].iloc[0]
        assert row1["s_397_0_mean"] == pytest.approx(15.0)

    def test_computed_std(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        row1 = df[df["vehicle_id"] == 1].iloc[0]
        assert not np.isnan(row1["s_397_0_std"])

    def test_nan_for_empty_sensor(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        row2 = df[df["vehicle_id"] == 2].iloc[0]
        assert np.isnan(row2["s_397_0_mean"])

    def test_returns_empty_df_for_unknown_family(self, sample_sensor_stats):
        df = _aggregate_sensor_family("999", sample_sensor_stats)
        assert len(df) == 0
        assert "vehicle_id" in df.columns

    def test_has_all_expected_columns(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        expected_suffixes = [
            "mean", "max", "min", "std", "last", "range",
            "missing_count", "trend_slope", "coeff_variation",
        ]
        for suffix in expected_suffixes:
            assert f"s_397_0_{suffix}" in df.columns


class TestDetectSensorFamilies:

    def test_detects_single_digit_families(self):
        cols = ["397_0", "397_1", "vehicle_id", "time_step"]
        result = _detect_sensor_families(cols)
        assert result == ["397"]

    def test_detects_multiple_families(self):
        cols = ["100_0", "397_0", "459_0", "vehicle_id"]
        result = _detect_sensor_families(cols)
        assert result == ["100", "397", "459"]

    def test_sorted_by_integer_value(self):
        cols = ["459_0", "100_0", "397_0"]
        result = _detect_sensor_families(cols)
        assert result == ["100", "397", "459"]

    def test_ignores_non_sensor_columns(self):
        cols = ["vehicle_id", "time_step", "Spec_0"]
        result = _detect_sensor_families(cols)
        assert result == []

    def test_empty_input(self):
        assert _detect_sensor_families([]) == []


class TestGenerateCompanyPool:

    def test_returns_dataframe(self, rng):
        df = _generate_company_pool(rng)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_companies(self, rng):
        df = _generate_company_pool(rng)
        assert len(df) == 25

    def test_has_required_columns(self, rng):
        df = _generate_company_pool(rng)
        for col in ["account_id", "company_name", "industry", "operating_prefecture"]:
            assert col in df.columns

    def test_account_ids_start_at_1(self, rng):
        df = _generate_company_pool(rng)
        assert df["account_id"].min() == 1
        assert df["account_id"].max() == 25

    def test_industry_values_valid(self, rng):
        df = _generate_company_pool(rng)
        valid_industries = {
            "Freight Transport", "Cold Chain Logistics",
            "Construction Materials", "Industrial Parts",
            "E-commerce Delivery", "Waste Management",
        }
        assert set(df["industry"].unique()).issubset(valid_industries)


class TestAssignVehiclesToAccounts:

    def test_returns_dataframe(self, rng):
        vehicle_ids = np.array([1, 2, 3, 4, 5])
        company_pool = pd.DataFrame({
            "account_id": [1, 2],
            "company_name": ["A", "B"],
        })
        df = _assign_vehicles_to_accounts(vehicle_ids, company_pool, rng)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, rng):
        vehicle_ids = np.array([1, 2, 3])
        company_pool = pd.DataFrame({
            "account_id": [1, 2],
            "company_name": ["A", "B"],
        })
        df = _assign_vehicles_to_accounts(vehicle_ids, company_pool, rng)
        assert len(df) == 3

    def test_has_required_columns(self, rng):
        vehicle_ids = np.array([1, 2, 3])
        company_pool = pd.DataFrame({
            "account_id": [1, 2],
            "company_name": ["A", "B"],
        })
        df = _assign_vehicles_to_accounts(vehicle_ids, company_pool, rng)
        for col in ["vehicle_id", "account_id", "vehicle_age_years", "vehicle_model", "engine_family", "manufacture_year"]:
            assert col in df.columns

    def test_account_ids_from_pool(self, rng):
        vehicle_ids = np.array(list(range(100)))
        company_pool = pd.DataFrame({
            "account_id": [1, 2],
            "company_name": ["A", "B"],
        })
        df = _assign_vehicles_to_accounts(vehicle_ids, company_pool, rng)
        assert set(df["account_id"].unique()).issubset({1, 2})

    def test_vehicle_age_range(self, rng):
        vehicle_ids = np.array([1, 2, 3])
        company_pool = pd.DataFrame({
            "account_id": [1],
            "company_name": ["A"],
        })
        df = _assign_vehicles_to_accounts(vehicle_ids, company_pool, rng)
        assert (df["vehicle_age_years"] >= 1).all()
        assert (df["vehicle_age_years"] <= 15).all()
