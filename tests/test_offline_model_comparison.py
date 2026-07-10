"""Tests for offline_model_comparison.py functions."""
import numpy as np
import pandas as pd
import pytest

from offline_model_comparison import (
    _aggregate_sensor_family,
    _cache_is_valid,
    _compute_slope_from_stats,
    _f2_score,
    _init_sensor_stats,
    _try_write_parquet,
)


class TestInitSensorStats:

    def test_returns_dict_with_all_keys(self):
        ss = _init_sensor_stats()
        expected_keys = {
            "count", "sum", "sum_sq", "min", "max",
            "missing", "sum_x", "sum_xy", "sum_x_sq", "last_val",
        }
        assert set(ss.keys()) == expected_keys

    def test_count_zero(self):
        assert _init_sensor_stats()["count"] == 0

    def test_sum_x_sq_initialized(self):
        assert _init_sensor_stats()["sum_x_sq"] == 0.0

    def test_last_val_nan(self):
        assert np.isnan(_init_sensor_stats()["last_val"])


class TestComputeSlopeFromStats:

    def test_perfect_linear(self):
        """y = 2x + 1 with 3 points: (0,1), (1,3), (2,5)"""
        ss = {
            "count": 3,
            "sum": 9.0,
            "sum_x": 3.0,
            "sum_xy": 13.0,
            "sum_x_sq": 5.0,
        }
        slope = _compute_slope_from_stats(ss)
        # n*sum_xy - sum_x*sum = 3*13 - 3*9 = 39 - 27 = 12
        # n*sum_x_sq - sum_x^2 = 3*5 - 9 = 6
        assert slope == pytest.approx(2.0)

    def test_zero_for_less_than_2(self):
        ss = {"count": 1, "sum": 1.0, "sum_x": 0, "sum_xy": 0, "sum_x_sq": 0}
        assert _compute_slope_from_stats(ss) == 0.0

    def test_zero_for_near_zero_denominator(self):
        ss = {"count": 3, "sum": 3.0, "sum_x": 3.0, "sum_xy": 3.0, "sum_x_sq": 3.0}
        assert _compute_slope_from_stats(ss) == 0.0

    def test_missing_sum_x_sq(self):
        ss = {"count": 3, "sum": 9.0, "sum_x": 3.0, "sum_xy": 13.0}
        denom = 3 * 0 - 3 ** 2
        expected = (3 * 13.0 - 3.0 * 9.0) / denom
        assert _compute_slope_from_stats(ss) == pytest.approx(expected)

    def test_returns_float_type(self):
        ss = {"count": 5, "sum": 10.0, "sum_x": 10.0, "sum_xy": 30.0, "sum_x_sq": 20.0}
        result = _compute_slope_from_stats(ss)
        assert isinstance(result, float)


class TestAggregateSensorFamily:

    def test_returns_dataframe(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        assert isinstance(df, pd.DataFrame)

    def test_has_vehicle_id_column(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        assert "vehicle_id" in df.columns

    def test_correct_row_count(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        assert len(df) == 2

    def test_mean_computed(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        row = df[df["vehicle_id"] == 1].iloc[0]
        assert row["s_397_0_mean"] == pytest.approx(15.0)

    def test_nan_for_empty_sensor(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        row = df[df["vehicle_id"] == 2].iloc[0]
        assert np.isnan(row["s_397_0_mean"])

    def test_empty_df_for_unknown_family(self, sample_sensor_stats):
        df = _aggregate_sensor_family("999", sample_sensor_stats)
        assert len(df) == 0

    def test_has_all_feature_columns(self, sample_sensor_stats):
        df = _aggregate_sensor_family("397", sample_sensor_stats)
        suffixes = ["mean", "max", "min", "std", "last", "range", "missing_count", "trend_slope", "coeff_variation"]
        for s in suffixes:
            assert f"s_397_0_{s}" in df.columns


class TestCacheIsValid:

    def test_false_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("offline_model_comparison.ETL_CACHE_PARQUET", str(tmp_path / "nonexistent.parquet"))
        monkeypatch.setattr("offline_model_comparison.TRAIN_READOUTS_CSV", str(tmp_path / "readouts.csv"))
        monkeypatch.setattr("offline_model_comparison.TRAIN_SPECS_CSV", str(tmp_path / "specs.csv"))
        monkeypatch.setattr("offline_model_comparison.TRAIN_TTE_CSV", str(tmp_path / "tte.csv"))
        # Create source files
        (tmp_path / "readouts.csv").touch()
        (tmp_path / "specs.csv").touch()
        (tmp_path / "tte.csv").touch()
        assert _cache_is_valid() is False

    def test_true_when_cache_newer(self, tmp_path, monkeypatch, tmp_path_factory):
        cache_file = tmp_path / "cache.parquet"
        readouts = tmp_path / "readouts.csv"
        specs = tmp_path / "specs.csv"
        tte = tmp_path / "tte.csv"

        # Create source files first (older)
        readouts.touch()
        specs.touch()
        tte.touch()

        import time
        time.sleep(0.05)

        # Create cache file (newer)
        pd.DataFrame({"a": [1]}).to_parquet(str(cache_file))

        monkeypatch.setattr("offline_model_comparison.ETL_CACHE_PARQUET", str(cache_file))
        monkeypatch.setattr("offline_model_comparison.TRAIN_READOUTS_CSV", str(readouts))
        monkeypatch.setattr("offline_model_comparison.TRAIN_SPECS_CSV", str(specs))
        monkeypatch.setattr("offline_model_comparison.TRAIN_TTE_CSV", str(tte))

        assert _cache_is_valid() is True

    def test_false_when_source_newer(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.parquet"
        readouts = tmp_path / "readouts.csv"
        specs = tmp_path / "specs.csv"
        tte = tmp_path / "tte.csv"

        pd.DataFrame({"a": [1]}).to_parquet(str(cache_file))

        import time
        time.sleep(0.05)
        readouts.touch()
        specs.touch()
        tte.touch()

        monkeypatch.setattr("offline_model_comparison.ETL_CACHE_PARQUET", str(cache_file))
        monkeypatch.setattr("offline_model_comparison.TRAIN_READOUTS_CSV", str(readouts))
        monkeypatch.setattr("offline_model_comparison.TRAIN_SPECS_CSV", str(specs))
        monkeypatch.setattr("offline_model_comparison.TRAIN_TTE_CSV", str(tte))

        assert _cache_is_valid() is False


class TestTryWriteParquet:

    def test_writes_parquet_file(self, tmp_path):
        path = str(tmp_path / "test.parquet")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        _try_write_parquet(df, path)
        assert pd.io.common.file_exists(path)
        loaded = pd.read_parquet(path)
        assert len(loaded) == 3

    def test_writes_csv_fallback_when_pyarrow_missing(self, tmp_path, monkeypatch):
        path = str(tmp_path / "test.parquet")
        df = pd.DataFrame({"a": [1, 2]})

        def mock_to_parquet(*args, **kwargs):
            raise ImportError("pyarrow not found")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)
        _try_write_parquet(df, path)
        fallback = path.rsplit(".", 1)[0] + ".csv"
        assert pd.io.common.file_exists(fallback)


class TestF2Score:

    def test_perfect_f2(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert _f2_score(y_true, y_pred) == pytest.approx(1.0)

    def test_zero_f2_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert _f2_score(y_true, y_pred) == pytest.approx(0.0)

    def test_favors_recall(self):
        """F2 weights recall 4x more than precision."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1, 0])
        # High recall, lower precision
        f2 = _f2_score(y_true, y_pred)
        assert 0.0 < f2 <= 1.0

    def test_returns_float(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = _f2_score(y_true, y_pred)
        assert isinstance(result, float)

    def test_all_negative_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        f2 = _f2_score(y_true, y_pred)
        assert f2 == pytest.approx(0.0)

    def test_all_positive_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        f2 = _f2_score(y_true, y_pred)
        assert 0.0 < f2 < 1.0
