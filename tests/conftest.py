"""Shared fixtures for the entire test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Seeded NumPy random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_readouts(rng):
    """Minimal operational-readout DataFrame (3 vehicles, 2 time steps)."""
    return pd.DataFrame({
        "vehicle_id": [1, 1, 2, 2, 3, 3],
        "time_step": [0, 1, 0, 1, 0, 1],
        "397_0": [10.0, 12.0, 20.0, 22.0, 5.0, 6.0],
        "397_1": [0.0, 1.0, 0.0, 2.0, 1.0, 1.5],
        "459_0": [100.0, 105.0, 200.0, 210.0, 50.0, 52.0],
    })


@pytest.fixture
def sample_specs():
    """Minimal specifications DataFrame."""
    return pd.DataFrame({
        "vehicle_id": [1, 2, 3],
        "Spec_0": ["A", "B", "A"],
        "Spec_1": ["X", "Y", "X"],
    })


@pytest.fixture
def sample_tte():
    """Minimal time-to-event label DataFrame."""
    return pd.DataFrame({
        "vehicle_id": [1, 2, 3],
        "in_study_repair": [1, 0, 0],
        "length_of_study_time_step": [100, 100, 100],
    })


@pytest.fixture
def sample_features_df(sample_specs, sample_tte):
    """Merged features DataFrame ready for model training."""
    df = sample_specs.merge(sample_tte, on="vehicle_id", how="inner")
    for i in range(2, 8):
        df[f"Spec_{i}"] = "Z"
    # Add a few numeric sensor-like columns
    df["s_397_0_mean"] = [11.0, 21.0, 5.5]
    df["s_397_1_mean"] = [0.5, 1.0, 1.25]
    df["s_459_0_mean"] = [102.5, 205.0, 51.0]
    return df


@pytest.fixture
def sample_sensor_stats(rng):
    """Pre-populated sensor stats dict for 2 vehicles, 1 sensor family."""
    return {
        1: {
            "397_0": {
                "count": 10,
                "sum": 150.0,
                "sum_sq": 2500.0,
                "min": 10.0,
                "max": 20.0,
                "missing": 0,
                "sum_x": 45.0,
                "sum_xy": 750.0,
                "sum_x_sq": 225.0,
                "last_val": 18.0,
            }
        },
        2: {
            "397_0": {
                "count": 0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "missing": 0,
                "sum_x": 0.0,
                "sum_xy": 0.0,
                "sum_x_sq": 0.0,
                "last_val": float("nan"),
            }
        },
    }
