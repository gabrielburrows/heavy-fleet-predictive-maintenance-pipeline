"""Tests for config.py constants and directory setup."""
import os

import config


class TestConfigPaths:
    """Verify all path constants resolve to valid string values."""

    def test_base_dir_exists(self):
        assert os.path.isdir(config.BASE_DIR)

    def test_data_dir_defined(self):
        assert isinstance(config.DATA_DIR, str)
        assert config.DATA_DIR.endswith("data")

    def test_train_readouts_csv_defined(self):
        assert config.TRAIN_READOUTS_CSV.endswith("train_operational_readouts.csv")

    def test_train_specs_csv_defined(self):
        assert config.TRAIN_SPECS_CSV.endswith("train_specifications.csv")

    def test_train_tte_csv_defined(self):
        assert config.TRAIN_TTE_CSV.endswith("train_tte.csv")

    def test_models_dir_created(self):
        assert os.path.isdir(config.MODELS_DIR)

    def test_outputs_dir_created(self):
        assert os.path.isdir(config.OUTPUTS_DIR)

    def test_logs_dir_created(self):
        assert os.path.isdir(config.LOGS_DIR)

    def test_model_path_defined(self):
        assert config.MODEL_PATH.endswith("random_forest.joblib")

    def test_evaluation_metrics_path_defined(self):
        assert config.EVALUATION_METRICS_PATH.endswith("evaluation_metrics.json")

    def test_feature_importance_path_defined(self):
        assert config.FEATURE_IMPORTANCE_PATH.endswith("feature_importance.csv")


class TestConfigConstants:
    """Verify hyperparameter and threshold constants have valid values."""

    def test_chunk_size_positive(self):
        assert config.CHUNK_SIZE > 0

    def test_random_state_non_negative(self):
        assert config.RANDOM_STATE >= 0

    def test_test_size_between_0_and_1(self):
        assert 0 < config.TEST_SIZE < 1

    def test_cv_folds_positive(self):
        assert config.CV_FOLDS > 1

    def test_n_estimators_positive(self):
        assert config.N_ESTIMATORS > 0

    def test_max_depth_positive(self):
        assert config.MAX_DEPTH > 0

    def test_min_samples_leaf_positive(self):
        assert config.MIN_SAMPLES_LEAF > 0

    def test_high_risk_threshold_positive(self):
        assert 0 < config.HIGH_RISK_THRESHOLD <= 1

    def test_medium_risk_threshold_positive(self):
        assert 0 < config.MEDIUM_RISK_THRESHOLD <= 1

    def test_high_risk_above_medium_risk(self):
        assert config.HIGH_RISK_THRESHOLD > config.MEDIUM_RISK_THRESHOLD

    def test_financial_constants_positive(self):
        assert config.CO2_PER_FAILURE_KG > 0
        assert config.MATERIALS_PER_FAILURE_KG > 0
        assert config.EXPECTED_BREAKDOWN_COST_JPY > 0
        assert config.PREVENTATIVE_MAINTENANCE_COST_JPY > 0

    def test_known_sensor_families_not_empty(self):
        assert len(config.KNOWN_SENSOR_FAMILIES) == 14

    def test_model_version_not_empty(self):
        assert len(config.MODEL_VERSION) > 0

    def test_algorithm_name_not_empty(self):
        assert len(config.ALGORITHM_NAME) > 0


class TestSensorDashboardMap:
    """Verify SENSOR_DASHBOARD_MAP is well-formed."""

    def test_is_dict(self):
        assert isinstance(config.SENSOR_DASHBOARD_MAP, dict)

    def test_not_empty(self):
        assert len(config.SENSOR_DASHBOARD_MAP) > 0

    def test_all_keys_strings(self):
        for k, v in config.SENSOR_DASHBOARD_MAP.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
