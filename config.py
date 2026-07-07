import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- GPU detection (legacy, kept for backward compatibility) ---
try:
    import lightgbm as lgb
    _m = lgb.LGBMClassifier(n_estimators=1, max_depth=2, device="gpu")
    _m.fit(np.random.randn(10, 3), np.random.randint(0, 2, 10))
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# --- Data paths ---
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_READOUTS_CSV = os.path.join(DATA_DIR, "train_operational_readouts.csv")
TRAIN_SPECS_CSV = os.path.join(DATA_DIR, "train_specifications.csv")
TRAIN_TTE_CSV = os.path.join(DATA_DIR, "train_tte.csv")
TABLEAU_OUTPUT_CSV = os.path.join(DATA_DIR, "tableau_fleet_analytics.csv")
ETL_CACHE_PARQUET = os.path.join(DATA_DIR, "engineered_features_cache.parquet")

# --- Output paths ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.joblib")
EVALUATION_METRICS_PATH = os.path.join(OUTPUTS_DIR, "evaluation_metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUTS_DIR, "feature_importance.csv")
MODEL_METRICS_PATH = os.path.join(DATA_DIR, "model_metrics.csv")
FEATURE_IMPORTANCE_TABLEAU_PATH = os.path.join(DATA_DIR, "feature_importance.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
RISK_TIER_SUMMARY_PATH = os.path.join(DATA_DIR, "risk_tier_summary.csv")

# --- ETL ---
CHUNK_SIZE = 100000
KNOWN_SENSOR_FAMILIES = [
    "100", "158", "167", "171", "272", "291",
    "309", "370", "397", "427", "459", "666", "835", "837",
]

# --- Model hyperparameters ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ESTIMATORS = 300
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 5
CLASS_WEIGHT = "balanced"

# --- Risk thresholds (optimized for XGBoost, threshold=0.05) ---
HIGH_RISK_THRESHOLD = 0.05
MEDIUM_RISK_THRESHOLD = 0.03

# --- Financial / sustainability constants ---
CO2_PER_FAILURE_KG = 350.0
MATERIALS_PER_FAILURE_KG = 45.0
EXPECTED_BREAKDOWN_COST_JPY = 500000.0
PREVENTATIVE_MAINTENANCE_COST_JPY = 30000.0

# --- Database tables ---
TABLE_ACCOUNTS = "crm_b2b_accounts"
TABLE_VEHICLES = "crm_fleet_vehicles"
TABLE_FEATURES = "vehicle_engineered_features"

# --- Model versioning ---
MODEL_VERSION = "1.0.0"
ALGORITHM_NAME = "XGBoost (hist)"
HYPERPARAMETER_SEARCH = "RandomizedSearchCV"

# --- Dashboard / Tableau export mappings ---
# NOTE: SCANIA Component X columns are all cumulative counters. 
# The aliases below map the most predictive families (from feature importance)
# to readable dashboard names.
SENSOR_DASHBOARD_MAP = {
    "total_operating_hours": "length_of_study_time_step",
    # Primary degradation histogram (36 bins, highest feature importance)
    "component_x_primary_hist_mean": "s_397_0_mean",
    "component_x_primary_hist_max": "s_397_0_max",
    # Secondary operational histogram (20 bins)
    "component_x_secondary_hist_mean": "s_459_0_mean",
    "component_x_secondary_hist_max": "s_459_0_max",
    # Tertiary histogram (11 bins)
    "component_x_tertiary_hist_mean": "s_291_0_mean",
    "component_x_tertiary_hist_max": "s_291_0_max",
    # Simple cumulative counters (single-bin families)
    "component_x_cumulative_counter_1": "s_100_0_mean",
    "component_x_cumulative_counter_2": "s_666_0_mean",
}

# --- Ensure directories ---
for _dir in (MODELS_DIR, OUTPUTS_DIR, LOGS_DIR):
    os.makedirs(_dir, exist_ok=True)
