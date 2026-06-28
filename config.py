import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data paths ---
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_READOUTS_CSV = os.path.join(DATA_DIR, "train_operational_readouts.csv")
TRAIN_SPECS_CSV = os.path.join(DATA_DIR, "train_specifications.csv")
TRAIN_TTE_CSV = os.path.join(DATA_DIR, "train_tte.csv")
TABLEAU_OUTPUT_CSV = os.path.join(DATA_DIR, "tableau_fleet_analytics.csv")

# --- Output paths ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.joblib")
EVALUATION_METRICS_PATH = os.path.join(OUTPUTS_DIR, "evaluation_metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUTS_DIR, "feature_importance.csv")

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

# --- Risk thresholds ---
HIGH_RISK_THRESHOLD = 0.75
MEDIUM_RISK_THRESHOLD = 0.40

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

# --- Ensure directories ---
for _dir in (MODELS_DIR, OUTPUTS_DIR, LOGS_DIR):
    os.makedirs(_dir, exist_ok=True)
