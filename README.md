# Heavy Fleet Predictive Maintenance Pipeline

Production-style predictive maintenance platform for Japanese heavy truck fleets. Uses sensor telemetry to predict DPF (Diesel Particulate Filter) failure risk, estimate maintenance costs, and quantify sustainability impact.

## Architecture

```
Raw CSV (1.1M rows)
    ↓
Chunked ETL (insert_data.py)
    ↓
Feature Engineering (956 features / vehicle)
    ↓
Supabase Analytics Warehouse (23,550 vehicles)
    ↓
Training Pipeline (train_predictive_model.py)
    ↓
Saved Model (random_forest.joblib)
    ↓
Prediction Pipeline (predict.py)
    ↓
Tableau Dataset (tableau_fleet_analytics.csv)
    ↓
Dashboard
```

## Quick Start

```bash
pip install -r requirements.txt
python insert_data.py            # ~9 min — ETL + upload to Supabase
python train_predictive_model.py # ~5 min — train + evaluate + save model
python predict.py                # ~1 min — inference → Tableau CSV
```

## Pipeline Steps

### 1. ETL — `insert_data.py`

Reads 1.1M telemetry rows in 100K chunks, aggregates to per-vehicle features, uploads to Supabase.

- **14 sensor families** auto-detected from column names
- **10 statistics** per sensor: mean, max, min, std, last, range, missing count, trend slope, coefficient of variation
- **24 realistic Japanese logistics companies** with prefecture, industry, and fleet size
- Storage reduced **~98%** (1.1M rows → 23,550 engineered features)

### 2. Training — `train_predictive_model.py`

Loads engineered features from Supabase, trains a RandomForest classifier with proper validation.

- **80/20 stratified train/test split** (no data leakage)
- **RandomizedSearchCV** — 10 iterations × 5-fold cross-validation
- **Full evaluation**: accuracy, precision, recall, F1, ROC AUC, PR AUC, confusion matrix
- Feature importance exported to `outputs/feature_importance.csv`
- Model persisted to `models/random_forest.joblib`

### 3. Prediction — `predict.py`

Loads saved model, scores entire fleet, produces Tableau-ready dataset.

- Failure probability per vehicle
- Maintenance priority: `CRITICAL DISPATCH` (≥75%), `PREVENTATIVE MONITOR` (≥40%), `STABLE OPERATION`
- Financial metrics: expected breakdown cost, maintenance ROI, net savings
- Sustainability metrics: expected CO₂ saved, materials saved

## Project Structure

```
├── config.py                   # Constants, thresholds, paths
├── insert_data.py              # Chunked ETL → Supabase
├── train_predictive_model.py   # Train, evaluate, save model
├── predict.py                  # Inference → Tableau CSV
├── requirements.txt
├── .env                        # SUPABASE_DB_URL
├── models/                     # Saved model artifacts
├── outputs/                    # evaluation_metrics.json, feature_importance.csv
├── logs/
└── data/
    ├── train_operational_readouts.csv  # Raw telemetry (1.1M rows)
    ├── train_specifications.csv        # Vehicle specs (23,550)
    ├── train_tte.csv                   # Time-to-event labels (23,550)
    └── tableau_fleet_analytics.csv     # Dashboard output
```

## Model Performance

| Metric | Test Set | 5-Fold CV |
|---|---|---|
| Accuracy | 90.96% | 90.88% |
| **ROC AUC** | **80.41%** | **78.37%** |
| Precision | 67.50% | 71.55% |
| Recall | 11.89% | 9.13% |
| F1 | 20.22% | 16.20% |

Top predictive features are **trend slopes** across DPF sensor families — gradual degradation patterns over time.

## Database Schema

| Table | Rows | Description |
|---|---|---|
| `crm_b2b_accounts` | 25 | Logistics companies with prefecture and fleet size |
| `crm_fleet_vehicles` | 23,550 | Vehicle metadata (age, model, engine family) |
| `vehicle_engineered_features` | 23,550 | Aggregated sensor features + failure labels |

## Tableau Dashboard Columns

`vehicle_id`, `company_name`, `operating_prefecture`, `fleet_size`, `vehicle_age_years`, `failure_probability`, `risk_percentage`, `maintenance_priority`, `expected_breakdown_cost_jpy`, `net_savings_jpy`, `expected_co2_saved_kg`, `expected_materials_saved_kg`, `expected_repair_avoided`, `feature_importance_rank`, `model_version`, `prediction_timestamp`

## Configuration

All constants live in `config.py`:

- **Thresholds**: `HIGH_RISK_THRESHOLD = 0.75`, `MEDIUM_RISK_THRESHOLD = 0.40`
- **Financial**: `EXPECTED_BREAKDOWN_COST_JPY = 500000`, `PREVENTATIVE_MAINTENANCE_COST_JPY = 30000`
- **Sustainability**: `CO2_PER_FAILURE_KG = 350`, `MATERIALS_PER_FAILURE_KG = 45`
- **Model**: `N_ESTIMATORS = 300`, `MAX_DEPTH = 10`, `MIN_SAMPLES_LEAF = 5`

## Dependencies

See `requirements.txt`: pandas, numpy, scikit-learn, scipy, joblib, python-dotenv, sqlalchemy, psycopg2-binary

## Environment

```bash
cp .env.example .env
# Edit SUPABASE_DB_URL with your connection string
```
