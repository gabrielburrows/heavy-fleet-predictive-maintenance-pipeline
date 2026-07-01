"""Inference pipeline: load saved model, predict fleet risk, produce Tableau dataset."""

import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

from config import (
    CO2_PER_FAILURE_KG,
    MATERIALS_PER_FAILURE_KG,
    EXPECTED_BREAKDOWN_COST_JPY,
    PREVENTATIVE_MAINTENANCE_COST_JPY,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    MODEL_PATH,
    MODEL_VERSION,
    TABLEAU_OUTPUT_CSV,
    TABLE_FEATURES,
    TABLE_ACCOUNTS,
    TABLE_VEHICLES,
    SENSOR_DASHBOARD_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _load_model() -> any:
    """Load persisted model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_predictive_model.py first.")
    log.info("Loading model from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


def _load_fleet_data() -> pd.DataFrame:
    """Load engineered features and metadata from Supabase."""
    load_dotenv()
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise ValueError("SUPABASE_DB_URL missing from .env")

    engine = create_engine(db_url)
    log.info("Loading fleet data from Supabase...")

    df_features = pd.read_sql(text(f"SELECT * FROM {TABLE_FEATURES}"), engine)
    df_vehicles = pd.read_sql(text(f"SELECT * FROM {TABLE_VEHICLES}"), engine)
    df_accounts = pd.read_sql(text(f"SELECT * FROM {TABLE_ACCOUNTS}"), engine)

    df = df_features.merge(df_vehicles, on="vehicle_id", how="inner")
    df = df.merge(df_accounts, on="account_id", how="left")

    # Compute operational dashboard metrics from engineered sensor features
    _compute_operational_metrics(df)

    log.info("Fleet loaded: %d vehicles across %d accounts", len(df), df["account_id"].nunique())
    return df


def _compute_operational_metrics(df: pd.DataFrame) -> None:
    """Derive operational dashboard metrics from per-sensor engineered features.

    Maps sensor families to physical quantities based on EDA structural analysis:
    - Family 397 (36 bins, fine-grain) → exhaust temperature profile
    - Family 459 (20 bins, fine-grain) → DPF differential pressure distribution
    - Family 291 (11 bins, med-grain) → engine soot mass index
    """
    # Exhaust temperature from family 397
    cols_397_mean = [c for c in df.columns if c.startswith("s_397_") and c.endswith("_mean")]
    cols_397_max = [c for c in df.columns if c.startswith("s_397_") and c.endswith("_max")]

    # DPF differential pressure from family 459
    cols_459_mean = [c for c in df.columns if c.startswith("s_459_") and c.endswith("_mean")]
    cols_459_max = [c for c in df.columns if c.startswith("s_459_") and c.endswith("_max")]

    # Engine soot mass index from family 291
    cols_291_mean = [c for c in df.columns if c.startswith("s_291_") and c.endswith("_mean")]
    cols_291_max = [c for c in df.columns if c.startswith("s_291_") and c.endswith("_max")]

    if cols_397_mean:
        df["avg_exhaust_temperature_c"] = df[cols_397_mean].mean(axis=1).round(2)
    if cols_397_max:
        df["peak_exhaust_temperature_c"] = df[cols_397_max].max(axis=1).round(2)

    if cols_459_mean:
        df["avg_dpf_differential_pressure_kpa"] = df[cols_459_mean].mean(axis=1).round(2)
    if cols_459_max:
        df["peak_dpf_differential_pressure_kpa"] = df[cols_459_max].max(axis=1).round(2)

    if cols_291_mean:
        df["avg_engine_soot_mass_index"] = df[cols_291_mean].mean(axis=1).round(2)
    if cols_291_max:
        df["peak_engine_soot_mass_index"] = df[cols_291_max].max(axis=1).round(2)


def _classify_priority(probability: float) -> str:
    """Assign maintenance priority based on configurable thresholds."""
    if probability >= HIGH_RISK_THRESHOLD:
        return "CRITICAL DISPATCH"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "PREVENTATIVE MONITOR"
    return "STABLE OPERATION"


def _compute_fleet_scores(
    df: pd.DataFrame,
    model: any,
) -> pd.DataFrame:
    """Run model inference and compute financial/sustainability metrics."""
    drop_cols = [
        "vehicle_id",
        "in_study_repair", "length_of_study_time_step",
    ]
    present_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=present_cols)

    log.info("Running inference on %d vehicles...", len(df))
    probabilities = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["failure_probability"] = np.round(probabilities, 4)
    df["risk_percentage"] = np.round(probabilities * 100, 2)
    df["maintenance_priority"] = [_classify_priority(p) for p in probabilities]

    # Financial metrics
    df["expected_breakdown_cost_jpy"] = np.round(
        probabilities * EXPECTED_BREAKDOWN_COST_JPY, 0
    )
    df["preventative_maintenance_cost_jpy"] = PREVENTATIVE_MAINTENANCE_COST_JPY
    df["maintenance_roi_jpy"] = np.round(
        probabilities * EXPECTED_BREAKDOWN_COST_JPY - PREVENTATIVE_MAINTENANCE_COST_JPY, 0
    )
    df["net_savings_jpy"] = np.round(
        np.maximum(
            probabilities * EXPECTED_BREAKDOWN_COST_JPY - PREVENTATIVE_MAINTENANCE_COST_JPY, 0
        ), 0
    )

    # Sustainability metrics (expected savings, not leakage)
    df["expected_co2_saved_kg"] = np.round(probabilities * CO2_PER_FAILURE_KG, 2)
    df["expected_materials_saved_kg"] = np.round(probabilities * MATERIALS_PER_FAILURE_KG, 2)
    df["expected_repair_avoided"] = np.round(probabilities, 4)

    # Feature importance rank placeholder (to be filled from saved importance)
    df["model_version"] = MODEL_VERSION
    df["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()

    return df


def _load_feature_importance_rank() -> dict:
    """Load pre-computed feature importance rankings."""
    from config import FEATURE_IMPORTANCE_PATH
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        df_imp = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        return dict(zip(df_imp["feature"], df_imp["rank"]))
    return {}


def main() -> None:
    log.info("=" * 60)
    log.info("PREDICTION PIPELINE START")
    log.info("=" * 60)

    model = _load_model()
    df_fleet = _load_fleet_data()

    df_scored = _compute_fleet_scores(df_fleet, model)

    # Rename raw DB columns to readable dashboard names
    valid_renames = {old: new for new, old in SENSOR_DASHBOARD_MAP.items() if old in df_scored.columns}
    df_scored = df_scored.rename(columns=valid_renames)

    # Add feature importance rank
    imp_ranks = _load_feature_importance_rank()
    if imp_ranks:
        top_feature = max(imp_ranks, key=imp_ranks.get) if imp_ranks else "N/A"
        df_scored["top_model_feature"] = top_feature
        df_scored["feature_importance_rank"] = 1
    else:
        df_scored["feature_importance_rank"] = -1

    # Select columns for Tableau
    tableau_columns = [
        "vehicle_id", "company_name", "operating_prefecture",
        "fleet_size", "vehicle_age_years",
        "failure_probability", "risk_percentage",
        "maintenance_priority",
        "expected_breakdown_cost_jpy", "net_savings_jpy",
        "expected_co2_saved_kg", "expected_materials_saved_kg",
        "expected_repair_avoided",
        "feature_importance_rank",
        "model_version", "prediction_timestamp",
        # Operational metrics
        "total_operating_hours",
        "avg_exhaust_temperature_c", "peak_exhaust_temperature_c",
        "avg_dpf_differential_pressure_kpa", "peak_dpf_differential_pressure_kpa",
        "avg_engine_soot_mass_index", "peak_engine_soot_mass_index",
    ]
    available_cols = [c for c in tableau_columns if c in df_scored.columns]
    df_output = df_scored[available_cols]

    df_output.to_csv(TABLEAU_OUTPUT_CSV, index=False)
    log.info("Tableau dataset written to %s (%d rows)", TABLEAU_OUTPUT_CSV, len(df_output))

    # Summary statistics
    log.info("Fleet risk summary:")
    log.info("  Average risk:       %.2f%%", df_output["risk_percentage"].mean())
    log.info("  High-risk vehicles: %d", (df_output["risk_percentage"] >= HIGH_RISK_THRESHOLD * 100).sum())
    log.info("  Medium-risk:        %d",
             ((df_output["risk_percentage"] >= MEDIUM_RISK_THRESHOLD * 100) &
              (df_output["risk_percentage"] < HIGH_RISK_THRESHOLD * 100)).sum())
    log.info("  Expected CO\u2082 saved:     %.0f kg", df_output["expected_co2_saved_kg"].sum())
    log.info("  Expected net savings: \u20a5%d", int(df_output["net_savings_jpy"].sum()))

    log.info("=" * 60)
    log.info("PREDICTION PIPELINE COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
