"""Export prediction details: predictions.csv and risk_tier_summary.csv for Tableau."""

import argparse
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from config import (
    CO2_PER_FAILURE_KG,
    EXPECTED_BREAKDOWN_COST_JPY,
    HIGH_RISK_THRESHOLD,
    MATERIALS_PER_FAILURE_KG,
    MODEL_PATH,
    MODEL_VERSION,
    PREDICTIONS_PATH,
    PREVENTATIVE_MAINTENANCE_COST_JPY,
    RISK_TIER_SUMMARY_PATH,
    MEDIUM_RISK_THRESHOLD,
    TABLE_ACCOUNTS,
    TABLE_FEATURES,
    TABLE_VEHICLES,
    TRAIN_SPECS_CSV,
    TRAIN_TTE_CSV,
    TRAIN_READOUTS_CSV,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _load_model() -> any:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_predictive_model.py first.")
    return joblib.load(MODEL_PATH)


def _load_fleet_data(skip_db: bool = False) -> pd.DataFrame:
    if skip_db:
        return _load_fleet_data_local()
    return _load_fleet_data_db()


def _load_fleet_data_db() -> pd.DataFrame:
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
    log.info("Fleet loaded: %d vehicles", len(df))
    return df


def _load_fleet_data_local() -> pd.DataFrame:
    """Fallback: merge local CSVs (skips Supabase, ~seconds vs minutes).

    Builds a minimal feature matrix from specifications + tte labels.
    Sensor features are filled with 0 — probabilities will be approximate
    since the model expects engineered sensor columns.
    """
    log.info("Loading fleet data from local CSVs (skipping Supabase)...")

    if not os.path.exists(TRAIN_SPECS_CSV):
        raise FileNotFoundError(f"Local specs not found at {TRAIN_SPECS_CSV}")

    df_specs = pd.read_csv(TRAIN_SPECS_CSV)

    # Load labels
    if os.path.exists(TRAIN_TTE_CSV):
        df_tte = pd.read_csv(TRAIN_TTE_CSV)
        label_col = "in_study_repair"
        if label_col in df_tte.columns:
            df_specs = df_specs.merge(
                df_tte[["vehicle_id", label_col, "length_of_study_time_step"]],
                on="vehicle_id",
                how="left",
            )
            df_specs[label_col] = df_specs[label_col].fillna(0).astype(int)
        else:
            df_specs[label_col] = 0
            df_specs["length_of_study_time_step"] = 0
    else:
        df_specs["in_study_repair"] = 0
        df_specs["length_of_study_time_step"] = 0

    log.info("Fleet loaded locally: %d vehicles", len(df_specs))
    return df_specs


def _get_model_columns(model: any) -> list:
    """Extract expected column names from a trained Pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    cat_cols = preprocessor.transformers_[1][2]
    num_cols = preprocessor.transformers_[0][2]
    return list(set(cat_cols + num_cols))


def _fill_missing_columns(X: pd.DataFrame, model: any) -> pd.DataFrame:
    """Add any missing columns expected by the model, filled with 0."""
    expected = _get_model_columns(model)
    missing = [c for c in expected if c not in X.columns]
    if missing:
        log.info("Filling %d missing feature columns with 0...", len(missing))
        missing_df = pd.DataFrame(0, index=X.index, columns=missing)
        X = pd.concat([X, missing_df], axis=1)
    return X


def _classify_priority(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "CRITICAL DISPATCH"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "PREVENTATIVE MONITOR"
    return "STABLE OPERATION"


def _export_predictions(df: pd.DataFrame, model: any) -> None:
    """Export per-vehicle predictions to predictions.csv."""
    drop_cols = ["vehicle_id", "in_study_repair", "length_of_study_time_step"]
    present_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=present_cols)

    # Fill in any missing columns expected by the model
    X = _fill_missing_columns(X, model)

    log.info("Running inference for predictions export (%d vehicles)...", len(X))
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    actual = df["in_study_repair"].copy()
    if actual.isna().any():
        actual = actual.fillna(-1)

    prediction_df = pd.DataFrame({
        "Vehicle Id": df["vehicle_id"].values,
        "Actual": actual.astype(int).values,
        "Predicted": predictions.astype(int),
        "Failure Probability": np.round(probabilities, 4),
        "Maintenance Priority": [_classify_priority(p) for p in probabilities],
        "Model Version": MODEL_VERSION,
        "Prediction Timestamp": datetime.now(),
    })

    prediction_df.to_csv(PREDICTIONS_PATH, index=False)
    log.info("Predictions exported to %s (%d rows)", PREDICTIONS_PATH, len(prediction_df))

    # Summary of misclassifications
    fp_mask = (prediction_df["Actual"] == 0) & (prediction_df["Predicted"] == 1)
    fn_mask = (prediction_df["Actual"] == 1) & (prediction_df["Predicted"] == 0)
    log.info("  False Positives:  %d", fp_mask.sum())
    log.info("  False Negatives:  %d", fn_mask.sum())


def _export_risk_tier_summary(df: pd.DataFrame, model: any) -> None:
    """Export risk tier summary to risk_tier_summary.csv."""
    drop_cols = ["vehicle_id", "in_study_repair", "length_of_study_time_step"]
    present_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=present_cols)
    X = _fill_missing_columns(X, model)

    probabilities = model.predict_proba(X)[:, 1]

    scored = df.copy()
    scored["failure_probability"] = np.round(probabilities, 4)
    scored["maintenance_priority"] = [_classify_priority(p) for p in probabilities]

    # Financial metrics
    scored["expected_breakdown_cost_jpy"] = np.round(
        probabilities * EXPECTED_BREAKDOWN_COST_JPY, 0
    )
    scored["net_savings_jpy"] = np.round(
        np.maximum(
            probabilities * EXPECTED_BREAKDOWN_COST_JPY - PREVENTATIVE_MAINTENANCE_COST_JPY, 0
        ), 0
    )

    # Sustainability metrics
    scored["expected_co2_saved_kg"] = np.round(probabilities * CO2_PER_FAILURE_KG, 2)

    tier_summary = (
        scored.groupby("maintenance_priority")
        .agg(
            {
                "vehicle_id": "count",
                "failure_probability": "mean",
                "expected_breakdown_cost_jpy": "mean",
                "net_savings_jpy": "mean",
                "expected_co2_saved_kg": "mean",
            }
        )
        .reset_index()
    )

    tier_summary.columns = [
        "Priority",
        "Vehicle Count",
        "Average Probability",
        "Average Expected Breakdown Cost JPY",
        "Average Net Savings JPY",
        "Average Expected CO2 Saved KG",
    ]

    tier_summary["Average Probability"] = tier_summary["Average Probability"].round(4)
    tier_summary["Average Expected Breakdown Cost JPY"] = tier_summary["Average Expected Breakdown Cost JPY"].round(0)
    tier_summary["Average Net Savings JPY"] = tier_summary["Average Net Savings JPY"].round(0)
    tier_summary["Average Expected CO2 Saved KG"] = tier_summary["Average Expected CO2 Saved KG"].round(2)

    tier_summary = tier_summary.sort_values("Average Probability", ascending=False).reset_index(drop=True)
    tier_summary.to_csv(RISK_TIER_SUMMARY_PATH, index=False)
    log.info("Risk tier summary exported to %s", RISK_TIER_SUMMARY_PATH)

    for _, row in tier_summary.iterrows():
        log.info(
            "  %-25s  Vehicles: %d  Avg Prob: %.4f",
            row["Priority"],
            row["Vehicle Count"],
            row["Average Probability"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prediction details for Tableau.")
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Use local CSVs instead of Supabase (fast, but no sensor features)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("PREDICTION DETAILS EXPORT START")
    log.info("=" * 60)

    model = _load_model()
    df_fleet = _load_fleet_data(skip_db=args.skip_db)

    _export_predictions(df_fleet, model)
    _export_risk_tier_summary(df_fleet, model)

    log.info("=" * 60)
    log.info("PREDICTION DETAILS EXPORT COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
