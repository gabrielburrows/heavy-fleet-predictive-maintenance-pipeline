"""Export model diagnostics: model_metrics.csv and feature_importance.csv for Tableau."""

import logging
import os
from datetime import datetime

import json
import numpy as np
import pandas as pd

from config import (
    ALGORITHM_NAME,
    CV_FOLDS,
    EVALUATION_METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_IMPORTANCE_TABLEAU_PATH,
    HYPERPARAMETER_SEARCH,
    MODEL_METRICS_PATH,
    MODEL_VERSION,
    TRAIN_SPECS_CSV,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _export_model_metrics() -> None:
    """Export model_metrics.csv with evaluation metrics and metadata."""
    with open(EVALUATION_METRICS_PATH, "r") as f:
        metrics = json.load(f)

    cm = metrics["confusion_matrix"]
    total_samples = cm["true_negative"] + cm["false_positive"] + cm["false_negative"] + cm["true_positive"]

    # Total training samples from specs file
    if os.path.exists(TRAIN_SPECS_CSV):
        specs = pd.read_csv(TRAIN_SPECS_CSV)
        training_samples = len(specs)
    else:
        training_samples = total_samples

    # Total engineered features from feature importance file
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        engineered_features = len(fi)
    else:
        engineered_features = 0

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC",
            "PR AUC",
            "True Negative",
            "False Positive",
            "False Negative",
            "True Positive",
            "Training Samples",
            "Engineered Features",
        ],
        "Value": [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
            metrics["pr_auc"],
            cm["true_negative"],
            cm["false_positive"],
            cm["false_negative"],
            cm["true_positive"],
            training_samples,
            engineered_features,
        ],
    })

    metrics_df["Algorithm"] = ALGORITHM_NAME
    metrics_df["Prediction Timestamp"] = datetime.now()
    metrics_df["Model Version"] = MODEL_VERSION

    # Hyperparameter search metadata
    extra = pd.DataFrame({
        "Metric": ["Hyperparameter Search", "CV Folds"],
        "Value": [HYPERPARAMETER_SEARCH, CV_FOLDS],
        "Algorithm": [ALGORITHM_NAME, ALGORITHM_NAME],
        "Prediction Timestamp": [datetime.now(), datetime.now()],
        "Model Version": [MODEL_VERSION, MODEL_VERSION],
    })
    metrics_df = pd.concat([metrics_df, extra], ignore_index=True)

    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)
    log.info("Model metrics exported to %s (%d rows)", MODEL_METRICS_PATH, len(metrics_df))


def _categorize_feature(feature: str) -> str:
    """Assign a human-readable category to a feature name.

    Sensor family mapping (from EDA):
      397 -> Exhaust Temperature
      459 -> DPF Pressure
      291 -> Soot
      Other families -> grouped by metric type
    """
    feature_lower = feature.lower()

    if "trend_slope" in feature_lower:
        if "s_397" in feature_lower:
            return "Exhaust Temperature"
        if "s_459" in feature_lower:
            return "DPF Pressure"
        if "s_291" in feature_lower:
            return "Soot"
        return "Trend"

    if "mean" in feature_lower:
        if "s_397" in feature_lower:
            return "Exhaust Temperature"
        if "s_459" in feature_lower:
            return "DPF Pressure"
        if "s_291" in feature_lower:
            return "Soot"
        return "Other"

    if "max" in feature_lower:
        if "s_397" in feature_lower:
            return "Exhaust Temperature"
        if "s_459" in feature_lower:
            return "DPF Pressure"
        if "s_291" in feature_lower:
            return "Soot"
        return "Other"

    if "std" in feature_lower:
        if "s_397" in feature_lower:
            return "Exhaust Temperature"
        if "s_459" in feature_lower:
            return "DPF Pressure"
        if "s_291" in feature_lower:
            return "Soot"
        return "Other"

    return "Other"


def _export_feature_importance() -> None:
    """Export feature_importance.csv with categories for Tableau."""
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        log.warning("Feature importance file not found at %s", FEATURE_IMPORTANCE_PATH)
        return

    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    importance = importance.rename(columns={"feature": "Feature", "importance": "Importance", "rank": "Rank"})
    importance = importance.sort_values("Importance", ascending=False).reset_index(drop=True)
    importance["Rank"] = np.arange(1, len(importance) + 1)

    importance["Category"] = importance["Feature"].apply(_categorize_feature)

    importance.to_csv(FEATURE_IMPORTANCE_TABLEAU_PATH, index=False)
    log.info("Feature importance exported to %s (%d features)", FEATURE_IMPORTANCE_TABLEAU_PATH, len(importance))

    # Log top 20
    log.info("Top 20 features:")
    for _, row in importance.head(20).iterrows():
        log.info("  #%d  %-45s  %.6f  [%s]", row["Rank"], row["Feature"], row["Importance"], row["Category"])


def main() -> None:
    log.info("=" * 60)
    log.info("MODEL DIAGNOSTICS EXPORT START")
    log.info("=" * 60)

    _export_model_metrics()
    _export_feature_importance()

    log.info("=" * 60)
    log.info("MODEL DIAGNOSTICS EXPORT COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
