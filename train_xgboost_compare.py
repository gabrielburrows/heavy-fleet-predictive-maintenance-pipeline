"""XGBoost training + comparison against Random Forest baseline.

Trains an XGBoost classifier, compares head-to-head with the saved
Random Forest model on the same held-out test set, and persists the
XGBoost model only if it outperforms on ROC AUC.

Usage:
    python train_xgboost_compare.py
"""

import json
import logging
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import __version__ as _scipy_ver
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine, text
import xgboost as xgb

from config import (
    CV_FOLDS,
    CLASS_WEIGHT,
    EVALUATION_METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
    MODEL_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
    TEST_SIZE,
    TABLE_FEATURES,
    TABLE_ACCOUNTS,
    TABLE_VEHICLES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

XGB_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "xgboost.joblib"
)
XGB_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs", "xgboost_metrics.json"
)
COMPARISON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs", "model_comparison.json"
)


# --- Data loading (shared with train_predictive_model.py) ---


def _load_features_from_db():
    load_dotenv()
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise ValueError("SUPABASE_DB_URL missing from .env")

    engine = create_engine(db_url)
    log.info("Connecting to Supabase...")

    df_features = pd.read_sql(text(f"SELECT * FROM {TABLE_FEATURES}"), engine)
    log.info("Features loaded: %d rows, %d columns", len(df_features), len(df_features.columns))

    df_accounts = pd.read_sql(text(f"SELECT * FROM {TABLE_ACCOUNTS}"), engine)
    df_vehicles = pd.read_sql(text(f"SELECT * FROM {TABLE_VEHICLES}"), engine)

    return df_features, df_accounts, df_vehicles


def _build_feature_matrix(df):
    drop_cols = ["vehicle_id", "in_study_repair", "length_of_study_time_step"]
    X = df.drop(columns=drop_cols)
    y = df["in_study_repair"]

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype in ("float64", "int64", "float")]

    log.info("Feature matrix: %d samples, %d features", len(X), len(X.columns))
    log.info("  Categorical columns: %d", len(cat_cols))
    log.info("  Numerical columns:  %d", len(num_cols))
    log.info("  Target distribution: %s", dict(y.value_counts()))

    return X, y, cat_cols, num_cols


# --- Pipeline builders ---


def _build_xgb_pipeline(cat_cols, num_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_child_weight=MIN_SAMPLES_LEAF,
            scale_pos_weight=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="auc",
            tree_method="hist",
        )),
    ])


def _build_rf_pipeline(cat_cols, num_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


# --- Hyperparameter search ---


def _xgb_hyperparameter_search(pipeline, X_train, y_train):
    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [4, 6, 8, 10, 12],
        "classifier__min_child_weight": [1, 3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__reg_alpha": [0, 0.01, 0.1, 1],
        "classifier__reg_lambda": [0.1, 1, 10],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=CV_FOLDS,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    log.info("Best XGBoost params: %s", search.best_params_)
    log.info("Best XGBoost CV ROC AUC: %.4f", search.best_score_)
    return search.best_estimator_


# --- Evaluation ---


def _evaluate_model(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }
    return metrics


def _extract_feature_importance(pipeline):
    feature_names = (
        pipeline.named_steps["preprocessor"]
        .transformers_[0][2]
        + list(
            pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["ohe"]
            .get_feature_names_out(
                pipeline.named_steps["preprocessor"].transformers_[1][2]
            )
        )
    )

    if hasattr(pipeline.named_steps["classifier"], "feature_importances_"):
        importances = pipeline.named_steps["classifier"].feature_importances_
    else:
        importances = pipeline.named_steps["classifier"].booster().get_score(
            importance_type="weight"
        )

    df = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# --- Comparison ---


def _load_rf_model():
    if not os.path.exists(MODEL_PATH):
        return None
    log.info("Loading existing Random Forest model from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


def _print_comparison(rf_metrics, xgb_metrics, model_name):
    separator = "=" * 70
    header = f" {'METRIC':<20} | {'Random Forest':>14} | {'XGBoost':>14} | {'Delta':>10}"
    rule = "-" * 70

    log.info("\n" + separator)
    log.info("         MODEL COMPARISON (held-out test set)")
    log.info(separator)
    log.info(header)
    log.info(rule)

    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for k in keys:
        rf_val = rf_metrics.get(k, 0)
        xgb_val = xgb_metrics.get(k, 0)
        delta = xgb_val - rf_val
        sign = "+" if delta >= 0 else ""
        log.info(
            f" {k:<20} | {rf_val:>14.4f} | {xgb_val:>14.4f} | {sign}{delta:>9.4f}"
        )

    log.info(rule)
    winner = "XGBoost" if xgb_metrics["roc_auc"] > rf_metrics["roc_auc"] else "Random Forest"
    log.info("WINNER (by ROC AUC): %s", winner)
    log.info(separator + "\n")


# --- Main ---


def main():
    log.info("=" * 60)
    log.info("XGBoost TRAINING + COMPARISON PIPELINE")
    log.info("=" * 60)

    # Load data
    df_features, df_accounts, df_vehicles = _load_features_from_db()
    df_master = df_features.merge(df_vehicles, on="vehicle_id", how="inner")
    df_master = df_master.merge(df_accounts, on="account_id", how="left")

    X, y, cat_cols, num_cols = _build_feature_matrix(df_master)

    valid_mask = y.notna()
    X, y = X[valid_mask], y[valid_mask].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    # ---- Train XGBoost ----
    log.info("\n--- Training XGBoost ---")
    xgb_pipeline = _build_xgb_pipeline(cat_cols, num_cols)
    log.info("Hyperparameter search (20 iterations, %d-fold CV)...", CV_FOLDS)
    best_xgb = _xgb_hyperparameter_search(xgb_pipeline, X_train, y_train)

    # CV predictions
    log.info("Generating cross-validated predictions (XGBoost)...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_cv_proba = cross_val_predict(best_xgb, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    y_cv_pred = (y_cv_proba >= 0.5).astype(int)
    xgb_cv_metrics = _evaluate_model(y_train.values, y_cv_pred, y_cv_proba)
    log.info("  XGBoost CV ROC AUC: %.4f", xgb_cv_metrics["roc_auc"])
    log.info("  XGBoost CV F1:      %.4f", xgb_cv_metrics["f1"])

    # Test set
    log.info("Evaluating XGBoost on held-out test set...")
    xgb_proba = best_xgb.predict_proba(X_test)[:, 1]
    xgb_pred = best_xgb.predict(X_test)
    xgb_metrics = _evaluate_model(y_test.values, xgb_pred, xgb_proba)

    log.info("XGBoost TEST SET METRICS:")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        log.info("  %-12s: %.4f", k, xgb_metrics[k])

    # ---- Evaluate Random Forest on the SAME test split ----
    log.info("\n--- Evaluating Random Forest baseline ---")
    rf_model = _load_rf_model()

    if rf_model is not None:
        log.info("Scoring Random Forest on the same held-out test set...")
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        rf_pred = rf_model.predict(X_test)
        rf_metrics = _evaluate_model(y_test.values, rf_pred, rf_proba)

        log.info("Random Forest TEST SET METRICS:")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            log.info("  %-12s: %.4f", k, rf_metrics[k])

        _print_comparison(rf_metrics, xgb_metrics, "comparison")

        xgb_wins = xgb_metrics["roc_auc"] > rf_metrics["roc_auc"]
    else:
        log.warning("No Random Forest model found at %s — XGBoost will be saved as default.", MODEL_PATH)
        rf_metrics = None
        xgb_wins = True

    # ---- Save XGBoost artifacts (always, for inspection) ----
    with open(XGB_METRICS_PATH, "w") as f:
        json.dump({**xgb_metrics, "cv_metrics": xgb_cv_metrics}, f, indent=2)
    log.info("XGBoost metrics saved to %s", XGB_METRICS_PATH)

    # Feature importance for XGBoost
    log.info("Extracting XGBoost feature importances...")
    try:
        xgb_feat_imp = _extract_feature_importance(best_xgb)
        xgb_imp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "outputs", "xgboost_feature_importance.csv"
        )
        xgb_feat_imp.to_csv(xgb_imp_path, index=False)
        log.info("XGBoost feature importance saved to %s", xgb_imp_path)
        log.info("Top 10 XGBoost features:")
        for _, row in xgb_feat_imp.head(10).iterrows():
            log.info("  %d. %s (%.4f)", row["rank"], row["feature"], row["importance"])
    except Exception as e:
        log.warning("XGBoost feature importance extraction failed: %s", e)

    # ---- Decision: replace model only if XGBoost wins ----
    if xgb_wins:
        # Train final XGBoost on full data with best params
        log.info("\n*** XGBoost outperforms Random Forest — retraining on full dataset ***")
        best_params = best_xgb.named_steps["classifier"].get_params()
        final_xgb = _build_xgb_pipeline(cat_cols, num_cols)
        final_xgb.set_params(**{f"classifier__{k}": v for k, v in best_params.items()})
        final_xgb.fit(X, y)
        joblib.dump(final_xgb, XGB_MODEL_PATH)
        log.info("Full XGBoost model saved to %s", XGB_MODEL_PATH)

        # Back up the old RF model before overwriting
        rf_backup = MODEL_PATH + ".backup"
        if os.path.exists(MODEL_PATH):
            joblib.dump(rf_model, rf_backup)
            log.info("Random Forest model backed up to %s", rf_backup)

        # Overwrite the primary model and metrics with XGBoost
        joblib.dump(final_xgb, MODEL_PATH)
        log.info("PRIMARY model replaced at %s (was Random Forest)", MODEL_PATH)

        # Overwrite metrics so downstream scripts pick up the new model
        save_metrics = {**xgb_metrics, "cv_metrics": xgb_cv_metrics, "algorithm": "XGBoost"}
        with open(EVALUATION_METRICS_PATH, "w") as f:
            json.dump(save_metrics, f, indent=2)
        log.info("Evaluation metrics updated at %s", EVALUATION_METRICS_PATH)

        # Overwrite feature importance
        final_imp = _extract_feature_importance(final_xgb)
        final_imp.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        log.info("Feature importance updated at %s", FEATURE_IMPORTANCE_PATH)

        # Save comparison report
        comparison_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "winner": "XGBoost",
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics,
            "xgboost_cv": xgb_cv_metrics,
            "best_xgb_params": best_xgb.named_steps["classifier"].get_params(),
            "action": "XGBoost replaced Random Forest as primary model",
        }
    else:
        log.info("\n*** Random Forest remains the primary model ***")
        log.info("XGBoost results saved separately for reference.")

        comparison_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "winner": "Random Forest",
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics,
            "xgboost_cv": xgb_cv_metrics,
            "best_xgb_params": best_xgb.named_steps["classifier"].get_params(),
            "action": "Random Forest retained; XGBoost saved as secondary model",
        }

    with open(COMPARISON_PATH, "w") as f:
        json.dump(comparison_report, f, indent=2)
    log.info("Comparison report saved to %s", COMPARISON_PATH)

    log.info("=" * 60)
    log.info("XGBoost PIPELINE COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
