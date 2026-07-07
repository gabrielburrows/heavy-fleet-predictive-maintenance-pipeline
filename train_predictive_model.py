"""Training pipeline: load engineered features, split, cross-validate, train, evaluate, and persist model."""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine, text

from config import (
    CV_FOLDS,
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_features_from_db():
    """Load engineered features, accounts, and vehicles from Supabase."""
    load_dotenv()
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise ValueError("SUPABASE_DB_URL missing from .env")

    engine = create_engine(db_url)
    log.info("Connecting to Supabase...")

    log.info("Loading engineered features...")
    df_features = pd.read_sql(text(f"SELECT * FROM {TABLE_FEATURES}"), engine)
    log.info("Features loaded: %d rows, %d columns", len(df_features), len(df_features.columns))

    log.info("Loading accounts and vehicles...")
    df_accounts = pd.read_sql(text(f"SELECT * FROM {TABLE_ACCOUNTS}"), engine)
    df_vehicles = pd.read_sql(text(f"SELECT * FROM {TABLE_VEHICLES}"), engine)

    return df_features, df_accounts, df_vehicles


def _build_feature_matrix(df: pd.DataFrame):
    """Separate features (X) and target (y), identify column types."""
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


# ---------------------------------------------------------------------------
# Model pipeline
# ---------------------------------------------------------------------------

def _build_pipeline(cat_cols: list, num_cols: list) -> Pipeline:
    """Build scikit-learn Pipeline with ColumnTransformer + XGBoost."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="passthrough",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_child_weight=MIN_SAMPLES_LEAF,
            scale_pos_weight=15,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


def _set_class_weight(pipeline: Pipeline, y: pd.Series) -> None:
    """Set scale_pos_weight on the XGBoost classifier inside the pipeline."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = float(neg / pos) if pos > 0 else 15.0
    pipeline.set_params(classifier__scale_pos_weight=spw)


def _hyperparameter_search(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
) -> Pipeline:
    """Run RandomizedSearchCV on a small hyperparameter space."""
    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [6, 8, 10, 12],
        "classifier__min_child_weight": [3, 5, 10],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=CV_FOLDS,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    log.info("Best params: %s", search.best_params_)
    log.info("Best CV ROC AUC: %.4f", search.best_score_)
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
) -> dict:
    """Compute comprehensive evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
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
            y_true, y_pred, output_dict=True, zero_division=0,
        ),
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extract and rank feature importances from trained pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = (
        preprocessor.transformers_[0][2]
        + list(
            preprocessor.named_transformers_["cat"]
            .named_steps["ohe"]
            .get_feature_names_out(preprocessor.transformers_[1][2])
        )
    )
    importances = pipeline.named_steps["classifier"].feature_importances_

    df = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("TRAINING PIPELINE START")
    log.info("=" * 60)

    # Load data
    df_features, df_accounts, df_vehicles = _load_features_from_db()
    df_master = df_features.merge(df_vehicles, on="vehicle_id", how="inner")
    df_master = df_master.merge(df_accounts, on="account_id", how="left")

    # Build feature matrix
    X, y, cat_cols, num_cols = _build_feature_matrix(df_master)

    # Drop rows with missing target
    valid_mask = y.notna()
    X, y = X[valid_mask], y[valid_mask].astype(int)

    log.info("XGBoost (hist tree method, %d samples)", len(X))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    log.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    # Build pipeline
    pipeline = _build_pipeline(cat_cols, num_cols)

    # Hyperparameter search on training set
    log.info("Running hyperparameter search (10 iterations, %d-fold CV)...", CV_FOLDS)
    _set_class_weight(pipeline, y_train)
    best_pipeline = _hyperparameter_search(pipeline, X_train, y_train)

    # Cross-validation predictions for unbiased evaluation
    log.info("Generating cross-validated predictions...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_cv_proba = cross_val_predict(
        best_pipeline, X_train, y_train, cv=cv, method="predict_proba",
    )[:, 1]
    y_cv_pred = (y_cv_proba >= 0.5).astype(int)

    log.info("Cross-validation metrics:")
    cv_metrics = _evaluate_model(y_train.values, y_cv_pred, y_cv_proba)
    log.info("  CV ROC AUC: %.4f", cv_metrics["roc_auc"])
    log.info("  CV F1:     %.4f", cv_metrics["f1"])

    # Test set evaluation
    log.info("Evaluating on held-out test set...")
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = best_pipeline.predict(X_test)
    metrics = _evaluate_model(y_test.values, y_pred, y_proba)

    log.info("TEST SET METRICS:")
    log.info("  Accuracy:  %.4f", metrics["accuracy"])
    log.info("  Precision: %.4f", metrics["precision"])
    log.info("  Recall:    %.4f", metrics["recall"])
    log.info("  F1:        %.4f", metrics["f1"])
    log.info("  ROC AUC:   %.4f", metrics["roc_auc"])
    log.info("  PR AUC:    %.4f", metrics["pr_auc"])
    log.info("  Confusion matrix: %s", metrics["confusion_matrix"])

    # Save evaluation metrics
    save_metrics = {**metrics, "cv_metrics": cv_metrics}
    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(save_metrics, f, indent=2)
    log.info("Metrics saved to %s", EVALUATION_METRICS_PATH)

    # Train final model on full data
    log.info("Training final model on full dataset...")
    final_pipeline = _build_pipeline(cat_cols, num_cols)
    _set_class_weight(final_pipeline, y)
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_PATH)
    log.info("Model saved to %s", MODEL_PATH)

    # Feature importance
    log.info("Extracting feature importances...")
    try:
        feat_imp = _extract_feature_importance(final_pipeline)
        feat_imp.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        log.info("Top 10 features:")
        for _, row in feat_imp.head(10).iterrows():
            log.info("  %d. %s (%.4f)", row["rank"], row["feature"], row["importance"])
        log.info("Feature importance saved to %s", FEATURE_IMPORTANCE_PATH)
    except Exception as e:
        log.warning("Feature importance extraction failed: %s", e)

    log.info("=" * 60)
    log.info("TRAINING PIPELINE COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
