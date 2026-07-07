"""Offline multi-model comparison — no Supabase I/O.

Reads raw CSVs from data/, runs the same in-memory ETL as insert_data.py,
caches engineered features to data/engineered_features_cache.parquet, then
benchmarks a set of tabular classifiers using stratified k-fold CV.

Usage:
    python offline_model_comparison.py          # uses cache if valid
    python offline_model_comparison.py --force-etl  # re-run ETL
"""

import argparse
import inspect
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from config import (
    CHUNK_SIZE,
    RANDOM_STATE,
    TRAIN_READOUTS_CSV,
    TRAIN_SPECS_CSV,
    TRAIN_TTE_CSV,
    ETL_CACHE_PARQUET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

CV_FOLDS = 5

# Aggressive class weights for 9:1 imbalance (21278 neg / 2272 pos)
AGGRESSIVE_CLASS_WEIGHT = {0: 1, 1: 5}


# ---------------------------------------------------------------------------
# In-memory ETL (mirrors insert_data.py logic, no DB calls)
# ---------------------------------------------------------------------------

def _init_sensor_stats() -> Dict[str, Any]:
    """Initialize running-statistics dict for a single sensor."""
    return {
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


def _compute_slope_from_stats(ss: Dict[str, Any]) -> float:
    """Compute linear regression slope from running statistics."""
    n = ss["count"]
    if n < 2:
        return 0.0
    sum_x = ss["sum_x"]
    sum_xy = ss["sum_xy"]
    sum_x_sq = ss.get("sum_x_sq", 0.0)
    denom = n * sum_x_sq - sum_x ** 2
    if abs(denom) < 1e-12:
        return 0.0
    return float((n * sum_xy - sum_x * ss["sum"]) / denom)


def _aggregate_sensor_family(
    family: str,
    stats: Dict[int, Dict[str, Dict[str, Any]]],
) -> pd.DataFrame:
    """Build per-vehicle feature DataFrame for a single sensor family."""
    sensor_keys: List[str] = []
    for _vid, sensor_dict in stats.items():
        for key in sensor_dict:
            if key.startswith(f"{family}_") and key not in sensor_keys:
                sensor_keys.append(key)
    sensor_keys.sort(key=lambda x: int(x.rsplit("_", 1)[1]))

    if not sensor_keys:
        return pd.DataFrame(columns=["vehicle_id"])

    results: List[pd.DataFrame] = []
    for vid, sensor_stats in stats.items():
        row: Dict[str, Any] = {"vehicle_id": vid}
        for key in sensor_keys:
            ss = sensor_stats.get(key)
            if not ss or ss["count"] == 0:
                row.update({
                    f"s_{key}_mean": np.nan,
                    f"s_{key}_max": np.nan,
                    f"s_{key}_min": np.nan,
                    f"s_{key}_std": np.nan,
                    f"s_{key}_last": np.nan,
                    f"s_{key}_range": np.nan,
                    f"s_{key}_missing_count": 0,
                    f"s_{key}_trend_slope": 0.0,
                    f"s_{key}_coeff_variation": np.nan,
                })
                continue

            mean_val = ss["sum"] / ss["count"]
            variance = max(ss["sum_sq"] / ss["count"] - mean_val ** 2, 0)
            std_val = variance ** 0.5
            slope = _compute_slope_from_stats(ss)
            cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 0.0

            row.update({
                f"s_{key}_mean": mean_val,
                f"s_{key}_max": ss["max"],
                f"s_{key}_min": ss["min"],
                f"s_{key}_std": std_val,
                f"s_{key}_last": ss["last_val"],
                f"s_{key}_range": ss["max"] - ss["min"],
                f"s_{key}_missing_count": ss["missing"],
                f"s_{key}_trend_slope": slope,
                f"s_{key}_coeff_variation": cv,
            })
        results.append(pd.DataFrame([row]))

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["vehicle_id"])


def _run_etl() -> pd.DataFrame:
    """Run full chunked ETL, returning merged per-vehicle feature DataFrame."""
    log.info("Loading specifications and TTE labels...")
    df_specs = pd.read_csv(TRAIN_SPECS_CSV)
    df_tte = pd.read_csv(TRAIN_TTE_CSV)

    log.info("Detecting sensor families from readouts header...")
    header = pd.read_csv(TRAIN_READOUTS_CSV, nrows=0).columns.tolist()
    families = [
        c.rsplit("_", 1)[0]
        for c in header
        if "_" in c and c.rsplit("_", 1)[1].isdigit() and c.rsplit("_", 1)[0].isdigit()
    ]
    sensor_families = sorted(set(families), key=int)
    log.info("Sensor families detected: %s", sensor_families)

    log.info("Streaming %s in chunks (chunk_size=%d)...", TRAIN_READOUTS_CSV, CHUNK_SIZE)
    stats: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for chunk_iter, chunk in enumerate(pd.read_csv(TRAIN_READOUTS_CSV, chunksize=CHUNK_SIZE)):
        for family in sensor_families:
            family_cols = [c for c in chunk.columns if c.startswith(f"{family}_")]
            if not family_cols:
                continue

            grouped = chunk.groupby("vehicle_id")
            for vid, grp in grouped:
                vid = int(vid)
                if vid not in stats:
                    stats[vid] = {}

                times = grp["time_step"].values
                for col in family_cols:
                    vals = grp[col].values
                    if col not in stats[vid]:
                        stats[vid][col] = _init_sensor_stats()

                    ss = stats[vid][col]
                    valid_mask = ~np.isnan(vals)
                    valid = vals[valid_mask]
                    valid_t = times[valid_mask]
                    n_valid = len(valid)

                    ss["count"] += n_valid
                    ss["sum"] += float(np.sum(valid))
                    ss["sum_sq"] += float(np.sum(valid ** 2))
                    ss["missing"] += int(np.isnan(vals).sum())
                    ss["sum_x"] += float(np.sum(valid_t))
                    ss["sum_xy"] += float(np.sum(valid_t * valid))
                    ss["sum_x_sq"] += float(np.sum(valid_t ** 2))

                    if n_valid > 0:
                        ss["min"] = min(ss["min"], float(np.min(valid)))
                        ss["max"] = max(ss["max"], float(np.max(valid)))
                        ss["last_val"] = float(valid[-1])

        log.info("  Chunk %d done (%d rows)", chunk_iter + 1, len(chunk))
        del chunk

    log.info("Building per-vehicle feature matrices...")
    sensor_dfs: List[pd.DataFrame] = []
    for family in sensor_families:
        log.info("  Aggregating family %s", family)
        sdf = _aggregate_sensor_family(family, stats)
        sensor_dfs.append(sdf)
        del sdf

    del stats

    log.info("Merging features, specs, and labels...")
    df_features = df_specs.merge(df_tte, on="vehicle_id", how="inner")
    for sdf in sensor_dfs:
        df_features = df_features.merge(sdf, on="vehicle_id", how="left")
        del sdf

    for i in range(8):
        df_features.rename(columns={f"Spec_{i}": f"spec_{i}"}, inplace=True)

    log.info("ETL complete: %d vehicles, %d columns", len(df_features), len(df_features.columns))
    return df_features


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_is_valid() -> bool:
    """Return True if the parquet cache exists and is newer than all source CSVs."""
    if not os.path.exists(ETL_CACHE_PARQUET):
        return False
    cache_mtime = os.path.getmtime(ETL_CACHE_PARQUET)
    for path in (TRAIN_READOUTS_CSV, TRAIN_SPECS_CSV, TRAIN_TTE_CSV):
        if not os.path.exists(path) or os.path.getmtime(path) > cache_mtime:
            return False
    return True


def _try_write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write parquet, falling back to CSV if pyarrow is unavailable."""
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except ImportError:
        base = path.rsplit(".", 1)[0]
        fallback = base + ".csv"
        log.info("pyarrow not found — writing CSV fallback to %s", fallback)
        df.to_csv(fallback, index=False)


def _load_or_run_etl() -> pd.DataFrame:
    """Load cached features if valid, otherwise run ETL and cache the result."""
    if _cache_is_valid():
        log.info("Loading cached engineered features from %s", ETL_CACHE_PARQUET)
        df = pd.read_parquet(ETL_CACHE_PARQUET)
        log.info("Cache hit: %d vehicles, %d columns", len(df), len(df.columns))
        return df

    log.info("Cache miss or stale — running ETL...")
    df = _run_etl()

    log.info("Writing cache to %s", ETL_CACHE_PARQUET)
    _try_write_parquet(df, ETL_CACHE_PARQUET)
    if os.path.exists(ETL_CACHE_PARQUET):
        log.info("Cache written (%.1f MB)", os.path.getsize(ETL_CACHE_PARQUET) / 1024 / 1024)
    return df


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _make_tree_pipeline(cls, **kwargs):
    """Return a callable that builds a Pipeline once it sees X (to detect col types)."""
    def _build(X):
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if X[c].dtype in ("float64", "int64", "float")]

        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ], remainder="drop")

        clf_kwargs = {"random_state": RANDOM_STATE, **kwargs}
        sig = inspect.signature(cls.__init__)
        if "n_jobs" in sig.parameters:
            clf_kwargs["n_jobs"] = -1

        return Pipeline([
            ("prep", preprocessor),
            ("clf", cls(**clf_kwargs)),
        ])

    return _build


def _make_linear_pipeline():
    """Return a callable that builds a logistic regression pipeline."""
    def _build(X):
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if X[c].dtype in ("float64", "int64", "float")]

        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
            ]), cat_cols),
        ], remainder="drop")

        return Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ])

    return _build


MODEL_SPECS = [
    ("RandomForest", _make_tree_pipeline(
        RandomForestClassifier,
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight=AGGRESSIVE_CLASS_WEIGHT,
    )),
    ("ExtraTrees", _make_tree_pipeline(
        ExtraTreesClassifier,
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight=AGGRESSIVE_CLASS_WEIGHT,
    )),
    ("GradientBoosting", _make_tree_pipeline(
        GradientBoostingClassifier,
        n_estimators=100, max_depth=6, min_samples_leaf=5, learning_rate=0.1,
    )),
    ("LogisticRegression", _make_linear_pipeline()),
]

# ---- Optional: XGBoost ----
try:
    from xgboost import XGBClassifier
    MODEL_SPECS.append(("XGBoost", _make_tree_pipeline(
        XGBClassifier,
        n_estimators=300, max_depth=8, min_child_weight=5,
        learning_rate=0.1, eval_metric="logloss", tree_method="hist",
        scale_pos_weight=15,
    )))
except ImportError:
    pass

# ---- Optional: LightGBM ----
try:
    from lightgbm import LGBMClassifier
    MODEL_SPECS.append(("LightGBM", _make_tree_pipeline(
        LGBMClassifier,
        n_estimators=300, max_depth=8, min_child_samples=10,
        learning_rate=0.1, verbosity=-1, scale_pos_weight=15,
    )))
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _cv_evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Run stratified k-fold CV and return mean metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
    }
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        k: float(results[f"test_{k}"].mean())
        for k in scoring
    }


def _cv_evaluate_with_timeout(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, timeout_seconds: int = 360,
) -> Dict[str, float]:
    """Run CV evaluation in a background thread. Raises TimeoutError if it exceeds the timeout."""
    result_container: List[Any] = [None]
    exception_container: List[Any] = [None]

    def _run():
        try:
            result_container[0] = _cv_evaluate(pipeline, X, y)
        except Exception as e:
            exception_container[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"CV evaluation exceeded {timeout_seconds}s")

    if exception_container[0] is not None:
        raise exception_container[0]

    return result_container[0]


def _f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F2 score — weights recall 4× more than precision."""
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    return 5 * p * r / (4 * p + r)


def _find_optimal_threshold(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
) -> Tuple[float, Dict[str, float]]:
    """Train on a held-out split, sweep thresholds, return best threshold + F2 metrics."""
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    fitted = clone(pipeline)
    fitted.fit(X_train, y_train)

    probas = fitted.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f2 = 0.0
    best_metrics: Dict[str, float] = {}

    for t_int in range(5, 55, 2):
        t = t_int / 100.0
        preds = (probas >= t).astype(int)
        f2 = _f2_score(y_val.values, preds)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t
            best_metrics = {
                "opt_threshold": round(t, 3),
                "opt_f2": round(f2, 4),
                "opt_precision": round(precision_score(y_val.values, preds, zero_division=0), 4),
                "opt_recall": round(recall_score(y_val.values, preds, zero_division=0), 4),
                "opt_f1": round(f1_score(y_val.values, preds), 4),
            }

    return best_threshold, best_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline model comparison")
    parser.add_argument("--force-etl", action="store_true", help="Re-run ETL even if cache is valid")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("OFFLINE MODEL COMPARISON")
    log.info("=" * 60)

    # ETL (cached)
    if args.force_etl:
        log.info("--force-etl requested, ignoring cache")
        df = _run_etl()
        log.info("Writing cache to %s", ETL_CACHE_PARQUET)
        _try_write_parquet(df, ETL_CACHE_PARQUET)
    else:
        df = _load_or_run_etl()

    # Feature matrix
    drop_cols = ["vehicle_id", "in_study_repair", "length_of_study_time_step"]
    X = df.drop(columns=drop_cols)
    y = df["in_study_repair"]

    valid_mask = y.notna()
    X, y = X[valid_mask], y[valid_mask].astype(int)

    log.info("Final dataset: %d samples, %d features", len(X), len(X.columns))
    log.info("Target distribution: %s", dict(y.value_counts().sort_index()))

    # Benchmark
    results: List[Dict[str, Any]] = []

    for name, pipeline_obj in MODEL_SPECS:
        log.info("-" * 40)
        log.info("Benchmarking: %s", name)

        if callable(pipeline_obj):
            pipeline = pipeline_obj(X)
        else:
            pipeline = pipeline_obj

        t0 = time.time()
        try:
            metrics = _cv_evaluate_with_timeout(pipeline, X, y, timeout_seconds=360)
        except TimeoutError as e:
            log.warning("  %s — SKIPPED", e)
            continue
        elapsed = time.time() - t0

        # Threshold optimization (F2: weights recall 4× more than precision)
        opt_threshold, opt_metrics = _find_optimal_threshold(pipeline, X, y)
        metrics.update(opt_metrics)

        metrics["model"] = name
        metrics["train_time_s"] = round(elapsed, 1)
        results.append(metrics)

        log.info("  Default (0.5) -> ROC AUC: %.4f  |  F1: %.4f  |  Recall: %.4f",
                 metrics["roc_auc"], metrics["f1"], metrics["recall"])
        log.info("  Optimized  (%.2f) -> F2: %.4f    |  P: %.4f    |  Recall: %.4f",
                 opt_threshold, opt_metrics["opt_f2"],
                 opt_metrics["opt_precision"], opt_metrics["opt_recall"])

    # ---- Print comparison table ----
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df_results["rank"] = range(1, len(df_results) + 1)

    separator = "=" * 90
    header_fmt = (
        " {:>4} | {:<18} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>7}"
        " | {:>8} | {:>8} | {:>8}"
    )
    row_fmt = (
        " {:>4} | {:<18} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f}"
        " | {:>6.0f}s | {:>8.4f} | {:>8.4f} | {:>5.3f}"
    )

    log.info("\n" + separator)
    log.info("         OFFLINE MODEL COMPARISON (%d-fold stratified CV)", CV_FOLDS)
    log.info(separator)
    log.info(header_fmt.format(
        "Rank", "Model", "ROC AUC", "PR AUC", "F1", "Precision", "Recall",
        "Accuracy", "Time", "Opt F2", "Opt Rec", "Opt Th",
    ))
    log.info("-" * 110)

    for _, r in df_results.iterrows():
        log.info(row_fmt.format(
            r["rank"], r["model"],
            r["roc_auc"], r["pr_auc"], r["f1"],
            r["precision"], r["recall"], r["accuracy"],
            r["train_time_s"],
            r.get("opt_f2", ""), r.get("opt_recall", ""), r.get("opt_threshold", ""),
        ))

    log.info(separator)

    best_auc = df_results.iloc[0]
    best_f2 = df_results.loc[df_results["opt_f2"].idxmax()]
    log.info("Best ROC AUC: %s (%.4f)", best_auc["model"], best_auc["roc_auc"])
    log.info("Best Opt F2:  %s (%.4f, threshold=%.3f)",
             best_f2["model"], best_f2["opt_f2"], best_f2["opt_threshold"])
    log.info(separator)

    # ---- Pairwise deltas vs best ROC AUC ----
    best_name = df_results.iloc[0]["model"]
    best_row = df_results.set_index("model").loc[best_name]

    log.info("\nDeltas vs best (%s):", best_name)
    log.info(" {:<18} | {:>10} | {:>10} | {:>10}".format("Model", "Δ ROC AUC", "Δ F1", "Δ PR AUC"))
    log.info("-" * 50)
    for _, r in df_results.iterrows():
        if r["model"] == best_name:
            continue
        d_auc = r["roc_auc"] - best_row["roc_auc"]
        d_f1 = r["f1"] - best_row["f1"]
        d_pr = r["pr_auc"] - best_row["pr_auc"]
        log.info(" {:<18} | {:>+10.4f} | {:>+10.4f} | {:>+10.4f}".format(
            r["model"], d_auc, d_f1, d_pr,
        ))

    # ---- Save summary ----
    summary_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", "offline_comparison_summary.json",
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    opt_records = []
    for _, r in df_results.iterrows():
        opt_records.append({
            "model": r["model"],
            "roc_auc": r["roc_auc"],
            "default_f1": r["f1"],
            "default_recall": r["recall"],
            "default_precision": r["precision"],
            "opt_threshold": r.get("opt_threshold", ""),
            "opt_f2": r.get("opt_f2", ""),
            "opt_recall": r.get("opt_recall", ""),
            "opt_precision": r.get("opt_precision", ""),
            "opt_f1": r.get("opt_f1", ""),
            "train_time_s": r["train_time_s"],
        })

    with open(summary_path, "w") as f:
        json.dump({
            "n_samples": len(X),
            "n_features": len(X.columns),
            "target_distribution": {
                str(k): int(v) for k, v in y.value_counts().sort_index().items()
            },
            "cv_folds": CV_FOLDS,
            "models": opt_records,
        }, f, indent=2)
    log.info("\nSummary saved to %s", summary_path)

    log.info("=" * 60)
    log.info("OFFLINE COMPARISON COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
