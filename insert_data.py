"""Chunked ETL pipeline: aggregate raw telemetry into engineered features, then upload to Supabase."""

import logging
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

from config import (
    CHUNK_SIZE,
    TRAIN_READOUTS_CSV,
    TRAIN_SPECS_CSV,
    TRAIN_TTE_CSV,
    TABLE_ACCOUNTS,
    TABLE_VEHICLES,
    TABLE_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

_JAPANESE_COMPANIES: List[Tuple[str, str, int]] = [
    ("Yamato Regional Freight", "Tokyo", 45),
    ("Kanto Express Logistics", "Kanagawa", 38),
    ("Tohoku Heavy Transport", "Miyagi", 52),
    ("Okayama Industrial Shipping", "Okayama", 30),
    ("Kyushu Fleet Services", "Fukuoka", 41),
    ("Nagoya Freight Systems", "Aichi", 35),
    ("Hokkaido Bulk Logistics", "Hokkaido", 48),
    ("Osaka Port Carriers", "Osaka", 28),
    ("Shikoku Cargo Lines", "Kagawa", 22),
    ("Chubu Transport Group", "Nagano", 33),
    ("Kyoto Distribution Co", "Kyoto", 25),
    ("Hiroshima Fleet Corp", "Hiroshima", 27),
    ("Sendai Express Haulage", "Miyagi", 20),
    ("Niigata Freight Network", "Niigata", 18),
    ("Fukuoka City Logistics", "Fukuoka", 31),
    ("Sapporo Cold Chain", "Hokkaido", 24),
    ("Kobe Marine Transport", "Hyogo", 36),
    ("Okayama Valley Shipping", "Okayama", 19),
    ("Shizuoka Port Logistics", "Shizuoka", 26),
    ("Matsuyama Cargo Express", "Ehime", 21),
    ("Kumamoto Heavy Haulage", "Kumamoto", 23),
    ("Kanazawa Freight Lines", "Ishikawa", 17),
    ("Utsunomiya Distribution", "Tochigi", 29),
    ("Kochi Industrial Transport", "Kochi", 15),
    ("Saga Fleet Management", "Saga", 16),
]

_ENGINE_FAMILIES = ["Cummins ISX15", "Mitsubishi P11C", "Isuzu 6SKY", "Mitsubishi P06C", "Cummins L9"]
_MANUFACTURERS = ["Isuzu", "Hino", "Mitsubishi Fuso", "UD Trucks", "Mercedes-Benz"]
_MODEL_SUFFIXES = ["FV Series", "500 Series", "Super Great", "Quon", "eCanter", "Fitter", "Gallop"]


def _generate_company_pool(
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate pool of companies expanded by fleet_size copies."""
    rows: List[Dict[str, Any]] = []
    for idx, (name, prefecture, fleet_size) in enumerate(_JAPANESE_COMPANIES):
        for _ in range(fleet_size):
            rows.append({
                "account_id": idx + 1,
                "company_name": name,
                "industry": rng.choice(
                    ["Freight Transport", "Cold Chain Logistics",
                     "Construction Materials", "Industrial Parts",
                     "E-commerce Delivery", "Waste Management"],
                ),
                "operating_prefecture": prefecture,
                "fleet_size": fleet_size,
            })
    return pd.DataFrame(rows)


def _assign_vehicles_to_accounts(
    vehicle_ids: np.ndarray,
    company_pool: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Randomly assign each vehicle to a company account."""
    account_ids = rng.choice(company_pool["account_id"].unique(), size=len(vehicle_ids))
    return pd.DataFrame({
        "vehicle_id": vehicle_ids,
        "account_id": account_ids,
        "vehicle_age_years": rng.uniform(1, 15, size=len(vehicle_ids)).round(1),
        "vehicle_model": rng.choice(_MODEL_SUFFIXES, size=len(vehicle_ids)),
        "engine_family": rng.choice(_ENGINE_FAMILIES, size=len(vehicle_ids)),
        "manufacture_year": rng.integers(2010, 2024, size=len(vehicle_ids)),
    })


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
        "last_val": float("nan"),
    }


def _update_sensor_stats(ss: Dict[str, Any], vals: np.ndarray, times: np.ndarray) -> None:
    """Update running statistics for a sensor with new observations."""
    valid_mask = ~np.isnan(vals)
    valid = vals[valid_mask]
    valid_t = times[valid_mask]
    n_valid = len(valid)

    ss["count"] += n_valid
    ss["sum"] += float(np.sum(valid))
    ss["sum_sq"] += float(np.sum(valid ** 2))
    ss["missing"] += int(valid_mask.sum() ^ len(vals))
    ss["missing"] = int(np.isnan(vals).sum()) + ss["missing"] - int(valid_mask.sum() ^ len(vals))
    ss["missing"] = ss["missing"] - int(n_valid) + int(np.isnan(vals).sum())

    if n_valid > 0:
        ss["min"] = min(ss["min"], float(np.min(valid)))
        ss["max"] = max(ss["max"], float(np.max(valid)))
        ss["sum_x"] += float(np.sum(valid_t))
        ss["sum_xy"] += float(np.sum(valid_t * valid))
        ss["last_val"] = float(valid[-1])


def _compute_slope_from_stats(ss: Dict[str, Any]) -> float:
    """Compute linear regression slope from running statistics (Welford-style)."""
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


def _detect_sensor_families(columns: List[str]) -> List[str]:
    """Detect sensor families from column names (pattern: FAMILY_INDEX)."""
    families = set()
    for col in columns:
        if "_" in col:
            parts = col.rsplit("_", 1)
            if parts[1].isdigit() and parts[0].isdigit():
                families.add(parts[0])
    return sorted(families, key=int)


def main() -> None:
    load_dotenv()
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise ValueError("SUPABASE_DB_URL missing from .env")

    engine = create_engine(db_url)
    rng = np.random.default_rng(42)

    log.info("Loading specifications and TTE labels...")
    df_specs = pd.read_csv(TRAIN_SPECS_CSV)
    df_tte = pd.read_csv(TRAIN_TTE_CSV)

    log.info("Detecting sensor families from readouts header...")
    header = pd.read_csv(TRAIN_READOUTS_CSV, nrows=0).columns.tolist()
    sensor_families = _detect_sensor_families(header)
    log.info("Sensor families detected: %s", sensor_families)

    log.info("Reading %s in chunks (chunk_size=%d)...", TRAIN_READOUTS_CSV, CHUNK_SIZE)

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
                    key = col
                    vals = grp[col].values
                    if key not in stats[vid]:
                        stats[vid][key] = _init_sensor_stats()
                        stats[vid][key]["sum_x_sq"] = 0.0

                    ss = stats[vid][key]
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

        log.info("  Chunk %d processed (%d rows)", chunk_iter + 1, len(chunk))
        del chunk

    log.info("Building per-vehicle feature matrices...")
    sensor_dfs: List[pd.DataFrame] = []
    for family in sensor_families:
        log.info("  Processing family: %s", family)
        sdf = _aggregate_sensor_family(family, stats)
        sensor_dfs.append(sdf)
        del sdf

    log.info("Memory cleanup: freeing running-stat dicts...")
    del stats

    log.info("Merging engineered features with specs and labels...")
    df_features = df_specs.merge(df_tte, on="vehicle_id", how="inner")

    for sdf in sensor_dfs:
        df_features = df_features.merge(sdf, on="vehicle_id", how="left")

    unique_vehicles = sorted(df_features["vehicle_id"].unique())
    log.info("Total unique vehicles: %d", len(unique_vehicles))

    company_pool = _generate_company_pool(rng)
    df_vehicles = _assign_vehicles_to_accounts(
        np.array(unique_vehicles), company_pool, rng
    )
    df_accounts = company_pool.drop_duplicates(subset="account_id").reset_index(drop=True)

    log.info("Uploading to Supabase (engineered features only)...")
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_FEATURES} CASCADE"))
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_VEHICLES} CASCADE"))
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_ACCOUNTS} CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS vehicle_sensor_telemetry CASCADE"))
        conn.commit()

        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_ACCOUNTS} (
                account_id INTEGER PRIMARY KEY,
                company_name TEXT,
                industry TEXT,
                operating_prefecture TEXT,
                fleet_size INTEGER
            )
        """))

        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_VEHICLES} (
                vehicle_id INTEGER PRIMARY KEY,
                account_id INTEGER REFERENCES {TABLE_ACCOUNTS}(account_id),
                vehicle_age_years NUMERIC,
                vehicle_model TEXT,
                engine_family TEXT,
                manufacture_year INTEGER
            )
        """))

        spec_cols = ", ".join(f"spec_{i} TEXT" for i in range(8))

        conn.commit()

        log.info("Inserting accounts...")
        df_accounts.to_sql(TABLE_ACCOUNTS, conn, if_exists="append", index=False, chunksize=100)

        log.info("Inserting vehicles...")
        df_vehicles.to_sql(TABLE_VEHICLES, conn, if_exists="append", index=False, method="multi", chunksize=100)

        log.info("Inserting engineered features...")
        df_upload = df_features.copy()
        for i in range(8):
            df_upload.rename(columns={f"Spec_{i}": f"spec_{i}"}, inplace=True)

        safe_df = df_upload.replace({np.inf: np.nan, -np.inf: np.nan})
        safe_df.to_sql(
            TABLE_FEATURES, conn, if_exists="replace", index=False,
            chunksize=100,
        )

    log.info("ETL complete. %d vehicles uploaded as engineered features.", len(unique_vehicles))


if __name__ == "__main__":
    main()
