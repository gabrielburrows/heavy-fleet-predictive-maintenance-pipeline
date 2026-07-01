"""EDA: Map SCANIA Component X prefixes to physical sensor roles."""
import logging

import numpy as np
import pandas as pd

from config import TRAIN_READOUTS_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def infer_sensor_role(family: str, n_bins: int, monotonicity: float) -> str:
    """Guess the physical nature of a sensor family based on structural heuristics."""
    if monotonicity > 0.95:
        return "CUMULATIVE COUNTER (Mileage / Hours / Cycles)"
    if n_bins >= 16:
        return "FINE-GRAIN HISTOGRAM (Likely Temperature or RPM profile)"
    if n_bins >= 10:
        return "MED-GRAIN HISTOGRAM (Likely Pressure or Load distribution)"
    if n_bins >= 5:
        return "COARSE HISTOGRAM (State tracking / Duty cycle)"
    return "SINGLE-VALUE GAUGE (Instantaneous reading)"


def run(csv_path: str = TRAIN_READOUTS_CSV):
    log.info("Loading data to map sensor families...")
    df = pd.read_csv(csv_path, nrows=15_000)

    exclude = {"vehicle_id", "time_step"}
    sensors = [c for c in df.columns if c not in exclude]

    # 1. Group columns by their family prefix (e.g., "158", "397")
    families = {}
    for col in sensors:
        prefix = col.rsplit("_", 1)[0]
        families.setdefault(prefix, []).append(col)

    log.info("Detected %d sensor families:\n", len(families))

    # 2. Calculate monotonicity and role for each family
    results = []
    for fam, cols in sorted(families.items(), key=lambda x: int(x[0])):
        # Average monotonicity across all bins in this family
        mono_scores = []
        for c in cols:
            scores = []
            for _, grp in df.groupby("vehicle_id"):
                diffs = grp[c].diff().dropna()
                if len(diffs) > 0:
                    scores.append((diffs >= 0).mean())
            mono_scores.append(np.mean(scores) if scores else 0.0)
        
        avg_mono = np.mean(mono_scores)
        role = infer_sensor_role(fam, len(cols), avg_mono)
        results.append((fam, len(cols), avg_mono, role, cols))

    # 3. Print structured report
    log.info(f"{'Family':<8} | {'Bins':<6} | {'Monotonicity':<14} | {'Inferred Role'}")
    log.info("-" * 75)
    for fam, n_bins, mono, role, cols in results:
        log.info(f"{fam:<8} | {n_bins:<6} | {mono:<14.4f} | {role}")
        log.info(f"           -> Bins: {sorted(cols)}\n")


if __name__ == "__main__":
    run()
