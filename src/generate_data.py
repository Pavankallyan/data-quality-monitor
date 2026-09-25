"""Synthetic IoT device telemetry generator for the data-quality-monitor demo.

Generates two batches of a FICTIONAL "AriaHome" smart-home sensor feed:
  - baseline: 5000 clean rows
  - new:      5000 rows with INJECTED quality issues (known ground truth):
      1. missing-value spike in temperature_c and humidity_pct (~8%)
      2. out-of-range values: humidity_pct up to 135, battery_pct down to -12
      3. ~2% exact duplicate rows
      4. schema drift: column renamed  status -> device_status
      5. new unexpected column: signal_quality
      6. distribution shift: temperature_c mean shifted +6 deg C

Fixed seeds make the batches fully reproducible.
All data is synthetic — no real devices or companies are involved.
"""

import numpy as np
import pandas as pd

N_ROWS = 5000
BASE_SEED = 42
NEW_SEED = 7
BASELINE_PATH = "data/baseline.csv"
NEW_PATH = "data/new.csv"

DEVICE_IDS = [f"AH-{i:05d}" for i in range(1, 201)]
FIRMWARES = ["v1.4.2", "v1.5.0", "v2.0.1", "v2.1.0"]
STATUSES = ["online", "offline", "maintenance"]


def make_baseline(n: int = N_ROWS, seed: int = BASE_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-06-01", periods=n, freq="min")
    df = pd.DataFrame({
        "device_id": rng.choice(DEVICE_IDS, size=n),
        "timestamp": rng.choice(ts, size=n, replace=True),
        "temperature_c": np.round(rng.normal(22.0, 3.0, n), 2),
        "humidity_pct": np.round(rng.normal(45.0, 8.0, n).clip(5, 95), 2),
        "battery_pct": np.round(rng.normal(78.0, 12.0, n).clip(0, 100), 1),
        "signal_dbm": np.round(rng.normal(-62.0, 8.0, n).clip(-100, -20), 1),
        "firmware_version": rng.choice(FIRMWARES, size=n, p=[0.1, 0.3, 0.4, 0.2]),
        "status": rng.choice(STATUSES, size=n, p=[0.85, 0.10, 0.05]),
    })
    # tiny bit of natural missingness (~0.2%) so the profiler has something real
    mask = rng.random(n) < 0.002
    df.loc[mask, "humidity_pct"] = np.nan
    return df.sort_values("timestamp").reset_index(drop=True)


def make_new(n: int = N_ROWS, seed: int = NEW_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-07-01", periods=n, freq="min")
    df = pd.DataFrame({
        "device_id": rng.choice(DEVICE_IDS, size=n),
        "timestamp": rng.choice(ts, size=n, replace=True),
        # 6) distribution shift: temperature mean +6 deg C
        "temperature_c": np.round(rng.normal(28.0, 3.5, n), 2),
        "humidity_pct": np.round(rng.normal(45.0, 8.0, n).clip(5, 95), 2),
        "battery_pct": np.round(rng.normal(78.0, 12.0, n).clip(0, 100), 1),
        "signal_dbm": np.round(rng.normal(-62.0, 8.0, n).clip(-100, -20), 1),
        "firmware_version": rng.choice(FIRMWARES, size=n, p=[0.05, 0.25, 0.45, 0.25]),
        "status": rng.choice(STATUSES, size=n, p=[0.80, 0.12, 0.08]),
    })
    # 1) missing-value spike: ~8% in temperature_c and humidity_pct
    for col in ("temperature_c", "humidity_pct"):
        df.loc[rng.random(n) < 0.08, col] = np.nan
    # 2) out-of-range values: humidity > 100 (up to 135), battery < 0 (down to -12)
    idx = rng.choice(n, size=120, replace=False)
    df.loc[idx, "humidity_pct"] = np.round(rng.uniform(101, 135, 120), 2)
    idx = rng.choice(n, size=90, replace=False)
    df.loc[idx, "battery_pct"] = np.round(rng.uniform(-12, -0.1, 90), 1)
    # 4) schema drift: rename status -> device_status
    df = df.rename(columns={"status": "device_status"})
    # 5) new unexpected column
    df["signal_quality"] = rng.choice(["good", "fair", "poor"], size=n, p=[0.7, 0.2, 0.1])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # 3) ~2% exact duplicate rows
    dupes = df.sample(frac=0.02, random_state=seed).copy()
    df = pd.concat([df, dupes], ignore_index=True)
    return df


def main() -> None:
    baseline = make_baseline()
    new = make_new()
    baseline.to_csv(BASELINE_PATH, index=False)
    new.to_csv(NEW_PATH, index=False)
    print(f"baseline: {baseline.shape} -> {BASELINE_PATH}")
    print(f"new:      {new.shape} -> {NEW_PATH}")


if __name__ == "__main__":
    main()
