"""Column-level and dataset-level profiling of a telemetry batch.

For every column computes:
  - completeness: fraction of non-null values
  - validity:     fraction of non-null values passing domain rules
                  (numeric ranges, firmware regex vX.Y.Z, allowed status values)
  - uniqueness:   fraction of distinct values
  - cardinality:  number of distinct values

Produces one JSON report per batch with a dataset-level summary.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Domain validity rules for the AriaHome telemetry feed.
RANGES = {
    "temperature_c": (-40.0, 85.0),
    "humidity_pct": (0.0, 100.0),
    "battery_pct": (0.0, 100.0),
    "signal_dbm": (-120.0, 0.0),
}
FIRMWARE_RE = re.compile(r"^v\d+\.\d+\.\d+$")
ALLOWED_STATUS = {"online", "offline", "maintenance"}


def validity_mask(series: pd.Series) -> pd.Series:
    """Boolean mask of valid non-null values for a column."""
    name = series.name
    valid = series.notna()
    vals = series[valid]
    if name in RANGES:
        lo, hi = RANGES[name]
        ok = pd.to_numeric(vals, errors="coerce").between(lo, hi)
    elif name == "firmware_version":
        ok = vals.astype(str).str.match(FIRMWARE_RE)
    elif name in ("status", "device_status"):
        ok = vals.isin(ALLOWED_STATUS)
    else:
        ok = pd.Series(True, index=vals.index)
    return ok.reindex(series.index, fill_value=False)


def profile_dataframe(df: pd.DataFrame, batch_name: str) -> dict:
    columns = {}
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        completeness = non_null / len(s) if len(s) else 0.0
        valid = int(validity_mask(s).sum())
        validity = valid / non_null if non_null else 0.0
        cardinality = int(s.nunique(dropna=True))
        uniqueness = cardinality / non_null if non_null else 0.0
        columns[col] = {
            "dtype": str(s.dtype),
            "count": len(s),
            "non_null": non_null,
            "nulls": int(s.isna().sum()),
            "completeness": round(completeness, 4),
            "validity": round(validity, 4),
            "cardinality": cardinality,
            "uniqueness": round(uniqueness, 4),
        }
    completeness_mean = float(np.mean([c["completeness"] for c in columns.values()])) if columns else 0.0
    validity_mean = float(np.mean([c["validity"] for c in columns.values()])) if columns else 0.0
    return {
        "batch": batch_name,
        "rows": len(df),
        "columns": columns,
        "summary": {
            "n_columns": len(columns),
            "mean_completeness": round(completeness_mean, 4),
            "mean_validity": round(validity_mean, 4),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_pct": round(float(df.duplicated().mean() * 100), 2),
        },
    }


def profile_csv(csv_path: str, batch_name: str, out_path: str | None = None) -> dict:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    report = profile_dataframe(df, batch_name)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
        print(f"profile written: {out_path}")
    return report


if __name__ == "__main__":
    profile_csv("data/baseline.csv", "baseline", "report/profile_baseline.json")
    profile_csv("data/new.csv", "new", "report/profile_new.json")
