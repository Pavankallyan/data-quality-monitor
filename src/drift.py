"""Drift detection between a baseline and a new batch.

Per shared column:
  - numeric:     PSI (Population Stability Index) + Kolmogorov-Smirnov test
  - categorical: PSI on category shares + chi-square test of independence

PSI thresholds (credit-risk industry standard):
  < 0.10  -> no significant change
  0.10-0.25 -> moderate change (investigate)
  > 0.25  -> significant change (action needed)

Also produces a schema diff: added / removed / renamed columns, dtype changes.
Renames are detected when a removed column and an added column share the same
distinct-value signature on common rows (value-overlap heuristic).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PSI_BINS = 10
EPS = 1e-4


def psi_numeric(base: pd.Series, new: pd.Series, bins: int = PSI_BINS) -> float:
    """PSI over quantile bins of the baseline distribution."""
    base = base.dropna().astype(float)
    new = new.dropna().astype(float)
    if base.empty or new.empty:
        return 0.0
    qs = np.linspace(0, 1, bins + 1)
    breaks = base.quantile(qs).unique()
    if len(breaks) < 3:
        return 0.0
    breaks = np.unique(np.concatenate(([-np.inf], breaks[1:-1], [np.inf])))
    b_counts, _ = np.histogram(base, bins=breaks)
    n_counts, _ = np.histogram(new, bins=breaks)
    b_pct = b_counts / b_counts.sum()
    n_pct = n_counts / n_counts.sum()
    b_pct = np.clip(b_pct, EPS, None)
    n_pct = np.clip(n_pct, EPS, None)
    return float(np.sum((n_pct - b_pct) * np.log(n_pct / b_pct)))


def psi_categorical(base: pd.Series, new: pd.Series) -> float:
    """PSI over category shares (union of categories)."""
    cats = pd.Index(base.dropna().unique()).union(pd.Index(new.dropna().unique()))
    if len(cats) == 0:
        return 0.0
    b = base.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    n = new.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    b = np.clip(b, EPS, None)
    n = np.clip(n, EPS, None)
    return float(np.sum((n - b) * np.log(n / b)))


def psi_band(psi: float) -> str:
    if psi < 0.10:
        return "no change"
    if psi <= 0.25:
        return "moderate"
    return "significant"


def ks_drift(base: pd.Series, new: pd.Series) -> dict:
    base = base.dropna().astype(float)
    new = new.dropna().astype(float)
    if base.empty or new.empty or base.nunique() < 2:
        return {"ks_stat": None, "p_value": None, "drifted": False}
    res = stats.ks_2samp(base, new)
    return {
        "ks_stat": round(float(res.statistic), 4),
        "p_value": float(res.pvalue),
        "drifted": bool(res.pvalue < 0.05),
    }


def chisq_drift(base: pd.Series, new: pd.Series) -> dict:
    cats = pd.Index(base.dropna().unique()).union(pd.Index(new.dropna().unique()))
    if len(cats) < 2:
        return {"chi2_stat": None, "p_value": None, "drifted": False}
    b = base.value_counts().reindex(cats, fill_value=0).to_numpy()
    n = new.value_counts().reindex(cats, fill_value=0).to_numpy()
    table = np.array([b, n])
    if (table.sum(axis=1) == 0).any():
        return {"chi2_stat": None, "p_value": None, "drifted": False}
    chi2, p, _, _ = stats.chi2_contingency(table)
    return {
        "chi2_stat": round(float(chi2), 4),
        "p_value": float(p),
        "drifted": bool(p < 0.05),
    }


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def column_drift(base: pd.DataFrame, new: pd.DataFrame, col: str) -> dict:
    b, n = base[col], new[col]
    result = {"column": col, "type": "numeric" if is_numeric(b) else "categorical"}
    if is_numeric(b):
        psi = psi_numeric(b, n)
        result["psi"] = round(psi, 4)
        result["psi_band"] = psi_band(psi)
        result.update(ks_drift(b, n))
    else:
        psi = psi_categorical(b, n)
        result["psi"] = round(psi, 4)
        result["psi_band"] = psi_band(psi)
        result.update(chisq_drift(b, n))
    return result


def _value_signature(s: pd.Series) -> frozenset:
    return frozenset(s.dropna().astype(str).unique()[:2000])


def schema_diff(base: pd.DataFrame, new: pd.DataFrame) -> dict:
    base_cols = list(base.columns)
    new_cols = list(new.columns)
    removed = [c for c in base_cols if c not in new_cols]
    added = [c for c in new_cols if c not in base_cols]
    renamed = []
    dtype_changes = []
    for c in base_cols:
        if c in new_cols and str(base[c].dtype) != str(new[c].dtype):
            dtype_changes.append({
                "column": c,
                "baseline_dtype": str(base[c].dtype),
                "new_dtype": str(new[c].dtype),
            })
    # rename heuristic: removed + added columns with highly overlapping value sets
    used_added = set()
    for r in removed:
        r_sig = _value_signature(base[r])
        for a in added:
            if a in used_added or not r_sig:
                continue
            a_sig = _value_signature(new[a])
            overlap = len(r_sig & a_sig) / max(len(r_sig | a_sig), 1)
            if overlap > 0.8:
                renamed.append({"from": r, "to": a, "value_overlap": round(overlap, 4)})
                used_added.add(a)
                break
    added = [c for c in added if c not in used_added]
    removed = [c for c in removed if c not in {x["from"] for x in renamed}]
    return {
        "added": added,
        "removed": removed,
        "renamed": renamed,
        "dtype_changes": dtype_changes,
    }


def detect_drift(
    baseline_path: str,
    new_path: str,
    out_path: str | None = None,
) -> dict:
    base = pd.read_csv(baseline_path, parse_dates=["timestamp"])
    new = pd.read_csv(new_path, parse_dates=["timestamp"])
    shared = [c for c in base.columns if c in new.columns and c != "timestamp"]
    columns = [column_drift(base, new, c) for c in shared]
    drift = {
        "baseline_rows": len(base),
        "new_rows": len(new),
        "shared_columns": shared,
        "columns": columns,
        "schema_diff": schema_diff(base, new),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(drift, indent=2))
        print(f"drift report written: {out_path}")
    return drift


if __name__ == "__main__":
    detect_drift("data/baseline.csv", "data/new.csv", "report/drift.json")
