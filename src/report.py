"""Combine profiles + drift into one quality report and charts.

Overall quality score (0-100, per batch):
    score = 100 * (0.4 * mean_completeness + 0.4 * mean_validity + 0.2 * (1 - duplicate_rate))
Completeness and validity dominate because missing/invalid telemetry is what
breaks downstream models; duplicates are penalized less.

Outputs:
  - report/quality_report.json  (score, failed checks table, drift findings table)
  - report/plots/missingness.png
  - report/plots/psi.png
  - report/plots/shift_temperature_c.png  (baseline vs new histograms of the
    significantly-shifted numeric column, if any)
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from profile import profile_csv  # noqa: E402
from drift import detect_drift  # noqa: E402

PLOTS_DIR = Path("report/plots")
W_COMP, W_VALID, W_DEDUP = 0.4, 0.4, 0.2
FAILED_COMPLETENESS = 0.95  # flag a column if completeness < 95%
FAILED_VALIDITY = 0.99      # flag a column if validity < 99%


def quality_score(profile: dict) -> float:
    s = profile["summary"]
    dup_rate = profile["columns"] and (s["duplicate_rows"] / max(profile["rows"], 1))
    score = 100 * (
        W_COMP * s["mean_completeness"]
        + W_VALID * s["mean_validity"]
        + W_DEDUP * (1 - min(dup_rate, 1.0))
    )
    return round(score, 2)


def failed_checks(profile: dict) -> list[dict]:
    fails = []
    for col, c in profile["columns"].items():
        if c["completeness"] < FAILED_COMPLETENESS:
            fails.append({
                "batch": profile["batch"],
                "column": col,
                "check": "completeness",
                "value": c["completeness"],
                "threshold": FAILED_COMPLETENESS,
                "detail": f"{c['nulls']} nulls of {c['count']} rows",
            })
        if c["validity"] < FAILED_VALIDITY:
            fails.append({
                "batch": profile["batch"],
                "column": col,
                "check": "validity",
                "value": c["validity"],
                "threshold": FAILED_VALIDITY,
                "detail": f"{round((1 - c['validity']) * c['non_null'])} invalid non-null values",
            })
    if profile["summary"]["duplicate_pct"] > 1.0:
        fails.append({
            "batch": profile["batch"],
            "column": "(dataset)",
            "check": "duplicates",
            "value": profile["summary"]["duplicate_pct"],
            "threshold": 1.0,
            "detail": f"{profile['summary']['duplicate_rows']} duplicate rows",
        })
    return fails


def drift_findings(drift: dict) -> list[dict]:
    findings = []
    for c in drift["columns"]:
        stat = "ks" if c["type"] == "numeric" else "chi2"
        findings.append({
            "column": c["column"],
            "type": c["type"],
            "psi": c["psi"],
            "psi_band": c["psi_band"],
            "test": stat,
            "p_value": c.get("p_value"),
            "drifted": bool(c.get("drifted")) or c["psi_band"] == "significant",
        })
    return findings


def plot_missingness(profile: dict, path: Path) -> None:
    cols = list(profile["columns"].keys())
    miss = [1 - profile["columns"][c]["completeness"] for c in cols]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#d62728" if m > 0.05 else "#2ca02c" for m in miss]
    ax.bar(cols, [m * 100 for m in miss], color=colors)
    ax.set_ylabel("Missing %")
    ax.set_title(f"Missing values per column — {profile['batch']} batch")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_psi(drift: dict, path: Path) -> None:
    cols = [c["column"] for c in drift["columns"]]
    psis = [c["psi"] for c in drift["columns"]]
    bands = [c["psi_band"] for c in drift["columns"]]
    color = {"no change": "#2ca02c", "moderate": "#ff7f0e", "significant": "#d62728"}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(cols, psis, color=[color[b] for b in bands])
    ax.axvline(0.10, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.25, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("PSI (baseline vs new)")
    ax.set_title("Population Stability Index per column")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_shift(base: pd.DataFrame, new: pd.DataFrame, col: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b = base[col].dropna().astype(float)
    n = new[col].dropna().astype(float)
    bins = np.histogram_bin_edges(pd.concat([b, n]), bins=40)
    ax.hist(b, bins=bins, alpha=0.6, label=f"baseline (mean={b.mean():.2f})")
    ax.hist(n, bins=bins, alpha=0.6, label=f"new (mean={n.mean():.2f})")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(f"Distribution shift — {col}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_report(
    baseline_path: str = "data/baseline.csv",
    new_path: str = "data/new.csv",
    out_path: str = "report/quality_report.json",
) -> dict:
    prof_base = profile_csv(baseline_path, "baseline")
    prof_new = profile_csv(new_path, "new")
    drift = detect_drift(baseline_path, new_path)

    scores = {"baseline": quality_score(prof_base), "new": quality_score(prof_new)}
    fails = failed_checks(prof_base) + failed_checks(prof_new)
    findings = drift_findings(drift)

    report = {
        "scores": scores,
        "checks_passed": {"baseline": None, "new": None},  # filled below
        "failed_checks": fails,
        "drift_findings": findings,
        "schema_diff": drift["schema_diff"],
        "profiles": {"baseline": prof_base, "new": prof_new},
    }
    for batch, prof in (("baseline", prof_base), ("new", prof_new)):
        n_checks = len(prof["columns"]) * 2 + 1  # completeness + validity per column + dup check
        n_failed = sum(1 for f in fails if f["batch"] == batch)
        report["checks_passed"][batch] = f"{n_checks - n_failed}/{n_checks}"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"quality report written: {out_path}")

    # plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_missingness(prof_new, PLOTS_DIR / "missingness.png")
    plot_psi(drift, PLOTS_DIR / "psi.png")
    base = pd.read_csv(baseline_path, parse_dates=["timestamp"])
    new = pd.read_csv(new_path, parse_dates=["timestamp"])
    shifted = [c for c in drift["columns"]
               if c["type"] == "numeric" and c["psi_band"] == "significant"]
    for c in shifted:
        plot_shift(base, new, c["column"], PLOTS_DIR / f"shift_{c['column']}.png")
    print(f"plots written to {PLOTS_DIR}/")

    print(f"\nQuality scores: baseline={scores['baseline']}  new={scores['new']}")
    print(f"Failed checks: {len(fails)}")
    sig = [f for f in findings if f['psi_band'] == 'significant']
    print(f"Significant drift (PSI>0.25): {[f['column'] for f in sig]}")
    return report


if __name__ == "__main__":
    build_report()
