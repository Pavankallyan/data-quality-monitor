# Data Quality Monitor

An automated data-quality watchdog for tabular feeds: profile a batch, compare it against a baseline for distribution drift and schema changes, and roll everything into a single scored quality report with charts. Ships with a Streamlit dashboard and a one-command end-to-end pipeline.

**Important:** all data in this project is synthetic. I generated it myself with a fixed-seed script — no real devices, companies, or customer data are involved. "AriaHome" is a fictional smart-home brand I invented for the demo.

## Overview

I built this to practice the thing I used to do as a test engineer, but for data: turn "the feed looks off" into concrete, repeatable checks. Given two batches of IoT sensor telemetry (a clean baseline and a new batch), the tool:

1. **Profiles** each batch — per-column completeness, validity (range/regex/allowed-value rules), uniqueness, cardinality.
2. **Detects drift** — PSI, Kolmogorov–Smirnov (numeric), chi-square (categorical), plus a schema diff (added / removed / renamed columns, dtype changes).
3. **Reports** — a weighted quality score (0–100), a failed-checks table, a drift-findings table, and PNG charts.
4. **Dashboards** — a Streamlit app that renders the report.

## Approach

- `src/generate_data.py` — synthetic telemetry generator (fixed seeds: 42, 7). The new batch has six injected defects whose ground truth I know, so I can verify the monitor catches them: a missing-value spike in `temperature_c`/`humidity_pct`, out-of-range `humidity_pct` (>100) and `battery_pct` (<0), ~2% duplicate rows, a renamed column (`status` → `device_status`), an unexpected new column (`signal_quality`), and a +6 °C mean shift in `temperature_c`.
- `src/profile.py` — column-level metrics + dataset summary → JSON per batch.
- `src/drift.py` — PSI over baseline quantile bins (numeric) / category shares (categorical); KS and chi-square tests; rename detection via distinct-value overlap.
- `src/report.py` — weighted score, thresholded checks, and matplotlib charts.
- `app.py` — Streamlit dashboard (imports streamlit lazily so the module is import-safe).
- `run.py` — one command: generate → profile both → drift → report + plots.

## Results on demo data

From the actual run (fixed seeds, so reproducible):

| Metric | Baseline batch | New batch |
|---|---|---|
| Rows × columns | 5,000 × 8 | 5,100 × 9 (100 duplicate rows added) |
| Quality score | **99.99**/100 | **98.7**/100 |
| Checks passed | 17/17 | 14/19 — **5 failed** |

Failed checks in the new batch:
- `temperature_c` completeness 0.919 (8% missing spike) — caught
- `humidity_pct` completeness 0.92 (8% missing spike) — caught
- `humidity_pct` validity 0.974 (values up to 135%, i.e. > 100) — caught
- `battery_pct` validity 0.9818 (values down to −12%) — caught
- 100 duplicate rows (1.96% of batch) — caught

Drift findings (baseline → new):
- `temperature_c` PSI **2.6944** — significant (the injected +6 °C shift)
- `humidity_pct`, `battery_pct` PSI < 0.1 (no change) but KS p < 0.05 — drifted at the tails
- `firmware_version` chi-square p ≈ 0 — drifted (category mix changed)
- `device_id`, `signal_dbm` — no drift

Schema diff:
- added: `signal_quality`
- renamed: `status` → `device_status` (value overlap 1.0)
- removed: none; dtype changes: none

All six injected defects were detected. Nothing in the clean baseline was flagged.

## Project structure

```
data-quality-monitor/
├── app.py                  # Streamlit dashboard
├── run.py                  # end-to-end pipeline (one command)
├── requirements.txt
├── README.md
├── src/
│   ├── generate_data.py    # synthetic batch generator (fixed seeds)
│   ├── profile.py          # column + dataset profiling -> JSON
│   ├── drift.py            # PSI, KS, chi-square, schema diff -> JSON
│   └── report.py           # quality score, failed checks, charts
├── data/                   # baseline.csv, new.csv (generated)
└── report/
    ├── quality_report.json
    └── plots/              # missingness.png, psi.png, shift_temperature_c.png
```

## How to run

```bash
cd data-quality-monitor
pip install -r requirements.txt

# full pipeline: generate -> profile -> drift -> report + plots
python run.py

# individual steps (from project root)
python src/generate_data.py
python src/profile.py
python src/drift.py
python src/report.py

# dashboard
streamlit run app.py
```

Python 3.10+ (developed on 3.12). The dashboard reads `report/quality_report.json`, so run `python run.py` first.

Note: if your system Python has pinned old numpy/pandas (as some Linux distros do), install into a venv first — `streamlit` wants its own dependency tree:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Design decisions

- **PSI thresholds** (< 0.1 / 0.1–0.25 / > 0.25): the credit-risk industry standard for population stability. They're interpretable by non-statisticians and symmetric, unlike p-values, which are sample-size sensitive (note the KS test flagged `humidity_pct` at p = 0.03 while PSI said "no change" — both are reported so the reader sees the disagreement).
- **Scoring weights** (0.4 completeness + 0.4 validity + 0.2 duplicate-rate): missing and invalid telemetry is what actually breaks downstream dashboards and models, so it dominates; duplicates are a pipeline hygiene issue and get a smaller penalty. Weights are explicit constants at the top of `report.py` — easy to retune.
- **Fixed seeds + injected ground truth**: because I know exactly which defects are in the new batch, I can prove the monitor catches them rather than hand-waving.
- **Lazy streamlit import**: `app.py` imports streamlit only inside `main()`, so `load_report()` and the module itself are testable without streamlit installed.
- **Limitations**: PSI on 10 quantile bins can miss tail-only drift (that's why KS is included); the rename heuristic is value-overlap based and could false-positive on low-cardinality columns; validity rules are hard-coded for this schema rather than config-driven; the pipeline runs locally on CSVs, not on a schedule or a live feed.
