"""Streamlit dashboard for the data-quality-monitor quality report.

Run with:  streamlit run app.py   (from the project root)
"""

import json
from pathlib import Path

REPORT_PATH = Path(__file__).parent / "report" / "quality_report.json"
PLOTS_DIR = Path(__file__).parent / "report" / "plots"


def load_report(path: str | Path = REPORT_PATH) -> dict:
    """Load the JSON quality report produced by src/report.py."""
    return json.loads(Path(path).read_text())


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Data Quality Monitor", layout="wide")
    st.title("Data Quality Monitor — AriaHome telemetry (synthetic demo)")

    try:
        report = load_report()
    except FileNotFoundError:
        st.error("report/quality_report.json not found. Run `python run.py` first.")
        return

    scores = report["scores"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline quality score", f"{scores['baseline']}/100")
    c2.metric("New batch quality score", f"{scores['new']}/100",
              delta=f"{round(scores['new'] - scores['baseline'], 2)} vs baseline")
    c3.metric("Checks passed (new batch)", report["checks_passed"]["new"])

    st.subheader("Failed checks")
    fails = report["failed_checks"]
    if fails:
        st.dataframe(fails, use_container_width=True)
    else:
        st.success("No failed checks.")

    st.subheader("Drift findings (baseline → new)")
    st.dataframe(report["drift_findings"], use_container_width=True)

    st.subheader("Schema diff")
    sd = report["schema_diff"]
    st.json(sd)

    st.subheader("Plots")
    for img in sorted(PLOTS_DIR.glob("*.png")):
        st.image(str(img), caption=img.stem.replace("_", " "), use_container_width=True)

    with st.expander("Dataset summaries"):
        for batch, prof in report["profiles"].items():
            st.write(f"**{batch}**: {prof['rows']} rows, "
                     f"{prof['summary']['n_columns']} columns, "
                     f"{prof['summary']['duplicate_pct']}% duplicates")


if __name__ == "__main__":
    main()
