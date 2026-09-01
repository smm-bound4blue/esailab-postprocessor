"""
Sail -- Home page.

Lets the user pick a project folder under data/sail/, run the ETL
pipeline (src.sail.pipeline.run_etl) to (re)populate the results
database, and shows a summary of what's already synced for this type.
"""

import logging

import streamlit as st

from app.sail.data import SIMULATION_TYPE, get_project_summary, run_etl_for_project
from src.config import get_project_path_for_type

logger = logging.getLogger(__name__)

st.title("Sail")
st.caption("Sail rotated through an AoA sweep at fixed TWS/RPM, no harbour geometry.")

try:
    data_root = get_project_path_for_type(SIMULATION_TYPE)
except ValueError as e:
    st.error(f"{e}. Set CFD_PROJECT_PATH in .env and restart.")
    st.stop()

st.subheader("Run ETL")

if not data_root.exists():
    st.warning(f"Data directory not found: {data_root}")
    candidate_names = []
else:
    candidate_names = sorted(d.name for d in data_root.iterdir() if d.is_dir())

if not candidate_names:
    st.info(f"No project folders found under {data_root}")
else:
    selected_name = st.selectbox("Project folder", candidate_names)
    if st.button("Run ETL", type="primary"):
        with st.spinner(f"Syncing '{selected_name}'..."):
            try:
                result = run_etl_for_project(data_root / selected_name)
                st.success(
                    f"Synced '{selected_name}': {result.cases_written} cases, "
                    f"{result.aoa_rows_written} AoA rows written, "
                    f"{result.aoa_rows_skipped} skipped (unchanged)"
                )
            except Exception as e:
                logger.exception("ETL run failed")
                st.error(f"ETL failed: {e}")

st.subheader("Sail projects in database")
summary_df = get_project_summary()
if summary_df.empty:
    st.info("No projects synced yet — run the ETL above.")
else:
    display_df = summary_df.assign(
        rpm_range=lambda d: d["rpm_min"].map("{:g}".format) + " – " + d["rpm_max"].map("{:g}".format),
        aws_range=lambda d: d["aws_min"].map("{:g}".format) + " – " + d["aws_max"].map("{:g}".format),
    )[["name", "case_count", "aoa_count", "rpm_range", "aws_range", "span", "chord", "last_synced"]]
    display_df.columns = [
        "Project", "Cases", "AoA rows", "RPM range", "AWS range (kts)", "Span (m)", "Chord (m)", "Last synced",
    ]
    st.dataframe(display_df, width="stretch", hide_index=True)
