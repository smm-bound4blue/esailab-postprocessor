"""
Sail -- Home page.

Lets the user pick a project folder under data/sail/, run the ETL
pipeline (src.sail.pipeline.run_etl) to (re)populate the results
database, and shows a summary of what's already synced for this type.
"""

import logging

import streamlit as st

from app.components.data import list_projects
from app.sail.data import SIMULATION_TYPE, run_etl_for_project
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
projects_df = list_projects(simulation_type=SIMULATION_TYPE)
if projects_df.empty:
    st.info("No projects synced yet — run the ETL above.")
else:
    st.dataframe(projects_df, width="stretch")
