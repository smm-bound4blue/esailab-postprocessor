"""Performance envelope: CL vs Cpow across RPM setpoints. Reads app.sail.data.get_polar_data()."""

import streamlit as st

from app.components.data import list_projects
from app.sail.data import SIMULATION_TYPE, get_polar_data
from app.sail.plotting import plot_performance_envelope

st.title("Performance")

projects_df = list_projects(simulation_type=SIMULATION_TYPE)
if projects_df.empty:
    st.info("No sail projects synced yet — run the ETL on the Home page.")
    st.stop()

project_row = st.selectbox("Project", projects_df.itertuples(), format_func=lambda r: r.name)
polar_df = get_polar_data(project_row.id)

if polar_df.empty:
    st.warning("No polar data found for this project.")
    st.stop()

if polar_df["cpow"].isna().all():
    st.warning("Cpow is not available — check that chord is set in config/config.yaml and re-sync.")
    st.stop()

st.plotly_chart(plot_performance_envelope(polar_df), width="stretch")
