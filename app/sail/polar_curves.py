"""Polar curves: CL/CD/E vs AoA, grouped by RPM. Reads app.sail.data.get_polar_data()."""

import streamlit as st

from app.components.data import list_projects
from app.sail.data import SIMULATION_TYPE, get_polar_data
from app.sail.plotting import plot_polar_curve

st.title("Polar Curves")

projects_df = list_projects(simulation_type=SIMULATION_TYPE)
if projects_df.empty:
    st.info("No sail projects synced yet — run the ETL on the Home page.")
    st.stop()

project_row = st.selectbox("Project", projects_df.itertuples(), format_func=lambda r: r.name)
polar_df = get_polar_data(project_row.id)

if polar_df.empty:
    st.warning("No polar data found for this project.")
    st.stop()

y_variable = st.selectbox("Variable", ["cl", "cd", "e"], format_func=str.upper)
st.plotly_chart(plot_polar_curve(polar_df, y=y_variable), width="stretch")
