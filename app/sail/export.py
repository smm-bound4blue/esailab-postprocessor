"""Export the transformed polar/performance data (from the database) to CSV."""

import streamlit as st

from app.components.data import list_projects
from app.sail.data import SIMULATION_TYPE, get_polar_data

st.title("Export")

projects_df = list_projects(simulation_type=SIMULATION_TYPE)
if projects_df.empty:
    st.info("No sail projects synced yet — run the ETL on the Home page.")
    st.stop()

project_row = st.selectbox("Project", projects_df.itertuples(), format_func=lambda r: r.name)
polar_df = get_polar_data(project_row.id)

if polar_df.empty:
    st.warning("No polar data found for this project.")
    st.stop()

st.dataframe(polar_df, width="stretch")
st.download_button(
    "Download CSV",
    data=polar_df.to_csv(index=False),
    file_name=f"{project_row.name}_polar_data.csv",
    mime="text/csv",
)
