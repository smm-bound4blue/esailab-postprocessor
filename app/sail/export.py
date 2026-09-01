"""Export the transformed polar/performance data (from the database) to CSV -- one or more projects."""

import streamlit as st

from app.sail.widgets import select_projects_and_load

st.title("Export")

polar_df = select_projects_and_load(key="export_projects")

st.dataframe(polar_df, width="stretch")

project_label = "_".join(sorted(polar_df["project_name"].unique())) if len(polar_df["project_name"].unique()) > 1 else polar_df["project_name"].iloc[0]
st.download_button(
    "Download CSV",
    data=polar_df.to_csv(index=False),
    file_name=f"{project_label}_polar_data.csv",
    mime="text/csv",
)
