"""
Performance: for each (Project, AWS, RPM) case, take the AoA point that
gives maximum CL, and plot how its CL/Cpow trend across RPM -- one trace
per (Project, AWS), connecting the CLmax point at each RPM.

For fan-specific analysis (reference curve vs simulated Static/Total
Pressure), see the dedicated Fan Performance page.
"""

import streamlit as st

from app.sail.analysis import compute_clmax_data
from app.sail.plotting import plot_polar_curve
from app.sail.widgets import render_plot_controls, select_projects_and_load

st.title("Performance")
st.caption("Each point is the AoA@CLmax operating point for one (Project, AWS, RPM) case.")

polar_df = select_projects_and_load(key="performance_projects")
plot_mode, legend_group_by = render_plot_controls(key="performance")

clmax_df = compute_clmax_data(polar_df)
trace_by = ["project_name", "aws"]  # connect CLmax points across RPM into one envelope line

if clmax_df["cpow"].isna().all():
    st.warning("Cpow is not available — check that chord is set in config/config.yaml and re-sync.")
else:
    st.plotly_chart(
        plot_polar_curve(
            clmax_df, y="cl", x="cpow", trace_by=trace_by,
            legend_group_by=legend_group_by, plot_mode=plot_mode,
        ),
        width="stretch",
    )
