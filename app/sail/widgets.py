"""
Shared UI pieces used identically by Polar Curves, Performance, and
Export: the project multiselect + data load, and the plot-mode/legend-
grouping controls.
"""

from typing import List, Tuple

import pandas as pd
import streamlit as st

from app.components.data import list_projects
from app.sail.data import SIMULATION_TYPE, get_polar_data
from app.sail.plotting import PLOT_MODES

_LEGEND_GROUPINGS = {
    "None": [],
    "Project": ["project_name"],
    "AWS": ["aws"],
    "Project + AWS": ["project_name", "aws"],
}


def select_projects_and_load(key: str) -> pd.DataFrame:
    """
    Renders a project multiselect plus AWS/RPM case filters (stops the
    script if nothing is selectable/selected/loaded) and returns the
    combined, filtered polar_df for the selected project(s). Trace/legend
    grouping is a separate, page-level concern -- see render_plot_controls().
    """
    projects_df = list_projects(simulation_type=SIMULATION_TYPE)
    if projects_df.empty:
        st.info("No sail projects synced yet — run the ETL on the Home page.")
        st.stop()

    rows = list(projects_df.itertuples())
    selected = st.multiselect("Projects", rows, default=rows[:1], format_func=lambda r: r.name, key=key)
    if not selected:
        st.info("Select at least one project.")
        st.stop()

    project_ids = tuple(r.id for r in selected)
    polar_df = get_polar_data(project_ids)
    if polar_df.empty:
        st.warning("No polar data found for the selected project(s).")
        st.stop()

    aws_options = sorted(polar_df["aws"].unique())
    rpm_options = sorted(polar_df["rpm"].unique())
    col_aws, col_rpm = st.columns(2)
    selected_aws = col_aws.multiselect(
        "AWS (kts)", aws_options, default=aws_options, format_func=lambda v: f"{v:g}", key=f"{key}_aws"
    )
    selected_rpm = col_rpm.multiselect(
        "RPM", rpm_options, default=rpm_options, format_func=lambda v: f"{v:g}", key=f"{key}_rpm"
    )
    if not selected_aws or not selected_rpm:
        st.info("Select at least one AWS and one RPM.")
        st.stop()

    polar_df = polar_df[polar_df["aws"].isin(selected_aws) & polar_df["rpm"].isin(selected_rpm)]
    if polar_df.empty:
        st.warning("No cases match the selected AWS/RPM filters.")
        st.stop()

    return polar_df


def render_plot_controls(key: str) -> Tuple[str, List[str]]:
    """Display-mode + legend-grouping selectors, shared by every chart on
    Polar Curves and Performance. Returns (plot_mode, legend_group_by)."""
    col_mode, col_group = st.columns(2)
    plot_mode = col_mode.selectbox("Display mode", list(PLOT_MODES.keys()), key=f"{key}_mode")
    grouping_label = col_group.selectbox(
        "Legend grouping", list(_LEGEND_GROUPINGS.keys()), key=f"{key}_grouping",
        help="Click a group's title in the legend to show/hide every trace in it at once.",
    )
    return plot_mode, _LEGEND_GROUPINGS[grouping_label]
