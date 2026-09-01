"""Polar curves: any result variable vs any other, one trace per (Project, AWS, RPM)
case, with optional legend grouping, std error bars, and stall-point highlighting."""

import streamlit as st

from app.sail.plotting import PLOTTABLE_COLUMNS, axis_label, plot_polar_curve
from app.sail.widgets import render_plot_controls, select_projects_and_load

st.title("Polar Curves")

polar_df = select_projects_and_load(key="polar_curves_projects")
plot_mode, legend_group_by = render_plot_controls(key="polar_curves")

col_y, col_x = st.columns(2)
y_variable = col_y.selectbox("Y variable", PLOTTABLE_COLUMNS, index=PLOTTABLE_COLUMNS.index("cl"), format_func=axis_label)
x_variable = col_x.selectbox("X variable", PLOTTABLE_COLUMNS, index=PLOTTABLE_COLUMNS.index("aoa"), format_func=axis_label)

std_available = f"{y_variable}_std" in polar_df.columns
col_err, col_stall = st.columns(2)
show_error_bars = col_err.checkbox(
    "Show error bars (std)", value=False, disabled=not std_available,
    help=None if std_available else "No std column for this variable (only CL/CD have one).",
)
highlight_stall = col_stall.checkbox("Highlight stall points", value=True)

st.plotly_chart(
    plot_polar_curve(
        polar_df, y=y_variable, x=x_variable,
        legend_group_by=legend_group_by, plot_mode=plot_mode,
        show_error_bars=show_error_bars, highlight_stall=highlight_stall,
        sort_by="aoa",
    ),
    width="stretch",
)
