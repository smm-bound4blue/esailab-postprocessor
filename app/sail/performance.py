"""
Performance: for each (Project, AWS, RPM) case, take the AoA point that
gives maximum CL, and plot how its performance metrics trend across RPM
-- one trace per (Project, AWS), connecting the CLmax point at each RPM.

The Interpolated Data tab resamples this same CLmax dataset onto a
uniform AoA grid per trace (monotonic PCHIP, ported from
esail-postprocessor's src/utils/interpolation.py) -- the real CLmax
points are sparse and non-uniformly spaced across RPM, so this smooths
the envelope curve without ever extrapolating past the real data.

For fan-specific analysis (reference curve vs simulated Static/Total
Pressure), see the dedicated Fan Performance page.
"""

import streamlit as st

from app.sail.analysis import compute_clmax_data
from app.sail.plotting import PLOTTABLE_COLUMNS, axis_label, plot_polar_curve
from app.sail.widgets import render_plot_controls, select_projects_and_load
from src.sail.interpolation import interpolate_performance_envelope

st.title("Performance")

polar_df = select_projects_and_load(key="performance_projects")
plot_mode, legend_group_by = render_plot_controls(key="performance")

full_polar_min_rpm = st.checkbox(
    "Full polar at min RPM", value=False,
    help="For each trace, replace its CLmax point at the trace's minimum RPM with every "
    "AoA point up to CLmax at that RPM -- traces the full polar curve at the lowest RPM "
    "instead of jumping straight to its CLmax point.",
)
clmax_df = compute_clmax_data(polar_df, full_polar_min_rpm=full_polar_min_rpm)
trace_by = ["project_name", "aws"]  # connect CLmax points across RPM into one envelope line

tab_envelope, tab_interpolated = st.tabs(["Performance Envelope", "Interpolated Data"])

with tab_envelope:
    st.caption("Each point is the AoA@CLmax operating point for one (Project, AWS, RPM) case.")

    col_y, col_x = st.columns(2)
    y_variable = col_y.selectbox(
        "Y variable", PLOTTABLE_COLUMNS, index=PLOTTABLE_COLUMNS.index("cl"), format_func=axis_label,
        key="envelope_y",
    )
    x_variable = col_x.selectbox(
        "X variable", PLOTTABLE_COLUMNS, index=PLOTTABLE_COLUMNS.index("cpow"), format_func=axis_label,
        key="envelope_x",
    )

    if clmax_df[y_variable].isna().all() or clmax_df[x_variable].isna().all():
        st.warning(
            f"{axis_label(y_variable)} or {axis_label(x_variable)} is not available for the "
            "selected project(s) — check that chord is set in config/config.yaml and re-sync."
        )
    else:
        st.plotly_chart(
            plot_polar_curve(
                clmax_df, y=y_variable, x=x_variable, trace_by=trace_by,
                legend_group_by=legend_group_by, plot_mode=plot_mode, sort_by="aoa",
            ),
            width="stretch",
        )

with tab_interpolated:
    st.caption(
        "The CLmax dataset above, resampled onto a uniform AoA grid per (Project, AWS) trace via "
        "monotonic PCHIP interpolation."
    )

    col_step, col_y2, col_x2 = st.columns(3)
    step = col_step.number_input("AoA step (deg)", min_value=0.1, value=1.0, step=0.5, key="interp_step")
    y_variable_interp = col_y2.selectbox(
        "Y variable", PLOTTABLE_COLUMNS, index=PLOTTABLE_COLUMNS.index("cl"), format_func=axis_label,
        key="interp_y",
    )
    x_variable_interp = col_x2.selectbox(
        "X variable", PLOTTABLE_COLUMNS, index=PLOTTABLE_COLUMNS.index("cpow"), format_func=axis_label,
        key="interp_x",
    )

    interpolated_df, interp_warnings = interpolate_performance_envelope(
        clmax_df, group_cols=trace_by, x_col="aoa", step=step,
    )

    if interpolated_df.empty:
        st.warning("No trace could be interpolated — each (Project, AWS) trace needs at least 2 CLmax points.")
    else:
        st.plotly_chart(
            plot_polar_curve(
                interpolated_df, y=y_variable_interp, x=x_variable_interp, trace_by=trace_by,
                legend_group_by=legend_group_by, plot_mode=plot_mode, sort_by="aoa",
            ),
            width="stretch",
        )

        if interp_warnings:
            with st.expander(f"{len(interp_warnings)} interpolation note(s)"):
                for msg in interp_warnings:
                    st.caption(f"⚠️ {msg}")

        st.download_button(
            "Download interpolated CSV",
            data=interpolated_df.to_csv(index=False),
            file_name="interpolated_performance.csv",
            mime="text/csv",
        )
