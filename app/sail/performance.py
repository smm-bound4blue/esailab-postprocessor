"""
Performance: for each (Project, AWS, RPM) case, take the AoA point that
gives maximum CL, and plot how its performance metrics trend across RPM
-- one trace per (Project, AWS), connecting the CLmax point at each RPM.

The Interpolated Data tab resamples this same CLmax dataset onto a
uniform AoA grid per trace (monotonic PCHIP, ported from
esail-postprocessor's src/utils/interpolation.py) -- the real CLmax
points are sparse and non-uniformly spaced across RPM, so this smooths
the envelope curve without ever extrapolating past the real data.

The Combined AWS Envelope tab pools every AWS/RPM trace within a project,
averages every variable within fixed-width AoA bins (absorbing AWS-to-AWS
scatter and overlapping/duplicate AoA values), then PCHIP-interpolates
through those bin averages -- see src.sail.interpolation for why this
replaced two earlier attempts (a scipy smoothing spline, then a
hand-rolled LOESS) that both turned out numerically fragile or prone to
overshoot on this project's real pooled data.

For fan-specific analysis (reference curve vs simulated Static/Total
Pressure), see the dedicated Fan Performance page.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.sail.analysis import compute_clmax_data, enforce_cl_monotonic
from app.sail.plotting import PLOTTABLE_COLUMNS, axis_label, plot_polar_curve
from app.sail.widgets import render_plot_controls, select_projects_and_load
from src.sail.interpolation import interpolate_performance_envelope, smooth_performance_envelope

st.title("Performance")

polar_df = select_projects_and_load(key="performance_projects")
plot_mode, legend_group_by = render_plot_controls(key="performance")

col_full_polar, col_monotonic = st.columns(2)
full_polar_enabled = col_full_polar.checkbox(
    "Full polar at min RPM", value=False,
    help="For each selected trace, replace its CLmax point at the trace's minimum RPM with "
    "every AoA point up to CLmax at that RPM -- traces the full polar curve at the lowest "
    "RPM instead of jumping straight to its CLmax point.",
)
cl_monotonic = col_monotonic.checkbox(
    "Only keep CL non-decreasing with AoA", value=True,
    help="Drops any (Project, AWS) trace point whose CL doesn't match or beat the running "
    "maximum CL at a lower AoA -- a higher-RPM operating point can have a slightly lower "
    "CLmax than a lower-RPM one before it, which otherwise reads as CL dropping as AoA "
    "increases. Real values are kept as-is; the offending row is dropped, not smoothed.",
)

full_polar_traces = []
if full_polar_enabled:
    available_traces = sorted(polar_df[["project_name", "aws"]].drop_duplicates().itertuples(index=False, name=None))
    trace_labels = {t: f"{t[0]} · AWS={t[1]:g}" for t in available_traces}
    full_polar_traces = st.multiselect(
        "Apply to", options=available_traces, default=available_traces, format_func=lambda t: trace_labels[t],
        key="full_polar_traces",
        help="Traces left unchecked here keep their plain one-point-per-RPM CLmax envelope.",
    )

clmax_df = compute_clmax_data(polar_df, full_polar_traces=full_polar_traces)
if cl_monotonic:
    clmax_df = enforce_cl_monotonic(clmax_df)
trace_by = ["project_name", "aws"]  # connect CLmax points across RPM into one envelope line

def _y_x_selectors(key_prefix: str):
    col_y, col_x = st.columns(2)
    y_variable = col_y.selectbox(
        "Y variable", PLOTTABLE_COLUMNS, index=PLOTTABLE_COLUMNS.index("cl"), format_func=axis_label,
        key=f"{key_prefix}_y",
    )
    x_variable = col_x.selectbox(
        "X variable", PLOTTABLE_COLUMNS, index=PLOTTABLE_COLUMNS.index("cpow"), format_func=axis_label,
        key=f"{key_prefix}_x",
    )
    return y_variable, x_variable


def _render_fit_result(fitted_df, group_cols: list, y_variable: str, x_variable: str, notes: list,
                        file_name: str, key_prefix: str, raw_df: pd.DataFrame = None) -> None:
    """Shared by the Interpolated Data and Combined AWS Envelope tabs -- plot a
    resampled/fitted DataFrame, surface any warnings, offer a CSV download.

    raw_df, if given, is overlaid as a single light-grey markers-only trace of
    the pre-fit points on the same x/y variables -- so the fit can be visually
    checked against what it was actually fit to."""
    if fitted_df.empty:
        st.warning("No trace produced enough points to plot — each trace needs at least 2 CLmax points.")
        return

    # legend_group_by is a page-level control that can include columns (e.g. "aws")
    # not in this tab's group_cols (the Combined tab groups by project only, having
    # pooled every AWS trace together) -- plot_polar_curve requires legend_group_by
    # to be a subset of trace_by, so drop anything that isn't.
    safe_legend_group_by = [c for c in legend_group_by if c in group_cols]

    fig = plot_polar_curve(
        fitted_df, y=y_variable, x=x_variable, trace_by=group_cols,
        legend_group_by=safe_legend_group_by, plot_mode=plot_mode, sort_by="aoa",
    )

    if raw_df is not None and not raw_df.empty and {x_variable, y_variable} <= set(raw_df.columns):
        raw_points = raw_df[[x_variable, y_variable]].dropna()
        if not raw_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=raw_points[x_variable], y=raw_points[y_variable], mode="markers",
                    name="Original points",
                    marker=dict(symbol="diamond-open", size=9, line=dict(width=2)),
                )
            )

    st.plotly_chart(fig, width="stretch")

    if notes:
        with st.expander(f"{len(notes)} note(s)"):
            for msg in notes:
                st.caption(f"⚠️ {msg}")

    st.download_button(
        "Download CSV", data=fitted_df.to_csv(index=False),
        file_name=file_name, mime="text/csv", key=f"{key_prefix}_download",
    )

    table_cols = [c for c in ["aoa", "cl", "cd", "cpow", "cq"] if c in fitted_df.columns]
    if table_cols:
        with st.expander("Show data table"):
            st.dataframe(fitted_df[table_cols], width="stretch")


tab_envelope, tab_interpolated, tab_combined = st.tabs(
    ["Performance Envelope", "Interpolated Data", "Combined AWS Envelope"]
)

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

    step = st.number_input("AoA step (deg)", min_value=0.1, value=1.0, step=0.5, key="interp_step")
    y_variable_interp, x_variable_interp = _y_x_selectors("interp")

    interpolated_df, interp_warnings = interpolate_performance_envelope(
        clmax_df, group_cols=trace_by, x_col="aoa", step=step,
    )
    _render_fit_result(
        interpolated_df, trace_by, y_variable_interp, x_variable_interp, interp_warnings,
        "interpolated_performance.csv", "interp",
    )

with tab_combined:
    st.caption(
        "Every AWS/RPM trace within a project pooled together, averaged within fixed-width AoA "
        "bins (absorbing AWS-to-AWS scatter and overlapping/duplicate AoA values), then "
        "PCHIP-interpolated through those bin averages onto a uniform AoA grid."
    )

    with st.expander("How does this work?"):
        st.markdown(
            """
1. **Pool every AWS/RPM trace together.** The "Performance Envelope" tab keeps one line per
   (Project, AWS); this tab drops the AWS split and treats every point from every AWS/RPM
   condition in the project as one combined dataset.
2. **Bin by AoA.** Points are grouped into fixed-width AoA bins (the *Bin width* below) — e.g.
   at 10°, all points with AoA in [20°, 30°) fall into the same bin, regardless of which AWS/RPM
   they came from.
3. **Average within each bin.** Every variable (CL, CD, Cpow, CQ, ...) is averaged across all
   points in a bin, collapsing AWS-to-AWS scatter and duplicate/overlapping AoA values into one
   representative point per bin — this is what lets traces with overlapping AoA ranges combine
   into a single curve instead of a zigzag.
4. **PCHIP-interpolate through the bin averages**, evaluated at the *AoA step* below. PCHIP is
   shape-preserving between its control points (unlike a plain line fit), so it can't invent a
   new dip or bump that the binned data doesn't already show. The very first and last bin are
   pinned back to the data's true min/max AoA, so the curve's endpoints always match the raw
   data regardless of bin width.
5. **Bin width is a real trade-off, not just cosmetic smoothing.** A narrow bin (e.g. 0.5°) barely
   averages anything, so the curve stays close to individual points. A wide bin absorbs more
   scatter but also blends together AWS/RPM conditions that don't actually share one physical
   relationship — e.g. Cpow depends mainly on RPM, not AoA alone, so pooling multiple RPM
   conditions can make a wide bin average across genuinely different operating regimes. That's a
   property of the underlying data, not something any curve-fitting method can fully undo — if a
   variable looks locally inconsistent even after binning, try a narrower bin width, or compare
   against the "Original points" markers and the per-AWS **Interpolated Data** tab.
            """
        )

    col_step_c, col_bin = st.columns(2)
    step_c = col_step_c.number_input("AoA step (deg)", min_value=0.1, value=1.0, step=0.5, key="combined_step")
    bin_width = col_bin.number_input(
        "Bin width (deg)", min_value=0.1, value=10.0, step=0.5, key="combined_bin_width",
        help="Width of each AoA bin that raw points are averaged within before fitting -- larger = "
        "fewer, coarser bins (more scatter absorbed, less faithful to individual points). Wider "
        "bins also pull the fitted curve's start/end further inward from the raw data's true "
        "min/max AoA, since the edge bins average in neighboring points too.",
    )
    y_variable_combined, x_variable_combined = _y_x_selectors("combined")

    combined_group_cols = ["project_name"]
    fitted_df, fit_warnings = smooth_performance_envelope(
        clmax_df, group_cols=combined_group_cols, x_col="aoa", step=step_c, bin_width=bin_width,
    )
    _render_fit_result(
        fitted_df, combined_group_cols, y_variable_combined, x_variable_combined, fit_warnings,
        "combined_smoothed_performance.csv", "combined", raw_df=clmax_df,
    )
