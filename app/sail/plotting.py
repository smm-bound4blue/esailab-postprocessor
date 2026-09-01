"""
Reusable Plotly chart builder for the Sail pages -- polar curves, the
performance envelope, and fan efficiency/CQ plots are all the same shape
(some Y vs some X, split into per-case traces, optionally clustered into
togglable legend groups), so one function covers all of them.

Expects the DataFrame shape returned by app.sail.data.get_polar_data()
(lowercase DB column names: aoa, cl, cd, e, rpm, cpow, cq, ..., plus
project_name and is_stall).
"""

from numbers import Number
from typing import Dict, List, Optional

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

_AXIS_LABELS = {
    "aoa": "AoA (deg)",
    "cl": "CL",
    "cd": "CD",
    "e": "E (CL/CD)",
    "cq": "CQ",
    "cpow": "Cpow",
    "fan_volumetric_flow": "Fan Volumetric Flow (m³/s)",
    "fan_total_pressure": "Fan Total Pressure (Pa)",
    "fan_static_pressure": "Fan Static Pressure (Pa)",
    "fan_static_efficiency": "Fan Static Efficiency",
    "fan_total_efficiency": "Fan Total Efficiency",
    "fan_power": "Fan Power (kW)",
    "rpm": "RPM",
    "aws": "AWS (kts)",
    "project_name": "Project",
}

# Curated set of columns worth plotting on an axis -- excludes bookkeeping
# columns (id, source_path, his_csv_mtime, pipeline_version, processed_at, ...).
PLOTTABLE_COLUMNS = [
    "aoa", "cl", "cd", "e",
    "fan_volumetric_flow", "fan_total_pressure", "fan_static_pressure",
    "fan_static_efficiency", "fan_total_efficiency", "fan_power",
    "cq", "cpow", "rpm", "aws",
]

# Every (project_name, aws, rpm) combination is one physical case -- one AoA
# sweep. Traces always split at this granularity so lines never silently mix
# data from two different operating points (see CLAUDE.md/conversation: this
# matters once AWS varies within a project, not just RPM).
CASE_COLS = ["project_name", "aws", "rpm"]

PLOT_MODES = {
    "Lines + Markers": ("lines+markers", 6),
    "Only Lines": ("lines", 0),
    "Only Markers": ("markers", 8),
}

_DASH_CYCLE = ["solid", "dash", "dot", "dashdot"]


def axis_label(col: str) -> str:
    return _AXIS_LABELS.get(col, col)


def _format_group_value(value) -> str:
    return f"{value:g}" if isinstance(value, Number) else str(value)


def _trace_name(row: dict, cols: List[str]) -> str:
    return " · ".join(f"{axis_label(c)}={_format_group_value(row[c])}" for c in cols)


def plot_polar_curve(
    df: pd.DataFrame,
    y: str,
    x: str = "aoa",
    trace_by: Optional[List[str]] = None,
    legend_group_by: Optional[List[str]] = None,
    plot_mode: str = "Lines + Markers",
    show_error_bars: bool = False,
    highlight_stall: bool = False,
) -> go.Figure:
    """
    y vs x, one trace per distinct trace_by combination (default: one per
    physical case -- CASE_COLS). legend_group_by (a subset of trace_by, e.g.
    ["project_name"] or ["project_name", "aws"]) clusters traces into
    togglable legend groups sharing one color; omit it to color every trace
    individually. show_error_bars adds a {y}_std error bar when that column
    exists (only cl/cd have one). highlight_stall overlays every is_stall
    row, across all traces, as a single red "Stall" marker trace.
    """
    trace_cols = trace_by or CASE_COLS
    legend_group_by = legend_group_by or []
    std_col = f"{y}_std"
    mode, marker_size = PLOT_MODES.get(plot_mode, PLOT_MODES["Lines + Markers"])

    color_cols = legend_group_by or trace_cols
    color_keys = sorted(df[color_cols].drop_duplicates().itertuples(index=False, name=None))
    palette = pc.qualitative.Plotly
    color_map = {key: palette[i % len(palette)] for i, key in enumerate(color_keys)}

    dash_index_by_group: dict = {}

    fig = go.Figure()
    for trace_values, subset in df.groupby(trace_cols, sort=True):
        if not isinstance(trace_values, tuple):
            trace_values = (trace_values,)
        row = dict(zip(trace_cols, trace_values))
        subset = subset.sort_values(x)

        color_key = tuple(row[c] for c in color_cols)
        legend_group = _trace_name(row, legend_group_by) if legend_group_by else None

        dash = "solid"
        if legend_group_by:
            dash_idx = dash_index_by_group.get(legend_group, 0)
            dash_index_by_group[legend_group] = dash_idx + 1
            dash = _DASH_CYCLE[dash_idx % len(_DASH_CYCLE)]

        trace_kwargs = dict(
            x=subset[x],
            y=subset[y],
            mode=mode,
            name=_trace_name(row, trace_cols),
            line=dict(color=color_map[color_key], dash=dash),
            marker=dict(size=marker_size),
        )
        if legend_group is not None:
            trace_kwargs["legendgroup"] = legend_group
            trace_kwargs["legendgrouptitle_text"] = legend_group
        if show_error_bars and std_col in subset.columns:
            trace_kwargs["error_y"] = dict(type="data", array=subset[std_col], visible=True)
        fig.add_trace(go.Scatter(**trace_kwargs))

    if highlight_stall and "is_stall" in df.columns:
        stall_df = df[df["is_stall"].astype(bool)]
        if not stall_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=stall_df[x],
                    y=stall_df[y],
                    mode="markers",
                    marker=dict(symbol="x", size=11, color="red", line=dict(width=2)),
                    name="Stall",
                )
            )

    fig.update_layout(
        xaxis_title=axis_label(x),
        yaxis_title=axis_label(y),
        template="plotly_dark",
        legend=dict(groupclick="togglegroup") if legend_group_by else {},
    )
    return fig


def plot_fan_performance_curve(
    sim_df: pd.DataFrame,
    reference_curves: Dict[float, pd.DataFrame],
    show_total: bool = True,
) -> go.Figure:
    """
    The manufacturer's fan performance curve -- one per selected RPM
    (reference_curves: rpm -> DataFrame from src.sail.fan.build_reference_curve,
    already affinity-scaled to that RPM) -- as reference lines, with the
    simulated Fan Static/Total Pressure overlaid as markers, one trace per
    (project, aws, rpm). Different AoA rows at a fixed RPM land at different
    flow rates (the sail's aerodynamic blockage is the fan's "system
    resistance"), tracing out where the CFD-derived operating points
    actually fall against each RPM's reference curve.

    Reference curves use a grey scale (darker = higher RPM), kept visually
    separate from the qualitative palette used for simulated (project, aws,
    rpm) traces, so the two "families" of lines don't blend together once
    multiple RPMs are shown at once. Both families still share one legend
    group per RPM (legendgroup="RPM={rpm}"), so toggling a group in the
    legend hides/shows that RPM's reference curve together with every
    simulated trace at that RPM, regardless of color.
    """
    fig = go.Figure()

    rpms_sorted = sorted(reference_curves.keys())
    n_rpms = len(rpms_sorted)
    grey_shades = (
        pc.sample_colorscale("Greys", [0.4 + 0.6 * (i / (n_rpms - 1)) for i in range(n_rpms)])
        if n_rpms > 1
        else ["black"]
    )
    ref_color_map = dict(zip(rpms_sorted, grey_shades))

    for rpm in rpms_sorted:
        curve = reference_curves[rpm]
        color = ref_color_map[rpm]
        legend_group = f"RPM={rpm:g}"
        fig.add_trace(
            go.Scatter(
                x=curve["flowrate"], y=curve["static_pressure"],
                mode="lines", name=f"Reference Fan Static Pressure ({rpm:g} RPM)", line=dict(color=color, width=3),
                legendgroup=legend_group, legendgrouptitle_text=legend_group,
            )
        )
        if show_total:
            fig.add_trace(
                go.Scatter(
                    x=curve["flowrate"], y=curve["total_pressure"],
                    mode="lines", name=f"Reference Fan Total Pressure ({rpm:g} RPM)",
                    line=dict(color=color, width=3, dash="dash"),
                    legendgroup=legend_group, legendgrouptitle_text=legend_group,
                )
            )

    group_cols = ["project_name", "aws", "rpm"]
    color_keys = sorted(sim_df[group_cols].drop_duplicates().itertuples(index=False, name=None))
    palette = pc.qualitative.Plotly
    color_map = {key: palette[i % len(palette)] for i, key in enumerate(color_keys)}

    for group_values, group in sim_df.groupby(group_cols, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values))
        group = group.sort_values("aoa")
        label = _trace_name(row, group_cols)
        color = color_map[group_values]
        legend_group = f"RPM={_format_group_value(row['rpm'])}"

        fig.add_trace(
            go.Scatter(
                x=group["fan_volumetric_flow"], y=group["fan_static_pressure"],
                mode="markers", name=f"{label} — Static (sim)",
                marker=dict(symbol="circle", color=color, size=12),
                customdata=group["aoa"],
                hovertemplate="Q=%{x:.2f} m³/s<br>FSP=%{y:.1f} Pa<br>AoA=%{customdata:.1f}°<extra></extra>",
                legendgroup=legend_group, legendgrouptitle_text=legend_group,
            )
        )
        if show_total:
            fig.add_trace(
                go.Scatter(
                    x=group["fan_volumetric_flow"], y=group["fan_total_pressure"],
                    mode="markers", name=f"{label} — Total (sim)",
                    marker=dict(symbol="diamond", color=color, size=12),
                    customdata=group["aoa"],
                    hovertemplate="Q=%{x:.2f} m³/s<br>FTP=%{y:.1f} Pa<br>AoA=%{customdata:.1f}°<extra></extra>",
                    legendgroup=legend_group, legendgrouptitle_text=legend_group,
                )
            )

    fig.update_layout(
        xaxis=dict(title="Volumetric Flow Rate (m³/s)", showgrid=True),
        yaxis=dict(title="Pressure (Pa)", showgrid=True),
        legend=dict(groupclick="togglegroup"),
        template="plotly_dark", height=600, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def plot_flow_rate_parity(df: pd.DataFrame) -> go.Figure:
    """
    Estimated (from two static-pressure probes, src.sail.flow_estimation)
    vs Simulated Fan Volumetric Flow, one trace per (project, aws, rpm),
    with a y=x reference line -- how well the two-probe estimate tracks
    the actual CFD flow rate across the AoA sweep. Rows where the estimate
    couldn't be computed (missing tables, non-physical inputs) are dropped.
    """
    df = df.dropna(subset=["estimated_flow_rate", "fan_volumetric_flow"])

    fig = go.Figure()
    if df.empty:
        return fig

    group_cols = ["project_name", "aws", "rpm"]
    color_keys = sorted(df[group_cols].drop_duplicates().itertuples(index=False, name=None))
    palette = pc.qualitative.Plotly
    color_map = {key: palette[i % len(palette)] for i, key in enumerate(color_keys)}

    for group_values, group in df.groupby(group_cols, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values))
        label = _trace_name(row, group_cols)
        color = color_map[group_values]
        fig.add_trace(
            go.Scatter(
                x=group["fan_volumetric_flow"], y=group["estimated_flow_rate"],
                mode="markers", name=label, marker=dict(color=color, size=10),
                customdata=group["aoa"],
                hovertemplate=(
                    "Simulated Q=%{x:.2f} m³/s<br>Estimated Q=%{y:.2f} m³/s<br>"
                    "AoA=%{customdata:.1f}°<extra></extra>"
                ),
            )
        )

    lo = min(df["fan_volumetric_flow"].min(), df["estimated_flow_rate"].min())
    hi = max(df["fan_volumetric_flow"].max(), df["estimated_flow_rate"].max())
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
            line=dict(color="black", dash="dash", width=2),
        )
    )

    fig.update_layout(
        xaxis=dict(title="Simulated Fan Volumetric Flow (m³/s)", showgrid=True),
        yaxis=dict(title="Estimated Fan Volumetric Flow (m³/s)", showgrid=True),
        template="plotly_dark", height=600, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig
