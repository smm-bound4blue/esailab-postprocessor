"""
Reusable Plotly chart builders for the Sail pages' polar curves and
performance envelope. Pure functions: DataFrame in, plotly.graph_objects.Figure
out -- no Streamlit calls here, so pages stay in charge of layout/widgets.

Expects the DataFrame shape returned by app.sail.data.get_polar_data()
(lowercase DB column names: aoa, cl, cd, e, rpm, cpow, cq, ...).
"""

import pandas as pd
import plotly.graph_objects as go

_AXIS_LABELS = {
    "aoa": "AoA (deg)",
    "cl": "CL",
    "cd": "CD",
    "e": "E (CL/CD)",
    "cq": "CQ",
    "cpow": "Cpow",
    "fan_static_efficiency": "Fan Static Efficiency",
    "fan_total_efficiency": "Fan Total Efficiency",
    "rpm": "RPM",
}


def _label(col: str) -> str:
    return _AXIS_LABELS.get(col, col)


def plot_polar_curve(df: pd.DataFrame, y: str, x: str = "aoa", group_by: str = "rpm") -> go.Figure:
    """e.g. y='cl' or y='cd' or y='e' vs AoA, one trace per RPM."""
    fig = go.Figure()
    for group_value in sorted(df[group_by].unique()):
        subset = df[df[group_by] == group_value].sort_values(x)
        fig.add_trace(
            go.Scatter(
                x=subset[x],
                y=subset[y],
                mode="lines+markers",
                name=f"{_label(group_by)} = {group_value:g}",
            )
        )
    fig.update_layout(
        xaxis_title=_label(x),
        yaxis_title=_label(y),
        template="plotly_dark",
        legend_title=_label(group_by),
    )
    return fig


def plot_performance_envelope(df: pd.DataFrame, x: str = "cpow", y: str = "cl") -> go.Figure:
    """CL vs Cpow performance envelope, one trace per RPM."""
    return plot_polar_curve(df, y=y, x=x, group_by="rpm")
