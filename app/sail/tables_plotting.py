"""
Plots for per-AoA spatial tables (src.sail.spatial_tables), grouped by
physical shape rather than table name -- our tables/ folder's naming
varies between data batches (see CLAUDE.md), but the column shapes fall
into three families:

  - height-profile (Accumulated Force Table): Position (m) is the
    vertical axis, a chosen force variable is horizontal.
  - Z-profile (esail_internal_*, esail_winglet_anemometer,
    wind_profile_*, Velocity and Pressure Probe Table): Z (m) is the
    vertical axis, a chosen scalar is horizontal. Any subset of these
    tables can be overlaid together since they share the same shape.
  - skin sections (esail_skin_sections_table): Cp vs x/c, or the XY
    airfoil cross-section colored by Cp -- one trace per Z (height)
    station either way.
"""

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

from app.sail.plotting import PLOT_MODES

_POSITION_COL = "Position (m)"
_Z_COL = "Z (m)"
_CP_COL = "Pressure Coefficient"


def is_profile_height_table(df: pd.DataFrame) -> bool:
    return _POSITION_COL in df.columns


def is_skin_sections_table(df: pd.DataFrame) -> bool:
    return _CP_COL in df.columns and {"X (m)", "Y (m)", _Z_COL} <= set(df.columns)


def is_z_profile_table(df: pd.DataFrame) -> bool:
    return _Z_COL in df.columns and not is_skin_sections_table(df) and not is_profile_height_table(df)


def numeric_columns(df: pd.DataFrame, exclude: List[str] = ()) -> List[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def plot_height_profile(df: pd.DataFrame, x_columns: List[str]) -> go.Figure:
    """Position (m) on the vertical axis, one or more chosen variables on the horizontal axis."""
    fig = go.Figure()
    for col in x_columns:
        fig.add_trace(go.Scatter(x=df[col], y=df[_POSITION_COL], mode="lines+markers", name=col))
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(title=_POSITION_COL, showgrid=True),
        template="plotly_dark", height=550, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def plot_z_profile(tables: Dict[str, pd.DataFrame], variable: str, plot_mode: str = "Lines + Markers") -> go.Figure:
    """Z (m) on the vertical axis, `variable` on the horizontal axis, one trace per table --
    only tables that actually contain `variable` are plotted (the families share most but not
    all column names, e.g. velocity component naming differs between them)."""
    mode, marker_size = PLOT_MODES.get(plot_mode, PLOT_MODES["Lines + Markers"])
    fig = go.Figure()
    for name, df in tables.items():
        if variable not in df.columns:
            continue
        df_sorted = df.sort_values(_Z_COL)
        fig.add_trace(
            go.Scatter(
                x=df_sorted[variable], y=df_sorted[_Z_COL], mode=mode,
                marker=dict(size=marker_size), name=name,
            )
        )
    fig.update_layout(
        xaxis=dict(title=variable, showgrid=True),
        yaxis=dict(title=_Z_COL, showgrid=True),
        template="plotly_dark", height=550, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def compute_x_over_chord(df: pd.DataFrame, chord: float) -> pd.Series:
    """x/c per Z station: (X - X_min at that station) / chord. Assumes X (m) is the
    sail's chordwise axis in its body-local frame (confirmed for this project -- see CLAUDE.md)."""
    x_min_by_z = df.groupby(_Z_COL)["X (m)"].transform("min")
    return (df["X (m)"] - x_min_by_z) / chord


def plot_cp_vs_xc(df: pd.DataFrame, chord: float, z_stations: List[float], invert_cp: bool = True) -> go.Figure:
    """Cp vs x/c, one trace per Z (height) station. Markers only (no connecting line) --
    upper and lower surface points overlap in x/c, so sorting by x/c alone (not full
    surface-perimeter order) would draw a line zigzagging between them rather than a
    clean curve."""
    df = df[df[_Z_COL].isin(z_stations)].copy()
    df["x/c"] = compute_x_over_chord(df, chord)

    fig = go.Figure()
    for z in sorted(z_stations):
        station = df[df[_Z_COL] == z].sort_values("x/c")
        fig.add_trace(go.Scatter(x=station["x/c"], y=station[_CP_COL], mode="markers", name=f"Z={z:g}m"))
    fig.update_layout(
        xaxis=dict(title="x/c", showgrid=True),
        yaxis=dict(title="Cp", showgrid=True, autorange="reversed" if invert_cp else True),
        template="plotly_dark", height=550, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def plot_skin_sections_xy(df: pd.DataFrame, z_stations: List[float]) -> go.Figure:
    """XY airfoil cross-section shape colored by Cp, one trace per Z station -- a shared
    color range across stations so colors are comparable, with a single shared colorbar."""
    df = df[df[_Z_COL].isin(z_stations)]
    cmin, cmax = df[_CP_COL].min(), df[_CP_COL].max()

    fig = go.Figure()
    for i, z in enumerate(sorted(z_stations)):
        station = df[df[_Z_COL] == z]
        marker = dict(size=5, color=station[_CP_COL], colorscale="Viridis", cmin=cmin, cmax=cmax)
        if i == 0:
            marker["showscale"] = True
            marker["colorbar"] = dict(title="Cp")
        fig.add_trace(go.Scatter(x=station["X (m)"], y=station["Y (m)"], mode="markers", name=f"Z={z:g}m", marker=marker))
    fig.update_layout(
        xaxis=dict(title="X (m)", showgrid=True),
        yaxis=dict(title="Y (m)", showgrid=True, scaleanchor="x", scaleratio=1),  # equal aspect -- true cross-section shape
        template="plotly_dark", height=650, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig
