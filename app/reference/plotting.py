"""
3D bar chart for the wind probability matrix. make_3d_bars() is adapted
from the same vectorized go.Mesh3d approach used in
../esail-fuel-savings-eedi/src/app.py for its true-wind probability
histogram -- one Mesh3d trace built from stacked box geometry rather than
one trace per bar, so it stays fast even for a 72x31-bar grid.
"""

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st

from app.reference.data import load_wind_probability_matrix


def make_3d_bars(
    x_centers: np.ndarray, y_centers: np.ndarray, z_heights: np.ndarray, dx: float, dy: float,
    x_name: str = "X", y_name: str = "Y", z_name: str = "Z",
    x_unit: str = "", y_unit: str = "", z_unit: str = "",
) -> go.Mesh3d:
    """Builds a single vectorized go.Mesh3d trace of 3D bars (boxes), one per
    (x_centers[i], y_centers[j]) grid cell with height z_heights[i, j].

    Each box gets its own hover label showing its X/Y bin range and Z value,
    rather than Plotly's default per-vertex x/y/z (which would show raw box
    *corner* coordinates -- ambiguous, since adjacent bars share corners, and
    a bottom-face vertex reads z=0 rather than the bar's actual height)."""
    xc, yc = np.meshgrid(x_centers, y_centers, indexing="ij")
    xc = xc.flatten()
    yc = yc.flatten()
    h = z_heights.flatten()
    n = len(h)

    hx, hy = dx / 2, dy / 2
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    vx = np.empty(n * 8)
    vy = np.empty(n * 8)
    vz = np.empty(n * 8)
    for idx, (ox, oy) in enumerate(corners):
        vx[idx::8] = xc + ox
        vy[idx::8] = yc + oy
        vz[idx::8] = 0.0
        vx[4 + idx::8] = xc + ox
        vy[4 + idx::8] = yc + oy
        vz[4 + idx::8] = h

    # 12 triangles per box (2 per face x 6 faces), as local vertex-index triplets
    local_tris = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),  # front
        (3, 2, 6), (3, 6, 7),  # back
        (0, 3, 7), (0, 7, 4),  # left
        (1, 2, 6), (1, 6, 5),  # right
    ]
    base = (np.arange(n) * 8)[:, None]
    tri_i = np.empty(n * 12, dtype=int)
    tri_j = np.empty(n * 12, dtype=int)
    tri_k = np.empty(n * 12, dtype=int)
    for t, (a, b, c) in enumerate(local_tris):
        tri_i[t::12] = (base + a).flatten()
        tri_j[t::12] = (base + b).flatten()
        tri_k[t::12] = (base + c).flatten()

    x_unit_sfx = f" {x_unit}" if x_unit else ""
    y_unit_sfx = f" {y_unit}" if y_unit else ""
    z_unit_sfx = f" {z_unit}" if z_unit else ""
    box_text = [
        f"{x_name}: [{x - hx:g}, {x + hx:g}){x_unit_sfx}<br>"
        f"{y_name}: [{y - hy:g}, {y + hy:g}){y_unit_sfx}<br>"
        f"{z_name}: {z:.3g}{z_unit_sfx}"
        for x, y, z in zip(xc, yc, h)
    ]
    vertex_text = np.repeat(box_text, 8)

    return go.Mesh3d(
        x=vx, y=vy, z=vz, i=tri_i, j=tri_j, k=tri_k,
        intensity=np.repeat(h, 8), colorscale="Viridis", showscale=True,
        flatshading=True,
        text=vertex_text,
        hovertemplate="%{text}<extra></extra>",
    )


@st.cache_data
def build_wind_probability_figure() -> go.Figure:
    """The wind probability matrix is a fixed constant, so this ~2200-bar 3D
    chart is built once and cached rather than rebuilt on every rerun."""
    matrix = load_wind_probability_matrix()
    twa_centers = matrix.index.to_numpy(dtype=float)
    tws_centers = matrix.columns.to_numpy(dtype=float)
    prob_pct = matrix.to_numpy() * 100

    fig = go.Figure(
        data=[
            make_3d_bars(
                twa_centers, tws_centers, prob_pct, dx=5.0, dy=1.0,
                x_name="True Wind Angle", y_name="True Wind Speed", z_name="Probability",
                x_unit="deg", y_unit="kts", z_unit="%",
            )
        ]
    )
    fig.update_layout(
        title="Barcelona Harbor — Wind Probability Distribution (True Wind)",
        scene=dict(
            xaxis_title="True Wind Angle (deg)",
            yaxis_title="True Wind Speed (kts)",
            zaxis_title="Probability (%)",
        ),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _group_speed_bins(matrix: pd.DataFrame, edges: Sequence[float]) -> pd.DataFrame:
    """Aggregate the matrix's 1kt-wide TWS columns into wider [edges[i], edges[i+1])
    speed bands, summing probability within each band -- 31 individual 1kt traces would
    make the wind rose's stacked legend unreadable, so group into ~6 bands instead."""
    tws = matrix.columns.to_numpy(dtype=float)
    labels = [f"{edges[i]:g}–{edges[i + 1]:g} kts" for i in range(len(edges) - 1)]
    grouped = pd.DataFrame(index=matrix.index, dtype=float)
    for label, lo, hi in zip(labels, edges[:-1], edges[1:]):
        cols = matrix.columns[(tws >= lo) & (tws < hi)]
        grouped[label] = matrix[cols].sum(axis=1)
    return grouped


@st.cache_data
def build_wind_rose_figure(speed_edges: Sequence[float] = (0, 5, 10, 15, 20, 25, 31)) -> go.Figure:
    """
    Classic wind rose: one angular sector per TWA bin (5deg wide), stacked
    radially by TWS speed band, radius = probability (%). Reads at a glance
    which TWA/TWS combinations dominate -- the pattern the 3D bar chart also
    shows, but harder to eyeball there since it requires rotating the view.

    TWA=0 is plotted at the top with angle increasing clockwise, matching the
    conventional bow-up layout for a vessel-relative wind angle (as opposed
    to a compass-referenced true wind direction, which TWA is not).
    """
    matrix = load_wind_probability_matrix()
    grouped = _group_speed_bins(matrix, speed_edges) * 100  # fraction -> percent
    twa = matrix.index.to_numpy(dtype=float)

    n_bands = len(grouped.columns)
    colors = pc.sample_colorscale("Turbo", [i / max(n_bands - 1, 1) for i in range(n_bands)])

    fig = go.Figure()
    for label, color in zip(grouped.columns, colors):
        fig.add_trace(
            go.Barpolar(
                r=grouped[label].to_numpy(),
                theta=twa,
                width=4.8,  # slightly under the 5deg bin width, for a thin gap between sectors
                name=label,
                marker_color=color,
                hovertemplate=f"TWA: %{{theta:.1f}}°<br>{label}<br>Probability: %{{r:.2f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Barcelona Harbor — Wind Rose (True Wind)",
        barmode="stack",
        polar=dict(
            angularaxis=dict(rotation=90, direction="clockwise", ticksuffix="°"),
            radialaxis=dict(ticksuffix="%"),
        ),
        legend_title="TWS",
        height=650,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig
