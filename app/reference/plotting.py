"""
3D bar chart for the wind probability matrix. make_3d_bars() is adapted
from the same vectorized go.Mesh3d approach used in
../esail-fuel-savings-eedi/src/app.py for its true-wind probability
histogram -- one Mesh3d trace built from stacked box geometry rather than
one trace per bar, so it stays fast even for a 72x31-bar grid.
"""

from typing import Optional, Sequence, Tuple

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
            # Orthographic, not perspective (the plotly default) -- bar heights need to
            # compare accurately regardless of distance from the camera.
            camera=dict(projection=dict(type="orthographic")),
        ),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _speed_bin_edges(min_speed: float, max_speed: float, width: float) -> list:
    """
    [min_speed, min_speed+width, ..., max_speed] -- full-width bins up to the
    last one, which absorbs whatever remainder is smaller than `width` (so a
    31kt range grouped in 5kt steps gives bins ...,20-25,25-31, not a
    trailing 30-31 sliver). Raises if width <= 0 or doesn't fit at least once.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    n_full_bins = int((max_speed - min_speed) // width)
    if n_full_bins < 1:
        raise ValueError(f"width {width} is wider than the data range [{min_speed}, {max_speed}]")
    edges = [min_speed + i * width for i in range(n_full_bins + 1)]
    if edges[-1] < max_speed:
        edges[-1] = max_speed
    return edges


def _group_speed_bins(matrix: pd.DataFrame, edges: Sequence[float]) -> pd.DataFrame:
    """Aggregate the matrix's 1kt-wide TWS columns into wider [edges[i], edges[i+1])
    speed bands, summing probability within each band -- 31 individual 1kt traces would
    make the wind rose's stacked legend unreadable, so group into wider bands instead."""
    tws = matrix.columns.to_numpy(dtype=float)
    labels = [f"{edges[i]:g}–{edges[i + 1]:g} kts" for i in range(len(edges) - 1)]
    grouped = pd.DataFrame(index=matrix.index, dtype=float)
    for label, lo, hi in zip(labels, edges[:-1], edges[1:]):
        cols = matrix.columns[(tws >= lo) & (tws < hi)]
        grouped[label] = matrix[cols].sum(axis=1)
    return grouped


def _filter_by_twa_range(matrix: pd.DataFrame, twa_range: Optional[Tuple[float, float]]) -> pd.DataFrame:
    """
    Restricts the matrix to TWA (index) bins within [lo, hi] inclusive. None
    means no filtering -- every bin kept, matching this function's use as an
    optional param default across build_wind_rose_figure/
    build_wind_speed_probability_figure.

    lo > hi means the range wraps through the +-180deg seam -- TWA is a
    circular quantity (180deg and -180deg are the same direction), so
    (170, -170) selects the 20deg wedge from 170deg through +-180deg down
    to -170deg, rather than being an invalid/empty range. Mirrors how a
    "10pm to 6am" time-of-day range is understood to cross midnight once
    the start is later than the end.
    """
    if twa_range is None:
        return matrix
    lo, hi = twa_range
    twa = matrix.index.to_numpy(dtype=float)
    if lo <= hi:
        return matrix.loc[(twa >= lo) & (twa <= hi)]
    return matrix.loc[(twa >= lo) | (twa <= hi)]


def _twa_range_label(twa_range: Optional[Tuple[float, float]]) -> str:
    """Chart-title suffix for a twa_range, e.g. ', TWA [170°, -170°] (wraps through ±180°)'."""
    if twa_range is None:
        return ""
    lo, hi = twa_range
    wrap_note = " (wraps through ±180°)" if lo > hi else ""
    return f", TWA [{lo:g}°, {hi:g}°]{wrap_note}"


def twa_range_coverage_pct(twa_range: Optional[Tuple[float, float]]) -> float:
    """% of total wind probability mass falling within twa_range (inclusive of both
    ends) -- None (no filter) is always 100%. Restricting to a TWA range necessarily
    drops some probability mass, so a filtered Wind Speed Probability CDF caps out
    below 100% -- this is what the app shows alongside it so that isn't mistaken for
    a bug."""
    if twa_range is None:
        return 100.0
    matrix = load_wind_probability_matrix()
    return float(_filter_by_twa_range(matrix, twa_range).to_numpy().sum() * 100)


@st.cache_data
def build_wind_rose_figure(speed_bin_width: float = 5.0, twa_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    """
    Classic wind rose: one angular sector per TWA bin (5deg wide), stacked
    radially by TWS speed band, radius = probability (%). Reads at a glance
    which TWA/TWS combinations dominate -- the pattern the 3D bar chart also
    shows, but harder to eyeball there since it requires rotating the view.

    speed_bin_width (kts) controls how many TWS bands the legend/stack has --
    the underlying matrix is always 1kt resolution (see load_wind_probability_matrix),
    grouped into wider bands here purely for a legible legend. Bin edges are derived
    from the matrix's own TWS range, not hardcoded, so any width divides it cleanly.

    twa_range, if given, restricts to that [lo, hi] TWA slice (see
    _filter_by_twa_range) -- sectors outside it simply don't appear, rather than
    being drawn as zero-height, so the rose becomes a wedge instead of a full circle.

    TWA=0 is plotted at the top with angle increasing clockwise, matching the
    conventional bow-up layout for a vessel-relative wind angle (as opposed
    to a compass-referenced true wind direction, which TWA is not).
    """
    matrix = _filter_by_twa_range(load_wind_probability_matrix(), twa_range)
    tws = matrix.columns.to_numpy(dtype=float)
    half_width = (tws[1] - tws[0]) / 2  # e.g. 0.5 for 1kt-wide source columns
    edges = _speed_bin_edges(tws.min() - half_width, tws.max() + half_width, speed_bin_width)

    grouped = _group_speed_bins(matrix, edges) * 100  # fraction -> percent
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
        title=f"Barcelona Harbor — Wind Rose (True Wind{_twa_range_label(twa_range)})",
        barmode="stack",
        polar=dict(
            angularaxis=dict(rotation=90, direction="clockwise", ticksuffix="°"),
            radialaxis=dict(ticksuffix="%"),
        ),
        legend_title="TWS",
        height=650,
        margin=dict(l=0, r=0, t=60, b=20),
    )
    return fig


#: The wind probability matrix's TWS values are measured at this height (m) --
#: standard meteorological reference height.
REFERENCE_HEIGHT_M = 10.0

# alpha (Hellmann exponent) presets for the power-law wind profile
# V(z) = V(z_ref) * (z / z_ref)^alpha. Values in common use for extrapolating
# a 10m reference measurement to another height.
TERRAIN_EXPONENTS = {
    "Open water (α=1/9)": 1 / 9,
    "Land (α=1/7)": 1 / 7,
}


def power_law_scale_factor(height_m: float, alpha: float, reference_height_m: float = REFERENCE_HEIGHT_M) -> float:
    """
    (z / z_ref)^alpha from V(z) = V(z_ref) * (z / z_ref)^alpha. This factor
    is the same for every wind speed (it doesn't depend on V(z_ref) itself),
    so applying it to a whole speed distribution is just a constant rescale
    of the speed axis -- the probabilities themselves don't change.
    """
    return (height_m / reference_height_m) ** alpha


@st.cache_data
def build_wind_speed_probability_figure(
    height_m: float = REFERENCE_HEIGHT_M, alpha: float = 1 / 7, twa_range: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """
    Marginal TWS probability (summed over every TWA, or just the TWA rows
    inside twa_range if given) as a PDF + CDF on a shared x-axis, dual
    y-axes -- the classic "wind speed probability" chart for reading off
    both the most likely speed and the cumulative fraction of time below
    any given speed.

    height_m/alpha extrapolate the 10m-reference TWS values to another
    height via the power-law wind profile (power_law_scale_factor) -- e.g.
    height_m=20 for the sail's top winglet anemometers. Probabilities are
    unchanged; only the TWS axis is rescaled (see power_law_scale_factor).

    twa_range restricts the sum to that TWA slice instead of every TWA row
    -- deliberately *not* renormalized back to 100%, so the CDF caps out at
    whatever share of total wind time that TWA range actually represents
    (see twa_range_coverage_pct, which the app shows alongside this chart
    so the capped-below-100% CDF isn't mistaken for a bug). Consistent with
    the wind rose and 3D chart, which are also absolute, not conditional.
    """
    matrix = _filter_by_twa_range(load_wind_probability_matrix(), twa_range)
    scale = power_law_scale_factor(height_m, alpha)
    tws_centers = matrix.columns.to_numpy(dtype=float) * scale
    pdf_pct = matrix.to_numpy().sum(axis=0) * 100  # sum over TWA rows, fraction -> percent
    cdf_pct = np.cumsum(pdf_pct)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tws_centers, y=pdf_pct, mode="lines+markers", name="Probability",
            hovertemplate="TWS: %{x:.2f} kts<br>Probability: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tws_centers, y=cdf_pct, mode="lines+markers", name="Cumulative Probability",
            yaxis="y2",
            hovertemplate="TWS: %{x:.2f} kts<br>Cumulative: %{y:.1f}%<extra></extra>",
        )
    )

    height_note = (
        f"at {REFERENCE_HEIGHT_M:g}m (reference)"
        if height_m == REFERENCE_HEIGHT_M
        else f"at {height_m:g}m (extrapolated from {REFERENCE_HEIGHT_M:g}m, α={alpha:.3f})"
    )
    fig.update_layout(
        title=f"Barcelona Harbor — Wind Speed Probability (True Wind, {height_note}{_twa_range_label(twa_range)})",
        xaxis_title="True Wind Speed (kts)",
        yaxis=dict(title="Probability (%)", rangemode="tozero"),
        yaxis2=dict(title="Cumulative Probability (%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550,
        margin=dict(l=0, r=0, t=100, b=0),
    )
    return fig
