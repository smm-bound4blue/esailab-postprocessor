"""Wind Climate: Barcelona harbor wind probability distribution (TWA/TWS) — wind rose,
wind speed probability, and 3D detail view."""

import streamlit as st

from app.reference.plotting import (
    REFERENCE_HEIGHT_M,
    TERRAIN_EXPONENTS,
    build_wind_probability_figure,
    build_wind_rose_figure,
    build_wind_speed_probability_figure,
    twa_range_coverage_pct,
)

st.title("Wind Climate")
st.caption(
    "Barcelona harbor true wind probability distribution — used to weight which TWS/TWA "
    "combinations matter most when planning harbour/interference simulations."
)


def _twa_range_input(key_prefix: str, help_text: str):
    """From/To TWA sliders (not one two-handle slider) -- TWA is circular, wrapping at
    +-180deg, so From > To is a valid selection meaning "wrap through +-180deg" (e.g.
    From=170, To=-170 selects the wedge straddling the seam), which a single ordered
    [lo, hi] slider can't express."""
    col_from, col_to = st.columns(2)
    twa_from = col_from.slider(
        "From", min_value=-180, max_value=180, value=-180, step=5, format="%d°", key=f"{key_prefix}_from",
        help=help_text,
    )
    twa_to = col_to.slider(
        "To", min_value=-180, max_value=180, value=180, step=5, format="%d°", key=f"{key_prefix}_to",
        help="If From > To, the range wraps through ±180° instead of being invalid.",
    )
    return twa_from, twa_to


tab_rose, tab_speed, tab_3d = st.tabs(["Wind Rose", "Wind Speed Probability", "3D Detail"])

with tab_rose:
    st.caption("TWA=0 at top, increasing clockwise (vessel-relative angle, not compass direction).")
    speed_bin_width = st.select_slider(
        "Speed bin width", options=[1, 2, 3, 5, 10, 15], value=2, format_func=lambda w: f"{w} kts"
    )
    twa_range_rose = _twa_range_input(
        "rose_twa_range", help_text="Restrict to a TWA slice -- sectors outside it just disappear from the rose."
    )
    st.plotly_chart(
        build_wind_rose_figure(speed_bin_width=speed_bin_width, twa_range=twa_range_rose), width="stretch"
    )
    st.caption(f"Selected TWA range covers {twa_range_coverage_pct(twa_range_rose):.1f}% of total wind time.")

with tab_speed:
    st.caption(
        "TWS probability marginalized over all TWA (PDF, left axis) and its cumulative sum (CDF, right axis). "
        f"Source data is at {REFERENCE_HEIGHT_M:g}m — extrapolate to another height with the power-law wind "
        "profile V(z) = V(10) · (z/10)^α."
    )
    col_height, col_terrain, col_alpha = st.columns(3)
    height_m = col_height.number_input(
        "Height (m)", min_value=1.0, value=20.0, step=1.0,
        help="20m is where the sail's top winglet anemometers sit.",
    )
    terrain = col_terrain.selectbox("Terrain", [*TERRAIN_EXPONENTS.keys(), "Custom"])
    if terrain == "Custom":
        alpha = col_alpha.number_input("α (exponent)", min_value=0.0, max_value=1.0, value=1 / 7, step=0.01, format="%.3f")
    else:
        alpha = TERRAIN_EXPONENTS[terrain]
        col_alpha.metric("α (exponent)", f"{alpha:.3f}")

    twa_range_speed = _twa_range_input(
        "speed_twa_range",
        help_text="Restrict the PDF/CDF to a TWA slice -- probabilities are NOT renormalized, so a "
        "narrower range means a smaller PDF and a CDF that caps out below 100%.",
    )

    st.plotly_chart(
        build_wind_speed_probability_figure(height_m=height_m, alpha=alpha, twa_range=twa_range_speed),
        width="stretch",
    )
    st.caption(f"Selected TWA range covers {twa_range_coverage_pct(twa_range_speed):.1f}% of total wind time.")

with tab_3d:
    st.caption("Full 5°×1kt resolution, ungrouped.")
    st.plotly_chart(build_wind_probability_figure(), width="stretch")
