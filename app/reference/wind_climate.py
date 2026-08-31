"""Wind Climate: Barcelona harbor wind probability distribution (TWA/TWS) — wind rose + 3D detail view."""

import streamlit as st

from app.reference.plotting import build_wind_probability_figure, build_wind_rose_figure

st.title("Wind Climate")
st.caption(
    "Barcelona harbor true wind probability distribution — used to weight which TWS/TWA "
    "combinations matter most when planning harbour/interference simulations."
)

tab_rose, tab_3d = st.tabs(["Wind Rose", "3D Detail"])

with tab_rose:
    st.caption("TWA=0 at top, increasing clockwise (vessel-relative angle, not compass direction).")
    speed_bin_width = st.select_slider(
        "Speed bin width", options=[1, 2, 3, 5, 10, 15], value=5, format_func=lambda w: f"{w} kts"
    )
    st.plotly_chart(build_wind_rose_figure(speed_bin_width=speed_bin_width), width="stretch")

with tab_3d:
    st.caption("Full 5°×1kt resolution, ungrouped.")
    st.plotly_chart(build_wind_probability_figure(), width="stretch")
