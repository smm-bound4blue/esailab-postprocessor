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
    st.plotly_chart(build_wind_rose_figure(), width="stretch")

with tab_3d:
    st.caption("Full 5°×1kt resolution, ungrouped.")
    st.plotly_chart(build_wind_probability_figure(), width="stretch")
