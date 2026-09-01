"""
Tables: browse per-AoA spatial tables for one Project/Case/AoA, read live
from data/ (see CLAUDE.md's "Deliberate exception"). Grouped by shape
into three categories, each with its own purpose-built plot -- see
app.sail.tables_plotting's module docstring.
"""

import streamlit as st

from app.components.data import list_projects
from app.sail.data import SIMULATION_TYPE, get_polar_data
from app.sail.plotting import PLOT_MODES
from app.sail.tables_plotting import (
    compute_x_over_chord,
    is_profile_height_table,
    is_skin_sections_table,
    is_z_profile_table,
    numeric_columns,
    plot_cp_vs_xc,
    plot_height_profile,
    plot_skin_sections_xy,
    plot_z_profile,
)
from src.config import load_config
from src.sail.spatial_tables import list_table_files, load_table

st.title("Tables")
st.caption(
    "Per-AoA spatial tables, read live from data/sail/ — not part of the results database, "
    "since these are raw point-cloud exports rather than the curated polar/performance grain."
)

projects_df = list_projects(simulation_type=SIMULATION_TYPE)
if projects_df.empty:
    st.info("No sail projects synced yet — run the ETL on the Home page.")
    st.stop()

project_rows = list(projects_df.itertuples())
project_row = st.selectbox("Project", project_rows, format_func=lambda r: r.name)

polar_df = get_polar_data((project_row.id,))
if polar_df.empty:
    st.warning("No data found for this project.")
    st.stop()

col_case, col_aoa = st.columns(2)
case_name = col_case.selectbox("Case", sorted(polar_df["case_name"].unique()))

case_df = polar_df[polar_df["case_name"] == case_name].sort_values("aoa")
aoa = col_aoa.selectbox("AoA", case_df["aoa"].tolist(), format_func=lambda a: f"{a:g}°")

source_path = case_df[case_df["aoa"] == aoa]["source_path"].iloc[0]

table_stems = list_table_files(source_path)
if not table_stems:
    st.warning(f"No tables/ folder found for this AoA — looked in {source_path}/tables")
    st.stop()

tables = {stem: load_table(source_path, stem) for stem in table_stems}
tables = {stem: df for stem, df in tables.items() if not df.empty}

height_profile_tables = {s: d for s, d in tables.items() if is_profile_height_table(d)}
z_profile_tables = {s: d for s, d in tables.items() if is_z_profile_table(d)}
skin_sections_tables = {s: d for s, d in tables.items() if is_skin_sections_table(d)}

categories = {}
if z_profile_tables:
    categories["Z-profile probes"] = "z_profile"
if height_profile_tables:
    categories["Accumulated Force"] = "height_profile"
if skin_sections_tables:
    categories["Skin Sections (Cp)"] = "skin_sections"

if not categories:
    st.warning("No recognized table shapes found for this AoA.")
    st.stop()

category_label = st.radio("Table type", list(categories.keys()), horizontal=True)
category = categories[category_label]

if category == "z_profile":
    selected_stems = st.multiselect(
        "Tables to overlay", list(z_profile_tables.keys()), default=list(z_profile_tables.keys())
    )
    if not selected_stems:
        st.info("Select at least one table.")
        st.stop()
    selected_tables = {s: z_profile_tables[s] for s in selected_stems}

    col_variable, col_mode = st.columns(2)
    variable_options = sorted(
        set().union(*(numeric_columns(d, exclude=["X (m)", "Y (m)", "Z (m)"]) for d in selected_tables.values()))
    )
    variable = col_variable.selectbox("Variable", variable_options)
    plot_mode = col_mode.selectbox("Display mode", list(PLOT_MODES.keys()))
    st.plotly_chart(plot_z_profile(selected_tables, variable=variable, plot_mode=plot_mode), width="stretch")

    with st.expander("Raw data"):
        for stem, df in selected_tables.items():
            st.caption(stem)
            st.dataframe(df, width="stretch")
            st.download_button(
                f"Download {stem}.csv", data=df.to_csv(index=False), file_name=f"{stem}.csv",
                mime="text/csv", key=f"download_{stem}",
            )

elif category == "height_profile":
    stem, df = next(iter(height_profile_tables.items()))  # normally just one
    y_options = numeric_columns(df, exclude=["Position (m)", "Profile Lower (m)", "Profile Upper (m)"])
    default_ys = [c for c in y_options if "Force Coefficient" in c] or y_options
    selected_ys = st.multiselect("Plot columns", y_options, default=default_ys)
    if selected_ys:
        st.plotly_chart(plot_height_profile(df, x_columns=selected_ys), width="stretch")

    with st.expander("Raw data"):
        st.dataframe(df, width="stretch")
        st.download_button(f"Download {stem}.csv", data=df.to_csv(index=False), file_name=f"{stem}.csv", mime="text/csv")

else:  # skin_sections
    stem, df = next(iter(skin_sections_tables.items()))  # normally just one
    config = load_config()
    chord = config.esail.chord
    if chord is None:
        st.warning("Chord is not set in config/config.yaml — cannot compute x/c.")
        st.stop()

    z_all = sorted(df["Z (m)"].unique())
    z_stations = st.multiselect("Height (Z) stations", z_all, default=z_all, format_func=lambda z: f"{z:g} m")
    if not z_stations:
        st.info("Select at least one Z station.")
        st.stop()

    plot_mode = st.radio("Plot", ["Cp vs x/c", "XY airfoil (colored by Cp)"], horizontal=True)
    if plot_mode == "Cp vs x/c":
        invert_cp = st.checkbox("Invert Cp axis (suction up)", value=True)
        st.plotly_chart(plot_cp_vs_xc(df, chord=chord, z_stations=z_stations, invert_cp=invert_cp), width="stretch")
    else:
        st.plotly_chart(plot_skin_sections_xy(df, z_stations=z_stations), width="stretch")

    with st.expander("Raw data"):
        st.dataframe(df, width="stretch")
        st.download_button(f"Download {stem}.csv", data=df.to_csv(index=False), file_name=f"{stem}.csv", mime="text/csv")
