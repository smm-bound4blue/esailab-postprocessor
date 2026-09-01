"""Tests for app.sail.tables_plotting: table-shape classification and the per-shape plotters."""

import pandas as pd
import pytest

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


def test_is_profile_height_table_detects_position_column():
    df = pd.DataFrame({"Position (m)": [0.0], "Force Coefficient": [1.0]})
    assert is_profile_height_table(df)
    assert not is_z_profile_table(df)
    assert not is_skin_sections_table(df)


def test_is_skin_sections_table_detects_cp_and_xyz():
    df = pd.DataFrame({"Pressure Coefficient": [0.1], "X (m)": [0.0], "Y (m)": [0.0], "Z (m)": [10.0]})
    assert is_skin_sections_table(df)
    assert not is_z_profile_table(df)
    assert not is_profile_height_table(df)


def test_is_z_profile_table_is_the_shape_fallback():
    df = pd.DataFrame({"Static Pressure (Pa)": [1.0], "X (m)": [0.0], "Y (m)": [0.0], "Z (m)": [10.0]})
    assert is_z_profile_table(df)
    assert not is_skin_sections_table(df)
    assert not is_profile_height_table(df)


def test_numeric_columns_excludes_given_columns_and_non_numeric():
    df = pd.DataFrame({"X (m)": [1.0], "Static Pressure (Pa)": [2.0], "label": ["a"]})
    assert numeric_columns(df, exclude=["X (m)"]) == ["Static Pressure (Pa)"]


def test_plot_height_profile_position_on_vertical_axis():
    df = pd.DataFrame({"Position (m)": [0.0, 1.0, 2.0], "Force Coefficient": [0.1, 0.2, 0.3]})
    fig = plot_height_profile(df, x_columns=["Force Coefficient"])
    trace = fig.data[0]
    assert list(trace.y) == [0.0, 1.0, 2.0]  # Position on Y (vertical)
    assert list(trace.x) == [0.1, 0.2, 0.3]  # Force Coefficient on X (horizontal)


def test_plot_z_profile_one_trace_per_table_z_on_vertical_axis():
    tables = {
        "esail_internal_center_table": pd.DataFrame({"Z (m)": [10.0, 12.0], "Static Pressure (Pa)": [100.0, 90.0]}),
        "wind_profile_inlet_table": pd.DataFrame({"Z (m)": [10.0, 12.0], "Static Pressure (Pa)": [50.0, 45.0]}),
    }
    fig = plot_z_profile(tables, variable="Static Pressure (Pa)")
    assert len(fig.data) == 2
    names = {t.name for t in fig.data}
    assert names == {"esail_internal_center_table", "wind_profile_inlet_table"}
    for trace in fig.data:
        assert list(trace.y) == [10.0, 12.0]  # Z on the vertical axis


def test_plot_z_profile_skips_tables_missing_the_variable():
    tables = {
        "has_it": pd.DataFrame({"Z (m)": [10.0], "Static Pressure (Pa)": [100.0]}),
        "missing_it": pd.DataFrame({"Z (m)": [10.0], "Velocity: Magnitude (m/s)": [5.0]}),
    }
    fig = plot_z_profile(tables, variable="Static Pressure (Pa)")
    assert len(fig.data) == 1
    assert fig.data[0].name == "has_it"


def test_plot_height_profile_has_gridlines_both_axes():
    df = pd.DataFrame({"Position (m)": [0.0, 1.0], "Force Coefficient": [0.1, 0.2]})
    fig = plot_height_profile(df, x_columns=["Force Coefficient"])
    assert fig.layout.yaxis.showgrid is True
    assert fig.layout.xaxis.showgrid is True


def test_plot_z_profile_has_gridlines_both_axes():
    tables = {"t": pd.DataFrame({"Z (m)": [10.0], "Static Pressure (Pa)": [100.0]})}
    fig = plot_z_profile(tables, variable="Static Pressure (Pa)")
    assert fig.layout.yaxis.showgrid is True
    assert fig.layout.xaxis.showgrid is True


def test_plot_z_profile_display_modes_control_mode_and_marker_size():
    tables = {"t": pd.DataFrame({"Z (m)": [10.0, 12.0], "Static Pressure (Pa)": [100.0, 90.0]})}
    lines_only = plot_z_profile(tables, variable="Static Pressure (Pa)", plot_mode="Only Lines")
    markers_only = plot_z_profile(tables, variable="Static Pressure (Pa)", plot_mode="Only Markers")
    both = plot_z_profile(tables, variable="Static Pressure (Pa)", plot_mode="Lines + Markers")

    assert lines_only.data[0].mode == "lines"
    assert markers_only.data[0].mode == "markers"
    assert both.data[0].mode == "lines+markers"


def test_compute_x_over_chord_normalizes_per_z_station():
    df = pd.DataFrame({"Z (m)": [10.0, 10.0, 20.0, 20.0], "X (m)": [0.0, 2.714, 5.0, 7.714]})
    xc = compute_x_over_chord(df, chord=2.714)
    assert xc.tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_plot_cp_vs_xc_one_trace_per_selected_z_station():
    df = pd.DataFrame(
        {
            "Z (m)": [10.0, 10.0, 20.0, 20.0],
            "X (m)": [0.0, 2.714, 0.0, 2.714],
            "Pressure Coefficient": [-1.0, 0.5, -2.0, 0.3],
        }
    )
    fig = plot_cp_vs_xc(df, chord=2.714, z_stations=[10.0, 20.0])
    assert len(fig.data) == 2
    names = {t.name for t in fig.data}
    assert names == {"Z=10m", "Z=20m"}


def test_plot_cp_vs_xc_respects_z_station_filter():
    df = pd.DataFrame(
        {
            "Z (m)": [10.0, 10.0, 20.0, 20.0],
            "X (m)": [0.0, 2.714, 0.0, 2.714],
            "Pressure Coefficient": [-1.0, 0.5, -2.0, 0.3],
        }
    )
    fig = plot_cp_vs_xc(df, chord=2.714, z_stations=[10.0])
    assert len(fig.data) == 1
    assert fig.data[0].name == "Z=10m"


def test_plot_cp_vs_xc_invert_toggle():
    df = pd.DataFrame({"Z (m)": [10.0], "X (m)": [0.0], "Pressure Coefficient": [-1.0]})
    inverted = plot_cp_vs_xc(df, chord=2.714, z_stations=[10.0], invert_cp=True)
    normal = plot_cp_vs_xc(df, chord=2.714, z_stations=[10.0], invert_cp=False)
    assert inverted.layout.yaxis.autorange == "reversed"
    assert normal.layout.yaxis.autorange != "reversed"


def test_plot_cp_vs_xc_is_markers_only():
    df = pd.DataFrame({"Z (m)": [10.0], "X (m)": [0.0], "Pressure Coefficient": [-1.0]})
    fig = plot_cp_vs_xc(df, chord=2.714, z_stations=[10.0])
    assert fig.data[0].mode == "markers"


def test_plot_cp_vs_xc_has_gridlines_both_axes():
    df = pd.DataFrame({"Z (m)": [10.0], "X (m)": [0.0], "Pressure Coefficient": [-1.0]})
    fig = plot_cp_vs_xc(df, chord=2.714, z_stations=[10.0])
    assert fig.layout.yaxis.showgrid is True
    assert fig.layout.xaxis.showgrid is True


def test_plot_skin_sections_xy_one_trace_per_z_with_shared_colorbar():
    df = pd.DataFrame(
        {
            "Z (m)": [10.0, 10.0, 20.0, 20.0],
            "X (m)": [0.0, 1.0, 0.0, 1.0],
            "Y (m)": [0.0, 0.5, 0.0, 0.5],
            "Pressure Coefficient": [-1.0, 0.5, -2.0, 0.3],
        }
    )
    fig = plot_skin_sections_xy(df, z_stations=[10.0, 20.0])
    assert len(fig.data) == 2
    # only the first trace carries the shared colorbar
    assert fig.data[0].marker.showscale is True
    assert fig.data[1].marker.showscale in (None, False)
    # color range is shared (global min/max), not per-trace
    assert fig.data[0].marker.cmin == fig.data[1].marker.cmin == -2.0
    assert fig.data[0].marker.cmax == fig.data[1].marker.cmax == 0.5


def test_plot_skin_sections_xy_has_gridlines_both_axes():
    df = pd.DataFrame(
        {"Z (m)": [10.0], "X (m)": [0.0], "Y (m)": [0.0], "Pressure Coefficient": [-1.0]}
    )
    fig = plot_skin_sections_xy(df, z_stations=[10.0])
    assert fig.layout.yaxis.showgrid is True
    assert fig.layout.xaxis.showgrid is True
