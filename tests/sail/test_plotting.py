"""Tests for app.sail.plotting: plot_polar_curve (trace granularity, legend grouping,
display modes, error bars, stall highlighting), plot_fan_performance_curve, and
plot_flow_rate_parity."""

import numpy as np
import pandas as pd
import pytest

from app.sail.plotting import plot_fan_performance_curve, plot_flow_rate_parity, plot_polar_curve


def _sample_df():
    # 2 projects x 2 AWS x 2 RPM x 2 AoA = 16 rows -> 8 distinct (project,aws,rpm) cases
    rows = []
    for project in ["P1", "P2"]:
        for aws in [15.0, 20.0]:
            for rpm in [500.0, 800.0]:
                for aoa, cl in [(0.0, 1.0), (10.0, 2.0)]:
                    rows.append(
                        dict(
                            project_name=project, aws=aws, rpm=rpm, aoa=aoa, cl=cl,
                            cl_std=0.01, is_stall=(aoa == 10.0 and rpm == 800.0),
                        )
                    )
    return pd.DataFrame(rows)


def test_plot_polar_curve_default_trace_grain_is_one_per_case():
    fig = plot_polar_curve(_sample_df(), y="cl", x="aoa")
    # 8 cases (2 projects x 2 aws x 2 rpm), no stall highlight requested
    assert len(fig.data) == 8


def test_plot_polar_curve_never_mixes_two_aws_into_one_trace():
    df = _sample_df()
    fig = plot_polar_curve(df, y="cl", x="aoa")
    for trace in fig.data:
        # each trace's AoA values must be strictly increasing (sorted, no
        # duplicate AoA from a different AWS/RPM accidentally merged in)
        assert list(trace.x) == sorted(trace.x)
        assert len(set(trace.x)) == len(trace.x)


def test_plot_polar_curve_legend_group_by_project_clusters_correctly():
    fig = plot_polar_curve(_sample_df(), y="cl", x="aoa", legend_group_by=["project_name"])
    groups = {t.legendgroup for t in fig.data}
    assert groups == {"Project=P1", "Project=P2"}
    # each project has 4 cases (2 aws x 2 rpm) -> 4 traces share its legendgroup
    assert sum(1 for t in fig.data if t.legendgroup == "Project=P1") == 4


def test_plot_polar_curve_no_legend_group_by_leaves_legendgroup_unset():
    fig = plot_polar_curve(_sample_df(), y="cl", x="aoa")
    assert all(t.legendgroup in (None, "") for t in fig.data)


def test_plot_polar_curve_display_modes_control_mode_and_marker_size():
    df = _sample_df()
    lines_only = plot_polar_curve(df, y="cl", x="aoa", plot_mode="Only Lines")
    markers_only = plot_polar_curve(df, y="cl", x="aoa", plot_mode="Only Markers")
    both = plot_polar_curve(df, y="cl", x="aoa", plot_mode="Lines + Markers")

    assert lines_only.data[0].mode == "lines"
    assert markers_only.data[0].mode == "markers"
    assert both.data[0].mode == "lines+markers"


def test_plot_polar_curve_trace_by_controls_how_many_traces():
    # collapsing to (project, aws) -- e.g. for the Performance envelope,
    # where each trace should connect across RPM instead of splitting by it
    fig = plot_polar_curve(_sample_df(), y="cl", x="aoa", trace_by=["project_name", "aws"])
    assert len(fig.data) == 4


def test_plot_polar_curve_error_bars_only_when_std_column_present():
    df = _sample_df()
    fig_with = plot_polar_curve(df, y="cl", x="aoa", show_error_bars=True)
    assert all(t.error_y.array is not None for t in fig_with.data)

    fig_without_std_col = plot_polar_curve(df.drop(columns=["cl_std"]), y="cl", x="aoa", show_error_bars=True)
    assert all(t.error_y.array is None for t in fig_without_std_col.data)


def test_plot_polar_curve_highlight_stall_adds_single_stall_trace():
    fig = plot_polar_curve(_sample_df(), y="cl", x="aoa", highlight_stall=True)
    stall_traces = [t for t in fig.data if t.name == "Stall"]
    assert len(stall_traces) == 1
    assert len(stall_traces[0].x) == 4  # one stall row per (project,aws) x 2 aws x 2 project at rpm=800


def test_plot_polar_curve_no_stall_trace_when_no_stall_rows():
    df = _sample_df()
    df["is_stall"] = False
    fig = plot_polar_curve(df, y="cl", x="aoa", highlight_stall=True)
    assert not any(t.name == "Stall" for t in fig.data)


def _sample_fan_sim_df(rpms=(500.0, 500.0, 500.0, 500.0)):
    return pd.DataFrame(
        {
            "project_name": ["P1", "P1", "P2", "P2"],
            "aws": [20.0, 20.0, 20.0, 20.0],
            "rpm": list(rpms),
            "aoa": [0.0, 10.0, 0.0, 10.0],
            "fan_volumetric_flow": [5.0, 6.0, 4.5, 5.5],
            "fan_static_pressure": [800.0, 750.0, 820.0, 770.0],
            "fan_total_pressure": [900.0, 850.0, 920.0, 870.0],
        }
    )


def _sample_reference_curve():
    return pd.DataFrame(
        {
            "flowrate": [4.0, 5.0, 6.0, 7.0],
            "static_pressure": [900.0, 800.0, 700.0, 600.0],
            "total_pressure": [950.0, 850.0, 750.0, 650.0],
        }
    )


def test_plot_fan_performance_curve_includes_reference_lines():
    fig = plot_fan_performance_curve(_sample_fan_sim_df(), {500.0: _sample_reference_curve()})
    names = {t.name for t in fig.data}
    assert "Reference Fan Static Pressure (500 RPM)" in names
    assert "Reference Fan Total Pressure (500 RPM)" in names


def test_plot_fan_performance_curve_reference_lines_share_rpm_legend_group_with_sim():
    fig = plot_fan_performance_curve(_sample_fan_sim_df(), {500.0: _sample_reference_curve()})
    ref_traces = [t for t in fig.data if "Reference" in t.name]
    sim_traces = [t for t in fig.data if "sim" in t.name]
    assert ref_traces and sim_traces
    assert {t.legendgroup for t in ref_traces} == {"RPM=500"}
    assert {t.legendgroup for t in sim_traces} == {"RPM=500"}


def test_plot_fan_performance_curve_show_total_false_omits_total_traces():
    fig = plot_fan_performance_curve(_sample_fan_sim_df(), {500.0: _sample_reference_curve()}, show_total=False)
    names = {t.name for t in fig.data}
    assert not any("Total" in n for n in names)
    assert any("Static" in n for n in names)


def test_plot_fan_performance_curve_one_sim_trace_pair_per_project():
    fig = plot_fan_performance_curve(_sample_fan_sim_df(), {500.0: _sample_reference_curve()}, show_total=True)
    sim_names = {t.name for t in fig.data if "sim" in t.name}
    assert sim_names == {
        "Project=P1 · AWS (kts)=20 · RPM=500 — Static (sim)",
        "Project=P1 · AWS (kts)=20 · RPM=500 — Total (sim)",
        "Project=P2 · AWS (kts)=20 · RPM=500 — Static (sim)",
        "Project=P2 · AWS (kts)=20 · RPM=500 — Total (sim)",
    }


def test_plot_fan_performance_curve_x_axis_is_flow_rate():
    fig = plot_fan_performance_curve(_sample_fan_sim_df(), {500.0: _sample_reference_curve()})
    assert "Flow Rate" in fig.layout.xaxis.title.text


def test_plot_fan_performance_curve_sim_traces_are_markers_only_with_large_size():
    fig = plot_fan_performance_curve(_sample_fan_sim_df(), {500.0: _sample_reference_curve()})
    sim_traces = [t for t in fig.data if "sim" in t.name]
    assert sim_traces  # sanity: there are simulated traces to check
    for t in sim_traces:
        assert t.mode == "markers"
        assert t.marker.size >= 10


def test_plot_fan_performance_curve_aoa_hover_uses_one_decimal():
    fig = plot_fan_performance_curve(_sample_fan_sim_df(), {500.0: _sample_reference_curve()})
    sim_traces = [t for t in fig.data if "sim" in t.name]
    for t in sim_traces:
        assert "AoA=%{customdata:.1f}" in t.hovertemplate


def test_plot_fan_performance_curve_multi_rpm_has_one_reference_pair_per_rpm():
    reference_curves = {500.0: _sample_reference_curve(), 800.0: _sample_reference_curve()}
    fig = plot_fan_performance_curve(_sample_fan_sim_df(rpms=(500.0, 500.0, 800.0, 800.0)), reference_curves)
    ref_names = {t.name for t in fig.data if "Reference" in t.name}
    assert ref_names == {
        "Reference Fan Static Pressure (500 RPM)",
        "Reference Fan Total Pressure (500 RPM)",
        "Reference Fan Static Pressure (800 RPM)",
        "Reference Fan Total Pressure (800 RPM)",
    }


def test_plot_fan_performance_curve_multi_rpm_reference_lines_use_distinct_greys():
    reference_curves = {500.0: _sample_reference_curve(), 800.0: _sample_reference_curve()}
    fig = plot_fan_performance_curve(_sample_fan_sim_df(rpms=(500.0, 500.0, 800.0, 800.0)), reference_curves)
    static_traces = {t.name: t.line.color for t in fig.data if "Reference Fan Static" in t.name}
    colors = set(static_traces.values())
    assert len(colors) == 2  # each RPM's reference curve gets its own distinct color


def test_plot_fan_performance_curve_multi_rpm_sim_traces_split_by_rpm_too():
    fig = plot_fan_performance_curve(
        _sample_fan_sim_df(rpms=(500.0, 500.0, 800.0, 800.0)), {500.0: _sample_reference_curve(), 800.0: _sample_reference_curve()}
    )
    sim_names = {t.name for t in fig.data if "sim" in t.name}
    assert "Project=P1 · AWS (kts)=20 · RPM=500 — Static (sim)" in sim_names
    assert "Project=P2 · AWS (kts)=20 · RPM=800 — Static (sim)" in sim_names


def test_plot_fan_performance_curve_sim_traces_grouped_by_rpm():
    # 4 sim rows: (P1,500), (P1,500), (P2,800), (P2,800) -- 2 distinct RPMs
    fig = plot_fan_performance_curve(
        _sample_fan_sim_df(rpms=(500.0, 500.0, 800.0, 800.0)),
        {500.0: _sample_reference_curve(), 800.0: _sample_reference_curve()},
    )
    sim_traces = [t for t in fig.data if "sim" in t.name]
    groups = {t.legendgroup for t in sim_traces}
    assert groups == {"RPM=500", "RPM=800"}
    # every trace's legendgrouptitle matches its own legendgroup
    for t in sim_traces:
        assert t.legendgrouptitle.text == t.legendgroup


def test_plot_fan_performance_curve_reference_and_sim_share_groups_per_rpm():
    # every trace in the figure (reference lines and sim markers alike) should
    # belong to one of exactly the selected RPMs' groups -- nothing left ungrouped
    fig = plot_fan_performance_curve(
        _sample_fan_sim_df(rpms=(500.0, 500.0, 800.0, 800.0)),
        {500.0: _sample_reference_curve(), 800.0: _sample_reference_curve()},
    )
    all_groups = {t.legendgroup for t in fig.data}
    assert all_groups == {"RPM=500", "RPM=800"}


def test_plot_fan_performance_curve_groupclick_togglegroup():
    fig = plot_fan_performance_curve(_sample_fan_sim_df(), {500.0: _sample_reference_curve()})
    assert fig.layout.legend.groupclick == "togglegroup"


def _sample_parity_df():
    return pd.DataFrame(
        {
            "project_name": ["P1", "P1", "P2"],
            "aws": [20.0, 20.0, 20.0],
            "rpm": [500.0, 500.0, 500.0],
            "aoa": [0.0, 10.0, 0.0],
            "fan_volumetric_flow": [5.0, 6.0, 4.5],
            "estimated_flow_rate": [5.2, 6.3, np.nan],  # last row has no valid estimate
        }
    )


def test_plot_flow_rate_parity_includes_yx_reference_line():
    fig = plot_flow_rate_parity(_sample_parity_df())
    names = {t.name for t in fig.data}
    assert "y = x" in names


def test_plot_flow_rate_parity_drops_rows_without_estimate():
    fig = plot_flow_rate_parity(_sample_parity_df())
    data_traces = [t for t in fig.data if t.name != "y = x"]
    total_points = sum(len(t.x) for t in data_traces)
    assert total_points == 2  # the NaN row is dropped


def test_plot_flow_rate_parity_one_trace_per_project_aws_rpm():
    # P2's only row has no valid estimate (see _sample_parity_df), so it
    # contributes no trace at all once dropna() removes it -- only P1 remains.
    fig = plot_flow_rate_parity(_sample_parity_df())
    names = {t.name for t in fig.data if t.name != "y = x"}
    assert names == {"Project=P1 · AWS (kts)=20 · RPM=500"}


def test_plot_flow_rate_parity_empty_input_returns_empty_figure():
    fig = plot_flow_rate_parity(pd.DataFrame({"project_name": [], "aws": [], "rpm": [], "aoa": [], "fan_volumetric_flow": [], "estimated_flow_rate": []}))
    assert len(fig.data) == 0
