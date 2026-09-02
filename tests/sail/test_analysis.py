"""Tests for app.sail.analysis.compute_clmax_data."""

import pandas as pd
import pytest

from app.sail.analysis import compute_clmax_data, enforce_cl_monotonic


def test_compute_clmax_data_picks_max_cl_row_per_case():
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 4,
            "aws": [20.0] * 4,
            "rpm": [500.0, 500.0, 800.0, 800.0],
            "aoa": [0.0, 10.0, 0.0, 10.0],
            "cl": [1.0, 2.0, 1.5, 1.2],
            "cd": [0.1, 0.4, 0.2, 0.3],
        }
    )
    result = compute_clmax_data(df)
    assert len(result) == 2  # one row per (project, aws, rpm)
    row_500 = result[result["rpm"] == 500.0].iloc[0]
    row_800 = result[result["rpm"] == 800.0].iloc[0]
    assert row_500["aoa"] == 10.0 and row_500["cl"] == 2.0
    assert row_800["aoa"] == 0.0 and row_800["cl"] == 1.5
    # every other column comes along with the winning row, not just cl/aoa
    assert row_500["cd"] == 0.4


def test_compute_clmax_data_keeps_aws_distinct():
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 4,
            "aws": [15.0, 15.0, 20.0, 20.0],
            "rpm": [500.0] * 4,
            "aoa": [0.0, 10.0, 0.0, 10.0],
            "cl": [1.0, 3.0, 2.0, 1.5],
        }
    )
    result = compute_clmax_data(df)
    assert len(result) == 2
    row_15 = result[result["aws"] == 15.0].iloc[0]
    row_20 = result[result["aws"] == 20.0].iloc[0]
    assert row_15["cl"] == 3.0
    assert row_20["cl"] == 2.0


def test_compute_clmax_data_empty_input():
    assert compute_clmax_data(pd.DataFrame()).empty


def test_compute_clmax_data_full_polar_traces_expands_min_rpm_only():
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 6,
            "aws": [20.0] * 6,
            "rpm": [500.0, 500.0, 500.0, 800.0, 800.0, 800.0],
            "aoa": [0.0, 5.0, 10.0, 0.0, 5.0, 10.0],
            "cl": [1.0, 1.5, 2.0, 1.5, 1.2, 0.8],  # min RPM=500 CLmax at aoa=10; RPM=800 CLmax at aoa=0
        }
    )
    result = compute_clmax_data(df, full_polar_traces=[("P1", 20.0)])

    # min RPM (500) is expanded to every AoA point up to its CLmax (aoa<=10 -> all 3 rows)
    rows_500 = result[result["rpm"] == 500.0].sort_values("aoa")
    assert list(rows_500["aoa"]) == [0.0, 5.0, 10.0]

    # the other RPM (800) still collapses to just its single CLmax row
    rows_800 = result[result["rpm"] == 800.0]
    assert len(rows_800) == 1
    assert rows_800.iloc[0]["aoa"] == 0.0 and rows_800.iloc[0]["cl"] == 1.5


def test_compute_clmax_data_full_polar_traces_per_trace_not_global():
    # min RPM differs per (project, aws) trace -- expansion must use each
    # trace's own minimum, not one global minimum across all traces.
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 4 + ["P1"] * 4,
            "aws": [15.0] * 4 + [20.0] * 4,
            "rpm": [600.0, 600.0, 900.0, 900.0] + [500.0, 500.0, 800.0, 800.0],
            "aoa": [0.0, 8.0, 0.0, 8.0] * 2,
            "cl": [1.0, 2.0, 1.5, 1.2] * 2,
        }
    )
    result = compute_clmax_data(df, full_polar_traces=[("P1", 15.0), ("P1", 20.0)])

    aws15 = result[result["aws"] == 15.0]
    assert set(aws15[aws15["rpm"] == 600.0]["aoa"]) == {0.0, 8.0}
    assert len(aws15[aws15["rpm"] == 900.0]) == 1

    aws20 = result[result["aws"] == 20.0]
    assert set(aws20[aws20["rpm"] == 500.0]["aoa"]) == {0.0, 8.0}
    assert len(aws20[aws20["rpm"] == 800.0]) == 1


def test_compute_clmax_data_full_polar_traces_only_affects_selected_trace():
    # the core new behavior: two traces both have a min RPM eligible for
    # expansion, but only the one named in full_polar_traces actually expands
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 4 + ["P1"] * 4,
            "aws": [15.0] * 4 + [20.0] * 4,
            "rpm": [600.0, 600.0, 900.0, 900.0] + [500.0, 500.0, 800.0, 800.0],
            "aoa": [0.0, 8.0, 0.0, 8.0] * 2,
            "cl": [1.0, 2.0, 1.5, 1.2] * 2,
        }
    )
    result = compute_clmax_data(df, full_polar_traces=[("P1", 15.0)])

    # aws=15's min RPM (600) expanded
    aws15 = result[result["aws"] == 15.0]
    assert set(aws15[aws15["rpm"] == 600.0]["aoa"]) == {0.0, 8.0}

    # aws=20 not in full_polar_traces -- every RPM stays a single CLmax point,
    # including its own min RPM (500), which would otherwise have expanded too
    aws20 = result[result["aws"] == 20.0]
    assert len(aws20[aws20["rpm"] == 500.0]) == 1
    assert len(aws20[aws20["rpm"] == 800.0]) == 1


def test_compute_clmax_data_full_polar_traces_empty_matches_default():
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 4,
            "aws": [20.0] * 4,
            "rpm": [500.0, 500.0, 800.0, 800.0],
            "aoa": [0.0, 10.0, 0.0, 10.0],
            "cl": [1.0, 2.0, 1.5, 1.2],
        }
    )
    pd.testing.assert_frame_equal(
        compute_clmax_data(df, full_polar_traces=[]), compute_clmax_data(df)
    )
    pd.testing.assert_frame_equal(
        compute_clmax_data(df, full_polar_traces=None), compute_clmax_data(df)
    )


def test_enforce_cl_monotonic_drops_points_that_decrease_cl():
    # matches the user's reported case: AoA 52,58,66,72,77 -> CL 5.85,6.25,6.75,7.6,7.3
    # the last point (AoA=77, CL=7.3) doesn't beat the running max (7.6) and is dropped
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 5,
            "aws": [10.0] * 5,
            "rpm": [500.0, 600.0, 800.0, 1150.0, 1450.0],
            "aoa": [52.0, 58.0, 66.0, 72.0, 77.0],
            "cl": [5.85, 6.25, 6.75, 7.6, 7.3],
        }
    )
    result = enforce_cl_monotonic(df)
    assert list(result["aoa"]) == [52.0, 58.0, 66.0, 72.0]
    assert list(result["cl"]) == [5.85, 6.25, 6.75, 7.6]


def test_enforce_cl_monotonic_keeps_ties_and_drops_only_true_decreases():
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 4,
            "aws": [10.0] * 4,
            "rpm": [500.0, 600.0, 800.0, 1150.0],
            "aoa": [0.0, 10.0, 20.0, 30.0],
            "cl": [1.0, 1.0, 0.9, 1.5],  # tie at 10 kept, dip at 20 dropped, new max at 30 kept
        }
    )
    result = enforce_cl_monotonic(df)
    assert list(result["aoa"]) == [0.0, 10.0, 30.0]
    assert list(result["cl"]) == [1.0, 1.0, 1.5]


def test_enforce_cl_monotonic_per_trace_independent():
    # a decrease in one (project, aws) trace must not affect another trace
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 4,
            "aws": [10.0, 10.0, 20.0, 20.0],
            "rpm": [500.0, 600.0, 500.0, 600.0],
            "aoa": [0.0, 10.0, 0.0, 10.0],
            "cl": [2.0, 1.0, 1.0, 2.0],  # aws=10 decreases, aws=20 increases
        }
    )
    result = enforce_cl_monotonic(df)
    aws10 = result[result["aws"] == 10.0]
    aws20 = result[result["aws"] == 20.0]
    assert list(aws10["aoa"]) == [0.0]
    assert list(aws20["aoa"]) == [0.0, 10.0]


def test_enforce_cl_monotonic_empty_input():
    assert enforce_cl_monotonic(pd.DataFrame()).empty


def test_enforce_cl_monotonic_group_cols_merges_across_aws():
    # two AWS traces with overlapping AoA and conflicting CL: at AoA=10 the aws=20
    # point (cl=1.6) beats aws=10's running max (1.0), so it survives; at AoA=20
    # aws=10's point (cl=1.2) does NOT beat the aws=20 point already seen at a lower
    # AoA (1.6), so it's dropped even though it doesn't conflict within its own trace.
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 4,
            "aws": [10.0, 10.0, 20.0, 20.0],
            "rpm": [500.0, 600.0, 700.0, 800.0],
            "aoa": [0.0, 20.0, 10.0, 30.0],
            "cl": [1.0, 1.2, 1.6, 1.8],
        }
    )
    result = enforce_cl_monotonic(df, group_cols=["project_name"])
    assert list(result["aoa"]) == [0.0, 10.0, 30.0]
    assert list(result["cl"]) == [1.0, 1.6, 1.8]
    assert list(result["aws"]) == [10.0, 20.0, 20.0]


def test_enforce_cl_monotonic_default_group_cols_unchanged():
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 2,
            "aws": [10.0] * 2,
            "rpm": [500.0, 600.0],
            "aoa": [0.0, 10.0],
            "cl": [1.0, 0.5],
        }
    )
    pd.testing.assert_frame_equal(enforce_cl_monotonic(df), enforce_cl_monotonic(df, group_cols=None))
