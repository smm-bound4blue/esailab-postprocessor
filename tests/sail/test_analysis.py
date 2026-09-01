"""Tests for app.sail.analysis.compute_clmax_data."""

import pandas as pd
import pytest

from app.sail.analysis import compute_clmax_data


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


def test_compute_clmax_data_full_polar_min_rpm_expands_min_rpm_only():
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 6,
            "aws": [20.0] * 6,
            "rpm": [500.0, 500.0, 500.0, 800.0, 800.0, 800.0],
            "aoa": [0.0, 5.0, 10.0, 0.0, 5.0, 10.0],
            "cl": [1.0, 1.5, 2.0, 1.5, 1.2, 0.8],  # min RPM=500 CLmax at aoa=10; RPM=800 CLmax at aoa=0
        }
    )
    result = compute_clmax_data(df, full_polar_min_rpm=True)

    # min RPM (500) is expanded to every AoA point up to its CLmax (aoa<=10 -> all 3 rows)
    rows_500 = result[result["rpm"] == 500.0].sort_values("aoa")
    assert list(rows_500["aoa"]) == [0.0, 5.0, 10.0]

    # the other RPM (800) still collapses to just its single CLmax row
    rows_800 = result[result["rpm"] == 800.0]
    assert len(rows_800) == 1
    assert rows_800.iloc[0]["aoa"] == 0.0 and rows_800.iloc[0]["cl"] == 1.5


def test_compute_clmax_data_full_polar_min_rpm_per_trace_not_global():
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
    result = compute_clmax_data(df, full_polar_min_rpm=True)

    aws15 = result[result["aws"] == 15.0]
    assert set(aws15[aws15["rpm"] == 600.0]["aoa"]) == {0.0, 8.0}
    assert len(aws15[aws15["rpm"] == 900.0]) == 1

    aws20 = result[result["aws"] == 20.0]
    assert set(aws20[aws20["rpm"] == 500.0]["aoa"]) == {0.0, 8.0}
    assert len(aws20[aws20["rpm"] == 800.0]) == 1


def test_compute_clmax_data_full_polar_min_rpm_false_matches_default():
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
        compute_clmax_data(df, full_polar_min_rpm=False), compute_clmax_data(df)
    )
