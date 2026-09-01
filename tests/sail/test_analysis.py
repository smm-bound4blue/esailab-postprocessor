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
