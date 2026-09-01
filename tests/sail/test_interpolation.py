"""Tests for src.sail.interpolation.interpolate_performance_envelope (ported from
esail-postprocessor's src/utils/interpolation.py, adapted to lowercase column names)."""

import numpy as np
import pandas as pd
import pytest

from src.sail.interpolation import interpolate_performance_envelope


def _sample_clmax_df():
    # 2 traces (P1/P2), each with points at non-uniform AoA spacing across "RPM"
    return pd.DataFrame(
        {
            "project_name": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "aws": [20.0] * 6,
            "rpm": [500.0, 800.0, 1450.0, 500.0, 800.0, 1450.0],
            "aoa": [20.0, 47.0, 67.0, 22.0, 45.0, 66.0],
            "cl": [2.4, 5.3, 6.8, 2.6, 5.1, 6.6],
            "cpow": [0.03, 0.10, 0.68, 0.03, 0.10, 0.65],
        }
    )


def test_interpolate_lands_exactly_on_integer_grid_within_range():
    df = _sample_clmax_df()
    result, warnings = interpolate_performance_envelope(df, group_cols=["project_name", "aws"], x_col="aoa", step=1.0)
    assert warnings == []
    p1 = result[result["project_name"] == "P1"]
    assert p1["aoa"].min() == pytest.approx(20.0)
    assert p1["aoa"].max() == pytest.approx(67.0)
    # every point is on an exact integer -- no extrapolation, no fractional drift
    assert (p1["aoa"] == p1["aoa"].round(0)).all()
    assert len(p1) == 48  # 20..67 inclusive, step 1


def test_interpolate_never_extrapolates_past_real_data():
    df = _sample_clmax_df()
    result, _ = interpolate_performance_envelope(df, group_cols=["project_name", "aws"], x_col="aoa", step=1.0)
    for (_, _), group in result.groupby(["project_name", "aws"]):
        assert group["aoa"].min() >= 20.0
        assert group["aoa"].max() <= 67.0


def test_interpolate_snaps_non_integer_range_endpoints_inward():
    # range [20.0, 52.5] with step=1.0 -> grid should stop at 52.0, not extrapolate to 53.0
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 3,
            "aws": [20.0] * 3,
            "aoa": [20.0, 35.0, 52.5],
            "cl": [2.0, 4.0, 5.5],
        }
    )
    result, _ = interpolate_performance_envelope(df, group_cols=["project_name", "aws"], x_col="aoa", step=1.0)
    assert result["aoa"].max() == pytest.approx(52.0)


def test_interpolate_one_trace_per_group():
    df = _sample_clmax_df()
    result, _ = interpolate_performance_envelope(df, group_cols=["project_name", "aws"], x_col="aoa", step=1.0)
    assert set(result["project_name"].unique()) == {"P1", "P2"}


def test_interpolate_skips_trace_with_fewer_than_two_points():
    df = pd.DataFrame(
        {"project_name": ["P1", "P2", "P2"], "aws": [20.0, 20.0, 20.0], "aoa": [20.0, 20.0, 40.0], "cl": [2.0, 2.0, 4.0]}
    )
    result, warnings = interpolate_performance_envelope(df, group_cols=["project_name", "aws"], x_col="aoa", step=1.0)
    assert "P1" not in set(result["project_name"].unique())
    assert any("fewer than 2" in w for w in warnings)


def test_interpolate_resolves_duplicate_aoa_via_cpow_shift():
    # two rows share aoa=40 within the same trace -- the higher-cpow one gets shifted
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 3,
            "aws": [20.0] * 3,
            "aoa": [20.0, 40.0, 40.0],
            "cl": [2.0, 4.0, 4.5],
            "cpow": [0.02, 0.05, 0.10],
        }
    )
    result, warnings = interpolate_performance_envelope(df, group_cols=["project_name", "aws"], x_col="aoa", step=1.0)
    assert any("Shifted highest-cpow duplicate" in w for w in warnings)
    assert not result.empty


def test_interpolate_empty_input_returns_empty_with_no_warnings():
    result, warnings = interpolate_performance_envelope(pd.DataFrame(), group_cols=["project_name"], x_col="aoa")
    assert result.empty
    assert warnings == []


def test_interpolate_missing_x_col_returns_empty():
    df = pd.DataFrame({"project_name": ["P1"], "cl": [2.0]})
    result, warnings = interpolate_performance_envelope(df, group_cols=["project_name"], x_col="aoa")
    assert result.empty


def test_interpolate_step_controls_grid_density():
    df = _sample_clmax_df()
    coarse, _ = interpolate_performance_envelope(df, group_cols=["project_name", "aws"], x_col="aoa", step=5.0)
    fine, _ = interpolate_performance_envelope(df, group_cols=["project_name", "aws"], x_col="aoa", step=1.0)
    assert len(fine) > len(coarse)
