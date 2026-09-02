"""Tests for src.sail.interpolation.interpolate_performance_envelope (ported from
esail-postprocessor's src/utils/interpolation.py, adapted to lowercase column names)."""

import numpy as np
import pandas as pd
import pytest

from src.sail.interpolation import interpolate_performance_envelope, smooth_performance_envelope


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


# --- smooth_performance_envelope (bin-average + PCHIP) ----------------------------


def test_smooth_spans_the_pooled_range_not_per_source_trace():
    # two "traces" pooled into one group (as the Combined AWS Envelope tab does) --
    # the grid should span the pooled min/max, not either trace's own narrower range
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 3 + ["P1"] * 3,
            "aws": [5.0] * 3 + [20.0] * 3,
            "aoa": [10.0, 20.0, 30.0, 25.0, 35.0, 45.0],
            "cl": [2.0, 3.0, 4.0, 3.5, 4.5, 5.5],
        }
    )
    result, warnings = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=1.0)
    assert warnings == []
    assert result["aoa"].min() == pytest.approx(10.0)
    assert result["aoa"].max() == pytest.approx(45.0)


def test_smooth_never_extrapolates_past_pooled_range():
    df = pd.DataFrame(
        {"project_name": ["P1"] * 4, "aoa": [10.0, 20.0, 30.5, 40.0], "cl": [2.0, 3.0, 4.0, 5.0]}
    )
    result, _ = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=1.0)
    assert result["aoa"].min() >= 10.0
    assert result["aoa"].max() <= 40.0


def test_smooth_endpoints_match_true_min_max_even_with_wide_bins():
    # regression test: the two lowest AoA points (20, 25) are 5 degrees apart -- a
    # bin_width of 10 puts them in the same bin, whose mean x (22.5) would otherwise
    # become the fitted curve's starting AoA instead of the true minimum (20)
    df = pd.DataFrame(
        {"project_name": ["P1"] * 5, "aoa": [20.0, 25.0, 47.0, 70.0, 87.0], "cl": [1.0, 1.5, 3.0, 5.0, 6.0]}
    )
    result, _ = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=1.0, bin_width=10.0)
    assert result["aoa"].min() == pytest.approx(20.0)
    assert result["aoa"].max() == pytest.approx(87.0)
    # the endpoint's x is pinned to the true min, but its y is still the bin
    # average that x_col=20 got grouped into (here bin [20, 30) -> {20, 25})
    assert result.loc[result["aoa"] == 20.0, "cl"].iloc[0] == pytest.approx(1.25)


def test_smooth_tolerates_duplicate_aoa_across_pooled_traces():
    # a shared aoa value across two AWS traces must not raise or produce NaN -- this
    # is the whole point of fitting instead of interpolating for the combined tab
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 4,
            "aoa": [10.0, 20.0, 20.0, 30.0],
            "cl": [2.0, 3.0, 3.4, 4.0],
        }
    )
    result, warnings = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=1.0)
    assert not result.empty
    assert warnings == []
    assert not result["cl"].isna().any()


def test_smooth_handles_wide_dynamic_range_without_nan():
    # regression test: scipy.interpolate.UnivariateSpline (tried first) silently
    # returned all-NaN for real Cpow data spanning ~0.03 to ~40 (three orders of
    # magnitude) with a duplicate AoA across two AWS traces -- the exact shape that
    # broke it, reproduced here to guard against reintroducing that fragility
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 14,
            "aoa": [20.0, 25.0, 47.0, 52.0, 57.0, 61.0, 65.0, 66.0, 67.0, 71.0, 72.0, 74.0, 87.0, 87.0],
            "cpow": [0.032, 0.056, 0.131, 0.249, 0.416, 0.356, 1.834, 0.929, 0.684, 3.112, 5.217, 7.199, 41.575, 20.985],
        }
    )
    result, warnings = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=1.0)
    assert warnings == []
    assert not result["cpow"].isna().any()
    assert (result["cpow"] > -0.5).all()  # no wild negative overshoot


def test_smooth_skips_trace_with_fewer_than_two_points():
    df = pd.DataFrame({"project_name": ["P1", "P2", "P2"], "aoa": [10.0, 10.0, 20.0], "cl": [2.0, 2.0, 3.0]})
    result, warnings = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=1.0)
    assert "P1" not in set(result["project_name"].unique())
    assert any("fewer than 2" in w for w in warnings)


def test_smooth_empty_input_returns_empty_with_no_warnings():
    result, warnings = smooth_performance_envelope(pd.DataFrame(), group_cols=["project_name"], x_col="aoa")
    assert result.empty
    assert warnings == []


def test_smooth_missing_x_col_returns_empty():
    df = pd.DataFrame({"project_name": ["P1"], "cl": [2.0]})
    result, warnings = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa")
    assert result.empty


def test_smooth_wide_bin_blends_a_local_dip_into_its_neighbors():
    # a deliberately noisy point at aoa=30 (a dip amid an otherwise rising trend) --
    # a wide bin_width merges it into the same bin as aoa=40, averaging the dip away
    df = pd.DataFrame(
        {"project_name": ["P1"] * 6, "aoa": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0], "cl": [1.0, 2.0, 3.0, 2.5, 5.0, 6.0]}
    )
    wide, _ = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=10.0, bin_width=15.0)
    fitted_at_30 = wide.loc[wide["aoa"] == 30.0, "cl"].iloc[0]
    assert fitted_at_30 != pytest.approx(2.5, abs=0.1)
    assert fitted_at_30 > 2.5  # pulled up toward the surrounding rising trend

    # a narrow bin_width (far smaller than the 10-unit point spacing) puts every
    # point in its own bin, so PCHIP interpolates through the exact raw values --
    # evaluated exactly at a raw point, it reproduces that point exactly
    tight, _ = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=10.0, bin_width=0.5)
    tight_at_30 = tight.loc[tight["aoa"] == 30.0, "cl"].iloc[0]
    assert tight_at_30 == pytest.approx(2.5, abs=1e-6)


def test_smooth_never_overshoots_beyond_binned_points_range():
    # regression test for the failure mode that motivated switching away from LOESS:
    # a real gap in the data (22 degrees, wider than a 5-degree bin) adjacent to a
    # steep-rising pair must not pull the fit below the gap's real neighbors, since
    # PCHIP through bin averages can't invent a new extremum the bins don't show
    df = pd.DataFrame(
        {
            "project_name": ["P1"] * 6,
            "aoa": [20.0, 25.0, 47.0, 52.0, 57.0, 61.0],
            "cpow": [0.032, 0.056, 0.131, 0.249, 0.416, 0.356],
        }
    )
    result, _ = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=1.0, bin_width=5.0)
    gap = result[(result["aoa"] >= 25.0) & (result["aoa"] <= 47.0)]
    assert gap["cpow"].min() >= 0.056 - 1e-9


def test_smooth_step_controls_grid_density():
    df = pd.DataFrame(
        {"project_name": ["P1"] * 5, "aoa": [0.0, 10.0, 20.0, 30.0, 40.0], "cl": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    coarse, _ = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=10.0)
    fine, _ = smooth_performance_envelope(df, group_cols=["project_name"], x_col="aoa", step=1.0)
    assert len(fine) > len(coarse)
