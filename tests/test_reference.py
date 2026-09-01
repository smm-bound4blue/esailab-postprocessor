"""Tests for app.reference: wind probability matrix loading, 3D bar mesh builder, and wind rose."""

import base64

import numpy as np
import pandas as pd
import pytest

from app.reference.data import load_wind_probability_matrix
from app.reference.plotting import (
    REFERENCE_HEIGHT_M,
    _group_speed_bins,
    _speed_bin_edges,
    build_wind_rose_figure,
    build_wind_speed_probability_figure,
    make_3d_bars,
    power_law_scale_factor,
)


def _trace_values(array_like) -> np.ndarray:
    """Plotly sometimes stores a trace's array as a compact {'dtype','bdata'}
    dict instead of a plain sequence (an internal optimization that can kick
    in on a cached/shared Figure once any trace's array is touched) -- decode
    that form so tests don't depend on which representation happened to be
    active."""
    if isinstance(array_like, dict) and "bdata" in array_like:
        return np.frombuffer(base64.b64decode(array_like["bdata"]), dtype=array_like["dtype"])
    return np.asarray(array_like)


def test_load_wind_probability_matrix_sums_to_one():
    matrix = load_wind_probability_matrix()
    assert matrix.to_numpy().sum() == pytest.approx(1.0, abs=1e-6)


def test_load_wind_probability_matrix_shape():
    matrix = load_wind_probability_matrix()
    assert matrix.shape == (72, 31)  # 5deg TWA bins x 1kt TWS bins


def test_make_3d_bars_vertex_and_triangle_counts():
    x = np.array([0.0, 5.0])
    y = np.array([0.0, 1.0])
    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    mesh = make_3d_bars(x, y, z, dx=5.0, dy=1.0)

    n_bars = 4
    assert len(mesh.x) == n_bars * 8
    assert len(mesh.i) == n_bars * 12
    assert max(mesh.z) == pytest.approx(4.0)
    assert min(mesh.z) == pytest.approx(0.0)


def test_group_speed_bins_preserves_total_probability():
    matrix = pd.DataFrame(
        {1.0: [0.1, 0.2], 6.0: [0.05, 0.05], 11.0: [0.3, 0.3]},
        index=[0.0, 5.0],
    )
    grouped = _group_speed_bins(matrix, edges=(0, 5, 10, 15))
    assert list(grouped.columns) == ["0–5 kts", "5–10 kts", "10–15 kts"]
    pd.testing.assert_series_equal(
        grouped.sum(axis=1), matrix.sum(axis=1), check_names=False
    )


def test_group_speed_bins_assigns_each_column_to_exactly_one_band():
    matrix = pd.DataFrame({4.9: [1.0], 5.0: [1.0]}, index=[0.0])
    grouped = _group_speed_bins(matrix, edges=(0, 5, 10))
    assert grouped["0–5 kts"].iloc[0] == pytest.approx(1.0)
    assert grouped["5–10 kts"].iloc[0] == pytest.approx(1.0)


def test_speed_bin_edges_last_bin_absorbs_remainder():
    edges = _speed_bin_edges(0, 31, width=5)
    assert edges == [0, 5, 10, 15, 20, 25, 31]


def test_speed_bin_edges_exact_division_has_no_remainder_bin():
    edges = _speed_bin_edges(0, 30, width=5)
    assert edges == [0, 5, 10, 15, 20, 25, 30]


def test_speed_bin_edges_rejects_width_wider_than_range():
    with pytest.raises(ValueError):
        _speed_bin_edges(0, 31, width=40)


def test_speed_bin_edges_rejects_nonpositive_width():
    with pytest.raises(ValueError):
        _speed_bin_edges(0, 31, width=0)


@pytest.mark.parametrize("speed_bin_width", [1, 2, 5, 10, 15])
def test_build_wind_rose_figure_preserves_total_probability(speed_bin_width):
    fig = build_wind_rose_figure(speed_bin_width=speed_bin_width)
    total_pct = sum(_trace_values(trace.r).sum() for trace in fig.data)
    assert total_pct == pytest.approx(100.0, abs=1e-3)


def test_build_wind_speed_probability_figure_pdf_sums_to_100():
    fig = build_wind_speed_probability_figure()
    pdf_trace = fig.data[0]
    assert pdf_trace.name == "Probability"
    assert _trace_values(pdf_trace.y).sum() == pytest.approx(100.0, abs=1e-3)


def test_build_wind_speed_probability_figure_cdf_is_monotonic_and_ends_at_100():
    fig = build_wind_speed_probability_figure()
    cdf_trace = fig.data[1]
    assert cdf_trace.name == "Cumulative Probability"
    cdf = _trace_values(cdf_trace.y)
    assert list(cdf) == sorted(cdf)  # monotonically non-decreasing
    assert cdf[-1] == pytest.approx(100.0, abs=1e-3)


def test_power_law_scale_factor_at_reference_height_is_one():
    assert power_law_scale_factor(REFERENCE_HEIGHT_M, alpha=1 / 7) == pytest.approx(1.0)
    assert power_law_scale_factor(REFERENCE_HEIGHT_M, alpha=1 / 9) == pytest.approx(1.0)


def test_power_law_scale_factor_at_20m_matches_hand_calc():
    # V(20) = V(10) * (20/10)^(1/7)
    assert power_law_scale_factor(20.0, alpha=1 / 7) == pytest.approx(2.0 ** (1 / 7))


def test_build_wind_speed_probability_figure_scales_x_axis_only():
    fig_ref = build_wind_speed_probability_figure(height_m=REFERENCE_HEIGHT_M, alpha=1 / 7)
    fig_20m = build_wind_speed_probability_figure(height_m=20.0, alpha=1 / 7)

    x_ref = _trace_values(fig_ref.data[0].x)
    x_20m = _trace_values(fig_20m.data[0].x)
    expected_scale = power_law_scale_factor(20.0, alpha=1 / 7)
    np.testing.assert_allclose(x_20m, x_ref * expected_scale)

    # probabilities themselves are untouched by the height rescale
    np.testing.assert_allclose(_trace_values(fig_20m.data[0].y), _trace_values(fig_ref.data[0].y))
