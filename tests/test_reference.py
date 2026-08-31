"""Tests for app.reference: wind probability matrix loading, 3D bar mesh builder, and wind rose."""

import numpy as np
import pandas as pd
import pytest

from app.reference.data import load_wind_probability_matrix
from app.reference.plotting import _group_speed_bins, _speed_bin_edges, build_wind_rose_figure, make_3d_bars


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
    total_pct = sum(trace.r.sum() for trace in fig.data)
    assert total_pct == pytest.approx(100.0, abs=1e-3)
