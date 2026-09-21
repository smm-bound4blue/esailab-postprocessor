"""Tests for app.reference: wind probability matrix loading, 3D bar mesh builder, and wind rose."""

import base64

import numpy as np
import pandas as pd
import pytest

from app.reference.data import load_wind_probability_matrix
from app.reference.plotting import (
    REFERENCE_HEIGHT_M,
    _filter_by_twa_range,
    _group_speed_bins,
    _speed_bin_edges,
    _twa_range_label,
    build_wind_rose_figure,
    build_wind_speed_probability_figure,
    compute_wind_speed_stats,
    make_3d_bars,
    power_law_scale_factor,
    twa_range_coverage_pct,
    wind_speed_pdf_cdf,
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


def test_twa_range_coverage_pct_full_range_is_100():
    assert twa_range_coverage_pct(None) == pytest.approx(100.0)


def test_twa_range_coverage_pct_matches_manual_filter_sum():
    matrix = load_wind_probability_matrix()
    twa = matrix.index.to_numpy(dtype=float)
    expected = matrix.loc[(twa >= 30) & (twa <= 60)].to_numpy().sum() * 100
    assert twa_range_coverage_pct((30, 60)) == pytest.approx(expected)


def test_twa_range_coverage_pct_narrower_range_is_smaller():
    assert twa_range_coverage_pct((30, 60)) < twa_range_coverage_pct((0, 180))


def test_filter_by_twa_range_wrap_selects_both_edges_not_the_middle():
    # From=170 > To=-170 wraps through +-180deg: keep 170<=twa<=180 and -180<=twa<=-170,
    # drop everything strictly in between (e.g. 0deg)
    matrix = load_wind_probability_matrix()
    filtered = _filter_by_twa_range(matrix, (170, -170))
    twa = filtered.index.to_numpy(dtype=float)
    assert len(twa) > 0
    assert ((twa >= 170) | (twa <= -170)).all()
    assert 0.0 not in twa


def test_twa_range_coverage_pct_wrap_matches_manual_or_filter():
    matrix = load_wind_probability_matrix()
    twa = matrix.index.to_numpy(dtype=float)
    expected = matrix.loc[(twa >= 170) | (twa <= -170)].to_numpy().sum() * 100
    assert twa_range_coverage_pct((170, -170)) == pytest.approx(expected)


def test_twa_range_label_notes_wrap_only_when_from_greater_than_to():
    assert "(wraps" not in _twa_range_label((30, 60))
    assert "(wraps" in _twa_range_label((170, -170))
    assert _twa_range_label(None) == ""


def test_build_wind_rose_figure_twa_range_drops_outside_sectors():
    fig = build_wind_rose_figure(twa_range=(30, 60))
    for trace in fig.data:
        theta = _trace_values(trace.theta)
        assert theta.min() >= 30 and theta.max() <= 60


def test_build_wind_rose_figure_twa_range_total_matches_coverage():
    twa_range = (30, 60)
    fig = build_wind_rose_figure(twa_range=twa_range)
    total_pct = sum(_trace_values(trace.r).sum() for trace in fig.data)
    assert total_pct == pytest.approx(twa_range_coverage_pct(twa_range), abs=1e-3)


def test_build_wind_rose_figure_wrap_range_keeps_only_edge_sectors():
    fig = build_wind_rose_figure(twa_range=(170, -170))
    any_sector_seen = False
    for trace in fig.data:
        theta = _trace_values(trace.theta)
        if len(theta) == 0:
            continue
        any_sector_seen = True
        assert ((theta >= 170) | (theta <= -170)).all()
    assert any_sector_seen


def test_build_wind_speed_probability_figure_twa_range_caps_below_full_total():
    twa_range = (30, 60)
    fig = build_wind_speed_probability_figure(twa_range=twa_range)
    cdf = _trace_values(fig.data[1].y)
    assert cdf[-1] == pytest.approx(twa_range_coverage_pct(twa_range), abs=1e-3)
    assert cdf[-1] < 100.0  # not renormalized -- a real slice of a smaller total


def test_wind_speed_pdf_cdf_matches_figure_traces():
    tws, pdf, cdf = wind_speed_pdf_cdf(height_m=20.0, alpha=1 / 7, twa_range=(30, 60))
    fig = build_wind_speed_probability_figure(height_m=20.0, alpha=1 / 7, twa_range=(30, 60))
    np.testing.assert_allclose(tws, _trace_values(fig.data[0].x))
    np.testing.assert_allclose(pdf, _trace_values(fig.data[0].y))
    np.testing.assert_allclose(cdf, _trace_values(fig.data[1].y))


def test_compute_wind_speed_stats_mode_and_peak_probability():
    stats = compute_wind_speed_stats()
    row = stats.set_index("Statistic")
    tws, pdf, _ = wind_speed_pdf_cdf()
    mode_idx = np.argmax(pdf)
    assert row.loc["Most probable TWS (mode)", "Value"] == f"{tws[mode_idx]:.2f} kts"
    assert row.loc["Peak probability", "Value"] == f"{pdf[mode_idx]:.2f} %"


def test_compute_wind_speed_stats_half_max_range_brackets_the_mode():
    stats = compute_wind_speed_stats()
    row = stats.set_index("Statistic")
    tws, pdf, _ = wind_speed_pdf_cdf()
    mode_tws = tws[np.argmax(pdf)]
    lo_str, hi_str = row.loc["TWS range at ≥50% of peak probability", "Value"].replace(" kts", "").split(" – ")
    assert float(lo_str) <= mode_tws <= float(hi_str)


def test_compute_wind_speed_stats_percentiles_are_nondecreasing():
    stats = compute_wind_speed_stats()
    row = stats.set_index("Statistic")
    values = [
        float(row.loc[f"TWS at {p}% cumulative probability", "Value"].replace(" kts", ""))
        for p in (10, 25, 50, 75, 90)
    ]
    assert values == sorted(values)


def test_compute_wind_speed_stats_unreached_percentile_reports_coverage_not_a_bogus_value():
    # a narrow TWA slice whose CDF caps well below 90% must not silently report the
    # last TWS bin as if it were a real 90th-percentile answer
    stats = compute_wind_speed_stats(twa_range=(30, 60))
    row = stats.set_index("Statistic")
    total_pct = twa_range_coverage_pct((30, 60))
    assert total_pct < 90
    value = row.loc["TWS at 90% cumulative probability", "Value"]
    assert "not reached" in value
    assert f"{total_pct:.1f}%" in value


def test_compute_wind_speed_stats_responds_to_twa_range():
    full = compute_wind_speed_stats()
    narrow = compute_wind_speed_stats(twa_range=(30, 60))
    full_row = full.set_index("Statistic")
    narrow_row = narrow.set_index("Statistic")
    assert full_row.loc["Mean TWS", "Value"] != narrow_row.loc["Mean TWS", "Value"]


def test_compute_wind_speed_stats_total_coverage_matches_twa_range_coverage_pct():
    twa_range = (30, 60)
    stats = compute_wind_speed_stats(twa_range=twa_range)
    row = stats.set_index("Statistic")
    reported = float(row.loc["Total coverage (cumulative probability)", "Value"].replace(" %", ""))
    assert reported == pytest.approx(twa_range_coverage_pct(twa_range), abs=0.05)


def test_compute_wind_speed_stats_empty_selection_returns_empty_frame():
    # (1, 2) falls strictly between two 5deg-spaced bin centers -- no TWA bin lands
    # inside it, so the selection is genuinely empty and must not crash
    tws, pdf, _ = wind_speed_pdf_cdf(twa_range=(1, 2))
    assert pdf.sum() == 0  # sanity-check the fixture is actually empty
    stats = compute_wind_speed_stats(twa_range=(1, 2))
    assert stats.empty
    assert list(stats.columns) == ["Statistic", "Value"]


def test_build_wind_speed_probability_figure_scales_x_axis_only():
    fig_ref = build_wind_speed_probability_figure(height_m=REFERENCE_HEIGHT_M, alpha=1 / 7)
    fig_20m = build_wind_speed_probability_figure(height_m=20.0, alpha=1 / 7)

    x_ref = _trace_values(fig_ref.data[0].x)
    x_20m = _trace_values(fig_20m.data[0].x)
    expected_scale = power_law_scale_factor(20.0, alpha=1 / 7)
    np.testing.assert_allclose(x_20m, x_ref * expected_scale)

    # probabilities themselves are untouched by the height rescale
    np.testing.assert_allclose(_trace_values(fig_20m.data[0].y), _trace_values(fig_ref.data[0].y))
