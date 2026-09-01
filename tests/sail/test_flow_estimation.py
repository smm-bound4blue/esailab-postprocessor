"""Tests for src.sail.flow_estimation: ellipse area, the Bernoulli flow-rate formula,
and P1/P2 extraction from spatial tables."""

import math

import pandas as pd
import pytest

from src.sail.flow_estimation import (
    ellipse_area,
    estimate_flow_rate,
    estimate_flow_rates,
    extract_p1_plenum_pressure,
    extract_p2_fan_duct_pressure,
)


def test_ellipse_area_hand_calc():
    # chord=2.714, thickness_ratio=0.66 -> A1 = pi * 2.714 * (0.66*2.714) / 4
    chord = 2.714
    expected = math.pi * chord * (0.66 * chord) / 4.0
    assert ellipse_area(chord, thickness_ratio=0.66) == pytest.approx(expected)


def test_estimate_flow_rate_hand_calc():
    # Q = N*A2*sqrt(2*(p1-p2) / (rho*[(1+xi) - (N*A2/A1)^2]))
    p1, p2, rho, a1, a2, xi, n = 1000.0, 900.0, 1.2, 3.8, 1.2, 0.43, 1
    expected = n * a2 * math.sqrt(2 * (p1 - p2) / (rho * ((1 + xi) - (n * a2 / a1) ** 2)))
    assert estimate_flow_rate(p1, p2, rho, a1, a2, xi, n) == pytest.approx(expected)


def test_estimate_flow_rate_nan_when_p1_not_greater_than_p2():
    assert math.isnan(estimate_flow_rate(900.0, 900.0, 1.2, 3.8, 1.2, 0.43))
    assert math.isnan(estimate_flow_rate(800.0, 900.0, 1.2, 3.8, 1.2, 0.43))


def test_estimate_flow_rate_nan_when_missing_inputs():
    assert math.isnan(estimate_flow_rate(None, 900.0, 1.2, 3.8, 1.2, 0.43))
    assert math.isnan(estimate_flow_rate(1000.0, None, 1.2, 3.8, 1.2, 0.43))


def test_estimate_flow_rate_nan_when_denominator_nonpositive():
    # A2/A1 ratio large enough that (1+xi) - (N*A2/A1)^2 <= 0
    assert math.isnan(estimate_flow_rate(1000.0, 900.0, 1.2, a1=1.0, a2=1.0, xi=0.0, n=1))


def test_extract_p2_averages_lower_z_station(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    df = pd.DataFrame(
        {
            "Static Pressure (Pa)": [100.0, 102.0, 98.0, 100.0, 200.0, 202.0, 198.0, 200.0],
            "Z (m)": [20.0, 20.0, 20.0, 20.0, 20.75, 20.75, 20.75, 20.75],
        }
    )
    df.to_csv(tables_dir / "esail_internal_fan_table.csv", index=False)

    p2 = extract_p2_fan_duct_pressure(str(tmp_path))
    assert p2 == pytest.approx(100.0)  # mean of the lower-Z (20.0) station only


def test_extract_p2_none_when_table_missing(tmp_path):
    assert extract_p2_fan_duct_pressure(str(tmp_path)) is None


def test_extract_p1_nearest_height(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    df = pd.DataFrame(
        {
            "Static Pressure (Pa)": [50.0, 60.0, 70.0],
            "Z (m)": [10.0, 18.5, 23.0],
        }
    )
    df.to_csv(tables_dir / "esail_internal_center_table.csv", index=False)

    assert extract_p1_plenum_pressure(str(tmp_path), target_height=18.5) == pytest.approx(60.0)
    assert extract_p1_plenum_pressure(str(tmp_path), target_height=17.0) == pytest.approx(60.0)  # nearest is still 18.5
    assert extract_p1_plenum_pressure(str(tmp_path), target_height=11.0) == pytest.approx(50.0)  # nearest is 10.0


def test_extract_p1_none_when_table_missing(tmp_path):
    assert extract_p1_plenum_pressure(str(tmp_path)) is None


def test_estimate_flow_rates_adds_columns_and_handles_missing_tables(tmp_path):
    case_with_tables = tmp_path / "case1"
    tables_dir = case_with_tables / "tables"
    tables_dir.mkdir(parents=True)
    pd.DataFrame({"Static Pressure (Pa)": [200.0, 200.0], "Z (m)": [20.0, 20.75]}).to_csv(
        tables_dir / "esail_internal_fan_table.csv", index=False
    )
    pd.DataFrame({"Static Pressure (Pa)": [1000.0], "Z (m)": [18.5]}).to_csv(
        tables_dir / "esail_internal_center_table.csv", index=False
    )

    case_without_tables = tmp_path / "case2"
    case_without_tables.mkdir()

    df = pd.DataFrame({"source_path": [str(case_with_tables), str(case_without_tables)], "aoa": [0.0, 10.0]})
    result = estimate_flow_rates(df, chord=2.714, duct_diameter=1.25, rho=1.2)

    assert {"p1_pa", "p2_pa", "estimated_flow_rate"} <= set(result.columns)
    assert result.loc[0, "p1_pa"] == pytest.approx(1000.0)
    assert result.loc[0, "p2_pa"] == pytest.approx(200.0)
    assert result.loc[0, "estimated_flow_rate"] > 0
    assert pd.isna(result.loc[1, "p1_pa"])
    assert pd.isna(result.loc[1, "estimated_flow_rate"])
