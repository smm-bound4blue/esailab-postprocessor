"""Tests for src.sail.fan: dynamic pressure formula and the derived reference curve."""

import numpy as np
import pandas as pd
import pytest

from src.sail.fan import FanCurve, build_reference_curve, dynamic_pressure


def _sample_curve():
    return FanCurve(
        reference_rpm=1465.0,
        flowrate=pd.Series([5.0, 10.0, 15.0]),
        pressure=pd.Series([1000.0, 700.0, 300.0]),  # static, per user confirmation
        power=pd.Series([15.0, 17.0, 16.0]),
    )


def test_dynamic_pressure_hand_calc():
    # duct_diameter=1.0 -> area = pi/4 ~ 0.7854 m^2; velocity = Q/area
    duct_diameter = 1.0
    duct_area = np.pi * (duct_diameter / 2) ** 2
    Q = 2.0
    velocity = Q / duct_area
    expected = 0.5 * 1.2 * velocity**2
    assert dynamic_pressure(Q, rho=1.2, duct_diameter=duct_diameter) == pytest.approx(expected)


def test_dynamic_pressure_uses_absolute_flow():
    assert dynamic_pressure(-3.0, rho=1.2, duct_diameter=1.25) == dynamic_pressure(3.0, rho=1.2, duct_diameter=1.25)


def test_dynamic_pressure_works_on_series():
    result = dynamic_pressure(pd.Series([1.0, 2.0]), rho=1.2, duct_diameter=1.25)
    assert isinstance(result, pd.Series)
    assert result.iloc[1] > result.iloc[0]


def test_build_reference_curve_total_is_static_plus_dynamic():
    curve = _sample_curve()
    ref = build_reference_curve(curve, rpm=1465.0, rho=1.2, duct_diameter=1.25)
    assert {"flowrate", "static_pressure", "total_pressure", "power"} <= set(ref.columns)
    expected_pv2 = dynamic_pressure(ref["flowrate"], 1.2, 1.25)
    pd.testing.assert_series_equal(
        ref["total_pressure"], ref["static_pressure"] + expected_pv2, check_names=False
    )
    # total pressure is always >= static (dynamic pressure is non-negative)
    assert (ref["total_pressure"] >= ref["static_pressure"]).all()


def test_build_reference_curve_at_reference_rpm_matches_raw_static_curve():
    curve = _sample_curve()
    ref = build_reference_curve(curve, rpm=curve.reference_rpm, rho=1.2, duct_diameter=1.25)
    pd.testing.assert_series_equal(ref["flowrate"], curve.flowrate, check_names=False)
    pd.testing.assert_series_equal(ref["static_pressure"], curve.pressure, check_names=False)


def test_build_reference_curve_scales_with_affinity_laws():
    curve = _sample_curve()
    ratio = 2.0
    ref = build_reference_curve(curve, rpm=curve.reference_rpm * ratio, rho=1.2, duct_diameter=1.25)
    pd.testing.assert_series_equal(ref["flowrate"], curve.flowrate * ratio, check_names=False)
    pd.testing.assert_series_equal(ref["static_pressure"], curve.pressure * ratio**2, check_names=False)
