"""
Fan performance curve + affinity-law scaling, and the fan pressure/power
helpers used to compute efficiency and Cpow. Single fan only (his.csv has
no "Fan 2 ..." columns for eSAILab) — no per-fan normalization needed.

Affinity laws (scaling from the reference RPM in config/fan_curve.csv to
the operating RPM): Q ∝ N, ΔP ∝ N², P ∝ N³.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_RPM_IN_LABEL_PATTERN = re.compile(r"RPM_(\d+(?:\.\d+)?)")


@dataclass
class FanCurve:
    reference_rpm: float
    flowrate: pd.Series  # m^3/s, at reference_rpm
    pressure: pd.Series  # Pa, STATIC pressure (confirmed for the HGT-125-4T-6 datasheet), at reference_rpm
    power: pd.Series     # kW, at reference_rpm


def load_fan_curve(path: Path) -> FanCurve:
    """Parse a fan_curve.csv (header line with the fan model name, then
    Flow Rate (m3/s), Pressure (Pa), Power (kW) columns) into a FanCurve.
    reference_rpm is parsed from the model name line, e.g. '...RPM_1465'."""
    with open(path, encoding="utf-8-sig") as f:
        label_line = f.readline().strip()

    match = _RPM_IN_LABEL_PATTERN.search(label_line)
    if not match:
        raise ValueError(f"Could not find reference RPM in fan curve label: '{label_line}'")
    reference_rpm = float(match.group(1))

    df = pd.read_csv(path, skiprows=1, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    return FanCurve(
        reference_rpm=reference_rpm,
        flowrate=df["Flow Rate (m3/s)"],
        pressure=df["Pressure (Pa)"],
        power=df["Power (kW)"],
    )


def scale_to_rpm(curve: FanCurve, rpm: float) -> pd.DataFrame:
    """Scale curve to the given operating rpm via affinity laws.
    Returns a DataFrame with flowrate/pressure/power columns at that rpm."""
    ratio = rpm / curve.reference_rpm
    return pd.DataFrame(
        {
            "flowrate": curve.flowrate * ratio,
            "pressure": curve.pressure * ratio**2,
            "power": curve.power * ratio**3,
        }
    )


def interpolate_power(curve: FanCurve, rpm: float, flowrate_m3s: float) -> float:
    """Power (kW) at the given rpm and flowrate, via affinity scaling + linear
    interpolation along the scaled curve."""
    scaled = scale_to_rpm(curve, rpm)
    return float(np.interp(flowrate_m3s, scaled["flowrate"], scaled["power"]))


def dynamic_pressure(volumetric_flow, rho: float, duct_diameter: float):
    """Pv2 = 0.5 * rho * velocity^2, velocity = |Q| / duct_area. Shared by both the
    CFD-derived dynamic pressure estimate (src.sail.transform, when no direct monitor is
    present) and the manufacturer fan curve's derived Total Pressure (build_reference_curve)
    -- same physics, one formula. `volumetric_flow` can be a scalar, Series, or ndarray."""
    duct_area = np.pi * (duct_diameter / 2) ** 2
    velocity = np.abs(volumetric_flow) / duct_area
    return 0.5 * rho * velocity**2


def build_reference_curve(curve: FanCurve, rpm: float, rho: float, duct_diameter: float) -> pd.DataFrame:
    """
    The manufacturer curve (static_pressure) scaled to `rpm`, with a derived Total
    Pressure curve alongside it: total_pressure = static_pressure + dynamic_pressure(flowrate).
    Mirrors the same FSP/FTP relationship used for CFD-derived fan data in
    src.sail.transform (there it's the reverse direction: FSP = FTP - dynamic pressure,
    since the CFD monitors report total pressure and static is derived from it; here the
    manufacturer curve reports static pressure directly, confirmed for the HGT-125-4T-6).

    Returns a DataFrame with flowrate/static_pressure/total_pressure/power columns.
    """
    scaled = scale_to_rpm(curve, rpm).rename(columns={"pressure": "static_pressure"})
    scaled["total_pressure"] = scaled["static_pressure"] + dynamic_pressure(scaled["flowrate"], rho, duct_diameter)
    return scaled
