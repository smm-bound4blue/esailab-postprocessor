"""
Flow rate estimation from two static-pressure measurements -- P1 inside
the sail plenum (several diameters upstream of the fan bellmouth) and P2
inside the fan duct (downstream of the bellmouth, upstream of the
motor/blades) -- via Bernoulli with losses between the two locations:

    p1 + 1/2*rho*v1^2 = p2 + 1/2*rho*v2^2 + xi * 1/2*rho*v2^2

combined with mass conservation (Q_total = v1*A1 = N*v2*A2) into the
closed-form relation (general form, N parallel fans feeding one plenum):

    Q_total = N * A2 * sqrt( 2*(p1 - p2) / (rho * [(1+xi) - (N*A2/A1)^2]) )

P1/P2 are extracted live from per-AoA spatial tables (src.sail.spatial_tables)
-- same "deliberate exception" as the Tables page and src.sail.fan's
reference curve: this is a live comparison tool, not part of the ETL/DB
pipeline, since it's evaluated on demand with a user-adjustable loss
coefficient rather than persisted.
"""

import math
from typing import List, Optional

import pandas as pd

from src.sail.fan import duct_area
from src.sail.spatial_tables import load_table

_STATIC_PRESSURE_COL = "Static Pressure (Pa)"
_Z_COL = "Z (m)"

#: esail's internal cross-section (A1), modeled as an ellipse: major axis =
#: chord, minor axis = thickness = this fraction of chord.
DEFAULT_THICKNESS_RATIO = 0.66

#: Bellmouth loss coefficient (xi), from wind tunnel/CFD characterization.
DEFAULT_XI = 0.43

#: Height (m) of the plenum static-pressure probe (P1), if the user doesn't override it.
DEFAULT_P1_HEIGHT_M = 18.5


def ellipse_area(chord: float, thickness_ratio: float = DEFAULT_THICKNESS_RATIO) -> float:
    """A1: sail cross-section modeled as an ellipse (major axis = chord,
    minor axis = thickness_ratio * chord). Area = pi * a * b with
    a = chord/2, b = (thickness_ratio * chord)/2."""
    return math.pi * chord * (thickness_ratio * chord) / 4.0


def estimate_flow_rate(
    p1: Optional[float], p2: Optional[float], rho: float, a1: float, a2: float, xi: float, n: int = 1
) -> float:
    """
    Closed-form Bernoulli-with-losses flow rate estimate (see module
    docstring). Returns NaN if the inputs don't support a physical
    solution: p1/p2 missing, p1 <= p2 (no pressure drop driving flow
    toward the fan), or the denominator term is non-positive (would need
    a complex square root).
    """
    if p1 is None or p2 is None or not math.isfinite(p1) or not math.isfinite(p2):
        return float("nan")
    if p1 <= p2:
        return float("nan")
    denom_term = (1 + xi) - (n * a2 / a1) ** 2
    if denom_term <= 0:
        return float("nan")
    return n * a2 * math.sqrt(2 * (p1 - p2) / (rho * denom_term))


def extract_p2_fan_duct_pressure(aoa_source_path: str) -> Optional[float]:
    """
    P2: mean Static Pressure (Pa) over the points at the lower of
    esail_internal_fan_table's two Z stations (each station is a fixed-Z
    plane of probe points, so an exact-value group-by is reliable here).
    None if the table or its expected columns aren't present.
    """
    df = load_table(aoa_source_path, "esail_internal_fan_table")
    if df.empty or _Z_COL not in df.columns or _STATIC_PRESSURE_COL not in df.columns:
        return None
    lower_z = df[_Z_COL].min()
    station = df[df[_Z_COL] == lower_z]
    return float(station[_STATIC_PRESSURE_COL].mean())


def extract_p1_plenum_pressure(aoa_source_path: str, target_height: float = DEFAULT_P1_HEIGHT_M) -> Optional[float]:
    """
    P1: Static Pressure (Pa) at the esail_internal_center_table point
    nearest target_height. None if the table or its expected columns
    aren't present.
    """
    df = load_table(aoa_source_path, "esail_internal_center_table")
    if df.empty or _Z_COL not in df.columns or _STATIC_PRESSURE_COL not in df.columns:
        return None
    idx = (df[_Z_COL] - target_height).abs().idxmin()
    return float(df.loc[idx, _STATIC_PRESSURE_COL])


def estimate_flow_rates(
    df: pd.DataFrame,
    chord: float,
    duct_diameter: float,
    rho: float,
    xi: float = DEFAULT_XI,
    target_height: float = DEFAULT_P1_HEIGHT_M,
    n: int = 1,
    thickness_ratio: float = DEFAULT_THICKNESS_RATIO,
) -> pd.DataFrame:
    """
    For every row in df (expects a source_path column -- the AoA folder
    path, as returned by app.sail.data.get_polar_data), extract P1/P2 from
    that AoA's spatial tables and compute an estimated flow rate. Returns
    a copy of df with p1_pa/p2_pa/estimated_flow_rate columns added (NaN
    where extraction or the physical solution fails for that row).
    """
    a1 = ellipse_area(chord, thickness_ratio)
    a2 = duct_area(duct_diameter)

    p1_values: List[Optional[float]] = []
    p2_values: List[Optional[float]] = []
    q_values: List[float] = []
    for source_path in df["source_path"]:
        p1 = extract_p1_plenum_pressure(source_path, target_height=target_height)
        p2 = extract_p2_fan_duct_pressure(source_path)
        p1_values.append(p1)
        p2_values.append(p2)
        q_values.append(estimate_flow_rate(p1, p2, rho=rho, a1=a1, a2=a2, xi=xi, n=n))

    result = df.copy()
    result["p1_pa"] = p1_values
    result["p2_pa"] = p2_values
    result["estimated_flow_rate"] = q_values
    return result
