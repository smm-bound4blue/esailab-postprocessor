"""
Transform: pure DataFrame functions turning extracted raw data into the
polar/performance grain persisted by src.sail.db. No sqlite3 imports
here beyond stat()'ing his.csv for its mtime — unit-testable without a
database.

Pipeline (see CLAUDE.md for full column rationale):
    1. convergence stats   -- mean/std of the last n_avg iterations per AoA
    2. derived columns      -- E = CL/CD, fan pressures, CQ, efficiency, Cpow
    3. is_stall              -- computed analytically per case, not from folder naming

Single-fan assumption: all "Fan N ..." columns are hardcoded to fan 1 (see
CLAUDE.md "Why this repo is simpler"). Revisit if a project ever has Fan 2.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.sail.extract import CaseFolder
from src.sail.fan import FanCurve, dynamic_pressure, interpolate_power

logger = logging.getLogger(__name__)

KNOTS_TO_MS = 0.514444

# his.csv "Force Coefficient" / "Moment Coefficient" columns dimensionalized
# back to Newtons/N*m by add_derived_columns (F = CF * q * S, M = CM * q * S
# * chord -- q = dynamic pressure, S = span*chord reference area, chord the
# reference length for all three moment axes, confirmed with the user).
_FORCE_COEF_COLS = ["CL", "CD", "CFX_SAIL", "CFX_SYS", "CFY_SAIL", "CFY_SYS", "CFZ_SAIL", "CFZ_SYS"]
_MOMENT_COEF_COLS = [
    "CMX_PILLAR_1.7D_GND", "CMX_PILLAR_1.7D_TRANS", "CMX_PILLAR_BASE", "CMX_PILLAR_HALF", "CMX_SAIL_BASE",
    "CMY_PILLAR_1.7D_GND", "CMY_PILLAR_1.7D_TRANS", "CMY_PILLAR_BASE", "CMY_PILLAR_HALF", "CMY_SAIL_BASE",
    "CMZ_PILLAR_1.7D_GND", "CMZ_PILLAR_1.7D_TRANS", "CMZ_PILLAR_BASE", "CMZ_PILLAR_HALF", "CMZ_SAIL_BASE",
]


def convergence_stats(aoa_data: pd.DataFrame, n_avg: int) -> pd.Series:
    """Mean of the last n_avg rows of one AoA's his.csv data, all numeric columns."""
    if n_avg > len(aoa_data):
        raise ValueError(f"n_avg ({n_avg}) exceeds available data ({len(aoa_data)} rows)")
    return aoa_data.iloc[-n_avg:].mean(numeric_only=True)


def convergence_std(aoa_data: pd.DataFrame, n_avg: int) -> pd.Series:
    """Std of the last n_avg rows of one AoA's his.csv data, all numeric columns."""
    if n_avg > len(aoa_data):
        raise ValueError(f"n_avg ({n_avg}) exceeds available data ({len(aoa_data)} rows)")
    return aoa_data.iloc[-n_avg:].std(numeric_only=True)


def build_polar_dataframe(cases: List[CaseFolder], n_avg: int) -> pd.DataFrame:
    """
    One row per case+AoA: identifiers (Case, AWS, RPM, AoA) + converged mean
    of every monitor column + {col}_std columns, plus bookkeeping columns
    (n_avg, n_iterations, source_path, his_csv_mtime) used later by
    src.sail.pipeline to upsert into sail_results.
    """
    rows = []

    for case in cases:
        for aoa in sorted(case.aoas, key=lambda a: a.aoa):
            if len(aoa.data) < n_avg:
                logger.warning(
                    f"Skipping {case.case_name}/AoA {aoa.aoa}° — only {len(aoa.data)} "
                    f"iterations (need {n_avg})"
                )
                continue

            stats_mean = convergence_stats(aoa.data, n_avg)
            stats_std = convergence_std(aoa.data, n_avg)

            his_path = aoa.folder_path / "his.csv"
            row = {
                "Case": case.case_name,
                "AWS": case.aws,
                "RPM": case.rpm,
                "AoA": aoa.aoa,
                "n_avg": n_avg,
                "n_iterations": len(aoa.data),
                "source_path": str(aoa.folder_path),
                "his_csv_mtime": his_path.stat().st_mtime if his_path.exists() else 0.0,
            }
            for col in stats_mean.index:
                row[col] = stats_mean[col]
            for col in stats_std.index:
                row[f"{col}_std"] = stats_std[col]

            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _get_fan_total_pressure(df: pd.DataFrame) -> Optional[pd.Series]:
    """FTP (Pa) — 'Pressure Drop'/'Total Pressure'/'Total Pressure Rise' all name the
    same physical quantity under different his.csv conventions (see CLAUDE.md)."""
    for col in ("Fan 1 Pressure Drop", "Fan 1 Total Pressure", "Fan 1 Total Pressure Rise"):
        if col in df.columns:
            return df[col].abs()
    return None


def _get_fan_dynamic_pressure(df: pd.DataFrame, rho: float, fan_duct_diameter: float) -> Optional[pd.Series]:
    """Pv2 (Pa) — real monitor if present, else estimated from volumetric flow + duct diameter
    (src.sail.fan.dynamic_pressure, shared with the manufacturer curve's derived Total Pressure)."""
    if "Fan 1 Dynamic Pressure (MFA)" in df.columns:
        return df["Fan 1 Dynamic Pressure (MFA)"].abs()
    if "Fan 1 Volumetric Flow" in df.columns:
        return dynamic_pressure(df["Fan 1 Volumetric Flow"], rho, fan_duct_diameter)
    return None


def add_derived_columns(
    polar_df: pd.DataFrame,
    fan_curve: Optional[FanCurve],
    span: float,
    chord: Optional[float],
    rho: float,
    fan_duct_diameter: float,
) -> pd.DataFrame:
    """
    Adds (on a copy):
      - E = CL / CD
      - Fan Total/Static Pressure (FTP/FSP), fan-curve-independent
      - Fan Volumetric Flow, CQ = flow / (AWS_m/s * span * chord); NaN if chord is None
      - Fan Static/Total Efficiency, Fan Power (kW), Cpow -- only if fan_curve is not None
        and chord is not None (need the reference area)
      - {col}_N for every _FORCE_COEF_COLS entry (Newtons) and {col}_Nm for every
        _MOMENT_COEF_COLS entry (N*m) -- F = CF * q * S, M = CM * q * S * chord;
        NaN whenever chord is None (same reference-area guard as CQ/Cpow)
      - Z_CP = mean(ZCP_NATIVE_XZ, ZCP_NATIVE_YZ) -- both his.csv columns are the
        same physical quantity (the center of pressure's Z coordinate/height),
        just computed from two different load projections (XZ-plane, YZ-plane),
        so they're averaged into one value rather than kept as two near-duplicates
    """
    if polar_df.empty:
        return polar_df

    df = polar_df.copy()

    df["E"] = np.where(df["CD"] != 0, df["CL"] / df["CD"], 0.0) if {"CL", "CD"} <= set(df.columns) else np.nan

    if {"ZCP_NATIVE_XZ", "ZCP_NATIVE_YZ"} <= set(df.columns):
        df["Z_CP"] = df[["ZCP_NATIVE_XZ", "ZCP_NATIVE_YZ"]].mean(axis=1)
    else:
        df["Z_CP"] = np.nan

    ftp = _get_fan_total_pressure(df)
    df["Fan Total Pressure"] = ftp if ftp is not None else np.nan
    if ftp is not None:
        pv2 = _get_fan_dynamic_pressure(df, rho, fan_duct_diameter)
        df["Fan Static Pressure"] = (ftp - pv2) if pv2 is not None else ftp
    else:
        df["Fan Static Pressure"] = np.nan

    ref_area = span * chord if chord is not None else None
    V = df["AWS"] * KNOTS_TO_MS

    q = 0.5 * rho * V**2
    for col in _FORCE_COEF_COLS:
        df[f"{col}_N"] = (df[col] * q * ref_area) if (ref_area is not None and col in df.columns) else np.nan
    for col in _MOMENT_COEF_COLS:
        df[f"{col}_Nm"] = (
            (df[col] * q * ref_area * chord) if (ref_area is not None and col in df.columns) else np.nan
        )

    if "Fan 1 Volumetric Flow" in df.columns:
        df["Fan Volumetric Flow"] = df["Fan 1 Volumetric Flow"].abs()
        df["CQ"] = np.where(V > 0, df["Fan Volumetric Flow"] / (V * ref_area), np.nan) if ref_area else np.nan
    else:
        df["Fan Volumetric Flow"] = np.nan
        df["CQ"] = np.nan

    df["Fan Static Efficiency"] = np.nan
    df["Fan Total Efficiency"] = np.nan
    df["Fan Power"] = np.nan
    df["Cpow"] = np.nan

    if fan_curve is not None and ref_area is not None:
        for idx in df.index:
            rpm = df.loc[idx, "RPM"]
            Q = df.loc[idx, "Fan Volumetric Flow"]
            if pd.isna(rpm) or pd.isna(Q):
                continue

            try:
                P_kW = interpolate_power(fan_curve, rpm, Q)
            except Exception as e:
                logger.warning(f"Could not interpolate fan power at row {idx}: {e}")
                continue
            P_W = P_kW * 1000.0
            df.loc[idx, "Fan Power"] = P_kW

            FTP = df.loc[idx, "Fan Total Pressure"]
            FSP = df.loc[idx, "Fan Static Pressure"]
            if P_W > 0 and not pd.isna(FTP):
                df.loc[idx, "Fan Total Efficiency"] = (Q * FTP) / P_W
            if P_W > 0 and not pd.isna(FSP):
                df.loc[idx, "Fan Static Efficiency"] = (Q * FSP) / P_W

            v = df.loc[idx, "AWS"] * KNOTS_TO_MS
            if v > 0:
                df.loc[idx, "Cpow"] = P_W / (0.5 * rho * v**3 * ref_area)

    return df


def compute_is_stall(case_df: pd.DataFrame) -> pd.Series:
    """is_stall = (AoA > AoA_at_CLmax) AND (CL < CLmax), within one case's own AoA sweep."""
    if len(case_df) < 2 or case_df["CL"].isna().all():
        return pd.Series(False, index=case_df.index)

    clmax_idx = case_df["CL"].idxmax()
    aoa_at_clmax = case_df.loc[clmax_idx, "AoA"]
    cl_max = case_df.loc[clmax_idx, "CL"]

    return (case_df["AoA"] > aoa_at_clmax) & (case_df["CL"] < cl_max)


def annotate_is_stall(polar_df: pd.DataFrame) -> pd.DataFrame:
    """Adds an is_stall column to a combined polar DataFrame, computed independently per Case."""
    result = polar_df.copy()
    if result.empty:
        result["is_stall"] = pd.Series(dtype=bool)
        return result

    result["is_stall"] = False
    for case_name in result["Case"].unique():
        mask = result["Case"] == case_name
        result.loc[mask, "is_stall"] = compute_is_stall(result.loc[mask])
    return result
