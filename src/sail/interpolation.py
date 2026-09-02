"""
Performance data interpolation: resamples performance-envelope traces
(e.g. the CLmax dataset app.sail.analysis.compute_clmax_data produces)
onto a uniform grid along a chosen independent variable (aoa) using
monotonic PCHIP interpolation, so every numeric column gets a smooth
value at every grid point -- the "Interpolated Data" tab on the
Performance page uses this to turn each trace's sparse CLmax-per-RPM
points into a smooth envelope curve.

Ported from esail-postprocessor's src/utils/interpolation.py (same
algorithm, adapted to this repo's lowercase column names: aoa, cpow).
Pure pandas/numpy/scipy, no Streamlit dependency.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def interpolate_performance_envelope(
    df: pd.DataFrame,
    group_cols: List[str],
    x_col: str = "aoa",
    step: float = 1.0,
    duplicate_shift: float = 1.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Resample each (group_cols) trace onto a uniform x_col grid via PCHIP.

    For each group, every numeric column (other than x_col and the group
    columns themselves) is fit against x_col with a PchipInterpolator and
    evaluated on a uniform grid of x_col. The grid's endpoints are snapped
    to the nearest step multiple *inside* the group's [min(x_col), max(x_col)]
    range (ceil for the lower bound, floor for the upper bound), so grid
    points land on clean numbers (e.g. integers for step=1.0) without ever
    requiring PCHIP to extrapolate past the real data (e.g. a 52.5 max AoA
    yields a grid ending at 52.0, not an extrapolated 53.0).
    Group columns are carried forward as constants instead of being interpolated.

    When two rows share the same x_col value (e.g. two operating conditions
    stalling at the same AoA), the one with the highest 'cpow' value has its
    x_col shifted by `duplicate_shift` to make the sequence strictly
    monotonic. No data is dropped. If 'cpow' is absent the last row in
    sort order is shifted instead.

    Args:
        df: Source DataFrame (e.g. the CLmax performance envelope data)
        group_cols: Columns identifying a single trace (e.g. ['project_name']
                    or ['project_name', 'aws'])
        x_col: Independent variable column to interpolate against
        step: Target spacing of the new x_col grid
        duplicate_shift: Amount added to the x_col value of the highest-cpow
                         row when duplicate x_col values are found. Positive
                         shifts it later (typical: higher RPM stalls slightly
                         later), negative shifts it earlier.

    Returns:
        Tuple of (resampled DataFrame, list of warning messages for groups
        that were skipped or had duplicate x_col values resolved)

    Example:
        >>> resampled, warnings = interpolate_performance_envelope(
        ...     clmax_df, group_cols=['project_name', 'aws'], x_col='aoa', step=1.0,
        ... )
    """
    warnings: List[str] = []

    if df.empty or x_col not in df.columns:
        return pd.DataFrame(), warnings

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    y_cols = [c for c in numeric_cols if c != x_col and c not in group_cols]

    result_rows = []

    for group_key, group_df in df.groupby(group_cols, dropna=False):
        group_label = group_key if isinstance(group_key, tuple) else (group_key,)
        label_str = ", ".join(f"{c}={v}" for c, v in zip(group_cols, group_label))

        group_df = group_df.sort_values(x_col).copy()

        # Resolve duplicate x_col values by shifting the highest-cpow row.
        dup_mask = group_df.duplicated(subset=x_col, keep=False)
        if dup_mask.any():
            n_dup_groups = group_df.loc[dup_mask, x_col].nunique()
            for x_val, dup_rows in group_df[dup_mask].groupby(x_col):
                if "cpow" in dup_rows.columns and not dup_rows["cpow"].isna().all():
                    shift_idx = dup_rows["cpow"].idxmax()
                else:
                    shift_idx = dup_rows.index[-1]
                group_df.loc[shift_idx, x_col] = x_val + duplicate_shift

            group_df = group_df.sort_values(x_col)

            still_dup = group_df.duplicated(subset=x_col, keep=False).sum()
            msg = (
                f"Shifted highest-cpow duplicate {x_col} by {duplicate_shift:+g} "
                f"({n_dup_groups} case(s)) for trace ({label_str})"
            )
            if still_dup:
                msg += f" — {still_dup} duplicate(s) remain (try a larger |duplicate_shift|)"
            warnings.append(msg)

        x_arr = group_df[x_col].to_numpy(dtype=float)
        if len(x_arr) < 2:
            warnings.append(f"Skipped trace ({label_str}): fewer than 2 unique {x_col} points")
            continue

        x_min, x_max = x_arr.min(), x_arr.max()

        # Snap endpoints to the nearest step multiple inside [x_min, x_max],
        # so the grid never extrapolates past the real data (tol guards
        # against float dust when an endpoint is already a step multiple).
        tol = 1e-9
        x_min_grid = np.ceil(x_min / step - tol) * step
        x_max_grid = np.floor(x_max / step + tol) * step

        if x_min_grid > x_max_grid:
            warnings.append(
                f"Skipped trace ({label_str}): no {x_col} step={step:g} grid "
                f"point falls within [{x_min:g}, {x_max:g}]"
            )
            continue

        n_points = round((x_max_grid - x_min_grid) / step) + 1
        x_new = np.round(np.linspace(x_min_grid, x_max_grid, n_points), 9)

        group_result = {x_col: x_new}
        for col, val in zip(group_cols, group_label):
            group_result[col] = val

        for y_col in y_cols:
            y_arr = group_df[y_col].to_numpy(dtype=float)
            if np.isnan(y_arr).any():
                group_result[y_col] = np.full_like(x_new, np.nan)
            else:
                group_result[y_col] = PchipInterpolator(x_arr, y_arr)(x_new)

        result_rows.append(pd.DataFrame(group_result))

    if not result_rows:
        return pd.DataFrame(), warnings

    return pd.concat(result_rows, ignore_index=True), warnings


def smooth_performance_envelope(
    df: pd.DataFrame,
    group_cols: List[str],
    x_col: str = "aoa",
    step: float = 1.0,
    bin_width: float = 10.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    For each group_cols group: bin x_col into fixed-width bins (bin_width),
    averaging every numeric column within each populated bin into one
    representative point, then PCHIP-interpolate through those bin points
    onto a uniform x_col grid at `step` spacing, clipped to the bin points'
    own [min, max] (PCHIP never extrapolates past its control points). The
    first/last bin's x-position is pinned back to the group's true min/max
    x_col regardless of bin_width -- otherwise a wide bin_width would pull
    those edge bins' mean x inward (their y stays the bin average either
    way), shrinking the fitted curve's range away from the real data's.

    Unlike interpolate_performance_envelope's plain per-point PCHIP, this
    averages first -- that matters when group_cols is coarser than one
    physical trace (e.g. ["project_name"], pooling every AWS/RPM trace
    together): different AWS conditions can have overlapping/duplicate
    x_col values with conflicting y values, and averaging within each bin
    collapses that scatter into one clean point *before* fitting, instead
    of needing PCHIP's exact-x-value duplicate_shift trick.

    Tried two other approaches first, both real numeric failures on this
    project's actual pooled multi-AWS data, not just theoretical concerns:
    scipy.interpolate.UnivariateSpline's automatic knot placement was
    fragile on a Cpow column spanning ~0.03 to ~40 (three orders of
    magnitude) with a duplicate AoA across two AWS traces -- silently
    all-NaN, or wildly oscillating negative. A hand-rolled Gaussian-kernel
    local-linear regression (LOESS) fixed that, but a local *line* has no
    bound on its own: in a real gap between data points wider than its
    bandwidth, adjacent to a region where the trend was steepening, it
    would extrapolate into a dip/bump below or above every point that went
    into it. Binning sidesteps both: there's no knot-placement step to
    destabilize, and PCHIP through bin averages is shape-preserving
    between its control points, so it can't invent a new local extremum
    the binned data doesn't already show.

    bin_width (same units as x_col, e.g. degrees for AoA): the width of
    each averaging bin. Larger = fewer, coarser bins (more scatter
    absorbed, less faithful to individual points); smaller = finer bins
    closer to interpolate_performance_envelope's per-point behavior.

    Returns (fitted DataFrame, warnings for any skipped group -- fewer
    than 2 populated bins, or no step-grid point inside the bin range).
    """
    warnings: List[str] = []

    if df.empty or x_col not in df.columns:
        return pd.DataFrame(), warnings

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    y_cols = [c for c in numeric_cols if c != x_col and c not in group_cols]

    result_rows = []

    for group_key, group_df in df.groupby(group_cols, dropna=False):
        group_label = group_key if isinstance(group_key, tuple) else (group_key,)
        label_str = ", ".join(f"{c}={v}" for c, v in zip(group_cols, group_label))

        group_df = group_df.sort_values(x_col)
        x_arr = group_df[x_col].to_numpy(dtype=float)

        if len(x_arr) < 2:
            warnings.append(f"Skipped trace ({label_str}): fewer than 2 {x_col} points")
            continue

        x_min = x_arr.min()
        bin_idx = np.floor((x_arr - x_min) / bin_width).astype(int)
        binned = group_df[[x_col] + y_cols].groupby(bin_idx).mean().sort_values(x_col)
        bx = binned[x_col].to_numpy(dtype=float, copy=True)

        if len(bx) < 2:
            warnings.append(
                f"Skipped trace ({label_str}): fewer than 2 populated bins at bin_width={bin_width:g}"
            )
            continue

        # Pin the first/last bin's x-position back to the trace's true endpoint
        # AoA -- their y stays the bin average, but a wide bin_width otherwise
        # pulls the *edge* bins' mean x inward from the real min/max (e.g. two
        # points 5 degrees apart both landing in one 10-degree bin averages
        # their x too), which would shrink the fitted curve's AoA range away
        # from the real one. Always safe/still strictly increasing since these
        # can only move bx[0] down and bx[-1] up, i.e. further from their
        # already-innermost neighbors, never past them.
        bx[0], bx[-1] = x_arr.min(), x_arr.max()

        tol = 1e-9
        x_min_grid = np.ceil(bx.min() / step - tol) * step
        x_max_grid = np.floor(bx.max() / step + tol) * step

        if x_min_grid > x_max_grid:
            warnings.append(
                f"Skipped trace ({label_str}): no {x_col} step={step:g} grid "
                f"point falls within [{bx.min():g}, {bx.max():g}]"
            )
            continue

        n_points = round((x_max_grid - x_min_grid) / step) + 1
        x_new = np.round(np.linspace(x_min_grid, x_max_grid, n_points), 9)

        group_result = {x_col: x_new}
        for col, val in zip(group_cols, group_label):
            group_result[col] = val

        for y_col in y_cols:
            by = binned[y_col].to_numpy(dtype=float)
            if np.isnan(by).any():
                group_result[y_col] = np.full_like(x_new, np.nan)
            else:
                group_result[y_col] = PchipInterpolator(bx, by)(x_new)

        result_rows.append(pd.DataFrame(group_result))

    if not result_rows:
        return pd.DataFrame(), warnings

    return pd.concat(result_rows, ignore_index=True), warnings
