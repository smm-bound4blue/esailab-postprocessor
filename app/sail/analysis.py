"""
Performance-envelope analysis: pure DataFrame transform, no DB/Streamlit
here so it's unit-testable on its own.
"""

from typing import List, Optional, Tuple

import pandas as pd

TRACE_COLS = ["project_name", "aws"]


def compute_clmax_data(
    polar_df: pd.DataFrame, full_polar_traces: Optional[List[Tuple]] = None
) -> pd.DataFrame:
    """
    For every (project_name, aws, rpm) case, keep only the AoA row with the
    maximum CL -- all its other columns (CD, E, fan variables, CQ, Cpow,
    ...) come along for that one row. Gives one "best operating point" per
    case, which is what the Performance page plots across RPM (CL vs Cpow,
    fan efficiency, CQ) -- as opposed to Polar Curves, which plots the full
    AoA sweep.

    full_polar_traces (ported from esail-postprocessor's
    SimulationProject.get_max_cl_data): an explicit list of (project_name,
    aws) trace keys to apply the "full polar at min RPM" treatment to --
    for each of those traces, replace its single CLmax point at that
    trace's minimum RPM with every AoA row up to (and including)
    AoA@CLmax from the raw polar sweep at that RPM, so the envelope traces
    the full polar curve at the lowest RPM instead of jumping straight to
    its CLmax point (every other RPM in the trace still contributes just
    its one CLmax point). Traces not listed -- and every trace, when this
    is None or empty -- keep the plain one-point-per-RPM behavior.
    """
    if polar_df.empty:
        return polar_df
    idx = polar_df.groupby(["project_name", "aws", "rpm"])["cl"].idxmax()
    clmax_df = polar_df.loc[idx].reset_index(drop=True)

    if not full_polar_traces:
        return clmax_df

    target_traces = set(full_polar_traces)
    replaced = []
    for trace_values, trace_clmax in clmax_df.groupby(TRACE_COLS, sort=False):
        if trace_values not in target_traces:
            replaced.append(trace_clmax)
            continue

        project_name, aws = trace_values
        min_rpm = trace_clmax["rpm"].min()
        max_aoa_at_min_rpm = trace_clmax.loc[trace_clmax["rpm"] == min_rpm, "aoa"].iloc[0]

        full_polar_rows = polar_df[
            (polar_df["project_name"] == project_name)
            & (polar_df["aws"] == aws)
            & (polar_df["rpm"] == min_rpm)
            & (polar_df["aoa"] <= max_aoa_at_min_rpm)
        ]

        kept = trace_clmax[trace_clmax["rpm"] != min_rpm]
        replaced.append(pd.concat([kept, full_polar_rows], ignore_index=True))

    result = pd.concat(replaced, ignore_index=True)
    return result.sort_values(TRACE_COLS + ["rpm", "aoa"]).reset_index(drop=True)


def enforce_cl_monotonic(clmax_df: pd.DataFrame, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Within each group_cols group (default TRACE_COLS, i.e. per (project_name,
    aws) trace), walking AoA ascending, drops any row whose CL doesn't match
    or beat the running maximum CL seen so far -- keeps only the "record-
    setting" points, so CL is non-decreasing with AoA.

    A real CLmax-per-RPM envelope occasionally has a higher-RPM operating
    point whose CLmax is slightly *lower* than an earlier, lower-AoA point's
    (CFD noise, or a genuinely worse-performing RPM) -- left in, that reads
    as CL dropping as AoA increases, which downstream consumers (e.g. a VPP)
    don't expect from an envelope curve. Dropping the offending row is more
    honest than smoothing/averaging the CFD-computed CL value to force it
    into line.

    Passing group_cols=["project_name"] (coarser than the TRACE_COLS default)
    additionally resolves conflicts *across* AWS/RPM traces: different AWS
    conditions can have overlapping AoA ranges with different CL at similar
    AoA, so walking AoA ascending across every trace in the project and
    keeping only the running-max point (regardless of which AWS/RPM it came
    from) picks the best-performing condition at each AoA and merges every
    trace into one combined envelope -- see the "Combined AWS Envelope" tab.
    """
    if clmax_df.empty:
        return clmax_df
    group_cols = group_cols or TRACE_COLS
    df = clmax_df.dropna(subset=["cl"]).sort_values(group_cols + ["aoa"]).reset_index(drop=True)
    running_max = df.groupby(group_cols)["cl"].cummax()
    return df[df["cl"] == running_max].reset_index(drop=True)
