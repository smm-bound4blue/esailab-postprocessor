"""
Performance-envelope analysis: pure DataFrame transform, no DB/Streamlit
here so it's unit-testable on its own.
"""

import pandas as pd

TRACE_COLS = ["project_name", "aws"]


def compute_clmax_data(polar_df: pd.DataFrame, full_polar_min_rpm: bool = False) -> pd.DataFrame:
    """
    For every (project_name, aws, rpm) case, keep only the AoA row with the
    maximum CL -- all its other columns (CD, E, fan variables, CQ, Cpow,
    ...) come along for that one row. Gives one "best operating point" per
    case, which is what the Performance page plots across RPM (CL vs Cpow,
    fan efficiency, CQ) -- as opposed to Polar Curves, which plots the full
    AoA sweep.

    full_polar_min_rpm=True (ported from esail-postprocessor's
    SimulationProject.get_max_cl_data): for each (project_name, aws) trace,
    replace its single CLmax point at that trace's minimum RPM with every
    AoA row up to (and including) AoA@CLmax from the raw polar sweep at
    that RPM -- so the envelope traces the full polar curve at the lowest
    RPM instead of jumping straight to its CLmax point, while every other
    RPM in the trace still contributes just its one CLmax point.
    """
    if polar_df.empty:
        return polar_df
    idx = polar_df.groupby(["project_name", "aws", "rpm"])["cl"].idxmax()
    clmax_df = polar_df.loc[idx].reset_index(drop=True)

    if not full_polar_min_rpm:
        return clmax_df

    replaced = []
    for trace_values, trace_clmax in clmax_df.groupby(TRACE_COLS, sort=False):
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
