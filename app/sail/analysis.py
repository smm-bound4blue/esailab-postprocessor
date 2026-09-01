"""
Performance-envelope analysis: pure DataFrame transform, no DB/Streamlit
here so it's unit-testable on its own.
"""

import pandas as pd


def compute_clmax_data(polar_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (project_name, aws, rpm) case, keep only the AoA row with the
    maximum CL -- all its other columns (CD, E, fan variables, CQ, Cpow,
    ...) come along for that one row. Gives one "best operating point" per
    case, which is what the Performance page plots across RPM (CL vs Cpow,
    fan efficiency, CQ) -- as opposed to Polar Curves, which plots the full
    AoA sweep.
    """
    if polar_df.empty:
        return polar_df
    idx = polar_df.groupby(["project_name", "aws", "rpm"])["cl"].idxmax()
    return polar_df.loc[idx].reset_index(drop=True)
