"""
Static reference (non-CFD) data shared across simulation types -- e.g. the
Barcelona harbor wind climate. Small, git-tracked, hand-provided datasets
like this live in config/ rather than data/ or a database: there's no
extract/transform pipeline for a single fixed matrix, so it's just read
and cached directly here.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

WIND_PROBABILITY_MATRIX_PATH = Path(__file__).parent.parent.parent / "config" / "wind_probability_matrix.csv"


@st.cache_data
def load_wind_probability_matrix() -> pd.DataFrame:
    """
    Barcelona harbor true wind probability matrix: rows are TWA bin centers
    (deg, 5° bins), columns are TWS bin centers (kts, 1 kt bins), values are
    probability mass (fractions summing to ~1 over the whole matrix). Used
    to weight which TWS/TWA combinations matter most for harbour/interference
    test planning.
    """
    df = pd.read_csv(WIND_PROBABILITY_MATRIX_PATH, index_col=0)
    df.columns = df.columns.astype(float)
    return df
