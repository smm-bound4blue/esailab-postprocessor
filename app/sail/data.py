"""
Cached DB query + ETL-trigger helpers for the "Sail" pages. This is the
ONLY way sail pages read results data -- always from src.sail.db / the
SQLite database, never from src.sail.extract or the raw data/ folder
directly.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.data import clear_project_caches
from src.config import get_default_db_path
from src.sail import db as sail_db
from src.sail.pipeline import ETLResult, run_etl

SIMULATION_TYPE = "sail"


@st.cache_data(ttl=3600)
def get_polar_data(project_id: int) -> pd.DataFrame:
    """sail_results joined with sail_cases for one project -- the combined
    polar/performance DataFrame that the Sail pages plot directly."""
    conn = sail_db.get_connection(get_default_db_path())
    try:
        return pd.read_sql(
            """
            SELECT c.case_name, c.aws, c.rpm, r.*
            FROM sail_results r
            JOIN sail_cases c ON r.case_id = c.id
            WHERE c.project_id = ?
            ORDER BY c.rpm, r.aoa
            """,
            conn,
            params=(project_id,),
        )
    finally:
        conn.close()


def run_etl_for_project(project_path: Path) -> ETLResult:
    """Wraps src.sail.pipeline.run_etl() for the Sail Home page's 'Run ETL'
    button, then clears caches so the next read is fresh."""
    result = run_etl(project_path)
    clear_project_caches()
    get_polar_data.clear()
    return result
