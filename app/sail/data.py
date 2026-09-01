"""
Cached DB query + ETL-trigger helpers for the "Sail" pages. This is the
ONLY way sail pages read results data -- always from src.sail.db / the
SQLite database, never from src.sail.extract or the raw data/ folder
directly.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from app.components.data import clear_project_caches
from src.config import get_default_db_path
from src.sail import db as sail_db
from src.sail.pipeline import ETLResult, run_etl

SIMULATION_TYPE = "sail"


@st.cache_data(ttl=3600)
def get_polar_data(project_ids: Tuple[int, ...], db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    sail_results joined with sail_cases and projects, for one or more
    projects -- the combined polar/performance DataFrame the Sail pages
    plot directly. Always includes a project_name column (even for a
    single project) so callers don't need to special-case the count.
    """
    if not project_ids:
        return pd.DataFrame()

    conn = sail_db.get_connection(db_path or get_default_db_path())
    try:
        placeholders = ",".join("?" * len(project_ids))
        return pd.read_sql(
            f"""
            SELECT p.name AS project_name, c.case_name, c.aws, c.rpm, r.*
            FROM sail_results r
            JOIN sail_cases c ON r.case_id = c.id
            JOIN projects p ON c.project_id = p.id
            WHERE c.project_id IN ({placeholders})
            ORDER BY p.name, c.rpm, r.aoa
            """,
            conn,
            params=project_ids,
        )
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def get_project_summary(db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    One row per sail project: case/AoA counts, RPM/AWS range, and the most
    recent result write (max sail_results.processed_at) as a "last synced"
    signal -- more accurate than projects.created_at, which is set once on
    first insert and untouched by later re-syncs.
    """
    conn = sail_db.get_connection(db_path or get_default_db_path())
    try:
        return pd.read_sql(
            """
            SELECT
                p.id, p.name, p.span, p.chord, p.created_at,
                COUNT(DISTINCT c.id) AS case_count,
                COUNT(r.id) AS aoa_count,
                MIN(c.rpm) AS rpm_min, MAX(c.rpm) AS rpm_max,
                MIN(c.aws) AS aws_min, MAX(c.aws) AS aws_max,
                MAX(r.processed_at) AS last_synced
            FROM projects p
            LEFT JOIN sail_cases c ON c.project_id = p.id
            LEFT JOIN sail_results r ON r.case_id = c.id
            WHERE p.simulation_type = ?
            GROUP BY p.id
            ORDER BY p.created_at DESC
            """,
            conn,
            params=(SIMULATION_TYPE,),
        )
    finally:
        conn.close()


def run_etl_for_project(project_path: Path) -> ETLResult:
    """Wraps src.sail.pipeline.run_etl() for the Sail Home page's 'Run ETL'
    button, then clears caches so the next read is fresh."""
    result = run_etl(project_path)
    clear_project_caches()
    get_polar_data.clear()
    get_project_summary.clear()
    return result
