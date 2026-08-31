"""
Shared, simulation-type-agnostic DB helpers for the Streamlit app.
Type-specific query helpers (e.g. the sail polar/performance query) live
in each type's own app/<type>/data.py, since the result columns differ
per type. This module only knows about the shared `projects` table.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src.config import get_default_db_path
from src.db import get_connection


@st.cache_data(ttl=3600)
def list_projects(simulation_type: Optional[str] = None) -> pd.DataFrame:
    """Rows from the projects table, optionally filtered to one simulation_type."""
    conn = get_connection(get_default_db_path())
    try:
        if simulation_type is not None:
            return pd.read_sql(
                "SELECT * FROM projects WHERE simulation_type = ? ORDER BY created_at DESC",
                conn,
                params=(simulation_type,),
            )
        return pd.read_sql("SELECT * FROM projects ORDER BY created_at DESC", conn)
    finally:
        conn.close()


def clear_project_caches() -> None:
    """Clear cached project/result queries after an ETL run so the next read is fresh.
    Each type's app/<type>/data.py should clear its own result-query cache too."""
    list_projects.clear()
