"""
sail-specific SQLite helpers: connection (shared schema + this
type's tables) and case/result upserts. No domain logic -- callers
(src.sail.pipeline) pass plain dicts whose keys match
src.sail.schema column names.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.db import get_connection as get_shared_connection
from src.sail.schema import create_schema

_RESULT_COLUMNS = [
    "aoa", "is_stall",
    "cl", "cd", "cl_std", "cd_std", "e",
    "fan_volumetric_flow", "fan_total_pressure", "fan_static_pressure",
    "fan_static_efficiency", "fan_total_efficiency", "fan_power", "cq", "cpow",
    "n_avg", "n_iterations",
    "source_path", "his_csv_mtime", "pipeline_version",
]


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """src.db.get_connection() (shared projects table) + sail's own tables."""
    conn = get_shared_connection(db_path)
    create_schema(conn)
    return conn


def upsert_case(conn: sqlite3.Connection, project_id: int, row: Dict[str, Any]) -> int:
    """Insert or update a sail_cases row by its unique (project_id, case_name). Returns case id."""
    values = {**row, "project_id": project_id}
    conn.execute(
        """
        INSERT INTO sail_cases (project_id, case_name, aws, rpm, source_path)
        VALUES (:project_id, :case_name, :aws, :rpm, :source_path)
        ON CONFLICT(project_id, case_name) DO UPDATE SET
            aws         = excluded.aws,
            rpm         = excluded.rpm,
            source_path = excluded.source_path
        """,
        values,
    )
    result = conn.execute(
        "SELECT id FROM sail_cases WHERE project_id = ? AND case_name = ?",
        (project_id, row["case_name"]),
    ).fetchone()
    return result["id"]


def upsert_result(conn: sqlite3.Connection, case_id: int, row: Dict[str, Any]) -> Tuple[int, bool]:
    """
    Insert or update one sail_results row, keyed on (case_id, aoa).
    Skips the write when the stored his_csv_mtime and pipeline_version
    already match. Returns (result_id, was_written).
    """
    existing = conn.execute(
        "SELECT id, his_csv_mtime, pipeline_version FROM sail_results WHERE case_id = ? AND aoa = ?",
        (case_id, row["aoa"]),
    ).fetchone()

    if (
        existing is not None
        and existing["his_csv_mtime"] == row["his_csv_mtime"]
        and existing["pipeline_version"] == row["pipeline_version"]
    ):
        return existing["id"], False

    values = {**row, "case_id": case_id}
    placeholders = ", ".join(f":{c}" for c in _RESULT_COLUMNS)
    update_clause = ", ".join(f"{c} = excluded.{c}" for c in _RESULT_COLUMNS if c != "aoa")
    conn.execute(
        f"""
        INSERT INTO sail_results (case_id, {', '.join(_RESULT_COLUMNS)})
        VALUES (:case_id, {placeholders})
        ON CONFLICT(case_id, aoa) DO UPDATE SET
            {update_clause},
            processed_at = datetime('now')
        """,
        values,
    )
    result = conn.execute(
        "SELECT id FROM sail_results WHERE case_id = ? AND aoa = ?",
        (case_id, row["aoa"]),
    ).fetchone()
    return result["id"], True
