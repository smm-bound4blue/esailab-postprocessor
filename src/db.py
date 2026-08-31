"""
Low-level SQLite connection + the shared `projects` upsert. Type-specific
tables (sail_cases, sail_results, ...) are created and upserted
by each simulation type's own db module (e.g. src.sail.db) -- this
module only knows about the schema every type shares.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from src.config import get_default_db_path
from src.schema import create_schema as create_shared_schema


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Open a connection (WAL mode, foreign keys on, row access by column
    name), ensuring the shared schema exists. Defaults to
    src.config.get_default_db_path().

    Does NOT create type-specific tables -- callers using a given
    simulation type's tables must also call that type's own
    schema.create_schema(conn) (src.sail.db.get_connection() does
    this for sail).
    """
    path = Path(db_path) if db_path is not None else get_default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    create_shared_schema(conn)
    return conn


def upsert_project(
    conn: sqlite3.Connection,
    name: str,
    simulation_type: str,
    source_path: str,
    span: Optional[float] = None,
    chord: Optional[float] = None,
) -> int:
    """Insert or update a project row by its unique name. Returns project id."""
    conn.execute(
        """
        INSERT INTO projects (name, simulation_type, source_path, span, chord)
        VALUES (:name, :simulation_type, :source_path, :span, :chord)
        ON CONFLICT(name) DO UPDATE SET
            simulation_type = excluded.simulation_type,
            source_path     = excluded.source_path,
            span            = excluded.span,
            chord           = excluded.chord
        """,
        {
            "name": name,
            "simulation_type": simulation_type,
            "source_path": source_path,
            "span": span,
            "chord": chord,
        },
    )
    row = conn.execute("SELECT id FROM projects WHERE name = ?", (name,)).fetchone()
    return row["id"]


def list_projects(conn: sqlite3.Connection, simulation_type: Optional[str] = None):
    """All project rows, optionally filtered to one simulation_type."""
    if simulation_type is not None:
        return conn.execute(
            "SELECT * FROM projects WHERE simulation_type = ? ORDER BY created_at DESC", (simulation_type,)
        ).fetchall()
    return conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
