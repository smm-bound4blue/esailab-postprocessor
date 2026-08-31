"""
Shared SQLite schema: just the `projects` table, the one thing every
simulation type has in common. Each simulation type owns its own result
tables in its own schema module (e.g. src.sail.schema) and its own
PIPELINE_VERSION, so one type's transform changes never force a reprocess
of another type's data.

    projects (simulation_type: 'sail' | 'harbour' | 'interference')
    -> sail_cases -> sail_results          (src.sail.schema)
    -> harbour_...                                (future)
    -> interference_...                                  (future)
"""

import sqlite3

SIMULATION_TYPES = ("sail", "harbour", "interference")

SCHEMA_DDL = f"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL UNIQUE,
    simulation_type  TEXT NOT NULL CHECK(simulation_type IN {SIMULATION_TYPES!r}),
    source_path      TEXT NOT NULL,
    span             REAL,   -- meters; NULL for harbour (no sail)
    chord            REAL,   -- meters; NULL for harbour, or sail/interference before it's configured
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the shared projects table if it doesn't already exist. Idempotent."""
    conn.executescript(SCHEMA_DDL)
    conn.commit()
