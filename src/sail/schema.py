"""
SQLite schema for the sail simulation type: sail rotated through an
AoA sweep at fixed AWS/RPM operating points, no harbour geometry. One row
per case+AoA (no fan_results child table -- single fan, see CLAUDE.md).

    projects (shared, src.schema) -> sail_cases -> sail_results
"""

import sqlite3

# Bump whenever this type's transform logic changes, so a re-sync with
# unchanged his.csv files still forces full reprocessing. Independent of
# other simulation types' pipeline versions.
PIPELINE_VERSION = "1"

SCHEMA_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sail_cases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    case_name    TEXT NOT NULL,
    aws          REAL NOT NULL,   -- knots (== TWS; sail is ground-fixed, only AoA rotates)
    rpm          REAL NOT NULL,
    source_path  TEXT NOT NULL,
    UNIQUE(project_id, case_name)
);

CREATE TABLE IF NOT EXISTS sail_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id             INTEGER NOT NULL REFERENCES sail_cases(id) ON DELETE CASCADE,
    aoa                 REAL NOT NULL,
    is_stall            INTEGER NOT NULL DEFAULT 0,

    cl                  REAL,
    cd                  REAL,
    cl_std              REAL,
    cd_std              REAL,
    e                   REAL,

    fan_volumetric_flow     REAL,   -- m^3/s
    fan_total_pressure      REAL,   -- Pa (FTP)
    fan_static_pressure     REAL,   -- Pa (FSP)
    fan_static_efficiency   REAL,   -- dimensionless fraction
    fan_total_efficiency    REAL,   -- dimensionless fraction
    fan_power               REAL,   -- kW
    cq                       REAL,   -- dimensionless flow coefficient
    cpow                      REAL,   -- dimensionless power coefficient

    n_avg               INTEGER,
    n_iterations        INTEGER,

    source_path          TEXT NOT NULL,
    his_csv_mtime         REAL NOT NULL,
    pipeline_version       TEXT NOT NULL,
    processed_at             TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(case_id, aoa)
);

CREATE INDEX IF NOT EXISTS idx_sail_cases_project    ON sail_cases(project_id);
CREATE INDEX IF NOT EXISTS idx_sail_results_case     ON sail_results(case_id);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create sail's tables/indexes if they don't already exist. Idempotent.
    Assumes the shared projects table already exists (src.db.get_connection ensures this)."""
    conn.executescript(SCHEMA_DDL)
    conn.commit()
