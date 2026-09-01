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
PIPELINE_VERSION = "5"

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
    cl_n                REAL,   -- N, CL * q * S
    cd_n                REAL,   -- N, CD * q * S
    e                   REAL,

    -- Force coefficients, his.csv report type "Force Coefficient" (mean only,
    -- no _std -- unlike CL/CD these aren't used for downstream derived
    -- columns, just raw values for later inspection/plotting), plus their
    -- dimensionalized {col}_n counterpart in Newtons (F = CF * q * S).
    cfx_sail            REAL,
    cfx_sail_n          REAL,
    cfx_sys              REAL,
    cfx_sys_n            REAL,
    cfy_sail            REAL,
    cfy_sail_n          REAL,
    cfy_sys              REAL,
    cfy_sys_n            REAL,
    cfz_sail            REAL,
    cfz_sail_n          REAL,
    cfz_sys              REAL,
    cfz_sys_n            REAL,

    -- Moment coefficients, his.csv report type "Moment Coefficient", at
    -- several stations (SAIL_BASE, PILLAR_BASE/HALF/1.7D_GND/1.7D_TRANS --
    -- "1.7D" sanitized to "1_7d" for the column name), plus their
    -- dimensionalized {col}_nm counterpart in N*m (M = CM * q * S * chord --
    -- chord is the reference length for all three axes, confirmed with the user).
    cmx_pillar_1_7d_gnd      REAL,
    cmx_pillar_1_7d_gnd_nm   REAL,
    cmx_pillar_1_7d_trans    REAL,
    cmx_pillar_1_7d_trans_nm REAL,
    cmx_pillar_base          REAL,
    cmx_pillar_base_nm       REAL,
    cmx_pillar_half          REAL,
    cmx_pillar_half_nm       REAL,
    cmx_sail_base            REAL,
    cmx_sail_base_nm         REAL,
    cmy_pillar_1_7d_gnd      REAL,
    cmy_pillar_1_7d_gnd_nm   REAL,
    cmy_pillar_1_7d_trans    REAL,
    cmy_pillar_1_7d_trans_nm REAL,
    cmy_pillar_base          REAL,
    cmy_pillar_base_nm       REAL,
    cmy_pillar_half          REAL,
    cmy_pillar_half_nm       REAL,
    cmy_sail_base            REAL,
    cmy_sail_base_nm         REAL,
    cmz_pillar_1_7d_gnd      REAL,
    cmz_pillar_1_7d_gnd_nm   REAL,
    cmz_pillar_1_7d_trans    REAL,
    cmz_pillar_1_7d_trans_nm REAL,
    cmz_pillar_base          REAL,
    cmz_pillar_base_nm       REAL,
    cmz_pillar_half          REAL,
    cmz_pillar_half_nm       REAL,
    cmz_sail_base            REAL,
    cmz_sail_base_nm         REAL,

    -- Center of pressure height, mean of his.csv's ZCP_NATIVE_XZ/ZCP_NATIVE_YZ
    -- "Center of Loads (m)" columns -- both are the same Z coordinate from two
    -- different load projections, averaged into one value (see transform.py).
    z_cp                REAL,   -- m

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
