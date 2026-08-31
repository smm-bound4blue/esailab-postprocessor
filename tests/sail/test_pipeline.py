"""Tests for src.sail.pipeline.run_etl: end-to-end extract -> transform -> load, and re-sync skip-unchanged behavior.

Uses the real config/config.yaml + config/fan_curve.csv (span/chord/fan curve are fast, static
inputs) against a synthetic project folder under tmp_path, with its own throwaway sqlite db.
"""

import sqlite3

import pandas as pd
import pytest

from src.sail.pipeline import run_etl

HIS_CSV_HEADER = (
    '"Iteration: Iteration","AoA: Expression","CL: Force Coefficient","CD: Force Coefficient",'
    '"Fan 1 Volumetric Flow: Expression (m^3/s)","Fan 1 Total Pressure Rise: Pressure Drop (Pa)"\n'
)


def _write_aoa(folder, aoa, cl, cd, n_rows=5):
    folder.mkdir(parents=True)
    lines = [HIS_CSV_HEADER]
    for i in range(1, n_rows + 1):
        lines.append(f"{i},{aoa},{cl},{cd},5.0,100.0\n")
    (folder / "his.csv").write_text("".join(lines))


@pytest.fixture
def synthetic_project(tmp_path):
    project = tmp_path / "TEST_PROJECT"
    case = project / "AWS_20kts_RPM_0500"
    _write_aoa(case / "AoA_00.0", aoa=0.0, cl=1.0, cd=0.2)
    _write_aoa(case / "AoA_10.0", aoa=10.0, cl=2.0, cd=0.4)
    return project


def test_run_etl_writes_expected_row_counts(synthetic_project, tmp_path):
    db_path = tmp_path / "test.db"
    result = run_etl(synthetic_project, db_path=db_path, n_avg=3)

    assert result.cases_written == 1
    assert result.aoa_rows_written == 2
    assert result.aoa_rows_skipped == 0

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM sail_results ORDER BY aoa", conn)
    conn.close()
    assert len(df) == 2
    assert df["cl"].tolist() == pytest.approx([1.0, 2.0])


def test_run_etl_skips_unchanged_aoa_rows_on_resync(synthetic_project, tmp_path):
    db_path = tmp_path / "test.db"
    first = run_etl(synthetic_project, db_path=db_path, n_avg=3)
    second = run_etl(synthetic_project, db_path=db_path, n_avg=3)

    assert second.aoa_rows_written == 0
    assert second.aoa_rows_skipped == first.aoa_rows_written
