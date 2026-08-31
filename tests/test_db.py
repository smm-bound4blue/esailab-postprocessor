"""Tests for the shared src.db module: connection setup and the projects table."""

import sqlite3

import pytest

from src.db import get_connection, list_projects, upsert_project


def test_upsert_project_stores_simulation_type(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    project_id = upsert_project(conn, name="proj1", simulation_type="sail", source_path="/data/proj1", span=12.0, chord=2.714)
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    assert row["simulation_type"] == "sail"
    assert row["span"] == 12.0


def test_upsert_project_rejects_invalid_simulation_type(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    with pytest.raises(sqlite3.IntegrityError):
        upsert_project(conn, name="proj1", simulation_type="not_a_real_type", source_path="/data/proj1")


def test_upsert_project_is_idempotent_by_name(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    id1 = upsert_project(conn, name="proj1", simulation_type="sail", source_path="/data/proj1")
    id2 = upsert_project(conn, name="proj1", simulation_type="sail", source_path="/data/proj1-moved")
    assert id1 == id2
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (id1,)).fetchone()
    assert row["source_path"] == "/data/proj1-moved"


def test_list_projects_filters_by_simulation_type(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    upsert_project(conn, name="sail1", simulation_type="sail", source_path="/data/sail1")
    upsert_project(conn, name="harbour1", simulation_type="harbour", source_path="/data/harbour1")

    sail_only = list_projects(conn, simulation_type="sail")
    assert [r["name"] for r in sail_only] == ["sail1"]

    all_projects = list_projects(conn)
    assert {r["name"] for r in all_projects} == {"sail1", "harbour1"}
