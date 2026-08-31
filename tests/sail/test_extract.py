"""Tests for src.sail.extract: his.csv cleaning, case/AoA folder name parsing, project scan."""

from pathlib import Path

import pandas as pd
import pytest

from src.sail.extract import load_his_csv, parse_aoa_folder_name, parse_case_name, scan_project


def test_load_his_csv_cleans_column_names(tmp_path):
    raw = tmp_path / "his.csv"
    raw.write_text(
        '"Iteration: Iteration","CL: Force Coefficient","Body 2.CD: Force Coefficient",'
        '"Iteration: Iteration"\n'
        "1,0.5,0.1,1\n2,0.6,0.11,2\n"
    )
    df = load_his_csv(raw)
    assert list(df.columns) == ["Iteration", "CL", "CD"]
    assert len(df) == 2


def test_parse_case_name_extracts_aws_and_rpm():
    assert parse_case_name("AWS_20kts_RPM_0500") == {"aws": 20.0, "rpm": 500.0}


def test_parse_case_name_raises_on_bad_name():
    with pytest.raises(ValueError):
        parse_case_name("not_a_case_folder")


def test_parse_aoa_folder_name_handles_stall_prefix():
    assert parse_aoa_folder_name("AoA_20.0") == {"aoa": 20.0, "is_stall_named": False}
    assert parse_aoa_folder_name("stall_AoA_21.0") == {"aoa": 21.0, "is_stall_named": True}


def test_scan_project_finds_all_cases_and_aoas(tmp_path):
    project = tmp_path / "proj"
    case_dir = project / "AWS_20kts_RPM_0500" / "AoA_00.0"
    case_dir.mkdir(parents=True)
    (case_dir / "his.csv").write_text('"CL: Force Coefficient"\n0.5\n0.6\n')

    # a folder that should be skipped (no AWS/RPM in name)
    (project / "AWS_20kts_RPM_0500" / "input").mkdir(parents=True)

    cases = scan_project(project)
    assert len(cases) == 1
    assert cases[0].case_name == "AWS_20kts_RPM_0500"
    assert cases[0].aws == 20.0
    assert cases[0].rpm == 500.0
    assert len(cases[0].aoas) == 1
    assert cases[0].aoas[0].aoa == 0.0
    assert isinstance(cases[0].aoas[0].data, pd.DataFrame)


def test_scan_project_raises_on_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_project(tmp_path / "does_not_exist")
