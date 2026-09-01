"""Tests for app.sail.data: multi-project polar queries and the Home page summary query."""

import pytest

from app.sail.data import get_polar_data, get_project_summary
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


def _make_project(tmp_path, name, rpm, cl_base):
    project = tmp_path / name
    case = project / f"AWS_20kts_RPM_{rpm:04d}"
    _write_aoa(case / "AoA_00.0", aoa=0.0, cl=cl_base, cd=0.2)
    _write_aoa(case / "AoA_10.0", aoa=10.0, cl=cl_base + 1.0, cd=0.4)
    return project


@pytest.fixture
def two_projects_db(tmp_path):
    db_path = tmp_path / "test.db"
    proj_a = _make_project(tmp_path / "data", "proj_a", rpm=500, cl_base=1.0)
    proj_b = _make_project(tmp_path / "data", "proj_b", rpm=800, cl_base=2.0)
    run_etl(proj_a, db_path=db_path, n_avg=3)
    run_etl(proj_b, db_path=db_path, n_avg=3)
    return db_path


def test_get_polar_data_empty_when_no_project_ids(two_projects_db):
    assert get_polar_data((), db_path=two_projects_db).empty


def test_get_polar_data_single_project_has_project_name_column(two_projects_db):
    summary = get_project_summary(db_path=two_projects_db)
    proj_a_id = int(summary[summary["name"] == "proj_a"]["id"].iloc[0])

    df = get_polar_data((proj_a_id,), db_path=two_projects_db)
    assert set(df["project_name"]) == {"proj_a"}
    assert len(df) == 2


def test_get_polar_data_multiple_projects_combines_rows(two_projects_db):
    summary = get_project_summary(db_path=two_projects_db)
    ids = tuple(int(i) for i in summary["id"])

    df = get_polar_data(ids, db_path=two_projects_db)
    assert set(df["project_name"]) == {"proj_a", "proj_b"}
    assert len(df) == 4


def _make_project_with_shared_rpm(tmp_path, name):
    """One project with two AWS conditions (5, 20) both at RPM=500, each with
    3 AoA points -- the exact scenario an ORDER BY missing `aws` breaks."""
    project = tmp_path / name
    for aws, cl_base in [(5.0, 1.0), (20.0, 2.0)]:
        case = project / f"AWS_{aws:02.0f}kts_RPM_0500"
        _write_aoa(case / "AoA_00.0", aoa=0.0, cl=cl_base, cd=0.2)
        _write_aoa(case / "AoA_05.0", aoa=5.0, cl=cl_base + 0.5, cd=0.3)
        _write_aoa(case / "AoA_10.0", aoa=10.0, cl=cl_base + 1.0, cd=0.4)
    return project


@pytest.fixture
def shared_rpm_db(tmp_path):
    db_path = tmp_path / "shared_rpm.db"
    project = _make_project_with_shared_rpm(tmp_path / "data", "proj_shared")
    run_etl(project, db_path=db_path, n_avg=3)
    return db_path


def test_get_polar_data_groups_by_aws_before_rpm_when_rpm_is_shared(shared_rpm_db):
    summary = get_project_summary(db_path=shared_rpm_db)
    project_id = int(summary["id"].iloc[0])

    df = get_polar_data((project_id,), db_path=shared_rpm_db)
    assert len(df) == 6  # 2 aws x 3 aoa

    for (_aws, _rpm), group in df.groupby(["aws", "rpm"], sort=False):
        # AoA ascending within the trace
        assert list(group["aoa"]) == sorted(group["aoa"])
        # a contiguous run of rows -- not interleaved with the other AWS
        # trace that shares the same RPM
        start = group.index[0]
        assert list(group.index) == list(range(start, start + len(group)))


def test_get_project_summary_counts_and_ranges(two_projects_db):
    summary = get_project_summary(db_path=two_projects_db)
    assert set(summary["name"]) == {"proj_a", "proj_b"}

    row_a = summary[summary["name"] == "proj_a"].iloc[0]
    assert row_a["case_count"] == 1
    assert row_a["aoa_count"] == 2
    assert row_a["rpm_min"] == row_a["rpm_max"] == 500.0
    assert row_a["aws_min"] == row_a["aws_max"] == 20.0
    assert row_a["last_synced"] is not None
