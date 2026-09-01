"""Tests for src.sail.spatial_tables: direct-from-disk table discovery/loading."""

from src.sail.spatial_tables import list_table_files, load_table, tables_dir


def test_tables_dir_is_source_path_slash_tables(tmp_path):
    assert tables_dir(str(tmp_path)) == tmp_path / "tables"


def test_list_table_files_empty_when_no_tables_dir(tmp_path):
    assert list_table_files(str(tmp_path)) == []


def test_list_table_files_sorted_stems(tmp_path):
    d = tmp_path / "tables"
    d.mkdir()
    (d / "b_table.csv").write_text("X (m)\n1\n")
    (d / "a_table.csv").write_text("X (m)\n1\n")
    assert list_table_files(str(tmp_path)) == ["a_table", "b_table"]


def test_load_table_returns_empty_df_when_missing(tmp_path):
    assert load_table(str(tmp_path), "does_not_exist").empty


def test_load_table_reads_csv(tmp_path):
    d = tmp_path / "tables"
    d.mkdir()
    (d / "esail_internal_center_table.csv").write_text("X (m),Y (m)\n1.0,2.0\n3.0,4.0\n")
    df = load_table(str(tmp_path), "esail_internal_center_table")
    assert list(df.columns) == ["X (m)", "Y (m)"]
    assert len(df) == 2
