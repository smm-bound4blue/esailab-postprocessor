"""Tests for src.sail.transform: convergence stats, derived columns, is_stall."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sail.extract import AoaFolder, CaseFolder
from src.sail.fan import FanCurve
from src.sail.transform import (
    add_derived_columns,
    annotate_is_stall,
    build_polar_dataframe,
    compute_is_stall,
    convergence_stats,
    convergence_std,
)


def _aoa_data(n=10, cl=1.0, cd=0.2, fan_flow=5.0, fan_dp=100.0):
    return pd.DataFrame(
        {
            "CL": [cl] * n,
            "CD": [cd] * n,
            "Fan 1 Volumetric Flow": [fan_flow] * n,
            "Fan 1 Total Pressure Rise": [fan_dp] * n,
        }
    )


def test_convergence_stats_mean_over_last_n_iterations():
    data = pd.DataFrame({"CL": [1.0, 2.0, 3.0, 4.0]})
    stats = convergence_stats(data, n_avg=2)
    assert stats["CL"] == pytest.approx(3.5)


def test_convergence_stats_raises_when_n_avg_too_large():
    data = pd.DataFrame({"CL": [1.0, 2.0]})
    with pytest.raises(ValueError):
        convergence_stats(data, n_avg=5)


def test_convergence_std():
    data = pd.DataFrame({"CL": [1.0, 3.0]})
    std = convergence_std(data, n_avg=2)
    assert std["CL"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


def _make_case(case_name="AWS_20kts_RPM_0500", aws=20.0, rpm=500.0):
    case = CaseFolder(case_name=case_name, aws=aws, rpm=rpm, folder_path=Path(f"/tmp/{case_name}"))
    for aoa, cl in [(0.0, 1.0), (10.0, 2.0), (20.0, 1.8)]:
        case.aoas.append(
            AoaFolder(
                aoa=aoa,
                is_stall_named=False,
                folder_path=Path(f"/tmp/{case_name}/AoA_{aoa}"),
                data=_aoa_data(cl=cl),
            )
        )
    return case


def test_build_polar_dataframe_one_row_per_case_aoa():
    df = build_polar_dataframe([_make_case()], n_avg=5)
    assert len(df) == 3
    assert set(df["AoA"]) == {0.0, 10.0, 20.0}
    assert "CL_std" in df.columns


def test_build_polar_dataframe_skips_aoa_with_too_few_iterations():
    case = _make_case()
    case.aoas[0].data = _aoa_data(n=2)  # fewer rows than n_avg
    df = build_polar_dataframe([case], n_avg=5)
    assert len(df) == 2


def test_add_derived_columns_e_is_cl_over_cd():
    polar_df = pd.DataFrame({"CL": [2.0], "CD": [0.5], "AWS": [20.0], "RPM": [500.0]})
    result = add_derived_columns(polar_df, fan_curve=None, span=12.0, chord=2.714, rho=1.2, fan_duct_diameter=1.25)
    assert result["E"].iloc[0] == pytest.approx(4.0)


def test_add_derived_columns_cq_is_nan_when_chord_missing():
    polar_df = pd.DataFrame(
        {"CL": [2.0], "CD": [0.5], "AWS": [20.0], "RPM": [500.0], "Fan 1 Volumetric Flow": [5.0]}
    )
    result = add_derived_columns(polar_df, fan_curve=None, span=12.0, chord=None, rho=1.2, fan_duct_diameter=1.25)
    assert pd.isna(result["CQ"].iloc[0])


def test_add_derived_columns_computes_fan_efficiency_with_curve_and_chord():
    polar_df = pd.DataFrame(
        {
            "CL": [2.0],
            "CD": [0.5],
            "AWS": [20.0],
            "RPM": [500.0],
            "Fan 1 Volumetric Flow": [5.0],
            "Fan 1 Total Pressure Rise": [100.0],
        }
    )
    curve = FanCurve(
        reference_rpm=1465.0,
        flowrate=pd.Series([1.0, 10.0, 20.0]),
        pressure=pd.Series([1000.0, 500.0, 100.0]),
        power=pd.Series([5.0, 10.0, 15.0]),
    )
    result = add_derived_columns(polar_df, fan_curve=curve, span=12.0, chord=2.714, rho=1.2, fan_duct_diameter=1.25)
    # Not asserting a physically realistic efficiency range here -- the synthetic
    # curve's magnitudes are arbitrary. Realism is checked separately against the
    # real eSAILab fan curve (see manual pipeline run notes in CLAUDE.md).
    assert result["Fan Power"].iloc[0] > 0
    assert result["Cpow"].iloc[0] > 0
    assert result["Fan Total Efficiency"].iloc[0] > 0


def test_compute_is_stall_flags_post_clmax_points():
    case_df = pd.DataFrame({"AoA": [0.0, 10.0, 20.0, 21.0], "CL": [1.0, 2.0, 2.5, 2.3]})
    is_stall = compute_is_stall(case_df)
    assert list(is_stall) == [False, False, False, True]


def test_annotate_is_stall_per_case():
    polar_df = pd.DataFrame(
        {
            "Case": ["A", "A", "B", "B"],
            "AoA": [0.0, 10.0, 0.0, 10.0],
            "CL": [1.0, 0.5, 1.0, 2.0],
        }
    )
    result = annotate_is_stall(polar_df)
    assert result[result["Case"] == "A"]["is_stall"].tolist() == [False, True]
    assert result[result["Case"] == "B"]["is_stall"].tolist() == [False, False]
