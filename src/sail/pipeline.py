"""
ETL orchestration for the sail simulation type, plus a CLI entry point.

    python -m src.sail.pipeline sync <project_path>

The Streamlit app never calls src.sail.extract or .transform
directly -- it only reads from the database via src.sail.db, and
triggers this same run_etl() from the "Sail" page's "Run ETL"
button. Keeps "load from transformed results only" true for both the app
and any future cron/CLI use.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import load_config
from src.db import upsert_project
from src.sail import db as sail_db
from src.sail.extract import scan_project
from src.sail.fan import load_fan_curve
from src.sail.schema import PIPELINE_VERSION
from src.sail.transform import add_derived_columns, annotate_is_stall, build_polar_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIMULATION_TYPE = "sail"


@dataclass
class ETLResult:
    project_id: int
    cases_written: int = 0
    aoa_rows_written: int = 0
    aoa_rows_skipped: int = 0


def _get(row: pd.Series, col: str) -> Optional[float]:
    if col not in row.index or pd.isna(row[col]):
        return None
    return float(row[col])


def _build_result_row(prow: pd.Series, pipeline_version: str) -> Dict[str, Any]:
    return {
        "aoa": float(prow["AoA"]),
        "is_stall": bool(prow.get("is_stall", False)),
        "cl": _get(prow, "CL"),
        "cd": _get(prow, "CD"),
        "cl_std": _get(prow, "CL_std"),
        "cd_std": _get(prow, "CD_std"),
        "e": _get(prow, "E"),
        "fan_volumetric_flow": _get(prow, "Fan Volumetric Flow"),
        "fan_total_pressure": _get(prow, "Fan Total Pressure"),
        "fan_static_pressure": _get(prow, "Fan Static Pressure"),
        "fan_static_efficiency": _get(prow, "Fan Static Efficiency"),
        "fan_total_efficiency": _get(prow, "Fan Total Efficiency"),
        "fan_power": _get(prow, "Fan Power"),
        "cq": _get(prow, "CQ"),
        "cpow": _get(prow, "Cpow"),
        "n_avg": int(prow["n_avg"]),
        "n_iterations": int(prow["n_iterations"]),
        "source_path": str(prow["source_path"]),
        "his_csv_mtime": float(prow["his_csv_mtime"]),
        "pipeline_version": pipeline_version,
    }


def run_etl(project_path: Path, db_path: Optional[Path] = None, n_avg: Optional[int] = None) -> ETLResult:
    """
    Full pipeline for one sail project folder (e.g.
    data/sail/2026-08-21_SMM_TEST_9M): src.sail.extract.scan_project
    -> src.sail.transform.build_polar_dataframe / add_derived_columns /
    annotate_is_stall -> src.sail.db upserts.
    """
    project_path = Path(project_path)
    config = load_config()
    n_avg = n_avg or config.pipeline.n_avg

    try:
        fan_curve = load_fan_curve(config.fan_curve_path)
    except Exception as e:
        logger.warning(f"Could not load fan curve from {config.fan_curve_path}: {e}. Fan efficiency/Cpow will be NaN.")
        fan_curve = None

    cases = scan_project(project_path)

    polar_df = build_polar_dataframe(cases, n_avg=n_avg)
    polar_df = add_derived_columns(
        polar_df,
        fan_curve=fan_curve,
        span=config.esail.span,
        chord=config.esail.chord,
        rho=config.pipeline.rho,
        fan_duct_diameter=config.pipeline.fan_duct_diameter,
    )
    polar_df = annotate_is_stall(polar_df)

    conn = sail_db.get_connection(db_path)
    try:
        project_id = upsert_project(
            conn,
            name=project_path.name,
            simulation_type=SIMULATION_TYPE,
            source_path=str(project_path),
            span=config.esail.span,
            chord=config.esail.chord,
        )
        result = ETLResult(project_id=project_id)

        for case in cases:
            case_id = sail_db.upsert_case(
                conn,
                project_id,
                {
                    "case_name": case.case_name,
                    "aws": case.aws,
                    "rpm": case.rpm,
                    "source_path": str(case.folder_path),
                },
            )
            result.cases_written += 1

            case_polar = polar_df[polar_df["Case"] == case.case_name] if not polar_df.empty else polar_df
            for _, prow in case_polar.iterrows():
                result_row = _build_result_row(prow, PIPELINE_VERSION)
                _, was_written = sail_db.upsert_result(conn, case_id, result_row)
                if was_written:
                    result.aoa_rows_written += 1
                else:
                    result.aoa_rows_skipped += 1

        conn.commit()
        logger.info(
            f"Synced '{project_path}': {result.cases_written} cases, "
            f"{result.aoa_rows_written} AoA rows written, {result.aoa_rows_skipped} skipped (unchanged)"
        )
        return result
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync a sail eSAILab CFD project into the results database.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Extract, transform, and load one project folder.")
    sync_parser.add_argument("project_path", type=Path, help="Path to a project folder under data/sail/")
    sync_parser.add_argument("--db-path", type=Path, default=None)
    sync_parser.add_argument("--n-avg", type=int, default=None)

    args = parser.parse_args()

    if args.command == "sync":
        result = run_etl(args.project_path, db_path=args.db_path, n_avg=args.n_avg)
        print(
            f"Synced '{args.project_path}'\n"
            f"  cases: {result.cases_written}\n"
            f"  AoA rows written: {result.aoa_rows_written}\n"
            f"  AoA rows skipped (unchanged): {result.aoa_rows_skipped}"
        )


if __name__ == "__main__":
    main()
