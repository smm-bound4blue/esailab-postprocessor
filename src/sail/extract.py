"""
Extract: walk the raw data/<project>/<case>/<AoA> folder tree and load
his.csv into DataFrames. No derived columns or statistics here — that's
src.sail.transform's job. Pure filesystem + pandas I/O.

Folder convention (see CLAUDE.md):
    data/<project_name>/<case_folder AWS_XXkts_RPM_YYYY>/<AoA_XX.X | stall_AoA_XX.X>/his.csv

The fan curve is NOT extracted from here — it's a git-tracked structural
input read directly from config/fan_curve.csv by src.sail.fan.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

_AWS_PATTERN = re.compile(r"AWS_(\d+(?:\.\d+)?)kts")
_RPM_PATTERN = re.compile(r"RPM_(\d+(?:\.\d+)?)")


@dataclass
class AoaFolder:
    aoa: float
    is_stall_named: bool  # True if folder was named stall_AoA_* (naming hint only, not authoritative)
    folder_path: Path
    data: pd.DataFrame  # cleaned his.csv contents


@dataclass
class CaseFolder:
    case_name: str
    aws: float
    rpm: float
    folder_path: Path
    aoas: List[AoaFolder] = field(default_factory=list)


def load_his_csv(path: Path) -> pd.DataFrame:
    """
    Load and clean one his.csv: strip whitespace, drop ' Monitor:' / ':...'
    suffixes and 'Body 2.' prefixes from column names, drop duplicate columns.
    Mirrors the cleaning esail-postprocessor's AOASimulation._load_his_csv does.
    """
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    df.rename(columns=lambda name: name.split(" Monitor:")[0], inplace=True)
    df.rename(columns=lambda name: name.split(":")[0], inplace=True)
    df.rename(columns=lambda name: name.replace("Body 2.", ""), inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df.columns = df.columns.str.strip()

    return df


def parse_case_name(case_name: str) -> Dict[str, float]:
    """Extract {'aws': ..., 'rpm': ...} from a case folder name like 'AWS_20kts_RPM_0500'."""
    aws_match = _AWS_PATTERN.search(case_name)
    rpm_match = _RPM_PATTERN.search(case_name)
    if not aws_match:
        raise ValueError(f"Could not extract AWS from case name: {case_name}")
    if not rpm_match:
        raise ValueError(f"Could not extract RPM from case name: {case_name}")
    return {"aws": float(aws_match.group(1)), "rpm": float(rpm_match.group(1))}


def parse_aoa_folder_name(folder_name: str) -> Dict[str, object]:
    """Extract {'aoa': float, 'is_stall_named': bool} from 'AoA_20.0' or 'stall_AoA_21.0'."""
    if folder_name.startswith("stall_AoA_"):
        return {"aoa": float(folder_name.replace("stall_AoA_", "")), "is_stall_named": True}
    if folder_name.startswith("AoA_"):
        return {"aoa": float(folder_name.replace("AoA_", "")), "is_stall_named": False}
    raise ValueError(f"Invalid AoA folder name format: {folder_name}")


def _is_case_folder(d: Path) -> bool:
    return d.is_dir() and "AWS_" in d.name and "RPM_" in d.name


def _is_aoa_folder(d: Path) -> bool:
    return d.is_dir() and (d.name.startswith("AoA_") or d.name.startswith("stall_AoA_"))


def scan_project(project_path: Path) -> List[CaseFolder]:
    """Walk project_path for case folders, and each case for AoA folders, loading his.csv
    for every AoA. Skips folders that don't match the AWS_.../AoA_... naming convention."""
    project_path = Path(project_path)
    if not project_path.exists():
        raise FileNotFoundError(f"Project folder not found: {project_path}")

    cases: List[CaseFolder] = []

    for case_dir in sorted(d for d in project_path.iterdir() if _is_case_folder(d)):
        try:
            parsed = parse_case_name(case_dir.name)
        except ValueError as e:
            logger.warning(f"Skipping case folder '{case_dir.name}': {e}")
            continue

        case = CaseFolder(
            case_name=case_dir.name,
            aws=parsed["aws"],
            rpm=parsed["rpm"],
            folder_path=case_dir,
        )

        for aoa_dir in sorted(d for d in case_dir.iterdir() if _is_aoa_folder(d)):
            his_path = aoa_dir / "his.csv"
            if not his_path.exists():
                logger.warning(f"Skipping AoA folder '{aoa_dir.name}' — no his.csv found")
                continue

            aoa_parsed = parse_aoa_folder_name(aoa_dir.name)
            try:
                data = load_his_csv(his_path)
            except Exception as e:
                logger.warning(f"Failed to load {his_path}: {e}")
                continue

            case.aoas.append(
                AoaFolder(
                    aoa=aoa_parsed["aoa"],
                    is_stall_named=aoa_parsed["is_stall_named"],
                    folder_path=aoa_dir,
                    data=data,
                )
            )

        if not case.aoas:
            logger.warning(f"No AoA folders with his.csv found in case '{case_dir.name}'")

        cases.append(case)

    if not cases:
        logger.warning(f"No case folders found in project '{project_path}'")

    return cases
