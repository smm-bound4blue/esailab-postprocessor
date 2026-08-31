"""
Project configuration: loads config/config.yaml and applies .env overrides
(CFD_PROJECT_PATH, CFD_DB_PATH). Single source of truth for the eSAILab
physical spec (span/chord) and pipeline constants (n_avg, rho, fan duct
diameter), shared across simulation types that use them (sail,
interference -- harbour has no sail and won't reference esail/pipeline).

CFD_PROJECT_PATH points at the data/ root, which is organized as
data/<simulation_type>/<project_name>/... -- see get_project_path_for_type().
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_FAN_CURVE_PATH = PROJECT_ROOT / "config" / "fan_curve.csv"
DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "cfd_results.db"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class EsailSpec:
    span: float
    chord: Optional[float]  # None disables CQ/Cpow (they come out NaN); CL/CD/E unaffected


@dataclass
class PipelineConfig:
    n_avg: int
    rho: float
    fan_duct_diameter: float


@dataclass
class AppConfig:
    esail: EsailSpec
    pipeline: PipelineConfig
    title: str
    fan_curve_path: Path


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Read config/config.yaml into an AppConfig. Raises if required keys are missing."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    esail_raw = raw["esail"]
    pipeline_raw = raw["pipeline"]
    app_raw = raw["app"]

    return AppConfig(
        esail=EsailSpec(
            span=float(esail_raw["span"]),
            chord=float(esail_raw["chord"]) if esail_raw.get("chord") is not None else None,
        ),
        pipeline=PipelineConfig(
            n_avg=int(pipeline_raw["n_avg"]),
            rho=float(pipeline_raw["rho"]),
            fan_duct_diameter=float(pipeline_raw["fan_duct_diameter"]),
        ),
        title=app_raw["title"],
        fan_curve_path=DEFAULT_FAN_CURVE_PATH,
    )


def get_default_project_path() -> Path:
    """CFD_PROJECT_PATH env var (the data/ root), else raises -- no hardcoded fallback
    (data/ is gitignored, every environment needs its own path)."""
    env_path = os.environ.get("CFD_PROJECT_PATH")
    if not env_path:
        raise ValueError("CFD_PROJECT_PATH is not set (check .env)")
    return Path(env_path)


def get_project_path_for_type(simulation_type: str) -> Path:
    """data/<simulation_type>/ -- e.g. data/sail/, data/harbour/."""
    return get_default_project_path() / simulation_type


def get_default_db_path() -> Path:
    """CFD_DB_PATH env var, else PROJECT_ROOT / db / cfd_results.db."""
    env_path = os.environ.get("CFD_DB_PATH")
    return Path(env_path) if env_path else DEFAULT_DB_PATH
