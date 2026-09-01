"""
Direct-from-disk access to per-AoA tables/ (spatial point-cloud and
spanwise-profile CSVs) -- mirrors esail-postprocessor's AOASimulation
.tables_dir / .list_table_files() / .load_table(). Deliberately NOT part
of the ETL pipeline: these are bulky raw exports, not the curated polar/
performance grain that src.sail.pipeline persists to the database (see
CLAUDE.md's "Core rule"). The app reads them live, keyed off
sail_results.source_path -- the AoA folder path already stored per synced
row, so no new schema is needed to locate them.
"""

from pathlib import Path
from typing import List

import pandas as pd


def tables_dir(aoa_source_path: str) -> Path:
    return Path(aoa_source_path) / "tables"


def list_table_files(aoa_source_path: str) -> List[str]:
    """Sorted stem names of tables/*.csv for one AoA folder. Empty if the
    AoA folder or its tables/ subfolder isn't present on disk (e.g. moved
    or deleted since the last sync -- the DB row can still exist)."""
    d = tables_dir(aoa_source_path)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.csv"))


def load_table(aoa_source_path: str, stem: str) -> pd.DataFrame:
    """Load tables/<stem>.csv. Returns an empty DataFrame if not found."""
    path = tables_dir(aoa_source_path) / f"{stem}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
