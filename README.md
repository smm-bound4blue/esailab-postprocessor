# eSAILab CFD Post-Processor

A minimal Streamlit tool for post-processing CFD simulation results from **eSAILab**, bound4blue's R&D
project centered on a real 12-meter-span sail built as a test bench on the Barcelona harbor (sensors,
instrumentation, and validation, alongside CFD). It's an ETL pipeline plus a thin viewer: raw STAR-CCM+
exports go in, a SQLite database of processed results comes out, and the Streamlit app reads polar/
performance results from that database only — with one deliberate exception (per-AoA spatial tables,
read live from disk; see CLAUDE.md).

## Simulation types

eSAILab runs three kinds of CFD simulation, each with its own raw data shape, its own ETL pipeline, and
its own set of Streamlit pages (grouped as sidebar sections):

| Type          | Swept parameters      | Notes                                                        |
|---------------|------------------------|---------------------------------------------------------------|
| `sail`        | TWS/RPM                | Sail alone, ground-fixed — only AoA rotates. TWS == AWS.       |
| `harbour`     | TWS/TWA                | Harbor alone, no sail — flow-field/environmental data only.    |
| `interference`| TWS/TWA/RPM            | Sail + harbor together — aerodynamic interference between them.|

Only `sail` has data and an implemented pipeline today. `harbour` and `interference` follow the same
`src/<type>/` + `app/<type>/` pattern once their raw data and result variables are known — see
CLAUDE.md.

## Project Structure

```
esailab-postprocessor/
├── config/
│   ├── config.yaml                     # eSAILab span/chord, air density, n_avg default
│   ├── fan_curve.csv                    # canonical fan performance curve (sail + interference types)
│   └── wind_probability_matrix.csv       # Barcelona harbor TWA/TWS wind climate (git-tracked reference data)
├── data/                    # raw CFD exports (gitignored)
│   └── <simulation_type>/<project>/...
├── db/
│   └── cfd_results.db       # SQLite — the transformed store (gitignored, ETL output)
├── src/
│   ├── config.py            # shared config loader
│   ├── schema.py             # shared `projects` table (simulation_type column)
│   ├── db.py                  # shared connection + projects upsert
│   └── sail/                  # sail-type ETL pipeline
│       ├── extract.py           # raw folder tree -> DataFrames
│       ├── fan.py                # fan curve + affinity-law scaling
│       ├── transform.py           # convergence stats, derived columns, is_stall
│       ├── schema.py               # sail_cases / sail_results tables
│       ├── db.py                    # sail-specific upserts
│       ├── spatial_tables.py         # live disk reads of per-AoA tables/ (not ETL'd, see CLAUDE.md)
│       └── pipeline.py                # orchestration + CLI entry point
├── app/                     # Streamlit app — reads from db/cfd_results.db (+ live tables/, see above)
│   ├── main.py               # entry point: st.navigation sidebar sections
│   ├── components/            # shared across all simulation types
│   ├── sail/                   # Sail pages + query/plotting helpers
│   └── reference/               # static, non-CFD context data (e.g. wind climate)
└── tests/
    ├── test_db.py            # shared schema/db tests
    ├── test_reference.py      # wind matrix + 3D bar chart tests
    └── sail/                   # sail pipeline tests
```

Note: `.gitignore` blanket-ignores `*.csv` (raw CFD exports), with an explicit `!config/*.csv`
exception — small, hand-curated reference inputs like `fan_curve.csv` and
`wind_probability_matrix.csv` are meant to be git-tracked.

## Data Directory Structure

```
data/<simulation_type>/<project_name>/         # e.g. data/sail/2026-08-21_SMM_TEST_9M
└── <case_folder>/                             # e.g. AWS_20kts_RPM_0500 (sail type)
    ├── input/fan_curve.csv                    # provenance copy (canonical one lives in config/)
    └── <aoa_folder>/                          # e.g. AoA_10.0 or stall_AoA_21.0
        └── his.csv                            # simulation monitor data
```

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # then set CFD_PROJECT_PATH to your data/ directory
```

## Usage

Run the ETL for a sail project:

```bash
python -m src.sail.pipeline sync data/sail/2026-08-21_SMM_TEST_9M
```

Launch the app (reads from the database only):

```bash
streamlit run app/main.py
```

## Testing

```bash
pytest tests/
```
