# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

eSAILab CFD Post-Processor: a minimal Streamlit ETL tool for post-processing CFD simulation results
from bound4blue's eSAILab R&D project — a real 12-meter-span sail built as a test bench on the
Barcelona harbor, used for sensor/instrumentation validation alongside CFD. It is a stripped-down
sibling of `../esail-postprocessor`, which handles multiple production eSail geometries — this repo
intentionally drops that generality since eSAILab is one evolving physical sail, not a catalog.

**Core rule: the Streamlit app never reads `data/` directly.** It only queries the SQLite database
(`db/cfd_results.db`) via each simulation type's `db.py`. All raw-folder parsing lives in each type's
`extract.py`, invoked only by that type's `pipeline.run_etl()` (from the CLI or a "Run ETL" button on
that type's Home page). This keeps "load from transformed results only" true regardless of who's
driving the pipeline.

**Deliberate exception: per-AoA spatial tables (`app/sail/tables.py`).** The `tables/` subfolder under
each AoA folder (point-cloud probes, wind profiles, spanwise force distribution — see "Data Directory
Structure") is read live from disk via `src.sail.spatial_tables`, keyed off `sail_results.source_path`
(the AoA folder path, already stored per synced row — no schema change needed to locate it). These are
bulky raw exports, not the curated polar/performance grain the ETL persists — same reasoning
`esail-postprocessor`'s `AOASimulation.load_table()` uses for its own Spatial Tables page. `src/sail/spatial_tables.py`
is intentionally separate from `src/sail/extract.py`: it's disk I/O the *app* calls directly for display,
never touched by `pipeline.run_etl()`.

## Simulation types

eSAILab runs three kinds of CFD simulation against the real sail/harbor test bench:

| `simulation_type` | Swept parameters | What it studies |
|---|---|---|
| `sail`          | TWS/RPM      | Sail alone, ground-fixed — only AoA rotates (no TWA; TWS == AWS since the sail can't yaw). |
| `harbour`        | TWS/TWA       | Harbor alone, no sail — the environmental flow field the sail sits in. |
| `interference`     | TWS/TWA/RPM    | Sail + harbor together — aerodynamic interference between the two. |

**Only `sail` has data and an implemented pipeline as of 2026-08-31.** `harbour` and `interference`
will each need their own raw folder convention (unknown until that data arrives — don't guess at it),
their own `extract`/`transform`/`schema` (their result variables are entirely different: harbour has no
CL/CD/fan data at all, just flow-field/environmental quantities), and their own Streamlit pages. Each
new type follows the same two-sided pattern as `sail`:

- `src/<type>/{extract,transform,schema,db,pipeline}.py` — mirror `src/sail/*`, adapted to that type's
  actual folder structure and result variables. Only `schema.py`'s `create_schema()` needs wiring into
  a `get_connection()` for that type (see `src/sail/db.py` for the pattern).
- `app/<type>/{home,...}.py` + `app/<type>/data.py` + `app/<type>/plotting.py` (as needed) — mirror
  `app/sail/*`. Register the new pages as a new section in `app/main.py`'s `st.navigation({...})` dict.

Don't build `harbour`/`interference` pipelines speculatively before their real data exists — the exact
tables/columns should be designed against actual raw output, not guessed in advance.

## Why `sail` is simpler than esail-postprocessor

- **No geometry layer.** `esail-postprocessor` nests `SimulationProject -> eSailGeometry ->
  SimulationCase -> AOASimulation` because it compares multiple geometric variants per model. eSAILab's
  sail data has no geometry folder at all (`data/sail/<project>/<case>/<AoA>/his.csv`) — the hierarchy
  here is just `project -> case -> AoA`.
- **No per-fan child table.** eSAILab's `his.csv` only ever has `Fan 1 ...` columns (confirmed against
  real data as of 2026-08-31). Fan columns live directly on `sail_results` instead of a normalized
  `fan_results` table. If a future test campaign adds a second fan, this assumption needs revisiting —
  don't silently ignore `Fan 2 ...` columns if they appear.
- **No hardcoded model catalog.** `esail-postprocessor`'s `eSail` class hardcodes span/chord/max_chord
  for ~13 production models. eSAILab is R&D — its geometry can change between campaigns — so span/chord
  live in `config/config.yaml` instead, edited by hand when the physical model changes.
- **Plain functions, not dataclasses.** `src/sail/` is a functional extract -> transform -> load
  pipeline over DataFrames and dicts, not an OOP class hierarchy. Chosen deliberately over the
  reference repo's dataclass style since there's no geometry/multi-fan nesting left to justify the
  classes.

## Data Directory Structure

```
data/<simulation_type>/<project_name>/         # e.g. data/sail/2026-08-21_SMM_TEST_9M
└── <case_folder>/                             # sail type: AWS_XXkts_RPM_YYYY, e.g. AWS_20kts_RPM_0500
    ├── input/fan_curve.csv                    # provenance copy — same file across every case in a
    │                                           # project (the canonical copy lives in config/, see below)
    └── <aoa_folder>/                          # AoA_XX.X or stall_AoA_XX.X (stall_ prefix is a human-
        └── his.csv                            # authored naming hint only, NOT used to compute is_stall)
```

Per-AoA `tables/` is browsable live (not ETL'd) via `app/sail/tables.py` / `src.sail.spatial_tables` — see
"Deliberate exception" above. Table *names* vary between data batches (`Accumulated Force Table.csv` vs
`accumulated_force_table.csv`; a later batch added `esail_skin_sections_table` and extra `Mean of Cp`/
`Mean of Velocity: Magnitude` columns on some internal-probe tables, confirmed 2026-09-01) — so
`app/sail/tables_plotting.py` classifies by column *shape*, not by table name, into three families, each
with a domain-specific plot (not a generic scatter — these are meaningful engineering views, worked out
directly with the user rather than guessed):

- **`is_profile_height_table()`** (has `Position (m)`, currently just Accumulated Force Table) —
  `plot_height_profile()`: Position on the **vertical** axis, a chosen force variable horizontal.
- **`is_z_profile_table()`** (has `Z (m)`, the shape fallback — `esail_internal_*`, `esail_winglet_anemometer`,
  `wind_profile_*`, `Velocity and Pressure Probe Table`) — `plot_z_profile()`: Z on the **vertical** axis, a
  chosen scalar horizontal. Any subset of these tables can be overlaid on one chart (multiselect on the
  page), since they share this shape; a table missing the selected variable is silently skipped rather
  than erroring.
- **`is_skin_sections_table()`** (has `Pressure Coefficient` + X/Y/Z, currently just
  `esail_skin_sections_table`) — two selectable modes, one trace per Z (height) station either way:
  `plot_cp_vs_xc()` (Cp vs `x/c`, Y-axis inverted by default — suction up, the aerodynamic convention) and
  `plot_skin_sections_xy()` (the XY airfoil cross-section shape, colored by Cp with one shared color range/
  colorbar across stations). `x/c = (X - X_min at that Z) / chord` (`compute_x_over_chord()`,
  `config/config.yaml`'s `esail.chord`) — confirmed with the user that X/Y are in the sail's body-local
  frame, so X is the chordwise axis at every AoA, not just AoA=0.

Other per-case subfolders (`logs/`, `macro/`, `scripts/`, `input/porosity.csv`, per-AoA `plot_img/`,
`plot_csv/`) remain **out of scope** — this tool's ETL only extracts `his.csv` (per type) and
`fan_curve.csv`; revisit if/when STAR-CCM+ image browsing is needed too. `data/` also had ~2300 WSL/NTFS
`:Zone.Identifier` artifact files (cleaned 2026-08-31) — if they reappear after a new data drop, `find
data -name '*:Zone.Identifier' -delete` is safe (pure OS junk, not simulation output).

`his.csv` column names use STAR-CCM+'s `"<name>: <report type> (<units>)"` format (e.g. `"CL: Force
Coefficient"`, `"Fan 1 Total Pressure Rise: Pressure Drop (Pa)"`) — `src.sail.extract.load_his_csv`
strips everything from the first `:` onward, same cleaning `esail-postprocessor`'s
`AOASimulation._load_his_csv` does, so both repos tolerate the same naming quirks (duplicate
`Iteration` columns, `Fan N Total Pressure Rise` naming the same physical quantity as `Fan N Pressure
Drop` elsewhere).

## `config/config.yaml` and `config/fan_curve.csv`

Shared across simulation types that involve the sail (`sail`, `interference` — `harbour` has no sail
and won't reference these):
- `esail.span` / `esail.chord` (meters) — reference area `S = span * chord` for CQ/Cpow. If `chord` is
  ever unset (`null`), CQ and Cpow come out NaN; CL/CD/E are unaffected either way.
- `pipeline.n_avg` — default convergence-averaging window (iterations).
- `pipeline.rho`, `pipeline.fan_duct_diameter` — same constants as `esail-postprocessor`'s
  `esail_geometry.RHO` / `FAN_DUCT_DIAMETER`.

`config/fan_curve.csv` is the canonical fan performance curve (single fan — see above), git-tracked
here rather than read from `data/sail/<case>/input/fan_curve.csv` at extract time, because `data/` is
gitignored raw output and shouldn't be a dependency for a stable structural input. It's byte-identical
to the copies STAR-CCM+ writes into every case's `input/` folder (verified 2026-08-31) — those are
per-case provenance copies, not per-case data. `src.sail.fan.load_fan_curve` always reads from
`config/fan_curve.csv`, never from `data/`.

`.gitignore` has a blanket `*.csv` rule (for raw CFD exports dropped outside `data/`, which is already
fully ignored on its own), with an explicit `!config/*.csv` exception carved out right after it — so
any small, hand-curated reference CSV placed in `config/` (like `fan_curve.csv` and
`wind_probability_matrix.csv`, see "Reference data" below) is git-tracked by default. This bit
`config/fan_curve.csv` once already (2026-08-31: the file existed and CLAUDE.md claimed it was
tracked, but the blanket rule was silently ignoring it until the exception was added) — if a new
config CSV isn't showing up in `git status`, check this rule before assuming something else is wrong.

## Reference data (`app/reference/`)

Static, non-CFD context data shared across simulation types — currently just the Barcelona harbor true
wind probability matrix (`config/wind_probability_matrix.csv`: rows are TWA bin centers in 5° bins,
columns are TWS bin centers in 1 kt bins, values are probability mass summing to ~1 over the whole
matrix). Used to weight which TWS/TWA combinations matter most when planning `harbour`/`interference`
test campaigns.

This does **not** go through a `src/<type>/` ETL pipeline or the database — it's one small, fixed,
hand-provided file, so `app/reference/data.py` reads and `@st.cache_data`-caches it directly, no
extract/transform/schema needed. The "Wind Climate" page (`app/reference/wind_climate.py`, its own
`st.navigation` section in `app/main.py`) renders it as a 3D bar chart via
`app/reference/plotting.py`'s `make_3d_bars()` — a single vectorized `go.Mesh3d` trace (stacked box
geometry, not one Plotly trace per bar) adapted from the same approach in
`../esail-fuel-savings-eedi/src/app.py`'s true-wind probability histogram, so it stays fast even at
72×31 = 2232 bars.

## Database Schema

Shared table (`src/schema.py`):
- `projects` — one row per synced project folder, any type. `simulation_type` (CHECK-constrained to
  `sail`/`harbour`/`interference`) says which type-specific tables to join against. `span`/`chord` are
  nullable (NULL for `harbour`, or before `chord` is configured).

Per-type tables, e.g. `sail` (`src/sail/schema.py`): `sail_cases -> sail_results`, one row per case+AoA
(no `fan_results` child table — see "Why `sail` is simpler" above). `is_stall` is computed analytically
per case (`src.sail.transform.compute_is_stall`: `AoA > AoA_at_CLmax AND CL < CLmax`), not from the
`stall_AoA_` folder-naming convention — mirrors `esail-postprocessor`'s `transform.compute_is_stall()`.

Upserts are keyed on each table's natural `UNIQUE` constraint (re-syncing an updated project overwrites
in place, never duplicates), and `sail_results` additionally skips the write when `his_csv_mtime` and
`pipeline_version` already match — same staleness-check pattern as `esail-postprocessor`'s
`src/processors/db.py`. Each type owns its own `PIPELINE_VERSION` (`src/sail/schema.py` for `sail`) —
bump it when that type's transform logic changes; it doesn't force reprocessing of other types.

`src.db.get_connection()` only ensures the shared `projects` table exists. Each type's own
`db.get_connection()` (e.g. `src.sail.db.get_connection`) wraps it and additionally ensures that type's
tables exist — always go through the type-specific `get_connection()`, not the shared one, when working
with a specific type's data.

## Pipeline Validation

`python -m src.sail.pipeline sync data/sail/2026-08-21_SMM_TEST_9M` has been run end-to-end against the
real eSAILab data (5 cases, 43 AoA rows) and the output sanity-checked: fan efficiency ~40–75%, CQ
~0.007–0.018, Cpow ~0.03, all physically plausible with chord=2.714m. `compute_is_stall` correctly
reproduces the `stall_AoA_` folder-naming hint on every case, and additionally flags AoA=49° in the
`AWS_20kts_RPM_0800` case (CLmax at 47°) that the folder naming alone missed — confirming the analytical
approach generalizes beyond the human-authored hint, as intended. Re-running the sync immediately after
skips all rows (unchanged `his_csv_mtime`), confirming the upsert/staleness logic. The Streamlit app
(`app/main.py`, all 4 Sail pages) was smoke-tested against this same synced data via
`streamlit.testing.v1.AppTest` — no exceptions, and each page's dataframe/chart elements actually
populate (not just an empty shell).

## Performance page: Combined AWS Envelope

The Sail "Performance" page (`app/sail/performance.py`) has three tabs: **Performance Envelope**
(one CLmax-per-RPM point per (Project, AWS) trace), **Interpolated Data** (that same per-trace data
PCHIP-resampled onto a uniform AoA grid, `src.sail.interpolation.interpolate_performance_envelope`),
and **Combined AWS Envelope** — every AWS/RPM trace within a project pooled into *one* curve, for
when AWS traces overlap in AoA/Cpow and a single combined performance curve is needed instead of one
per AWS.

`compute_clmax_data`'s optional `full_polar_traces` (`app/sail/analysis.py`) lets the user pick which
specific (Project, AWS) traces get the "full polar at min RPM" treatment (replace that trace's CLmax
point at its minimum RPM with every AoA point up to CLmax there) rather than an all-or-nothing toggle.
`enforce_cl_monotonic` drops any point whose CL doesn't beat the running max at a lower AoA — pass
`group_cols=["project_name"]` (coarser than its default per-trace grouping) to resolve conflicts
*across* pooled AWS traces too, not just within one.

**Combined AWS Envelope's fitting method (`src.sail.interpolation.smooth_performance_envelope`) went
through two failed approaches before landing on bin-average + PCHIP** — real numeric failures against
this project's actual pooled multi-AWS data, not just theoretical concerns, so don't re-attempt either
without re-reading this:
1. `scipy.interpolate.UnivariateSpline` (cubic smoothing spline): its automatic knot placement was
   numerically fragile on a Cpow column spanning ~0.03 to ~40 (three orders of magnitude) with a
   duplicate AoA across two AWS traces — silently returned all-NaN, or a wildly oscillating fit dipping
   negative, no exception raised either way.
2. A hand-rolled Gaussian-kernel local-linear regression (LOESS): fixed the numerical fragility, but a
   local *line* has no bound of its own — in a real ~22°-wide gap between AoA data points (wider than
   its bandwidth) adjacent to a region where the trend was steepening, it would extrapolate into a dip
   well below every point that went into it. Clamping the local-linear result to its window's [min, max]
   fixed that specific case, but a wider bandwidth just widens the window (and the clamp bound) enough
   to pull in points from multiple genuinely different regimes at once, so the wobble came back at a
   different scale — a narrow fix, not a general one.

The current approach bins AoA into fixed-width bins (`bin_width`, default 10°), averages every
variable within each populated bin (collapsing AWS-to-AWS scatter and overlapping/duplicate AoA values
*before* fitting), then PCHIP-interpolates through those bin averages. No knot-placement step to
destabilize, and PCHIP is shape-preserving between control points, so it can't invent a new local
extremum the binned data doesn't already show. The first/last bin's x-position is pinned back to the
group's true min/max AoA regardless of `bin_width` (only its y stays the bin average) — otherwise a
wide bin_width shrinks the fitted curve's endpoints inward from the real data (confirmed on real data:
two AoA points 5° apart both landing in one 10°-wide edge bin pulled the start from 20° to 22.5° before
this fix).

**Known remaining limitation, not fixable by the fitting method:** pooling multiple AWS conditions that
happen to share the same minimum RPM (e.g. "full polar at min RPM" applied to several traces at once)
can put genuinely different Cpow "shelves" (Cpow is driven mainly by RPM, not AoA alone) over the same
AoA range — no 1D curve fit can resolve that into one physically meaningful value, since Cpow really is
multi-valued at a given AoA once RPM varies independently. When that shows up, either apply "full polar
at min RPM" to fewer traces at once (the `full_polar_traces` multiselect), or use the per-AWS
**Interpolated Data** tab instead, which doesn't pool across AWS.

## Commands

```bash
# Run the ETL for a sail project
python -m src.sail.pipeline sync data/sail/<project_name>

# Run the Streamlit app
streamlit run app/main.py

# Tests
pytest tests/
```
