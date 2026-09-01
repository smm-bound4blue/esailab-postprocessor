"""
Fan Performance: two ways to sanity-check the fan's CFD-derived behavior
against independent references -- the manufacturer's performance curve,
and a Bernoulli-based flow-rate estimate from two static-pressure probes.
Its own page (not a Performance tab) since there's room to grow here --
more fan-specific views can join it later.
"""

from pathlib import Path

import streamlit as st

from app.sail.data import get_flow_rate_estimates
from app.sail.plotting import plot_fan_performance_curve, plot_flow_rate_parity
from app.sail.widgets import select_projects_and_load
from src.config import load_config
from src.sail.fan import build_reference_curve, duct_area, load_fan_curve
from src.sail.flow_estimation import DEFAULT_P1_HEIGHT_M, DEFAULT_XI, ellipse_area

_ASSETS_DIR = Path(__file__).parent / "assets"
SAIL_CUTAWAY_IMAGE_PATH = _ASSETS_DIR / "sail_cutaway.webp"
P1_P2_IMAGE_PATH = _ASSETS_DIR / "p1_p2_probe_locations.webp"

st.title("Fan Performance")

polar_df = select_projects_and_load(key="fan_performance_projects")

if polar_df["fan_static_pressure"].isna().all() and polar_df["fan_total_pressure"].isna().all():
    st.warning("Fan pressure data is not available for the selected project(s).")
    st.stop()

config = load_config()

tab_reference, tab_flow_estimation = st.tabs(["Reference Curve", "Flow Rate Estimation"])

with tab_reference:
    st.caption(
        "Simulated Fan Static/Total Pressure (full AoA sweep) overlaid on the manufacturer's "
        "reference curve, for one or more fan speeds."
    )
    try:
        fan_curve = load_fan_curve(config.fan_curve_path)
    except Exception as e:
        st.warning(f"Could not load config/fan_curve.csv: {e}")
        fan_curve = None

    if fan_curve is None:
        st.warning("No reference fan curve available.")
    else:
        col_rpm, col_total = st.columns(2)
        rpm_options = sorted(polar_df["rpm"].unique())
        selected_rpms = col_rpm.multiselect(
            "Fan speed (RPM)", rpm_options, default=rpm_options[:1], format_func=lambda r: f"{r:g}"
        )
        show_total = col_total.checkbox("Show Fan Total Pressure", value=True)

        if not selected_rpms:
            st.info("Select at least one fan speed.")
        else:
            sim_df = polar_df[polar_df["rpm"].isin(selected_rpms)]
            reference_curves = {
                rpm: build_reference_curve(
                    fan_curve, rpm=rpm, rho=config.pipeline.rho, duct_diameter=config.pipeline.fan_duct_diameter
                )
                for rpm in selected_rpms
            }
            st.plotly_chart(
                plot_fan_performance_curve(sim_df, reference_curves, show_total=show_total),
                width="stretch",
            )

            rpm_list_str = ", ".join(f"{r:g}" for r in sorted(selected_rpms))
            with st.expander("How is this calculated?"):
                st.markdown(
                    f"""
**Reference curves (grey/black lines, darker = higher RPM)** — the manufacturer's
datasheet for this fan, `config/fan_curve.csv` ({fan_curve.reference_rpm:g} RPM reference),
one solid/dashed pair per selected RPM ({rpm_list_str}):

1. The datasheet gives **Static Pressure vs Volumetric Flow Rate** at
   {fan_curve.reference_rpm:g} RPM — that's the raw curve in the CSV.
2. Scaled to each selected RPM using the fan affinity laws (`N` = RPM ratio):
   `Flow ∝ N`, `Pressure ∝ N²`, `Power ∝ N³`.
3. **Total Pressure** (dashed) = Static Pressure + Dynamic Pressure, where
   `Dynamic Pressure = 0.5 × ρ × V²` and `V = Flow Rate / Duct Area`
   (`config.yaml`: `pipeline.rho` = {config.pipeline.rho:g} kg/m³,
   `pipeline.fan_duct_diameter` = {config.pipeline.fan_duct_diameter:g} m).

**Simulated points (● Static, ♦ Total)** — one color per (Project, AWS, RPM); every
marker is one AoA from that case's CFD sweep. Both families share one legend group per
RPM, so toggling a group hides/shows that RPM's reference curve together with every
simulated trace at that RPM:

1. **Fan Volumetric Flow** and **Fan Total Pressure** come straight from the his.csv fan
   monitor (`Fan 1 Pressure Drop` / `Fan 1 Total Pressure` / `Fan 1 Total Pressure Rise` —
   whichever STAR-CCM+ exported for that project; they're the same physical quantity under
   different names).
2. **Fan Static Pressure** = Fan Total Pressure − Dynamic Pressure, using the *simulated*
   flow rate in the same Dynamic Pressure formula above.
3. Different AoA change the sail's aerodynamic blockage — effectively the fan's external
   "system resistance" — so even at a fixed RPM, each AoA lands at a different flow rate.
   That's why the markers spread out along the X axis rather than sitting at one point.

If the simulated markers fall close to their RPM's reference line, the CFD-derived fan
behavior matches the manufacturer's curve at that operating point.
"""
                )

with tab_flow_estimation:
    st.caption(
        "Flow rate estimated from two static-pressure probes (plenum P1, fan-duct P2) via "
        "Bernoulli with losses, compared against the simulated Fan Volumetric Flow."
    )

    if config.esail.chord is None:
        st.warning("Chord is not set in config/config.yaml — cannot compute A1 (sail cross-section area).")
    else:
        col_height, col_xi = st.columns(2)
        target_height = col_height.number_input(
            "P1 probe height (m)", min_value=0.0, value=DEFAULT_P1_HEIGHT_M, step=0.5,
            help="Nearest point in esail_internal_center_table to this height.",
        )
        xi = col_xi.number_input(
            "Bellmouth loss coefficient ξ", min_value=0.0, value=DEFAULT_XI, step=0.01, format="%.3f"
        )

        enriched_df = get_flow_rate_estimates(
            polar_df, chord=config.esail.chord, duct_diameter=config.pipeline.fan_duct_diameter,
            rho=config.pipeline.rho, xi=xi, target_height=target_height,
        )
        valid_df = enriched_df.dropna(subset=["estimated_flow_rate"])

        if valid_df.empty:
            st.warning(
                "Could not estimate flow rate for any AoA — check that esail_internal_fan_table and "
                "esail_internal_center_table exist under tables/ for these cases."
            )
        else:
            col_chart, col_image = st.columns([2, 1])
            with col_chart:
                st.plotly_chart(plot_flow_rate_parity(enriched_df), width="stretch")
            with col_image:
                # "stretch" so it always fits the column's actual width rather than a
                # guessed fixed pixel size that can overflow past the column boundary
                st.image(str(P1_P2_IMAGE_PATH), caption="P1/P2 probe locations", width="stretch")

            st.caption(
                f"{len(valid_df)} of {len(enriched_df)} AoA points had a valid estimate "
                f"({len(enriched_df) - len(valid_df)} skipped — missing tables or a non-physical P1/P2 pair)."
            )

            with st.expander("How is this calculated?"):
                a1_val = ellipse_area(config.esail.chord)
                a2_val = duct_area(config.pipeline.fan_duct_diameter)

                col_text, col_image = st.columns([2, 1])
                with col_text:
                    st.markdown(
                        f"""
**Setup**: two static-pressure measurements per fan — **P1** several diameters upstream of
the bellmouth (inside the plenum) and **P2** downstream of the bellmouth, upstream of the
motor/blades (inside the fan duct) — see the P1/P2 diagram above.

**Physical principle**: Bernoulli with losses between the two locations:

```
p1 + ½ρv1² = p2 + ½ρv2² + ξ·½ρv2²
```

combined with mass conservation (`Q = v1·A1 = N·v2·A2`) into the closed-form relation:

```
Q = N · A2 · sqrt( 2·(p1 - p2) / (ρ · [(1+ξ) - (N·A2/A1)²]) )
```

**Where the numbers come from:**

- **P2** — mean Static Pressure over the points at the *lower* of `esail_internal_fan_table`'s
  two Z stations (4 points, per fan).
- **P1** — Static Pressure at the `esail_internal_center_table` point nearest
  **{target_height:g} m** height.
- **A1** = π × chord × (0.66 × chord) / 4 ≈ **{a1_val:.3f} m²** — the sail cross-section,
  modeled as a 66%-thickness ellipse (chord = {config.esail.chord:g} m from `config.yaml`).
- **A2** = π × (fan_duct_diameter / 2)² ≈ **{a2_val:.3f} m²** (fan_duct_diameter =
  {config.pipeline.fan_duct_diameter:g} m from `config.yaml`).
- **ξ** = {xi:.3f} (adjustable above; {DEFAULT_XI:g} is the CFD/wind-tunnel-characterized default).
- **N** = 1 (single fan).

A point sitting on the **y = x** line means the two-probe estimate matches the simulated
flow rate exactly at that AoA. Points are dropped (not plotted) when either table is
missing for that AoA, or when P1 ≤ P2 / the pressure-versus-loss balance has no physical
(real-valued) solution.
"""
                    )
                with col_image:
                    # width chosen from the image's own aspect ratio (244x860) to land at ~500px tall
                    st.image(
                        str(SAIL_CUTAWAY_IMAGE_PATH), caption="Sail section — probe locations in context",
                        width=200,
                    )
