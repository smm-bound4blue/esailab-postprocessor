"""
Performance: for each (Project, AWS, RPM) case, take the AoA point that
gives maximum CL, and plot how its CL/Cpow trend across RPM -- one trace
per (Project, AWS), connecting the CLmax point at each RPM. The Fan
Performance tab is the exception: it uses the full AoA sweep (not just
CLmax) at a chosen RPM, since different AoA shift the sail's aerodynamic
blockage -- the fan's effective "system resistance" -- so they land at
different flow rates worth comparing against the reference curve.
"""

import streamlit as st

from app.sail.analysis import compute_clmax_data
from app.sail.plotting import plot_fan_performance_curve, plot_polar_curve
from app.sail.widgets import render_plot_controls, select_projects_and_load
from src.config import load_config
from src.sail.fan import build_reference_curve, load_fan_curve

st.title("Performance")
st.caption("Each point is the AoA@CLmax operating point for one (Project, AWS, RPM) case.")

polar_df = select_projects_and_load(key="performance_projects")
plot_mode, legend_group_by = render_plot_controls(key="performance")

clmax_df = compute_clmax_data(polar_df)
trace_by = ["project_name", "aws"]  # connect CLmax points across RPM into one envelope line

tab_envelope, tab_fan = st.tabs(["CL vs Cpow", "Fan Performance"])

with tab_envelope:
    if clmax_df["cpow"].isna().all():
        st.warning("Cpow is not available — check that chord is set in config/config.yaml and re-sync.")
    else:
        st.plotly_chart(
            plot_polar_curve(
                clmax_df, y="cl", x="cpow", trace_by=trace_by,
                legend_group_by=legend_group_by, plot_mode=plot_mode,
            ),
            width="stretch",
        )

with tab_fan:
    if polar_df["fan_static_pressure"].isna().all() and polar_df["fan_total_pressure"].isna().all():
        st.warning("Fan pressure data is not available for the selected project(s).")
    else:
        try:
            config = load_config()
            fan_curve = load_fan_curve(config.fan_curve_path)
        except Exception as e:
            st.warning(f"Could not load config/fan_curve.csv: {e}")
            fan_curve = None

        if fan_curve is None:
            st.warning("No reference fan curve available.")
        else:
            col_rpm, col_total = st.columns(2)
            rpm_options = sorted(polar_df["rpm"].unique())
            selected_rpm = col_rpm.selectbox("Fan speed (RPM)", rpm_options, format_func=lambda r: f"{r:g}")
            show_total = col_total.checkbox("Show Fan Total Pressure", value=True)

            sim_df = polar_df[polar_df["rpm"] == selected_rpm]
            reference_curve = build_reference_curve(
                fan_curve, rpm=selected_rpm, rho=config.pipeline.rho, duct_diameter=config.pipeline.fan_duct_diameter
            )
            st.plotly_chart(
                plot_fan_performance_curve(sim_df, reference_curve, rpm=selected_rpm, show_total=show_total),
                width="stretch",
            )

            with st.expander("How is this calculated?"):
                st.markdown(
                    f"""
**Reference curve (solid/dashed black lines)** — the manufacturer's datasheet for this
fan, `config/fan_curve.csv` ({fan_curve.reference_rpm:g} RPM reference):

1. The datasheet gives **Static Pressure vs Volumetric Flow Rate** at
   {fan_curve.reference_rpm:g} RPM — that's the raw curve in the CSV.
2. Scaled to the selected **{selected_rpm:g} RPM** using the fan affinity laws
   (`N` = RPM ratio): `Flow ∝ N`, `Pressure ∝ N²`, `Power ∝ N³`.
3. **Total Pressure** (dashed) = Static Pressure + Dynamic Pressure, where
   `Dynamic Pressure = 0.5 × ρ × V²` and `V = Flow Rate / Duct Area`
   (`config.yaml`: `pipeline.rho` = {config.pipeline.rho:g} kg/m³,
   `pipeline.fan_duct_diameter` = {config.pipeline.fan_duct_diameter:g} m).

**Simulated points (● Static, ♦ Total)** — one color per (Project, AWS); every marker is
one AoA from that case's CFD sweep at {selected_rpm:g} RPM:

1. **Fan Volumetric Flow** and **Fan Total Pressure** come straight from the his.csv fan
   monitor (`Fan 1 Pressure Drop` / `Fan 1 Total Pressure` / `Fan 1 Total Pressure Rise` —
   whichever STAR-CCM+ exported for that project; they're the same physical quantity under
   different names).
2. **Fan Static Pressure** = Fan Total Pressure − Dynamic Pressure, using the *simulated*
   flow rate in the same Dynamic Pressure formula above.
3. Different AoA change the sail's aerodynamic blockage — effectively the fan's external
   "system resistance" — so even at a fixed RPM, each AoA lands at a different flow rate.
   That's why the markers spread out along the X axis rather than sitting at one point.

If the simulated markers fall close to the reference line, the CFD-derived fan behavior
matches the manufacturer's curve at that operating point.
"""
                )
