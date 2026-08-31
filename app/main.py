"""
eSAILab CFD Post-Processor -- app entry point.

Defines the sidebar navigation, grouped into a section per simulation
type (see CLAUDE.md: sail / harbour / interference), plus a "Reference"
section for static, non-CFD context data (e.g. the wind climate) shared
across types. Only "Sail" has an ETL pipeline/pages today -- add a
section here once a harbour or interference pipeline and its pages
exist, following the same app/<type>/ pattern as app/sail/.

Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

# Add project root to path so `import src...` / `import app...` work --
# Streamlit sets sys.path[0] to this file's directory (app/) otherwise.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

st.set_page_config(page_title="eSAILab CFD Post-Processor", layout="wide")

sail_home = st.Page("sail/home.py", title="Home", default=True)
sail_polar_curves = st.Page("sail/polar_curves.py", title="Polar Curves")
sail_performance = st.Page("sail/performance.py", title="Performance")
sail_export = st.Page("sail/export.py", title="Export")

reference_wind_climate = st.Page("reference/wind_climate.py", title="Wind Climate")

pg = st.navigation(
    {
        "Sail": [sail_home, sail_polar_curves, sail_performance, sail_export],
        "Reference": [reference_wind_climate],
    }
)
pg.run()
