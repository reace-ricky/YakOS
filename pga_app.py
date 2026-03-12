"""YakOS PGA DFS Optimizer – standalone multi-page app.

Separate entry point for PGA. Uses ``pages_pga/`` directory and
PGA-specific session state keys + persistence paths so NBA app
reboots never wipe PGA data.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st

st.set_page_config(
    page_title="YakOS · PGA",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    [
        st.Page("pages_pga/1_the_lab.py", title="The Lab", icon="🧪"),
        st.Page("pages_pga/2_build_publish.py", title="Build & Publish", icon="🏗️"),
        st.Page("pages_pga/3_ricky.py", title="Right Angle Ricky", icon="📐"),
    ]
)
pg.run()
