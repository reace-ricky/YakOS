"""YakOS DFS Optimizer – single-page Streamlit app.

Entry point.  Uses ``st.tabs()`` for navigation:
  - Edge Analysis (public)
  - Optimizer (public)
  - The Lab (admin)
  - Dashboard (admin)
  - Calibration Lab (admin)

Sport toggle (NBA/PGA) and admin password gate live in the sidebar.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so yak_core is importable on Streamlit Cloud
_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st

st.set_page_config(
    page_title="YakOS · Right Angle Ricky",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────
sport = st.sidebar.radio("Sport", ["NBA", "PGA"], key="sport_toggle")

from app.auth import check_admin_password

is_admin = check_admin_password()

# ── Tabs ──────────────────────────────────────────────────────────────────
if is_admin:
    tab_edge, tab_optimizer, tab_lab, tab_dashboard, tab_cal_lab = st.tabs(
        ["📐 Ricky's Edge Analysis", "⚙️ Optimizer", "🧪 The Lab", "📊 Dashboard", "🔬 Calibration Lab"]
    )
else:
    tab_edge, tab_optimizer = st.tabs(["📐 Ricky's Edge Analysis", "⚙️ Optimizer"])
    tab_lab = None
    tab_dashboard = None
    tab_cal_lab = None

# ── Render tabs ───────────────────────────────────────────────────────────
from app.edge_tab import render_edge_tab
from app.optimizer_tab import render_optimizer_tab

with tab_edge:
    render_edge_tab(sport)

with tab_optimizer:
    render_optimizer_tab(sport)

if is_admin and tab_lab is not None:
    from app.lab_tab import render_lab_tab
    from app.dashboard_tab import render_dashboard_tab
    from app.calibration_lab import render_calibration_lab

    with tab_lab:
        render_lab_tab(sport)

    with tab_dashboard:
        render_dashboard_tab(sport)

    with tab_cal_lab:
        render_calibration_lab(sport)
