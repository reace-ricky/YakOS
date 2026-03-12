"""Right Angle Ricky — standalone app for friends.

Reads published data from disk. Completely independent session
from the main YakOS app. Friends can browse edge analysis and
build their own lineups without interference.
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st

st.set_page_config(
    page_title="Right Angle Ricky",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

pg = st.navigation(
    [
        st.Page("pages_ricky/1_right_angle_ricky.py", title="Right Angle Ricky", icon="📐"),
    ]
)
pg.run()
