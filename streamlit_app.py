"""YakOS DFS Optimizer – Sprint 1 multi-page shell.

Entry point for the Streamlit app.  Uses ``st.navigation`` to route between
the five Sprint 1 pages:

  1. Slate Hub         – load DK contests, configure slate, publish
  2. Ricky Edge        – tag players / games / stacks, edge labels
  3. The Lab           – sims, calibration, contest gauges
  4. Build & Publish   – build lineups, export CSV, publish to Edge Share
  5. Friends / Edge Share – read-only lineup view + friend builder

All shared state lives in ``yak_core/state.py`` (SlateState,
RickyEdgeState, LineupSetState, SimState).

See YAKOS_BUILD_RULES.md for the full Sprint 1 build contract.
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
    page_title="YakOS DFS Optimizer",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation ─────────────────────────────────────────────────────────────
pg = st.navigation(
    [
        st.Page("pages/1_slate_hub.py", title="Slate Hub", icon="🏀"),
        st.Page("pages/2_ricky_edge.py", title="Ricky Edge", icon="🎯"),
        st.Page("pages/3_the_lab.py", title="The Lab", icon="🧪"),
        st.Page("pages/4_build_publish.py", title="Build & Publish", icon="🏗️"),
        st.Page("pages/5_friends_edge_share.py", title="Friends / Edge Share", icon="👥"),
    ]
)
pg.run()


def _merge_external_proj(base_df, external_df, source_name: str):
    """TEMP: placeholder to satisfy tests during Sprint 1 scaffold."""
    raise NotImplementedError("_merge_external_proj not wired to new projection merge logic yet")
