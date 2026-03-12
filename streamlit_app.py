"""YakOS DFS Optimizer – multi-page shell.

Entry point for the Streamlit app.  Uses ``st.navigation`` to route between
the pages:

  1. The Lab           – load slate, sims, calibration, edge analysis, learning status
  2. Ricky's Edge      – signal-driven edge analysis + approval gate
  3. Build & Publish   – build lineups, export CSV
  4. Right Angle Ricky – public showcase

All shared state lives in ``yak_core/state.py`` (SlateState,
RickyEdgeState, LineupSetState, SimState).
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

# ── Navigation ─────────────────────────────────────────────────────────────
pg = st.navigation(
    [
        st.Page("pages/1_the_lab.py", title="The Lab", icon="🧪"),
        st.Page("pages/2_ricky_edge.py", title="Ricky's Edge Analysis", icon="🎯"),
        st.Page("pages/3_build_publish.py", title="Build & Publish", icon="🏗️"),
        st.Page("pages/4_right_angle_ricky.py", title="Right Angle Ricky", icon="📐"),
        st.Page("pages/5_right_angle_ricky_share.py", title="Right Angle Ricky (Share)", icon="📡"),
    ]
)
pg.run()


def _merge_external_proj(base_df, source_name: str, rg_df=None, fp_df=None):
  """Merge external projections into player pool, adding proj_fp column."""
  import pandas as pd
  out = base_df.copy()
  out["proj_fp"] = out["proj"].copy()
  if source_name == "RotoGrinders" and rg_df is not None:
    ext = rg_df.rename(columns={"proj": "_proj_ext"})
    merged = out.merge(ext[["player_name", "_proj_ext"]], on="player_name", how="left")
    mask = merged["_proj_ext"].notna()
    out.loc[mask.values, "proj_fp"] = merged.loc[mask, "_proj_ext"].values
  elif source_name == "FantasyPros" and fp_df is not None:
    ext = fp_df.rename(columns={"proj": "_proj_ext"})
    merged = out.merge(ext[["player_name", "_proj_ext"]], on="player_name", how="left")
    mask = merged["_proj_ext"].notna()
    out.loc[mask.values, "proj_fp"] = merged.loc[mask, "_proj_ext"].values
  elif source_name == "Blended" and rg_df is not None and fp_df is not None:
    rg_ext = rg_df.rename(columns={"proj": "_proj_rg"})
    fp_ext = fp_df.rename(columns={"proj": "_proj_fp"})
    merged = out.merge(rg_ext[["player_name", "_proj_rg"]], on="player_name", how="left")
    merged = merged.merge(fp_ext[["player_name", "_proj_fp"]], on="player_name", how="left")
    both = merged["_proj_rg"].notna() & merged["_proj_fp"].notna()
    rg_only = merged["_proj_rg"].notna() & merged["_proj_fp"].isna()
    out.loc[both.values, "proj_fp"] = ((merged.loc[both, "_proj_rg"].values + merged.loc[both, "_proj_fp"].values) / 2)
    out.loc[rg_only.values, "proj_fp"] = merged.loc[rg_only, "_proj_rg"].values
  return out
