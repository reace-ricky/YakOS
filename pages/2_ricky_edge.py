"""Ricky's Edge Analysis – pre-filled from Lab sims, approve before lineup build.

Flow:
  1. The Lab loads a slate, runs sims → edge_df is computed & stored.
  2. This page reads the edge_df and auto-classifies players into
     Core / Value / Leverage / Fade tiers.
  3. User reviews, overrides tags if needed, and approves the Edge Check.
  4. Build & Publish is gated on the Edge Check approval.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import (  # noqa: E402
    get_slate_state,
    get_edge_state,
    set_edge_state,
)
from yak_core.edge import compute_edge_metrics  # noqa: E402
from yak_core.right_angle import (  # noqa: E402
    compute_game_environment_cards,
    compute_tiered_stack_alerts,
)
from yak_core.context import get_slate_context, get_lab_analysis  # noqa: E402

# ── Classification thresholds ──────────────────────────────────────────
_CORE_SMASH = 0.20
_VALUE_SMASH = 0.10
_LEVERAGE_THRESH = 1.3
_BUST_THRESH = 0.30
_FADE_OWN = 25.0  # ownership >= this AND bust >= 0.25 → fade

_TAG_COLORS = {
    "core": "🟢", "secondary": "🔵", "value": "🟡",
    "leverage": "⚡", "punt": "⚪", "fade": "🔴",
}
_PLAYER_TAGS = ["core", "secondary", "value", "leverage", "punt", "fade"]


def _auto_classify(row: pd.Series) -> str:
    """Classify a player into a tier based on edge metrics."""
    smash = float(row.get("smash_prob", 0) or 0)
    bust = float(row.get("bust_prob", 0) or 0)
    own = float(row.get("own_pct", 5) or 5)
    lev = float(row.get("leverage", 0) or 0)

    # Fade: high bust + high ownership
    if bust >= 0.25 and own >= _FADE_OWN:
        return "fade"
    # Core: high smash probability
    if smash >= _CORE_SMASH:
        return "core"
    # Leverage: good proj-to-ownership ratio, lower owned
    if pd.notna(lev) and lev >= _LEVERAGE_THRESH and own < 20:
        return "leverage"
    # Value: moderate smash upside
    if smash >= _VALUE_SMASH:
        return "value"
    # Bust risk → fade
    if bust >= _BUST_THRESH:
        return "fade"
    return ""


def main() -> None:
    st.title("Ricky's Edge Analysis")
    st.caption("Auto-populated from Lab sims. Review, override, and approve before building lineups.")

    slate = get_slate_state()
    edge = get_edge_state()

    if not slate.is_ready():
        st.warning("No slate loaded yet. Go to **The Lab** and load a slate first.")
        return

    # ── Get pool & edge data ──────────────────────────────────────────
    _analysis = get_lab_analysis()
    pool: pd.DataFrame = _analysis["pool"] if not _analysis["pool"].empty else (
        slate.player_pool if slate.player_pool is not None else pd.DataFrame()
    )

    if pool.empty:
        st.warning("No player pool available. Load a slate in The Lab first.")
        return

    # Compute edge if not already available
    edge_df = slate.edge_df if hasattr(slate, "edge_df") and slate.edge_df is not None and not slate.edge_df.empty else None
    if edge_df is None:
        try:
            edge_df = compute_edge_metrics(pool, calibration_state=slate.calibration_state)
        except Exception:
            edge_df = pool.copy()

    # Auto-classify players
    if "smash_prob" in edge_df.columns:
        edge_df["auto_tier"] = edge_df.apply(_auto_classify, axis=1)
    else:
        edge_df["auto_tier"] = ""

    player_names = sorted(edge_df["player_name"].dropna().tolist()) if "player_name" in edge_df.columns else []

    # ── Slate header ──────────────────────────────────────────────────
    st.markdown(f"**Slate:** {slate.slate_date} | **Contest:** {slate.contest_name} | **Players:** {len(edge_df)}")

    st.divider()

    # =====================================================================
    # SECTION 1: AUTO-FILLED EDGE TIERS
    # =====================================================================
    st.subheader("Edge Tiers")
    st.caption("Auto-classified from sim results. Override any player below.")

    # Core plays
    cores = edge_df[edge_df["auto_tier"] == "core"].head(8)
    if not cores.empty:
        st.markdown("**🟢 Core** — high smash probability, build around these")
        _show_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "own_pct", "smash_prob", "edge_score"] if c in cores.columns]
        _fmt = {c: "{:.1f}" for c in ["proj", "own_pct", "edge_score"] if c in cores.columns}
        _fmt.update({c: "{:.2f}" for c in ["smash_prob"] if c in cores.columns})
        st.dataframe(cores[_show_cols].style.format(_fmt), use_container_width=True, hide_index=True)

    # Leverage plays
    leverages = edge_df[edge_df["auto_tier"] == "leverage"].head(8)
    if not leverages.empty:
        st.markdown("**⚡ Leverage** — strong projection, low ownership, high upside-per-dollar-of-ownership")
        _show_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "own_pct", "leverage", "smash_prob"] if c in leverages.columns]
        _fmt = {c: "{:.1f}" for c in ["proj", "own_pct", "leverage"] if c in leverages.columns}
        _fmt.update({c: "{:.2f}" for c in ["smash_prob"] if c in leverages.columns})
        st.dataframe(leverages[_show_cols].style.format(_fmt), use_container_width=True, hide_index=True)

    # Value plays
    values = edge_df[edge_df["auto_tier"] == "value"].head(8)
    if not values.empty:
        st.markdown("**🟡 Value** — moderate upside, salary efficient")
        _show_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "own_pct", "smash_prob", "edge_score"] if c in values.columns]
        _fmt = {c: "{:.1f}" for c in ["proj", "own_pct", "edge_score"] if c in values.columns}
        _fmt.update({c: "{:.2f}" for c in ["smash_prob"] if c in values.columns})
        st.dataframe(values[_show_cols].style.format(_fmt), use_container_width=True, hide_index=True)

    # Fades
    fades = edge_df[edge_df["auto_tier"] == "fade"].head(8)
    if not fades.empty:
        st.markdown("**🔴 Fade** — bust risk, over-owned, or both")
        _show_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "own_pct", "bust_prob", "leverage"] if c in fades.columns]
        _fmt = {c: "{:.1f}" for c in ["proj", "own_pct", "leverage"] if c in fades.columns}
        _fmt.update({c: "{:.2f}" for c in ["bust_prob"] if c in fades.columns})
        st.dataframe(fades[_show_cols].style.format(_fmt), use_container_width=True, hide_index=True)

    # Empty tiers
    _filled = sum(1 for t in ["core", "leverage", "value", "fade"] if not edge_df[edge_df["auto_tier"] == t].empty)
    if _filled == 0:
        st.info("No edge tiers populated. Run sims in The Lab first to generate edge data.")

    st.divider()

    # =====================================================================
    # SECTION 2: TOP STACKS
    # =====================================================================
    st.subheader("Top Stacks")
    try:
        stack_alerts = compute_tiered_stack_alerts(pool)
        if stack_alerts:
            _stack_df = pd.DataFrame(stack_alerts).head(5)
            _stack_cols = [c for c in ["team", "tier", "conditions_met", "key_players", "implied_total"] if c in _stack_df.columns]
            st.dataframe(_stack_df[_stack_cols], use_container_width=True, hide_index=True)

            # Auto-define stacks from top teams if no manual stacks exist
            if not edge.stacks:
                for srow in stack_alerts[:3]:
                    team = srow.get("team", "")
                    if team and not pool.empty and "player_name" in pool.columns:
                        team_players = pool[pool["team"] == team].nlargest(3, "proj")["player_name"].tolist()
                        if len(team_players) >= 2:
                            edge.add_stack(team, team_players[:3], f"Auto: {srow.get('tier', '')} ({srow.get('conditions_met', 0)} conditions)")
                set_edge_state(edge)
        else:
            st.info("No stack data available.")
    except Exception:
        st.info("Run sims in The Lab to generate stack analysis.")

    # Show defined stacks
    if edge.stacks:
        st.caption(f"{len(edge.stacks)} stack(s) defined")
        rows = []
        for i, s in enumerate(edge.stacks):
            rows.append({
                "Team": s.get("team", ""),
                "Players": ", ".join(s.get("players", [])),
                "Rationale": s.get("rationale", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # =====================================================================
    # SECTION 3: MANUAL OVERRIDES
    # =====================================================================
    with st.expander("Override Player Tags", expanded=False):
        st.caption("Change any player's tier classification.")

        col_left, col_right = st.columns([2, 3])

        with col_left:
            if player_names:
                pick_player = st.selectbox("Player", [""] + player_names, key="_re_pick_player")
            else:
                pick_player = st.text_input("Player name", key="_re_pick_player_text")

            pick_tag = st.selectbox(
                "Tag",
                _PLAYER_TAGS,
                format_func=lambda t: f"{_TAG_COLORS.get(t, '')} {t.title()}",
                key="_re_pick_tag",
            )
            pick_conviction = st.select_slider(
                "Conviction", options=[1, 2, 3, 4, 5],
                format_func=lambda v: {1: "1 – Low", 2: "2", 3: "3 – Mid", 4: "4", 5: "5 – High"}[v],
                value=3,
                key="_re_pick_conv",
            )

            col_add, col_rm = st.columns(2)
            with col_add:
                if st.button("Tag", key="_re_add_tag") and pick_player:
                    edge.tag_player(pick_player, pick_tag, pick_conviction)
                    set_edge_state(edge)
                    st.success(f"Tagged {pick_player} → {pick_tag} ({pick_conviction})")
            with col_rm:
                if st.button("Remove", key="_re_rm_tag") and pick_player:
                    edge.remove_tag(pick_player)
                    set_edge_state(edge)
                    st.info(f"Removed tag for {pick_player}")

        with col_right:
            if edge.player_tags:
                st.caption("Manual overrides")
                rows = [
                    {
                        "Player": p,
                        "Tag": f"{_TAG_COLORS.get(v['tag'], '')} {v['tag'].title()}",
                        "Conviction": v.get("conviction", 3),
                    }
                    for p, v in sorted(edge.player_tags.items())
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No manual overrides. Auto-tiers are active.")

    # =====================================================================
    # SECTION 4: SLATE NOTES
    # =====================================================================
    with st.expander("Slate Notes", expanded=False):
        notes = st.text_area(
            "Notes for this slate",
            value=edge.slate_notes,
            height=80,
            key="_re_notes",
            placeholder="e.g. 'DEN in a pace-up spot, target Jokic stack. Fade CLE backs.'",
        )
        if notes != edge.slate_notes:
            edge.slate_notes = notes
            set_edge_state(edge)

    st.divider()

    # =====================================================================
    # SECTION 5: EDGE CHECK GATE
    # =====================================================================
    st.subheader("Edge Check")
    st.caption(
        "Approve to unlock lineup building in Build & Publish. "
        "Review the tiers above before approving."
    )

    if edge.ricky_edge_check:
        st.success(f"Approved at {edge.edge_check_ts}")
        if st.button("Revoke", key="_re_revoke"):
            edge.revoke_edge_check()
            set_edge_state(edge)
            st.warning("Edge Check revoked.")
    else:
        st.error("Not approved")
        if st.button("Approve Edge Check", type="primary", key="_re_approve"):
            _ts = datetime.now(timezone.utc).isoformat()
            edge.approve_edge_check(_ts)
            set_edge_state(edge)
            st.success(f"Approved at {_ts}")


main()
