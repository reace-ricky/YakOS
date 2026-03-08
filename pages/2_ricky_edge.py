"""Ricky's Edge Analysis -- auto-populated edge blueprint from Lab data.

Layout:
  1. Slate overview (4-5 bullets + recommendation)
  2. Top edges ranked by composite signal strength
  3. Top stacks
  4. Manual overrides (collapsed)
  5. Approval gate (checkboxes before publishing)
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
    compute_tiered_stack_alerts,
)
from yak_core.context import get_slate_context, get_lab_analysis  # noqa: E402
from yak_core.ricky_signals import (  # noqa: E402
    compute_ricky_signals,
    generate_slate_overview,
    SIGNAL_BADGES,
)

_TAG_COLORS = {
    "core": "\U0001f7e2", "secondary": "\U0001f535", "value": "\U0001f7e1",
    "leverage": "\u26a1", "punt": "\u26aa", "fade": "\U0001f534", "neutral": "\u26aa",
}
_PLAYER_TAGS = ["core", "secondary", "value", "leverage", "neutral", "fade"]


def main() -> None:
    st.title("🎯 Ricky's Edge Analysis")
    st.caption("Where the angles get sharp. Process-driven, data-backed, no vibes.")

    slate = get_slate_state()
    edge = get_edge_state()

    if not slate.is_ready():
        st.warning("Ricky needs a slate first. Head to **The Lab** and load one up.")
        return

    # ── Get pool + merge sim data ─────────────────────────────────────
    _analysis = get_lab_analysis()
    pool: pd.DataFrame = _analysis["pool"] if not _analysis["pool"].empty else (
        slate.player_pool if slate.player_pool is not None else pd.DataFrame()
    )

    if pool.empty:
        st.warning("No player pool available. Load a slate in The Lab first.")
        return

    # Compute edge metrics if not already on pool
    if "smash_prob" not in pool.columns:
        try:
            edge_df = compute_edge_metrics(pool, calibration_state=slate.calibration_state)
            # Merge back
            merge_cols = [c for c in ["player_name", "smash_prob", "bust_prob", "leverage", "edge_score"]
                          if c in edge_df.columns and c not in pool.columns]
            if merge_cols and "player_name" in edge_df.columns:
                pool = pool.merge(
                    edge_df[["player_name"] + [c for c in merge_cols if c != "player_name"]],
                    on="player_name", how="left",
                )
        except Exception:
            pass

    # Compute Ricky's edge signals
    signals_df = compute_ricky_signals(pool)
    contest_type = slate.contest_name or "GPP"

    # =====================================================================
    # SECTION 1: SLATE OVERVIEW
    # =====================================================================
    overview = generate_slate_overview(pool, signals_df, contest_type=contest_type)

    st.markdown(
        f"**{slate.slate_date}** &nbsp;\u00b7&nbsp; **{slate.contest_name}** "
        f"&nbsp;\u00b7&nbsp; {len(pool)} players"
    )

    # Bullets
    for bullet in overview["bullets"]:
        st.markdown(f"- {bullet}")

    # Recommendation
    if overview["recommendation"]:
        st.info(f"\U0001f4d0 **Recommendation:** {overview['recommendation']}")

    st.divider()

    # =====================================================================
    # SECTION 2: TOP EDGES (ranked by composite signal strength)
    # =====================================================================
    st.subheader("Top Edges")
    st.caption("Ranked by converging signal strength. More badges = more edge.")

    # Top 15 edges
    top_edges = signals_df[signals_df["edge_composite"] > 0].head(15)

    if top_edges.empty:
        st.info("No significant edges detected. Run sims in The Lab for better signal coverage.")
    else:
        # Build clean display table
        display_cols = []
        display_df = pd.DataFrame()
        display_df["Player"] = top_edges["player_name"].values
        if "pos" in top_edges.columns:
            display_df["Pos"] = top_edges["pos"].values
        if "team" in top_edges.columns:
            display_df["Team"] = top_edges["team"].values
        display_df["Salary"] = top_edges["salary"].values if "salary" in top_edges.columns else 0
        display_df["Proj"] = top_edges["proj"].values if "proj" in top_edges.columns else 0

        # Ownership
        own_col = "ownership" if "ownership" in top_edges.columns else "own_pct"
        if own_col in top_edges.columns:
            display_df["Own%"] = top_edges[own_col].values

        # Edge composite score scaled to 100
        display_df["Edge"] = (top_edges["edge_composite"].values * 100).round(0).astype(int)

        # Signals firing
        display_df["Signals"] = top_edges["signal_badges"].values

        # Injury bump if present
        if "injury_bump_fp" in top_edges.columns:
            bump = top_edges["injury_bump_fp"].values
            display_df["Inj+"] = [f"+{b:.1f}" if b > 0 else "" for b in bump]

        _fmt = {"Salary": "${:,.0f}", "Proj": "{:.1f}"}
        if "Own%" in display_df.columns:
            _fmt["Own%"] = "{:.1f}%"

        st.dataframe(
            display_df.style.format(_fmt, na_rep=""),
            use_container_width=True,
            hide_index=True,
            height=min(35 * len(display_df) + 40, 560),
        )

    st.divider()

    # =====================================================================
    # SECTION 3: TOP STACKS
    # =====================================================================
    st.subheader("Top Stacks")

    try:
        edge_for_stacks = signals_df if "smash_prob" in signals_df.columns else None
        stack_alerts = compute_tiered_stack_alerts(pool, edge_df=edge_for_stacks)
        if stack_alerts:
            _stack_df = pd.DataFrame(stack_alerts).head(5)
            _stack_cols = [c for c in ["team", "tier", "conditions_met", "key_players", "implied_total"]
                          if c in _stack_df.columns]
            if _stack_cols:
                st.dataframe(_stack_df[_stack_cols], use_container_width=True, hide_index=True)

                # Surface flagged-player warnings
                _warned = [a for a in stack_alerts[:5] if a.get("leverage_warning")]
                for w in _warned:
                    st.caption(f"{w['team']}: {w['leverage_warning']}")

            # Auto-define stacks from top teams if none exist
            if not edge.stacks:
                for srow in stack_alerts[:3]:
                    team = srow.get("team", "")
                    if team and not pool.empty and "player_name" in pool.columns:
                        team_players = pool[pool["team"] == team].nlargest(3, "proj")["player_name"].tolist()
                        if len(team_players) >= 2:
                            _rationale = f"Auto: {srow.get('tier', '')} ({srow.get('conditions_met', 0)} conditions)"
                            edge.add_stack(team, team_players[:3], _rationale)
                set_edge_state(edge)
        else:
            st.info("No stack data available.")
    except Exception:
        st.info("Run sims in The Lab to generate stack analysis.")

    if edge.stacks:
        st.caption(f"{len(edge.stacks)} stack(s) defined")
        rows = []
        for s in edge.stacks:
            rows.append({
                "Team": s.get("team", ""),
                "Players": ", ".join(s.get("players", [])),
                "Rationale": s.get("rationale", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # =====================================================================
    # SECTION 4: MANUAL OVERRIDES (collapsed)
    # =====================================================================
    player_names = sorted(signals_df["player_name"].dropna().tolist()) if "player_name" in signals_df.columns else []

    with st.expander("Override Player Tags", expanded=False):
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
                format_func=lambda v: {1: "1 - Low", 2: "2", 3: "3 - Mid", 4: "4", 5: "5 - High"}[v],
                value=3,
                key="_re_pick_conv",
            )

            col_add, col_rm = st.columns(2)
            with col_add:
                if st.button("Tag", key="_re_add_tag") and pick_player:
                    edge.tag_player(pick_player, pick_tag, pick_conviction)
                    set_edge_state(edge)
                    st.success(f"Tagged {pick_player} \u2192 {pick_tag} ({pick_conviction})")
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
                st.info("No manual overrides. Auto-signals are active.")

    # =====================================================================
    # SECTION 5: SLATE NOTES (collapsed)
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
    # SECTION 6: APPROVAL GATE
    # =====================================================================
    st.subheader("Approve Edge Analysis")
    st.caption("Check each box to confirm, then approve to unlock lineup building.")

    # Approval checkboxes
    _ck_edges = st.checkbox("Top edges reviewed", key="_re_ck_edges")
    _ck_stacks = st.checkbox("Stacks confirmed", key="_re_ck_stacks")
    _ck_fades = st.checkbox("Fades and chalk traps noted", key="_re_ck_fades")

    _all_checked = _ck_edges and _ck_stacks and _ck_fades

    if edge.ricky_edge_check:
        st.success(f"Approved at {edge.edge_check_ts}")
        if st.button("Revoke", key="_re_revoke"):
            edge.revoke_edge_check()
            set_edge_state(edge)
            st.warning("Edge Check revoked.")
    else:
        if not _all_checked:
            st.warning("Review all sections above and check each box before approving.")
        if st.button(
            "Approve Edge Analysis",
            type="primary",
            key="_re_approve",
            disabled=not _all_checked,
        ):
            _ts = datetime.now(timezone.utc).isoformat()
            edge.approve_edge_check(_ts)
            set_edge_state(edge)
            st.success(f"Approved at {_ts}")


main()
