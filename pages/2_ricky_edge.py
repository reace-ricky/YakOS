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
    compute_breakout_candidates,
)
from yak_core.context import get_slate_context, get_lab_analysis  # noqa: E402

# ── Classification thresholds ──────────────────────────────────────────
# Calibrated from 21-slate backtest (Feb 7 – Mar 5 2026, 3512 player-slates).
# Old smash_prob thresholds were useless: _compute_smash_bust produced
# constant ~0.069 smash for ALL players.  New model uses salary + ownership
# + value efficiency which are the actual predictors of smash/bust.
_FADE_SALARY = 8000   # $8K+ chalk over-projects by +2.54 FP
_CORE_MAX_SALARY = 6000  # sub-$6K has 37% smash rate
_CORE_MIN_VAL = 2.0   # minimum FP/$1K for core
_CORE_MAX_OWN = 15.0  # cores must be low-owned
_LEV_MAX_OWN = 12.0   # leverage plays under 12% owned
_LEV_MIN_VAL = 2.5    # min FP/$1K for leverage
_LEV_MAX_SAL = 7500   # leverage capped at $7.5K
_VALUE_MIN_VAL = 3.0  # min FP/$1K for value tier

_TAG_COLORS = {
    "core": "🟢", "secondary": "🔵", "value": "🟡",
    "leverage": "⚡", "punt": "⚪", "fade": "🔴", "neutral": "⚪",
}
_PLAYER_TAGS = ["core", "secondary", "value", "leverage", "neutral", "fade"]


def _auto_classify(row: pd.Series) -> str:
    """Classify a player into a tier using empirically calibrated rules.

    Based on 21-slate backtest:
    - FADE ($8K+ chalk): over-projects +2.54 FP, 11.4% bust, 12.2% smash
    - CORE (cheap, low-owned): 30.7% smash, 46.8% outperform rate
    - LEVERAGE (low-own, good value): 50% smash (small sample), 61% outperform
    - VALUE (good FP/$1K): 19.2% smash, solid baseline
    - NEUTRAL (mid-tier): 33.8% smash, 51.5% outperform — hidden edge
    """
    sal = float(row.get("salary", 6000) or 6000)
    own = float(row.get("own_pct", 15) or 15)
    proj = float(row.get("proj", 15) or 15)
    val = proj / max(sal / 1000.0, 1.0)

    # FADE: expensive chalk — biggest source of over-projection
    if sal >= _FADE_SALARY:
        return "fade"

    # CORE: cheap players with good value efficiency and low ownership
    if sal < _CORE_MAX_SALARY and val >= _CORE_MIN_VAL and own < _CORE_MAX_OWN:
        return "core"

    # LEVERAGE: low ownership + decent value + mid salary
    if own < _LEV_MAX_OWN and val >= _LEV_MIN_VAL and sal < _LEV_MAX_SAL:
        return "leverage"

    # VALUE: reasonable value efficiency
    if val >= _VALUE_MIN_VAL:
        return "value"

    # NEUTRAL: everything else (actually outperforms 51.5% of the time)
    return "neutral"


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

    # Core plays — cheap, low-owned, high smash
    cores = edge_df[edge_df["auto_tier"] == "core"].head(8)
    if not cores.empty:
        st.markdown("**🟢 Core** — low salary, low ownership, high ceiling rate (30.7% smash in backtest)")
        _show_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "own_pct", "smash_prob", "edge_score"] if c in cores.columns]
        _fmt = {c: "{:.1f}" for c in ["proj", "own_pct", "edge_score"] if c in cores.columns}
        _fmt.update({c: "{:.2f}" for c in ["smash_prob"] if c in cores.columns})
        st.dataframe(cores[_show_cols].style.format(_fmt), use_container_width=True, hide_index=True)

    # Leverage plays — low owned, good value
    leverages = edge_df[edge_df["auto_tier"] == "leverage"].head(8)
    if not leverages.empty:
        st.markdown("**⚡ Leverage** — under 12% owned, strong value efficiency, GPP differentiators")
        _show_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "own_pct", "leverage", "smash_prob"] if c in leverages.columns]
        _fmt = {c: "{:.1f}" for c in ["proj", "own_pct", "leverage"] if c in leverages.columns}
        _fmt.update({c: "{:.2f}" for c in ["smash_prob"] if c in leverages.columns})
        st.dataframe(leverages[_show_cols].style.format(_fmt), use_container_width=True, hide_index=True)

    # Value plays — good FP/$1K
    values = edge_df[edge_df["auto_tier"] == "value"].head(8)
    if not values.empty:
        st.markdown("**🟡 Value** — solid FP/$1K efficiency, reliable baseline producers")
        _show_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "own_pct", "smash_prob", "edge_score"] if c in values.columns]
        _fmt = {c: "{:.1f}" for c in ["proj", "own_pct", "edge_score"] if c in values.columns}
        _fmt.update({c: "{:.2f}" for c in ["smash_prob"] if c in values.columns})
        st.dataframe(values[_show_cols].style.format(_fmt), use_container_width=True, hide_index=True)

    # Neutral — hidden edge, highest outperform rate
    neutrals = edge_df[edge_df["auto_tier"] == "neutral"].head(8)
    if not neutrals.empty:
        st.markdown("**⚪ Neutral** — mid-tier with hidden edge (51.5% outperform rate, 33.8% smash in backtest)")
        _show_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "own_pct", "smash_prob", "edge_score"] if c in neutrals.columns]
        _fmt = {c: "{:.1f}" for c in ["proj", "own_pct", "edge_score"] if c in neutrals.columns}
        _fmt.update({c: "{:.2f}" for c in ["smash_prob"] if c in neutrals.columns})
        st.dataframe(neutrals[_show_cols].style.format(_fmt), use_container_width=True, hide_index=True)

    # Fades — expensive chalk
    fades = edge_df[edge_df["auto_tier"] == "fade"].head(8)
    if not fades.empty:
        st.markdown("**🔴 Fade** — $8K+ chalk trap, over-projects by +2.54 FP on average")
        _show_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "own_pct", "bust_prob", "leverage"] if c in fades.columns]
        _fmt = {c: "{:.1f}" for c in ["proj", "own_pct", "leverage"] if c in fades.columns}
        _fmt.update({c: "{:.2f}" for c in ["bust_prob"] if c in fades.columns})
        st.dataframe(fades[_show_cols].style.format(_fmt), use_container_width=True, hide_index=True)

    # Empty tiers
    _filled = sum(1 for t in ["core", "leverage", "value", "neutral", "fade"] if not edge_df[edge_df["auto_tier"] == t].empty)
    if _filled == 0:
        st.info("No edge tiers populated. Run sims in The Lab first to generate edge data.")

    st.divider()

    # =====================================================================
    # SECTION 1.5: BREAKOUT CANDIDATES
    # =====================================================================
    st.subheader("Breakout Candidates")
    st.caption(
        "Players with converging breakout signals: minutes surge, underpriced role, "
        "usage consolidation, soft matchup, and volatility."
    )

    try:
        # Check data quality: warn if rolling stats are missing
        _has_rolling = (
            "rolling_fp_5" in pool.columns
            and pool["rolling_fp_5"].notna().any()
            and "rolling_min_5" in pool.columns
            and pool["rolling_min_5"].notna().any()
        )
        if not _has_rolling:
            st.warning(
                "Rolling game-log stats not loaded — breakout model is running on "
                "salary value + matchup only (35% of full signal). "
                "Re-load the player pool with a Tank01 API key for full breakout detection."
            )

        breakout_df = compute_breakout_candidates(pool, top_n=10)
        if not breakout_df.empty:
            # Group by salary tier for clean display
            for tier_label, tier_emoji in [("Cheap", "\u2b06"), ("Mid", "\U0001F4C8"), ("Stud", "\U0001F525")]:
                tier_rows = breakout_df[breakout_df["salary_tier"] == tier_label]
                if tier_rows.empty:
                    continue
                tier_desc = {
                    "Cheap": "Underpriced minute spikes — role changes not yet priced in",
                    "Mid": "Usage consolidation targets — under-owned with peripherals upside",
                    "Stud": "Environment ceiling plays — pace-up + soft defense",
                }
                st.markdown(f"**{tier_emoji} {tier_label}** — {tier_desc.get(tier_label, '')}")
                _bo_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "breakout_score", "archetype", "breakout_signals"] if c in tier_rows.columns]
                _bo_fmt = {c: "{:.1f}" for c in ["proj", "breakout_score"] if c in tier_rows.columns}
                if "salary" in tier_rows.columns:
                    _bo_fmt["salary"] = "${:,.0f}"
                st.dataframe(
                    tier_rows[_bo_cols].style.format(_bo_fmt),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No breakout candidates detected. Load rolling stats in The Lab for better signal detection.")
    except Exception as exc:
        st.info(f"Breakout detection requires rolling game-log stats from The Lab. ({exc})")

    st.divider()

    # =====================================================================
    # SECTION 2: TOP STACKS
    # =====================================================================
    st.subheader("Top Stacks")
    try:
        stack_alerts = compute_tiered_stack_alerts(pool, edge_df=edge_df)
        if stack_alerts:
            _stack_df = pd.DataFrame(stack_alerts).head(5)
            _stack_cols = [c for c in ["team", "tier", "conditions_met", "key_players", "implied_total", "game_ou", "leverage_warning"] if c in _stack_df.columns]
            # Drop leverage_warning column if all empty (no edge data available)
            if "leverage_warning" in _stack_df.columns and _stack_df["leverage_warning"].str.strip().eq("").all():
                _stack_cols = [c for c in _stack_cols if c != "leverage_warning"]
            st.dataframe(_stack_df[_stack_cols], use_container_width=True, hide_index=True)

            # Surface flagged-player warnings beneath the table
            _warned = [a for a in stack_alerts[:5] if a.get("leverage_warning")]
            for w in _warned:
                st.caption(f"{w['team']}: {w['leverage_warning']}")

            # Auto-define stacks from top teams if no manual stacks exist
            if not edge.stacks:
                for srow in stack_alerts[:3]:
                    team = srow.get("team", "")
                    if team and not pool.empty and "player_name" in pool.columns:
                        team_players = pool[pool["team"] == team].nlargest(3, "proj")["player_name"].tolist()
                        if len(team_players) >= 2:
                            _rationale = f"Auto: {srow.get('tier', '')} ({srow.get('conditions_met', 0)} conditions)"
                            if srow.get("leverage_warning"):
                                _rationale += f" | {srow['leverage_warning']}"
                            edge.add_stack(team, team_players[:3], _rationale)
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
