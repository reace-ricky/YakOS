"""Tab 1: Edge Analysis (public).

Displays the pre-computed edge analysis from data/published/{sport}/:
  - Slate header (sport, date, pool size)
  - 4-box dashboard: Core, Leverage, Value, Fades
  - Analysis bullets and recommendation
  - PGA wave split summary
  - Published lineups by contest type
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _render_player_card(player: Dict[str, Any], is_pga: bool) -> None:
    """Render a single player card within an edge box."""
    name = player.get("player_name", "")
    salary = player.get("salary", 0)
    proj = player.get("proj", 0)
    own = player.get("ownership", 0)
    edge = player.get("edge", 0)
    value = player.get("value", 0)

    line = f"**{name}** — ${salary:,} | {proj:.1f} pts | {own:.1f}% own | edge {edge:.2f} | {value:.2f} pts/$1K"
    if is_pga:
        wave = player.get("wave", "")
        teetime = player.get("r1_teetime", "")
        if wave:
            line += f" | {wave}"
        if teetime:
            line += f" ({teetime})"
    st.markdown(line)


def _render_edge_box(title: str, players: List[Dict], color: str, is_pga: bool) -> None:
    """Render one of the 4 edge classification boxes."""
    st.markdown(
        f'<div style="border-left: 4px solid {color}; padding-left: 12px; margin-bottom: 16px;">'
        f"<h4>{title} ({len(players)})</h4></div>",
        unsafe_allow_html=True,
    )
    if not players:
        st.caption("No players in this category.")
        return
    for p in players:
        _render_player_card(p, is_pga)


def render_edge_tab(sport: str) -> None:
    """Render the Edge Analysis tab."""
    from app.data_loader import load_published_data

    try:
        meta, pool, edge_analysis, edge_state, lineups = load_published_data(sport)
    except Exception as e:
        st.error(f"Could not load {sport} data: {e}")
        return

    if not meta:
        st.info(f"No published {sport} data found. Run the pipeline first.")
        return

    is_pga = sport.upper() == "PGA"

    # ── Slate header ──
    slate_date = meta.get("date", "")
    pool_size = meta.get("pool_size", len(pool))
    st.markdown(f"### {sport} — {slate_date} — {pool_size} players")

    # ── 4-box dashboard ──
    col1, col2 = st.columns(2)
    with col1:
        _render_edge_box(
            "Core Plays",
            edge_analysis.get("core_plays", []),
            "#2196F3",
            is_pga,
        )
        _render_edge_box(
            "Value Plays",
            edge_analysis.get("value_plays", []),
            "#4CAF50",
            is_pga,
        )
    with col2:
        _render_edge_box(
            "Leverage Plays",
            edge_analysis.get("leverage_plays", []),
            "#FF9800",
            is_pga,
        )
        _render_edge_box(
            "Fades",
            edge_analysis.get("fade_candidates", []),
            "#f44336",
            is_pga,
        )

    # ── Analysis bullets ──
    bullets = edge_analysis.get("bullets", [])
    if bullets:
        st.markdown("---")
        st.markdown("#### Analysis")
        for b in bullets:
            st.markdown(f"- {b}")

    # ── Recommendation ──
    rec = edge_analysis.get("recommendation", "")
    if rec:
        st.info(rec)

    # ── PGA wave split ──
    if is_pga:
        wave_split = edge_state.get("wave_split")
        if wave_split:
            st.markdown("---")
            st.markdown("#### Wave Split")
            wc1, wc2 = st.columns(2)
            with wc1:
                st.metric("Early Wave", f"{wave_split.get('early_count', 0)} players",
                           f"{wave_split.get('early_avg_proj', 0):.1f} avg proj")
            with wc2:
                st.metric("Late Wave", f"{wave_split.get('late_count', 0)} players",
                           f"{wave_split.get('late_avg_proj', 0):.1f} avg proj")
            early_top = wave_split.get("early_players", [])
            late_top = wave_split.get("late_players", [])
            if early_top:
                st.caption(f"Top Early: {', '.join(early_top)}")
            if late_top:
                st.caption(f"Top Late: {', '.join(late_top)}")

    # ── Published lineups ──
    if lineups:
        st.markdown("---")
        st.markdown("#### Published Lineups")
        for contest_slug, ldf in lineups.items():
            label = contest_slug.replace("_", " ").title()
            with st.expander(f"{label} ({ldf['lineup_index'].nunique() if 'lineup_index' in ldf.columns else 0} lineups)"):
                if "lineup_index" not in ldf.columns:
                    st.dataframe(ldf)
                    continue
                for idx in sorted(ldf["lineup_index"].unique()):
                    lu = ldf[ldf["lineup_index"] == idx]
                    total_sal = int(pd.to_numeric(lu.get("salary", 0), errors="coerce").fillna(0).sum())
                    total_proj = float(pd.to_numeric(lu.get("proj", 0), errors="coerce").fillna(0).sum())
                    st.markdown(f"**Lineup {idx + 1}** — ${total_sal:,} sal | {total_proj:.1f} proj")
                    display_cols = ["player_name", "pos", "salary", "proj"]
                    if "slot" in lu.columns:
                        display_cols = ["slot", "player_name", "pos", "salary", "proj"]
                    avail = [c for c in display_cols if c in lu.columns]
                    st.dataframe(lu[avail].reset_index(drop=True), use_container_width=True, hide_index=True)
