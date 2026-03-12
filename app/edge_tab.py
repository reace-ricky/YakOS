"""Tab 1: Ricky's Edge Analysis (public).

Displays the pre-computed edge analysis from data/published/{sport}/:
  - Analysis bullets + recommendation up top
  - 4-box dashboard: Core, Leverage, Value, Fades (boxed cards)
  - PGA wave split summary
  - Published lineups by contest type
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


# ── Styled card CSS ───────────────────────────────────────────────────────
_CARD_CSS = """
<style>
.edge-box {
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
    background: rgba(255,255,255,0.03);
}
.edge-box h4 {
    margin: 0 0 12px 0;
    font-size: 1.05rem;
}
.player-card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    background: rgba(255,255,255,0.02);
}
.player-card .name {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.player-card .stats {
    font-size: 0.82rem;
    color: rgba(240,240,240,0.7);
}
.player-card .wave-badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 6px;
}
.wave-early { background: #1565C0; color: #fff; }
.wave-late { background: #E65100; color: #fff; }
.rec-box {
    border: 1px solid #1E90FF;
    border-radius: 8px;
    padding: 14px 18px;
    background: rgba(30,144,255,0.08);
    margin-bottom: 16px;
}
.bullet-list {
    margin: 0;
    padding-left: 20px;
}
.bullet-list li {
    margin-bottom: 4px;
    font-size: 0.92rem;
}
</style>
"""

# ── Box colors + emojis ──────────────────────────────────────────────────
_BOX_CONFIG = {
    "core_plays": {"title": "Core Plays", "emoji": "🎯", "color": "#2196F3"},
    "leverage_plays": {"title": "Leverage Plays", "emoji": "💎", "color": "#FF9800"},
    "value_plays": {"title": "Value Plays", "emoji": "💰", "color": "#4CAF50"},
    "fade_candidates": {"title": "Fades", "emoji": "👋", "color": "#f44336"},
}


def _render_player_card_html(player: Dict[str, Any], is_pga: bool) -> str:
    """Build HTML for a single player card."""
    name = player.get("player_name", "")
    salary = player.get("salary", 0)
    proj = player.get("proj", 0)
    own = player.get("ownership", 0)
    edge = player.get("edge", 0)
    value = player.get("value", 0)

    stats_parts = [
        f"${salary:,}",
        f"{proj:.1f} pts",
        f"{own:.1f}% own",
        f"{edge:.2f} edge",
        f"{value:.2f} pts/$1K",
    ]

    wave_html = ""
    if is_pga:
        wave = player.get("wave", "")
        teetime = player.get("r1_teetime", "")
        if wave == "Early":
            wave_html = f'<span class="wave-badge wave-early">☀️ Early{" · " + teetime if teetime else ""}</span>'
        elif wave == "Late":
            wave_html = f'<span class="wave-badge wave-late">🌙 Late{" · " + teetime if teetime else ""}</span>'

    return (
        f'<div class="player-card">'
        f'<div class="name">{name}{wave_html}</div>'
        f'<div class="stats">{" · ".join(stats_parts)}</div>'
        f'</div>'
    )


def _render_edge_box(key: str, players: List[Dict], is_pga: bool) -> None:
    """Render one of the 4 edge classification boxes."""
    cfg = _BOX_CONFIG[key]
    title = cfg["title"]
    emoji = cfg["emoji"]
    color = cfg["color"]

    cards_html = ""
    if not players:
        cards_html = '<p style="color:rgba(240,240,240,0.4); font-size:0.85rem;">No players in this category.</p>'
    else:
        for p in players:
            cards_html += _render_player_card_html(p, is_pga)

    box_html = (
        f'<div class="edge-box" style="border-left: 4px solid {color};">'
        f'<h4>{emoji} {title} ({len(players)})</h4>'
        f'{cards_html}'
        f'</div>'
    )
    st.markdown(box_html, unsafe_allow_html=True)


def render_edge_tab(sport: str) -> None:
    """Render Ricky's Edge Analysis tab."""
    from app.data_loader import load_published_data

    # Inject CSS once
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    try:
        meta, pool, edge_analysis, edge_state, lineups = load_published_data(sport)
    except Exception as e:
        st.error(f"Could not load {sport} data: {e}")
        return

    if not meta:
        st.info(f"No published {sport} data found. Run the pipeline first.")
        return

    is_pga = sport.upper() == "PGA"
    slate_date = meta.get("date", "")
    pool_size = meta.get("pool_size", len(pool))

    # ── Header ──
    st.markdown(f"## 📐 Right Angle Ricky — {sport}")
    st.caption(f"{slate_date} · {pool_size} players · DraftKings")

    # ── Analysis + Recommendation up top ──
    rec = edge_analysis.get("recommendation", "")
    bullets = edge_analysis.get("bullets", [])

    if rec:
        st.markdown(f'<div class="rec-box">📋 <strong>{rec}</strong></div>', unsafe_allow_html=True)

    if bullets:
        bullet_items = "".join(f"<li>{b}</li>" for b in bullets)
        st.markdown(f'<ul class="bullet-list">{bullet_items}</ul>', unsafe_allow_html=True)

    st.markdown("")  # spacer

    # ── PGA wave split (above the 4-box if present) ──
    if is_pga:
        wave_split = edge_state.get("wave_split")
        if wave_split:
            wc1, wc2 = st.columns(2)
            with wc1:
                st.metric(
                    "☀️ Early Wave",
                    f"{wave_split.get('early_count', 0)} players",
                    f"{wave_split.get('early_avg_proj', 0):.1f} avg proj",
                )
                early_top = wave_split.get("early_players", [])
                if early_top:
                    st.caption(f"Top: {', '.join(early_top[:5])}")
            with wc2:
                st.metric(
                    "🌙 Late Wave",
                    f"{wave_split.get('late_count', 0)} players",
                    f"{wave_split.get('late_avg_proj', 0):.1f} avg proj",
                )
                late_top = wave_split.get("late_players", [])
                if late_top:
                    st.caption(f"Top: {', '.join(late_top[:5])}")
            st.markdown("")

    # ── 4-box dashboard ──
    col1, col2 = st.columns(2)
    with col1:
        _render_edge_box("core_plays", edge_analysis.get("core_plays", []), is_pga)
        _render_edge_box("value_plays", edge_analysis.get("value_plays", []), is_pga)
    with col2:
        _render_edge_box("leverage_plays", edge_analysis.get("leverage_plays", []), is_pga)
        _render_edge_box("fade_candidates", edge_analysis.get("fade_candidates", []), is_pga)

    # ── Published lineups ──
    if lineups:
        st.markdown("---")
        st.markdown("### 📋 Published Lineups")
        for contest_slug, ldf in lineups.items():
            label = contest_slug.replace("_", " ").title()
            n_lu = ldf["lineup_index"].nunique() if "lineup_index" in ldf.columns else 0
            with st.expander(f"{label} — {n_lu} lineups"):
                if "lineup_index" not in ldf.columns:
                    st.dataframe(ldf, use_container_width=True, hide_index=True)
                    continue
                for idx in sorted(ldf["lineup_index"].unique()):
                    lu = ldf[ldf["lineup_index"] == idx]
                    total_sal = int(pd.to_numeric(lu.get("salary", 0), errors="coerce").fillna(0).sum())
                    total_proj = float(pd.to_numeric(lu.get("proj", 0), errors="coerce").fillna(0).sum())
                    st.markdown(f"**Lineup {idx + 1}** — ${total_sal:,} sal · {total_proj:.1f} proj")
                    display_cols = ["slot", "player_name", "pos", "salary", "proj"]
                    if is_pga:
                        display_cols = ["slot", "player_name", "salary", "proj", "wave", "r1_teetime"]
                        if "wave" not in lu.columns and "early_late_wave" in lu.columns:
                            lu = lu.copy()
                            lu["wave"] = lu["early_late_wave"].map({0: "Early", 1: "Late"})
                    avail = [c for c in display_cols if c in lu.columns]
                    st.dataframe(
                        lu[avail].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )
