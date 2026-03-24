"""Tab 1: Ricky's Edge Analysis (public).

Displays the pre-computed edge analysis from data/published/{sport}/:
  - The Board (unified Ricky voice): Last Slate Recap → Today's Edge
    (stacks, snipers, fades, injury edges, bust call — all in voice)
  - 3-box dashboard: Core, Leverage, Value (boxed cards)
  - PGA wave split summary
  - Published lineups by contest type
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


# ── Styled card CSS ─────────────────────────────────────────────────────────
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
.bullet-list {
    margin: 0;
    padding-left: 20px;
}
.bullet-list li {
    margin-bottom: 4px;
    font-size: 0.92rem;
}
.the-board {
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
    background: rgba(255,255,255,0.02);
}
.the-board h3 {
    margin: 0 0 14px 0;
}
.the-board-last-night {
    color: rgba(240,240,240,0.65);
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.the-board-edge-callout {
    font-size: 0.92rem;
    line-height: 1.5;
    margin-bottom: 8px;
}
.the-board-bust-call {
    border: 1px solid rgba(244,67,54,0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 14px;
    background: rgba(244,67,54,0.06);
    font-size: 0.95rem;
    line-height: 1.5;
}
</style>
"""

# ── Box colors + emojis ──────────────────────────────────────────────────────
_BOX_CONFIG = {
    "core_plays": {"title": "Core Plays", "emoji": "🎯", "color": "#2196F3"},
    "leverage_plays": {"title": "Leverage Plays", "emoji": "💎", "color": "#FF9800"},
    "value_plays": {"title": "Value Plays", "emoji": "💰", "color": "#4CAF50"},
}


def _render_player_card_html(player: Dict[str, Any], is_pga: bool, cleared_players: list | None = None) -> str:
    """Build HTML for a single player card."""
    name = player.get("player_name", "")
    salary = player.get("salary", 0)
    proj = player.get("proj", 0)
    own = player.get("ownership", 0)
    edge = player.get("edge", 0)
    value = player.get("value", 0)

    proj_min = player.get("proj_minutes", 0)
    sim90 = player.get("sim90th", 0)

    stats_parts = [
        f"${salary:,}",
        f"{proj:.1f} pts",
        f"{proj_min:.0f} min" if proj_min > 0 else None,
        f"{own:.1f}% own",
        f"{edge:.2f} edge",
        f"{value:.2f} pts/$1K",
        f"{sim90:.1f} ceil" if sim90 > 0 else None,
    ]
    stats_parts = [s for s in stats_parts if s is not None]

    wave_html = ""
    if is_pga:
        wave = player.get("wave", "")
        teetime = player.get("r1_teetime", "")
        if wave == "Early":
            wave_html = f'<span class="wave-badge wave-early">☀️ Early{". " + teetime if teetime else ""}</span>'
        elif wave == "Late":
            wave_html = f'<span class="wave-badge wave-late">🌙 Late{". " + teetime if teetime else ""}</span>'

    cleared_html = ""
    if cleared_players and name in cleared_players:
        cleared_html = ' <span style="color:#4CAF50;font-size:0.8rem;font-weight:600;">✅ Cleared</span>'

    return (
        f'<div class="player-card">'
        f'<div class="name">{name}{cleared_html}{wave_html}</div>'
        f'<div class="stats">{" . ".join(stats_parts)}</div>'
        f'</div>'
    )


def _render_edge_box(key: str, players: List[Dict], is_pga: bool, cleared_players: list | None = None) -> None:
    """Render one of the edge classification boxes."""
    cfg = _BOX_CONFIG[key]
    title = cfg["title"]
    emoji = cfg["emoji"]
    color = cfg["color"]

    cards_html = ""
    if not players:
        cards_html = '<p style="color:rgba(240,240,240,0.4); font-size:0.85rem;">No players in this category.</p>'
    else:
        for p in players:
            cards_html += _render_player_card_html(p, is_pga, cleared_players)

    box_html = (
        f'<div class="edge-box" style="border-left: 4px solid {color};">'
        f'<h4>{emoji} {title} ({len(players)})</h4>'
        f'{cards_html}'
        f'</div>'
    )
    st.markdown(box_html, unsafe_allow_html=True)


def _sniper_reason(p: Dict[str, Any], pool: pd.DataFrame) -> str:
    """Generate a player-specific reason for a Ricky's Play pick.

    Checks the pool for what makes this player unique — injury cascade,
    pace-up spot, salary value, minutes certainty, etc. Returns a 1-sentence
    reason, never a generic closer.
    """
    name = p.get("player_name", "")
    proj = p.get("proj", 0)
    ceil = p.get("ceil", 0)
    own = p.get("own_pct", 0)
    sal = p.get("salary", 0)

    # Try to find the player in the pool for richer data
    row = None
    if not pool.empty and "player_name" in pool.columns:
        match = pool[pool["player_name"] == name]
        if not match.empty:
            row = match.iloc[0]

    # Check each signal and return the first strong one
    if row is not None:
        # Injury cascade bump
        bump = float(row.get("injury_bump_fp", 0) or 0)
        if bump > 2.0:
            return (f"Picking up {bump:.1f} extra FP from an injury cascade. "
                    f"{own:.1f}% owned — the field hasn't adjusted.")

        # High minutes in a pace-up game
        mins = float(row.get("proj_minutes", 0) or 0)
        vegas = float(row.get("vegas_total", 0) or 0)
        if mins >= 30 and vegas >= 225:
            return (f"{mins:.0f} projected minutes in a {vegas:.0f}-total game. "
                    f"Only {own:.1f}% owned — salary is suppressing his ownership.")

        # Big ceiling-to-projection gap
        if ceil > 0 and proj > 0 and (ceil - proj) / proj > 0.35:
            return (f"{ceil:.0f} ceiling on a {proj:.1f} projection — "
                    f"{((ceil - proj) / proj * 100):.0f}% upside gap at {own:.1f}% owned.")

        # Low salary relative to projection
        if sal > 0 and proj > 0:
            pts_per_k = proj / (sal / 1000)
            if pts_per_k >= 6.5:
                return (f"{pts_per_k:.1f} pts/$1K at ${sal:,}. "
                        f"{proj:.1f} projected, {own:.1f}% owned. Pure value.")

    # Fallback: use what we have
    own_str = f"{own:.1f}%" if own > 0 else "low"
    return (f"{proj:.1f} projected with a {ceil:.0f} ceiling at {own_str} ownership. "
            f"Underpriced at ${sal:,}.")


def _render_the_board(sport: str, pool: pd.DataFrame, edge_analysis: Dict[str, Any], slate_date: str = "") -> None:
    """Render The Board -- tight, curated Ricky voice briefing.

    Structure:
      1. Last Slate: 1-2 sentences -- what hit, what missed
      2. Slate Read: 2-3 sentences -- game environment, pace, blowouts, where upside lives
      3. Ricky's Plays: 2-3 non-obvious edges the field won't find
      4. The Fade: 1 chalk trap with a specific reason
    """
    from yak_core.board import compute_stack_targets, compute_sniper_spots, compute_fades
    from yak_core.rickys_take import generate_bust_call, generate_last_night, reset_rotator

    reset_rotator(slate_date=slate_date or None)

    st.markdown("### \U0001f4cb The Board")

    parts: list = []

    # -- 1. Last Slate (1-2 sentences) --
    recap = None
    try:
        from yak_core.slate_recap import get_previous_slate_recap
        recap = get_previous_slate_recap(sport)
    except Exception:
        pass

    last_night = generate_last_night(recap, sport=sport)
    if last_night:
        recap_date = recap.get("slate_date", "") if recap else ""
        date_label = f" ({recap_date})" if recap_date else ""
        parts.append(
            f'<div style="margin-bottom:8px;font-weight:600;font-size:0.88rem;'
            f'color:rgba(240,240,240,0.5);">Last Slate{date_label}</div>'
            f'<div class="the-board-last-night">{last_night}</div>'
        )

    # -- 2. Slate Read (game environment -- 2-3 sentences) --
    slate_read_lines: list = []

    # Best stack game
    stacks = compute_stack_targets(pool, edge_analysis)
    if stacks:
        s = stacks[0]
        slate_read_lines.append(
            f"{s['team1']}-{s['team2']} is the game tonight. "
            f"{s['vegas_total']:.0f} total \u2014 that's where the ceiling lives."
        )

    # Blowout risk
    if "spread" in pool.columns:
        _spread_col = pd.to_numeric(pool["spread"], errors="coerce").fillna(0)
        _blowout_mask = _spread_col.abs() > 10
        if _blowout_mask.any():
            _bo_idx = _blowout_mask.idxmax()
            _bo_team = pool.loc[_bo_idx, "team"]
            _bo_sp = abs(_spread_col.loc[_bo_idx])
            slate_read_lines.append(
                f"Blowout watch: {_bo_team} in a {_bo_sp:.0f}-point spread game. "
                f"Starters could see reduced 4th-quarter run."
            )

    # Injury cascade opportunity
    if "injury_bump_fp" in pool.columns:
        _bump_col = pd.to_numeric(pool["injury_bump_fp"], errors="coerce").fillna(0)
        _bump_mask = _bump_col > 3.0
        if _bump_mask.any():
            _bumped = pool.loc[_bump_mask].nlargest(1, "injury_bump_fp").iloc[0]
            slate_read_lines.append(
                f"Injury edge: {_bumped['player_name']} picks up "
                f"{_bumped['injury_bump_fp']:.1f} extra FP from a cascade. "
                f"The field hasn't priced it in."
            )

    if slate_read_lines:
        parts.append(
            '<div style="margin-top:12px;margin-bottom:4px;font-weight:600;font-size:0.88rem;">'
            'Slate Read</div>'
        )
        for line in slate_read_lines[:3]:
            parts.append(f'<div class="the-board-edge-callout">{line}</div>')

    # -- 3. Ricky's Plays (2-3 non-obvious picks) --
    snipers = compute_sniper_spots(pool, edge_analysis)
    if snipers:
        parts.append(
            '<div style="margin-top:12px;margin-bottom:4px;font-weight:600;font-size:0.88rem;">'
            "Ricky's Plays</div>"
        )
        for p in snipers[:3]:
            own_str = f"{p['own_pct']:.1f}%" if p.get("own_pct", 0) > 0 else "low"
            reason = _sniper_reason(p, pool)
            parts.append(
                f'<div class="the-board-edge-callout">'
                f"{p['player_name']} ({p['team']}, ${p['salary']:,}) \u2014 {reason}"
                f'</div>'
            )

    # -- 4. The Fade (1 chalk trap) --
    _pos_names: set = set()
    for _tier in ("core_plays", "leverage_plays", "value_plays"):
        for _p in edge_analysis.get(_tier, []):
            _pos_names.add(_p.get("player_name", ""))
    _pos_names.discard("")

    bust = generate_bust_call(pool, edge_analysis.get("fade_candidates"), positive_tier_names=_pos_names or None)
    if bust:
        parts.append(
            f'<div class="the-board-bust-call">'
            f'<strong>\U0001f480 Fade of the slate: {bust["name"]} (${bust["salary"]:,}).</strong> '
            f'{bust["explanation"]}'
            f'</div>'
        )
    else:
        fades = compute_fades(pool, edge_analysis)
        if fades:
            f = fades[0]
            parts.append(
                f'<div class="the-board-bust-call">'
                f'<strong>\U0001f480 The Fade: {f["player_name"]} ({f["own_pct"]:.1f}% owned).</strong> '
                f'{f.get("reasoning", "Model says pass.")}'
                f'</div>'
            )

    # Render
    if parts:
        st.markdown(
            '<div class="the-board">' + "".join(parts) + '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No strong reads on this slate")


def _render_late_swap_alerts(alerts: list, sport: str, lineups: dict | None = None) -> None:
    """Render late swap alert boxes above The Board.

    - Red (high impact): st.error with cascade, pivots, lineup flags
    - Yellow (medium): st.warning one-liner
    - Cleared: stored in session state for inline badge rendering
    - Hidden/low: suppressed entirely
    - No changes: tiny caption confirming check ran
    """
    if not alerts:
        st.caption("Injury check complete (0 impactful changes)")
        return

    high = [a for a in alerts if a.get("impact") == "high"]
    med = [a for a in alerts if a.get("impact") == "medium"]
    cleared = [a for a in alerts if a.get("impact") == "cleared"]
    # "low" impact alerts are suppressed entirely

    timestamp = alerts[0].get("timestamp", "") if alerts else ""

    if high:
        n = len(high)
        with st.container():
            st.error(f"🔴 LATE SWAP ALERT — {n} high-impact change{'s' if n != 1 else ''} since pool load ({timestamp})")
            for a in high:
                note = f" ({a['injury_note']})" if a.get("injury_note") else ""
                st.markdown(f"**{a['player_name']}** → {a['new_status']}{note} — was {a['old_status']} at load")

                # Cascade beneficiaries
                for b in a.get("cascade_beneficiaries", [])[:3]:
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;└ {b['name']} (${b['salary']:,}, "
                        f"+{b['extra_minutes']:.0f} min, +{b['fp_bump']:.1f} FP)"
                    )

                # Replacement pivots
                if a.get("cash_pivot"):
                    cp = a["cash_pivot"]
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;└ **Cash pivot:** {cp['name']} "
                        f"(${cp['salary']:,}, proj {cp['proj']:.1f}, {cp['ownership']:.0f}% owned) — safe floor"
                    )
                if a.get("gpp_pivot"):
                    gp = a["gpp_pivot"]
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;└ **GPP pivot:** {gp['name']} "
                        f"(${gp['salary']:,}, proj {gp['proj']:.1f}, {gp['ownership']:.0f}% owned) — leverage play"
                    )

                # Lineup impact
                in_lu = a.get("in_lineups", [])
                if in_lu:
                    lu_str = ", ".join(f"Lineup {i + 1}" for i in in_lu)
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;└ ⚠️ **IN YOUR LINEUPS:** {lu_str}")
                    # "View affected lineups" button
                    btn_key = f"view_affected_{a['player_name']}_{sport}"
                    if st.button(f"View affected lineups for {a['player_name']}", key=btn_key):
                        st.session_state[f"filter_lineups_player_{sport}"] = a["player_name"]
                else:
                    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;└ Not in any saved lineups")

    if med:
        n = len(med)
        lines = []
        for a in med:
            note = f" ({a.get('injury_note', '')})" if a.get("injury_note") else ""
            lines.append(
                f"**{a['player_name']}** → {a['new_status']}{note} "
                f"(proj {a['proj']:.1f}, ${a['salary']:,}) — low rotation impact"
            )
        st.warning(f"🟡 STATUS UPDATE — {n} change{'s' if n != 1 else ''} since pool load\n\n" + "\n\n".join(lines))

    # Cleared players: store in session state for inline badge rendering
    if cleared:
        st.session_state[f"cleared_players_{sport}"] = [a["player_name"] for a in cleared]

    if not high and not med:
        st.caption(f"Injury check complete (0 impactful changes) · Last checked {timestamp}")


def render_edge_tab(sport: str) -> None:
    """Render Ricky's Edge Analysis tab."""
    from app.data_loader import invalidate_published_cache, load_published_data

    # Inject CSS once
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    # Manual refresh button so user can force-reload after new lineups are built
    if st.button("🔄 Refresh", key=f"edge_refresh_{sport}"):
        invalidate_published_cache()
        st.rerun()

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
    pool_size = len(pool)  # actual filtered pool, not raw DK draftables

    # ── Header ──
    st.markdown(f"## 📐 Right Angle Ricky — {sport}")
    st.caption(f"{slate_date} · {pool_size} players · DraftKings")

    # PGA: Under Construction banner
    if is_pga:
        st.info("🚧 PGA — Under Construction. Course fit, SG breakdowns, and PGA-specific callouts coming soon.")

    # ── Late Swap Alerts (above The Board) ────────────────────────────
    _late_swap = edge_analysis.get("late_swap_alerts", [])
    _render_late_swap_alerts(_late_swap, sport, lineups)

    # ── The Board ─────────────────────────────────────────────────────
    _render_the_board(sport, pool, edge_analysis, slate_date=slate_date)

    st.markdown("")  # spacer

    # ── PGA wave split (above the 3-box if present) ──
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

    # ── 3-box dashboard ──
    _cleared = st.session_state.get(f"cleared_players_{sport}", [])
    col1, col2, col3 = st.columns(3)
    with col1:
        _render_edge_box("core_plays", edge_analysis.get("core_plays", []), is_pga, _cleared)
    with col2:
        _render_edge_box("leverage_plays", edge_analysis.get("leverage_plays", []), is_pga, _cleared)
    with col3:
        _render_edge_box("value_plays", edge_analysis.get("value_plays", []), is_pga, _cleared)

    # ── Published lineups ──
    if lineups:
        st.markdown("---")
        st.markdown("### 📋 Published Lineups")

        # Check for "View affected lineups" filter
        _filter_player = st.session_state.get(f"filter_lineups_player_{sport}", "")
        if _filter_player:
            st.info(f"Showing lineups containing **{_filter_player}**")
            if st.button("Clear filter", key=f"clear_lineup_filter_{sport}"):
                del st.session_state[f"filter_lineups_player_{sport}"]
                st.rerun()

        # Load matchup labels from per-contest meta files for better display
        from app.data_loader import DATA_DIR
        _contest_meta_dir = DATA_DIR / sport.lower()

        for contest_slug, ldf in lineups.items():
            # Try to read matchup from the contest's meta file
            _meta_file = _contest_meta_dir / f"{contest_slug}_meta.json"
            _matchup_label = ""
            if _meta_file.exists():
                try:
                    import json as _json
                    _cmeta = _json.loads(_meta_file.read_text())
                    _matchup_label = _cmeta.get("matchup", "")
                except Exception:
                    pass

            if _matchup_label:
                label = f"Showdown — {_matchup_label}"
            else:
                label = contest_slug.replace("_", " ").title()
            n_lu = ldf["lineup_index"].nunique() if "lineup_index" in ldf.columns else 0
            with st.expander(f"{label} — {n_lu} lineups"):
                if "lineup_index" not in ldf.columns:
                    st.dataframe(ldf, use_container_width=True, hide_index=True)
                    continue

                # Determine which lineup indices to show
                _all_indices = sorted(ldf["lineup_index"].unique())
                if _filter_player and "player_name" in ldf.columns:
                    _affected = ldf[ldf["player_name"] == _filter_player]["lineup_index"].unique()
                    _show_indices = sorted(set(_all_indices) & set(_affected))
                    if not _show_indices:
                        st.caption(f"{_filter_player} not in any {label} lineups")
                        continue
                else:
                    _show_indices = _all_indices

                for idx in _show_indices:
                    lu = ldf[ldf["lineup_index"] == idx].copy()
                    total_sal = int(pd.to_numeric(lu["salary"], errors="coerce").fillna(0).sum()) if "salary" in lu.columns else 0
                    total_proj = float(pd.to_numeric(lu["proj"], errors="coerce").fillna(0).sum()) if "proj" in lu.columns else 0.0
                    total_ceil = float(pd.to_numeric(lu["ceil"], errors="coerce").fillna(0).sum()) if "ceil" in lu.columns else 0.0
                    ceil_part = f" · {total_ceil:.1f} ceil" if total_ceil > 0 else ""
                    st.markdown(f"**Lineup {idx + 1}** — ${total_sal:,} sal · {total_proj:.1f} proj{ceil_part}")

                    # Apply cleared badge and highlight filtered player
                    if "player_name" in lu.columns:
                        if _cleared:
                            lu["player_name"] = lu["player_name"].apply(
                                lambda n: f"✅ {n}" if n in _cleared else n
                            )
                        if _filter_player:
                            # Show salary remaining if the filtered player were removed
                            _filt_sal = pd.to_numeric(
                                lu.loc[lu["player_name"].str.contains(_filter_player, na=False), "salary"],
                                errors="coerce",
                            ).sum() if "salary" in lu.columns else 0
                            if _filt_sal > 0:
                                st.caption(f"Salary freed if {_filter_player} removed: ${int(_filt_sal):,}")
                            lu["player_name"] = lu["player_name"].apply(
                                lambda n: f"🔴 {n}" if _filter_player in str(n) else n
                            )

                    display_cols = ["slot", "player_name", "pos", "salary", "proj", "ceil"]
                    if is_pga:
                        display_cols = ["slot", "player_name", "salary", "proj", "wave", "r1_teetime"]
                        if "wave" not in lu.columns and "early_late_wave" in lu.columns:
                            lu["wave"] = lu["early_late_wave"].map(
                                {0: "Early", 1: "Late", "Early": "Early", "Late": "Late"}
                            )
                    avail = [c for c in display_cols if c in lu.columns]
                    _lu_edge_disp = lu[avail].reset_index(drop=True)
                    _fmt = {}
                    for _rc in ["proj", "ceil", "floor", "gpp_score", "own_pct"]:
                        if _rc in _lu_edge_disp.columns:
                            _lu_edge_disp[_rc] = pd.to_numeric(_lu_edge_disp[_rc], errors="coerce").round(2)
                            _fmt[_rc] = "{:.2f}"
                    st.dataframe(
                        _lu_edge_disp.style.format(_fmt, na_rep="") if _fmt else _lu_edge_disp,
                        use_container_width=True,
                        hide_index=True,
                    )
