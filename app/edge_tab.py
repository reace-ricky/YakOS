"""Tab 1: Ricky's Edge Analysis (public).

Displays the pre-computed edge analysis from data/published/{sport}/:
  - The Board: Last Slate Recap, Today's Edge (voice-driven callouts), Bust Call
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


def _generate_board_callouts(
    stacks: List[Dict[str, Any]],
    snipers: List[Dict[str, Any]],
    fades: List[Dict[str, Any]],
    pool: pd.DataFrame,
) -> List[str]:
    """Convert structured board data into Ricky-voice text callouts.

    Returns 3-6 callouts depending on what's relevant. Uses the
    _TemplateRotator from rickys_take.py for daily variety.
    """
    from yak_core.rickys_take import _pick_template, _pick_template_by_key

    callouts: List[str] = []

    # ── Sniper alerts ──
    _SNIPER_TEMPLATES = [
        "The field is sleeping on {name} tonight. Ranked #{rank} in my model at just {own:.1f}% ownership — {ceil:.0f} ceiling in a {total:.0f}-total game.",
        "{name} at ${salary:,} is my favorite misfire tonight. {own:.1f}% owned, {proj:.1f} projected, {ceil:.0f} ceiling. The field doesn't see it.",
        "If you're looking for a pivot, {name} checks every box. Top-20 projection, sub-10% ownership, in a game totaling {total:.0f}.",
    ]
    for i, p in enumerate(snipers):
        # Look up vegas total for this player's game
        game_total = 0.0
        if not pool.empty and "player_name" in pool.columns:
            prow = pool[pool["player_name"] == p["player_name"]]
            if not prow.empty:
                for vc in ("over_under", "vegas_total", "total"):
                    if vc in prow.columns and prow[vc].notna().any():
                        game_total = float(prow[vc].iloc[0])
                        break
        # Rank = position in top-N by proj (1-indexed)
        rank = i + 1
        tmpl = _pick_template(
            _SNIPER_TEMPLATES, p["player_name"], "board_sniper",
        )
        callouts.append(tmpl.format(
            name=p["player_name"], rank=rank, own=p["own_pct"],
            ceil=p["ceil"], total=game_total, salary=p["salary"],
            proj=p["proj"],
        ))

    # ── Fades ──
    _FADE_TEMPLATES = [
        "{name} at {own:.1f}% ownership? Pass. My model has him ranked #{rank} — there are better plays at ${salary:,}.",
        "The field loves {name} at {own:.1f}% tonight. I don't. Ranked #{rank} with a {proj:.1f} projection. You're paying for the name.",
        "Ownership trap alert: {name} ({own:.1f}%). Model says he's a fade — ranked #{rank}, ceiling doesn't justify the chalk.",
    ]
    for p in fades:
        # Extract rank from reasoning ("Ranked #30 by Ricky vs #5 by salary")
        rank = 0
        reasoning = p.get("reasoning", "")
        if "Ranked #" in reasoning:
            try:
                rank = int(reasoning.split("Ranked #")[1].split(" ")[0])
            except (ValueError, IndexError):
                pass
        tmpl = _pick_template(
            _FADE_TEMPLATES, p["player_name"], "board_fade",
        )
        callouts.append(tmpl.format(
            name=p["player_name"], own=p["own_pct"], rank=rank,
            salary=p["salary"], proj=p["proj"],
        ))

    # ── Stack callouts ──
    _STACK_TEMPLATES = [
        "{team1}-{team2} is the stack tonight. {total:.0f} total, {p1} + {p2} combine for a {ceil:.0f} ceiling.",
        "Game environment alert: {team1} at {team2} with a {total:.0f} total. Stack {p1} and {p2} — that's where the ceiling lives.",
    ]
    for s in stacks:
        key = f"{s['team1']}:{s['team2']}"
        tmpl = _pick_template_by_key(
            _STACK_TEMPLATES, key, "board_stack",
        )
        callouts.append(tmpl.format(
            team1=s["team1"], team2=s["team2"],
            total=s["vegas_total"], p1=s["top_player1"],
            p2=s["top_player2"], ceil=s["combined_ceil"],
        ))

    # ── Injury edges (players with injury_bump_fp > 0) ──
    _INJURY_TEMPLATES = [
        "{name} picks up an extra {bump:.1f} FP from the {out_player} absence. The field hasn't priced this in yet.",
    ]
    if not pool.empty and "injury_bump_fp" in pool.columns:
        bump_col = pd.to_numeric(pool["injury_bump_fp"], errors="coerce").fillna(0)
        bumped = pool[bump_col > 0].copy()
        bumped["_bump"] = bump_col[bumped.index]
        if not bumped.empty:
            top_bump = bumped.nlargest(1, "_bump").iloc[0]
            out_player = top_bump.get("injury_source", "an injury")
            tmpl = _pick_template(
                _INJURY_TEMPLATES, str(top_bump.get("player_name", "")), "board_injury",
            )
            callouts.append(tmpl.format(
                name=top_bump.get("player_name", "?"),
                bump=float(top_bump["_bump"]),
                out_player=out_player,
            ))

    # ── Blowout flags (spread > 10) ──
    _BLOWOUT_TEMPLATES = [
        "Blowout watch: {team} is a {spread:.0f}-point favorite. Starters could see reduced 4th-quarter minutes.",
    ]
    if not pool.empty and "spread" in pool.columns:
        spread_col = pd.to_numeric(pool["spread"], errors="coerce").fillna(0)
        blowout = pool[spread_col.abs() > 10].copy()
        if not blowout.empty:
            blowout["_spread"] = spread_col[blowout.index].abs()
            top_bo = blowout.nlargest(1, "_spread").iloc[0]
            team = top_bo.get("team", "?")
            tmpl = _pick_template_by_key(
                _BLOWOUT_TEMPLATES, str(team), "board_blowout",
            )
            callouts.append(tmpl.format(
                team=team, spread=float(top_bo["_spread"]),
            ))

    # Cap at 6, minimum is whatever data produced (don't force empty)
    return callouts[:6]


def _render_the_board(sport: str, pool: pd.DataFrame, edge_analysis: Dict[str, Any], slate_date: str = "") -> None:
    """Render The Board: Last Slate recap, Today's Edge callouts, Bust Call."""
    from yak_core.board import compute_stack_targets, compute_sniper_spots, compute_fades
    from yak_core.rickys_take import generate_bust_call, generate_last_night, reset_rotator

    # Reset the template rotator so intra-post dedup starts fresh and the
    # seed is pinned to the current slate date (deterministic output).
    reset_rotator(slate_date=slate_date or None)

    st.markdown('<div class="the-board"><h3>📋 The Board</h3>', unsafe_allow_html=True)

    # ── 1. Last Slate Recap (moved to top) ────────────────────────────────
    recap = None
    try:
        from yak_core.slate_recap import get_previous_slate_recap
        recap = get_previous_slate_recap(sport)
    except Exception as exc:
        print(f"[edge_tab] Slate recap error: {exc}")

    last_night = generate_last_night(recap, sport=sport)
    if last_night:
        recap_date = recap.get("slate_date", "") if recap else ""
        date_label = f' <span style="color:rgba(240,240,240,0.4);font-size:0.8rem;">({recap_date})</span>' if recap_date else ""
        st.markdown(
            f'<div style="margin-bottom:4px;font-weight:600;font-size:0.88rem;color:rgba(240,240,240,0.5);">Last Slate{date_label}</div>'
            f'<div class="the-board-last-night">{last_night}</div>',
            unsafe_allow_html=True,
        )

    # ── 2. Today's Edge — voice-driven callouts ──────────────────────────
    stacks = compute_stack_targets(pool, edge_analysis)
    snipers = compute_sniper_spots(pool, edge_analysis)
    fades = compute_fades(pool, edge_analysis)

    callouts = _generate_board_callouts(stacks, snipers, fades, pool)

    st.markdown(
        '<div style="margin-top:8px;margin-bottom:4px;font-weight:700;font-size:1.0rem;">Today\'s Edge</div>',
        unsafe_allow_html=True,
    )
    if callouts:
        for callout in callouts:
            st.markdown(
                f'<div class="the-board-edge-callout">{callout}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No strong edges on this slate")

    # ── 3. Bust Call ──────────────────────────────────────────────────────
    _pos_names = set()
    for _tier in ("core_plays", "leverage_plays", "value_plays"):
        for _p in edge_analysis.get(_tier, []):
            _pos_names.add(_p.get("player_name", ""))
    _pos_names.discard("")
    bust = generate_bust_call(pool, edge_analysis.get("fade_candidates"), positive_tier_names=_pos_names or None)
    if bust:
        st.markdown(
            f'<div class="the-board-bust-call">'
            f'<strong>💀 Fade of the slate: {bust["name"]} (${bust["salary"]:,}).</strong> '
            f'{bust["explanation"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


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
