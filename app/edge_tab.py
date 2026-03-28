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


# ---------------------------------------------------------------------------
# Signal-based play classification for The Board
# ---------------------------------------------------------------------------

# Signal IDs (each play gets exactly one)
_SIG_USAGE = "usage"       # injury cascade / minutes bump
_SIG_PACE = "pace"         # pace-up / blowout equity
_SIG_CEILING = "ceiling"   # high ceiling-to-floor spread
_SIG_VALUE = "value"       # price-to-production ratio

# Role labels per signal
_ROLE_MAP = {
    _SIG_USAGE:  ("\U0001f512 Core", "core"),
    _SIG_PACE:   ("\u26a1 Pivot", "pivot"),
    _SIG_CEILING:("\U0001f3b0 GPP Dart", "dart"),
    _SIG_VALUE:  ("\U0001f512 Core", "core"),  # value can also be Core
}


def _classify_signal(p: Dict[str, Any], pool: pd.DataFrame) -> tuple:
    """Return (signal_id, reason_str) for a sniper candidate.

    Picks the strongest applicable signal and writes a data-specific reason.
    No two players should get the same reason string.
    """
    name = p.get("player_name", "")
    proj = p.get("proj", 0)
    ceil = p.get("ceil", 0)
    floor_val = p.get("floor", 0)
    sal = p.get("salary", 0)
    own = p.get("own_pct", 0)
    bump = p.get("injury_bump_fp", 0)
    mins = p.get("proj_minutes", 0)
    r5 = p.get("rolling_fp_5", 0)

    # Look up pool row for vegas/spread if available
    vegas = 0.0
    spread = 0.0
    if not pool.empty and "player_name" in pool.columns:
        match = pool[pool["player_name"] == name]
        if not match.empty:
            row = match.iloc[0]
            for vc in ("vegas_total", "over_under", "total"):
                if vc in row.index:
                    vegas = float(row.get(vc, 0) or 0)
                    if vegas > 0:
                        break
            spread = abs(float(row.get("spread", 0) or 0))

    # Score each signal
    scores = {}

    # Usage/minutes signal
    if bump > 2.0:
        scores[_SIG_USAGE] = bump * 3.0
    elif mins >= 32:
        scores[_SIG_USAGE] = mins * 0.5

    # Pace/matchup signal
    if vegas >= 225 and mins >= 28:
        scores[_SIG_PACE] = (vegas - 220) * 0.3
    if spread >= 8 and mins >= 28:
        scores[_SIG_PACE] = scores.get(_SIG_PACE, 0) + spread * 0.5

    # Ceiling signal
    if ceil > 0 and proj > 0:
        ceil_spread = (ceil - proj) / proj
        if ceil_spread > 0.30:
            scores[_SIG_CEILING] = ceil_spread * 10

    # Value signal
    if sal > 0 and proj > 0:
        pts_per_k = proj / (sal / 1000)
        if pts_per_k >= 6.0:
            scores[_SIG_VALUE] = pts_per_k

    if not scores:
        scores[_SIG_CEILING] = 1.0  # fallback

    # Pick strongest signal
    sig = max(scores, key=scores.get)

    # Build data-specific reason
    if sig == _SIG_USAGE:
        if bump > 2.0:
            reason = f"Usage spike — +{bump:.0f} FP from injury cascade. Averaging {r5:.0f} over last 5. Price hasn't moved."
        else:
            reason = f"{mins:.0f} projected minutes, {own:.0f}% owned. Track the game log — {r5:.0f} avg last 5."
    elif sig == _SIG_PACE:
        if spread >= 8:
            reason = f"Blowout equity — {spread:.0f}-point spread. Pace-up + closing time if it's a corpse."
        else:
            reason = f"Pace-up spot — {vegas:.0f} total. {mins:.0f} minutes, {own:.0f}% owned. Field is looking elsewhere."
    elif sig == _SIG_CEILING:
        reason = f"Ceiling-to-floor spread is {ceil:.0f} vs {floor_val:.0f}. GPP-only — never in cash."
    elif sig == _SIG_VALUE:
        pts_per_k = proj / (sal / 1000) if sal > 0 else 0
        reason = f"${sal:,} for {proj:.0f} projected — {pts_per_k:.1f} pts/$1K. Clearance pricing."
    else:
        reason = f"{own:.0f}% owned, {proj:.0f} projected. Numbers are clean."

    return sig, reason


def _assign_tiered_plays(snipers: list, pool: pd.DataFrame) -> list:
    """Assign each sniper a signal-based tier. Enforce one play per signal.

    Returns list of (player_dict, signal_id, role_label, reason) tuples,
    max 3 plays, each with a unique signal.
    """
    candidates = []
    for p in snipers:
        sig, reason = _classify_signal(p, pool)
        candidates.append((p, sig, reason))

    # Dedup: one play per signal, take the first (highest priority) candidate
    used_signals = set()
    plays = []
    for p, sig, reason in candidates:
        if sig in used_signals:
            # Try to find an alternate signal for this player
            # If they only have one signal, skip them
            continue
        used_signals.add(sig)
        role_label, role_key = _ROLE_MAP.get(sig, ("\u26a1 Pivot", "pivot"))

        # If we already have a Core, downgrade subsequent usage/value to Pivot
        if role_key == "core" and any(r == "core" for _, _, r, _ in plays):
            role_label = "\u26a1 Pivot"
            role_key = "pivot"

        plays.append((p, sig, role_key, role_label, reason))
        if len(plays) >= 3:
            break

    # If we have fewer than 3, allow duplicate signals from remaining candidates
    if len(plays) < 3:
        for p, sig, reason in candidates:
            if any(p["player_name"] == pp["player_name"] for pp, *_ in plays):
                continue
            role_label = "\U0001f3b0 GPP Dart"
            plays.append((p, sig, "dart", role_label, reason))
            if len(plays) >= 3:
                break

    return plays


def _render_the_board(sport: str, pool: pd.DataFrame, edge_analysis: Dict[str, Any], slate_date: str = "") -> None:
    """Render The Board -- redesigned structure:

      1. Last Slate Recap (personal tone + verdict)
      2. The Setup (sharp narrative paragraph)
      3. Ricky's Plays (tiered: Core / Pivot / GPP Dart, signal-deduped)
      4. The Trap Stack (public narrative fade)
      5. Fade of the Slate (skull box)
    """
    from yak_core.board import compute_stack_targets, compute_sniper_spots, compute_fades
    from yak_core.rickys_take import generate_bust_call, generate_last_night, reset_rotator

    reset_rotator(slate_date=slate_date or None)

    st.markdown("### \U0001f4cb The Board")

    parts: list = []

    # ── 1. Last Slate Recap ─────────────────────────────────────────────
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

    # ── 2. The Setup (replaces Slate Read) ──────────────────────────────
    # One sharp paragraph: dominant narrative, the trap, the edge.
    setup_parts: list = []

    # Stack game
    stacks = compute_stack_targets(pool, edge_analysis)
    if stacks:
        s = stacks[0]
        setup_parts.append(
            f"{s['team1']}-{s['team2']} is the highest total at {s['vegas_total']:.0f}."
        )

    # Blowout spot
    if "spread" in pool.columns:
        _spread_col = pd.to_numeric(pool["spread"], errors="coerce").fillna(0)
        _blowout_mask = _spread_col.abs() > 8
        if _blowout_mask.any():
            _bo_rows = pool.loc[_blowout_mask].copy()
            _bo_rows["_abs_spread"] = _spread_col[_blowout_mask].abs()
            _bo_row = _bo_rows.nlargest(1, "_abs_spread").iloc[0]
            _bo_team = _bo_row["team"]
            _bo_sp = _bo_row["_abs_spread"]
            setup_parts.append(
                f"{_bo_team} is a {_bo_sp:.0f}-point spread — starters hit the bench in the 4th."
            )

    # Injury cascade narrative
    _cascade_name = ""
    if "injury_bump_fp" in pool.columns:
        _bump_col = pd.to_numeric(pool["injury_bump_fp"], errors="coerce").fillna(0)
        _bump_mask = _bump_col > 3.0
        if _bump_mask.any():
            _bumped = pool.loc[_bump_mask].nlargest(1, "injury_bump_fp").iloc[0]
            _cascade_name = _bumped["player_name"]
            _bump_fp = _bump_col.loc[_bumped.name]
            setup_parts.append(
                f"{_cascade_name} is the real minutes beneficiary — "
                f"+{_bump_fp:.0f} FP from the cascade. "
                f"The public will pile into the obvious name and miss this."
            )

    # Edge summary
    n_core = len(edge_analysis.get("core_plays", []))
    n_leverage = len(edge_analysis.get("leverage_plays", []))
    if n_core + n_leverage > 0:
        setup_parts.append(
            f"The edge is in the {n_core} core plays and {n_leverage} leverage spots the field isn't checking."
        )

    if setup_parts:
        setup_text = " ".join(setup_parts)
        parts.append(
            '<div style="margin-top:12px;margin-bottom:4px;font-weight:600;font-size:0.88rem;">'
            'The Setup</div>'
            f'<div class="the-board-edge-callout">{setup_text}</div>'
        )

    # ── 3. Ricky's Plays (tiered, signal-deduped) ──────────────────────
    snipers = compute_sniper_spots(pool, edge_analysis)
    tiered_plays = _assign_tiered_plays(snipers, pool) if snipers else []

    if tiered_plays:
        parts.append(
            '<div style="margin-top:12px;margin-bottom:4px;font-weight:600;font-size:0.88rem;">'
            "Ricky's Plays</div>"
        )
        for p, sig, role_key, role_label, reason in tiered_plays:
            parts.append(
                f'<div class="the-board-edge-callout">'
                f'<span style="color:rgba(240,240,240,0.5);font-size:0.8rem;">{role_label}</span> '
                f"<strong>{p['player_name']}</strong> ({p['team']}, ${p['salary']:,}) "
                f"\u2014 {reason}"
                f'</div>'
            )

    # ── 4. The Trap Stack ──────────────────────────────────────────────
    # Warn about the narrative everyone else is building.
    # Find the highest-owned non-core, non-sniper player who underperforms by model.
    _board_names = set()
    for _tier in ("core_plays", "leverage_plays", "value_plays"):
        for _p in edge_analysis.get(_tier, []):
            _board_names.add(_p.get("player_name", ""))
    for pp, *_ in tiered_plays:
        _board_names.add(pp.get("player_name", ""))
    _board_names.discard("")

    if "ownership" in pool.columns and "proj" in pool.columns:
        _own_col = pd.to_numeric(pool.get("ownership", pool.get("own_proj", 0)), errors="coerce").fillna(0)
        if _own_col.max() <= 1.0:
            _own_col = _own_col * 100
        _proj_col = pd.to_numeric(pool["proj"], errors="coerce").fillna(0)
        _sal_col = pd.to_numeric(pool.get("salary", 0), errors="coerce").fillna(0)
        _r5_col = pd.to_numeric(pool.get("rolling_fp_5", 0), errors="coerce").fillna(0)

        # Candidates: >8% owned, not on the board, and pts/$1K below 5.5
        _trap_mask = (
            (_own_col > 8)
            & (~pool["player_name"].isin(_board_names))
            & (_sal_col > 0)
            & (_proj_col / (_sal_col / 1000) < 5.5)
        )
        if _trap_mask.any():
            _trap_df = pool[_trap_mask].copy()
            _trap_df["_own"] = _own_col[_trap_mask]
            _trap_row = _trap_df.nlargest(1, "_own").iloc[0]
            _trap_name = _trap_row["player_name"]
            _trap_own = float(_trap_df.loc[_trap_row.name, "_own"])
            _trap_proj = float(_proj_col.loc[_trap_row.name])
            _trap_sal = int(_sal_col.loc[_trap_row.name])
            _trap_r5 = float(_r5_col.loc[_trap_row.name])
            _trap_ppk = _trap_proj / (_trap_sal / 1000) if _trap_sal > 0 else 0

            parts.append(
                '<div style="margin-top:12px;margin-bottom:4px;font-weight:600;font-size:0.88rem;">'
                '\u26a0\ufe0f The Trap Stack</div>'
                f'<div class="the-board-edge-callout">'
                f'The field is going to talk themselves into '
                f'<strong>{_trap_name}</strong> ({_trap_own:.0f}% owned, ${_trap_sal:,}). '
                f'Averaging {_trap_r5:.0f} over his last 5 at {_trap_ppk:.1f} pts/$1K. '
                f'That\'s not a leverage play \u2014 it\'s a trap.'
                f'</div>'
            )
            _board_names.add(_trap_name)

    # ── 5. Fade of the Slate ───────────────────────────────────────────
    bust = generate_bust_call(pool, edge_analysis.get("fade_candidates"), positive_tier_names=_board_names or None)
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
            fades = [f for f in fades if f.get("player_name", "") not in _board_names]
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
        st.caption("Nothing worth forcing. Play the board or sit tonight out.")


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
