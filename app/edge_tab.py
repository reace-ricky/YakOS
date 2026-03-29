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
/* ── The Board (accessible: bright headers, solid pills, high contrast) ── */
.the-board {
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 22px 24px;
    margin-bottom: 16px;
    background: #0f172a;
}
.tb-section-label {
    text-transform: uppercase;
    font-size: 14px;
    letter-spacing: 2px;
    color: #f59e0b;
    margin-top: 24px;
    margin-bottom: 10px;
    font-weight: 700;
}
.tb-recap {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 14px 16px;
    font-size: 15px;
    color: #e2e8f0;
    line-height: 1.7;
    margin-bottom: 18px;
}
.tb-setup {
    font-size: 15px;
    color: #e2e8f0;
    line-height: 1.7;
    margin-bottom: 18px;
}
.tb-play-row {
    margin-bottom: 12px;
    font-size: 15px;
    line-height: 1.6;
    padding: 8px 12px;
    background: #1e293b;
    border-radius: 6px;
}
.tb-play-row .tb-pill {
    display: inline-block;
    background: #f59e0b;
    color: #000000;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-right: 8px;
    vertical-align: middle;
}
.tb-play-row .tb-pill-dart {
    background: #8b5cf6;
    color: #ffffff;
}
.tb-play-row .tb-pill-pivot {
    background: #14b8a6;
    color: #000000;
}
.tb-play-row .tb-name {
    font-weight: 700;
    color: #ffffff;
    font-size: 16px;
}
.tb-play-row .tb-meta {
    color: #e2e8f0;
}
.tb-danger-box {
    background: #1e293b;
    border-left: 4px solid #ef4444;
    border-radius: 6px;
    padding: 14px 18px;
    margin-top: 18px;
    font-size: 15px;
    color: #fde2e2;
    line-height: 1.7;
}
.tb-danger-box .tb-divider {
    border-top: 1px solid #475569;
    margin: 10px 0;
}
</style>
"""

# ── Box colors + emojis ──────────────────────────────────────────────────────
_BOX_CONFIG = {
    "core_plays": {"title": "Core Plays", "emoji": "\U0001f3af", "color": "#2196F3"},
    "leverage_plays": {"title": "Leverage Plays", "emoji": "\U0001f48e", "color": "#FF9800"},
    "value_plays": {"title": "Value Plays", "emoji": "\U0001f4b0", "color": "#4CAF50"},
}


def _render_player_card_html(player: Dict[str, Any], is_pga: bool, cleared_players: list | None = None) -> str:
    name = player.get("player_name", "?")
    sal = player.get("salary", 0)
    proj = player.get("proj", 0)
    edge = player.get("edge", 0)
    own = player.get("ownership", 0)
    risk = player.get("risk_score", 0)
    mins = player.get("proj_minutes", 0)
    ceil_val = player.get("ceil", 0)

    cleared = cleared_players or []
    cleared_badge = ""
    if name in cleared:
        cleared_badge = (
            ' <span style="background:#1b5e20;color:#a5d6a7;padding:1px 6px;'
            'border-radius:3px;font-size:0.72rem;font-weight:600;margin-left:4px;">CLEARED</span>'
        )

    if is_pga:
        stats = f"${sal:,} \u00b7 Proj {proj:.1f} \u00b7 Edge {edge:.2f} \u00b7 Own {own:.1f}% \u00b7 Risk {risk:.0f}"
    else:
        stats = f"${sal:,} \u00b7 Proj {proj:.1f} \u00b7 Edge {edge:.2f} \u00b7 Own {own:.1f}% \u00b7 Mins {mins:.0f} \u00b7 Ceil {ceil_val:.0f}"

    return (
        f'<div class="player-card">'
        f'<div class="name">{name}{cleared_badge}</div>'
        f'<div class="stats">{stats}</div>'
        f'</div>'
    )


def _render_edge_box(key: str, players: List[Dict], is_pga: bool, cleared_players: list | None = None) -> None:
    cfg = _BOX_CONFIG.get(key, {"title": key, "emoji": "\U0001f4ca", "color": "#9E9E9E"})
    if not players:
        return
    cards_html = "\n".join(_render_player_card_html(p, is_pga, cleared_players) for p in players)
    box_html = (
        f'<div class="edge-box" style="border-color: {cfg["color"]}40;">'
        f'<h4>{cfg["emoji"]} {cfg["title"]}</h4>'
        f'{cards_html}'
        f'</div>'
    )
    st.markdown(box_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Signal-based play classification for The Board
# ---------------------------------------------------------------------------

_SIG_USAGE = "usage"
_SIG_PACE = "pace"
_SIG_CEILING = "ceiling"
_SIG_VALUE = "value"

_ROLE_MAP = {
    _SIG_USAGE:   ("\U0001f512 Core", "core"),
    _SIG_PACE:    ("\u26a1 Pivot", "pivot"),
    _SIG_CEILING: ("\U0001f3b0 GPP Dart", "dart"),
    _SIG_VALUE:   ("\U0001f512 Core", "core"),
}

# Reason templates per signal -- each uses unique data fields so no two players
# can produce the same string
_REASON_BUILDERS = {
    _SIG_USAGE: lambda p, bump, r5, own, mins, **_: (
        f"+{bump:.0f} FP bump, {own:.0f}% owned. "
        f"Averaging {r5:.0f} over last 5 \u2014 salary hasn't moved."
        if bump > 2.0
        else f"{mins:.0f} projected minutes at {own:.0f}% owned. "
             f"Last 5 average is {r5:.0f} \u2014 field is asleep on the log."
    ),
    _SIG_PACE: lambda p, spread, vegas, mins, own, **_: (
        f"{spread:.0f}-point spread, {mins:.0f} minutes. "
        f"Garbage time equity is real \u2014 ownership won't reflect it."
        if spread >= 8
        else f"Game total is {vegas:.0f}. {mins:.0f} minutes, {own:.0f}% owned. "
             f"Pace bump is priced in everywhere except here."
    ),
    _SIG_CEILING: lambda p, ceil, floor_val, proj, own, **_: (
        f"Ceiling {ceil:.0f}, floor {floor_val:.0f}, proj {proj:.0f}. "
        f"That spread at {own:.0f}% owned is GPP leverage \u2014 never touch in cash."
    ),
    _SIG_VALUE: lambda p, proj, sal, pts_per_k, r5, **_: (
        f"${sal:,} for {proj:.0f} projected ({pts_per_k:.1f} pts/$1K). "
        f"Last 5 average is {r5:.0f}. Clearance pricing the model can't ignore."
    ),
}


def _classify_signal(p: Dict[str, Any], pool: pd.DataFrame) -> tuple:
    name = p.get("player_name", "")
    proj = float(p.get("proj", 0) or 0)
    ceil = float(p.get("ceil", 0) or 0)
    floor_val = float(p.get("floor", 0) or 0)
    sal = int(p.get("salary", 0) or 0)
    own = float(p.get("own_pct", 0) or 0)
    bump = float(p.get("injury_bump_fp", 0) or 0)
    mins = float(p.get("proj_minutes", 0) or 0)
    r5 = float(p.get("rolling_fp_5", 0) or 0)

    vegas = 0.0
    spread = 0.0
    if not pool.empty and "player_name" in pool.columns:
        match = pool[pool["player_name"] == name]
        if not match.empty:
            row = match.iloc[0]
            for vc in ("vegas_total", "over_under", "total"):
                if vc in row.index and row[vc]:
                    vegas = float(row[vc] or 0)
                    if vegas > 0:
                        break
            spread = abs(float(row.get("spread", 0) or 0))

    scores = {}
    if bump > 2.0:
        scores[_SIG_USAGE] = bump * 3.0
    elif mins >= 32:
        scores[_SIG_USAGE] = mins * 0.5
    if vegas >= 225 and mins >= 28:
        scores[_SIG_PACE] = (vegas - 220) * 0.3
    if spread >= 8 and mins >= 28:
        scores[_SIG_PACE] = scores.get(_SIG_PACE, 0) + spread * 0.5
    if ceil > 0 and proj > 0 and (ceil - proj) / proj > 0.30:
        scores[_SIG_CEILING] = ((ceil - proj) / proj) * 10
    pts_per_k = proj / (sal / 1000) if sal > 0 else 0
    if pts_per_k >= 6.0:
        scores[_SIG_VALUE] = pts_per_k
    if not scores:
        scores[_SIG_CEILING] = 1.0

    sig = max(scores, key=scores.get)
    reason = _REASON_BUILDERS[sig](
        p, bump=bump, r5=r5, own=own, mins=mins,
        spread=spread, vegas=vegas, ceil=ceil,
        floor_val=floor_val, proj=proj, sal=sal,
        pts_per_k=pts_per_k,
    )
    return sig, reason


def _assign_tiered_plays(snipers: list, pool: pd.DataFrame) -> list:
    candidates = []
    for p in snipers:
        sig, reason = _classify_signal(p, pool)
        candidates.append((p, sig, reason))

    used_signals = set()
    plays = []
    for p, sig, reason in candidates:
        if sig in used_signals:
            continue
        used_signals.add(sig)
        role_label, role_key = _ROLE_MAP.get(sig, ("\u26a1 Pivot", "pivot"))
        if role_key == "core" and any(r == "core" for _, _, r, _, _ in plays):
            role_label, role_key = "\u26a1 Pivot", "pivot"
        plays.append((p, sig, role_key, role_label, reason))
        if len(plays) >= 3:
            break

    if len(plays) < 3:
        for p, sig, reason in candidates:
            if any(p["player_name"] == pp["player_name"] for pp, *_ in plays):
                continue
            plays.append((p, sig, "dart", "\U0001f3b0 GPP Dart", reason))
            if len(plays) >= 3:
                break

    return plays


def _render_the_board(sport: str, pool: pd.DataFrame, edge_analysis: Dict[str, Any], slate_date: str = "") -> None:
    from yak_core.board import compute_stack_targets, compute_sniper_spots, compute_fades
    from yak_core.rickys_take import generate_bust_call, generate_last_night, reset_rotator

    reset_rotator(slate_date=slate_date or None)

    st.markdown("### \U0001f4cb The Board")

    parts: list = []

    # -- 1. Last Slate Recap ------------------------------------------------
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
            f'<div class="tb-section-label">LAST SLATE{date_label}</div>'
            f'<div class="tb-recap">{last_night}</div>'
        )

    # -- 2. The Setup -------------------------------------------------------
    setup_parts: list = []
    stacks = compute_stack_targets(pool, edge_analysis)
    if stacks:
        s = stacks[0]
        setup_parts.append(f"{s['team1']}-{s['team2']} is the highest total at {s['vegas_total']:.0f}.")
    if "spread" in pool.columns:
        _spread_col = pd.to_numeric(pool["spread"], errors="coerce").fillna(0)
        _blowout_mask = _spread_col.abs() > 8
        if _blowout_mask.any():
            _bo_rows = pool.loc[_blowout_mask].copy()
            _bo_rows["_abs_spread"] = _spread_col[_blowout_mask].abs()
            _bo_row = _bo_rows.nlargest(1, "_abs_spread").iloc[0]
            setup_parts.append(
                f"{_bo_row['team']} is a {_bo_row['_abs_spread']:.0f}-point spread \u2014 "
                f"starters hit the bench in the 4th."
            )
    _cascade_name = ""
    if "injury_bump_fp" in pool.columns:
        _bump_col = pd.to_numeric(pool["injury_bump_fp"], errors="coerce").fillna(0)
        _bump_mask = _bump_col > 3.0
        if _bump_mask.any():
            _bumped = pool.loc[_bump_mask].nlargest(1, "injury_bump_fp").iloc[0]
            _cascade_name = _bumped["player_name"]
            _bump_fp = _bump_col.loc[_bumped.name]
            setup_parts.append(
                f"{_cascade_name} is the real minutes beneficiary \u2014 "
                f"+{_bump_fp:.0f} FP from the cascade. "
                f"The public will pile into the obvious name and miss this."
            )
    n_core = len(edge_analysis.get("core_plays", []))
    n_leverage = len(edge_analysis.get("leverage_plays", []))
    if n_core + n_leverage > 0:
        setup_parts.append(
            f"The edge is in the {n_core} core plays and {n_leverage} leverage spots the field isn't checking."
        )
    if setup_parts:
        parts.append(
            '<div class="tb-section-label">THE SETUP</div>'
            f'<div class="tb-setup">{" ".join(setup_parts)}</div>'
        )

    # -- 3. Ricky's Plays ---------------------------------------------------
    snipers = compute_sniper_spots(pool, edge_analysis)
    tiered_plays = _assign_tiered_plays(snipers, pool) if snipers else []

    if tiered_plays:
        parts.append('<div class="tb-section-label">RICKY\'S PLAYS</div>')
        for p, sig, role_key, role_label, reason in tiered_plays:
            pill_cls = "tb-pill"
            if role_key == "dart":
                pill_cls = "tb-pill tb-pill-dart"
            elif role_key == "pivot":
                pill_cls = "tb-pill tb-pill-pivot"
            pill_text = role_label.split(" ", 1)[-1].upper() if " " in role_label else role_label.upper()
            parts.append(
                f'<div class="tb-play-row">'
                f'<span class="{pill_cls}">{pill_text}</span>'
                f'<span class="tb-name">{p["player_name"]}</span> '
                f'<span class="tb-meta">({p["team"]}, ${p["salary"]:,}) \u2014 {reason}</span>'
                f'</div>'
            )

    # -- 4+5. Merged Danger Box: Trap Stack + Fade -------------------------
    _board_names = set()
    for _tier in ("core_plays", "leverage_plays", "value_plays"):
        for _p in edge_analysis.get(_tier, []):
            _board_names.add(_p.get("player_name", ""))
    for pp, *_ in tiered_plays:
        _board_names.add(pp.get("player_name", ""))
    _board_names.discard("")

    _trap_html = ""
    _fade_html = ""

    if "ownership" in pool.columns and "proj" in pool.columns:
        _own_col = pd.to_numeric(pool.get("ownership", pool.get("own_proj", 0)), errors="coerce").fillna(0)
        if _own_col.max() <= 1.0:
            _own_col = _own_col * 100
        _proj_col = pd.to_numeric(pool["proj"], errors="coerce").fillna(0)
        _sal_col = pd.to_numeric(pool.get("salary", 0), errors="coerce").fillna(0)
        _r5_col = pd.to_numeric(pool.get("rolling_fp_5", 0), errors="coerce").fillna(0)
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
            _tn = _trap_row["player_name"]
            _to = float(_trap_df.loc[_trap_row.name, "_own"])
            _ts = int(_sal_col.loc[_trap_row.name])
            _tr5 = float(_r5_col.loc[_trap_row.name])
            _tppk = float(_proj_col.loc[_trap_row.name]) / (_ts / 1000) if _ts > 0 else 0
            _trap_html = (
                f"\u26a0\ufe0f <strong>THE TRAP:</strong> "
                f"{_tn} is going to be over-owned tonight ({_to:.0f}%, ${_ts:,}). "
                f"Last 5 average is {_tr5:.0f} at {_tppk:.1f} pts/$1K. "
                f"The ownership is narrative-driven, not math-driven."
            )
            _board_names.add(_tn)

    bust = generate_bust_call(pool, edge_analysis.get("fade_candidates"), positive_tier_names=_board_names or None)
    fades = []
    if bust:
        _fade_html = (
            f"\U0001f480 <strong>FADE: {bust['name']} (${bust['salary']:,}).</strong> "
            f"{bust['explanation']}"
        )
    else:
        fades = compute_fades(pool, edge_analysis)
        if fades:
            fades = [f for f in fades if f.get("player_name", "") not in _board_names]
        if fades:
            f0 = fades[0]
            _fade_html = (
                f"\U0001f480 <strong>FADE: {f0['player_name']} ({f0['own_pct']:.1f}% owned).</strong> "
                f"{f0.get('reasoning', 'Model says pass.')}"
            )

    if _trap_html or _fade_html:
        danger_inner = ""
        if _trap_html:
            danger_inner += _trap_html
        if _trap_html and _fade_html:
            danger_inner += '<div class="tb-divider"></div>'
        if _fade_html:
            danger_inner += _fade_html
        parts.append(f'<div class="tb-danger-box">{danger_inner}</div>')

    # -- Auto-write fades to bias -------------------------------------------
    try:
        from yak_core.bias import save_bias
        _bias = st.session_state.setdefault("ricky_bias", {})
        _fade_names = []
        if bust:
            _fade_names.append(bust["name"])
        if fades:
            _fade_names.extend(f.get("player_name", "") for f in fades[:1])
        for _fn in _fade_names:
            if _fn:
                _bias.setdefault(_fn, {})["max_exposure"] = 0.0
        if _fade_names:
            save_bias(_bias)
    except Exception:
        pass

    if parts:
        st.markdown('<div class="the-board">' + "".join(parts) + '</div>', unsafe_allow_html=True)
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



# ── Ricky's Manual Adjustments ──────────────────────────────────────────
def _render_bias_panel(pool: pd.DataFrame) -> None:
    """Render the manual projection/exposure adjustment panel."""
    from yak_core.bias import save_bias

    with st.expander("🎛️ Ricky's Manual Adjustments", expanded=False):
        bias = st.session_state.setdefault("ricky_bias", {})

        player_names = sorted(pool["player_name"].dropna().unique().tolist()) if not pool.empty else []
        if not player_names:
            st.caption("Load a pool first.")
            return

        col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1])
        with col1:
            name = st.selectbox("Player", options=player_names, key="bias_player_select")
        with col2:
            existing_adj = bias.get(name, {}).get("proj_adj", 0.0) if name else 0.0
            adj = st.number_input("Proj +/-", value=float(existing_adj), step=0.5, key="bias_proj_adj")
        with col3:
            existing_exp = bias.get(name, {}).get("max_exposure") if name else None
            default_exp = int(existing_exp * 100) if existing_exp is not None else 35
            exp = st.number_input("Max Exp %", value=default_exp, min_value=0, max_value=100, key="bias_max_exp")
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Set", key="bias_set_btn"):
                entry = {}
                if adj != 0.0:
                    entry["proj_adj"] = adj
                if exp < 100:
                    entry["max_exposure"] = exp / 100.0
                if entry:
                    bias[name] = entry
                elif name in bias:
                    del bias[name]
                save_bias(bias)
                st.rerun()

        if bias:
            import pandas as _pd
            rows = []
            for pname, settings in sorted(bias.items()):
                rows.append({
                    "Player": pname,
                    "Proj +/-": settings.get("proj_adj", 0.0),
                    "Max Exp %": f"{settings['max_exposure'] * 100:.0f}%" if "max_exposure" in settings else "—",
                })
            st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if bias and st.button("🗑️ Clear All Bias", key="bias_clear_btn"):
            st.session_state["ricky_bias"] = {}
            save_bias({})
            st.rerun()


def render_edge_tab(sport: str) -> None:
    """Render Ricky's Edge Analysis tab."""
    from app.data_loader import invalidate_published_cache, load_published_data

    # Load Ricky's bias overrides into session state (persisted to disk)
    from yak_core.bias import load_bias, save_bias
    if "ricky_bias" not in st.session_state:
        st.session_state["ricky_bias"] = load_bias()

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

    # Manual bias adjustments panel
    _render_bias_panel(pool)

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
