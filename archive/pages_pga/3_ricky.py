"""Right Angle Ricky — PGA standalone page.

Two tabs:

  Tab 1 – Ricky's Edge Analysis
    PGA course/weather/waves/history info cards, 4-box dashboard
    (Core/Leverage/Value/Fade), top PGA GPP / Cash / Showdown lineup
    from Build & Publish.

  Tab 2 – Optimizer
    FantasyPros-style optimizer with st.data_editor player pool
    table, inline Lock / Exclude checkboxes, contest-type-first
    flow, build settings, summary metrics, paged lineup cards,
    DK CSV export.

State read:  SlateState, RickyEdgeState, LineupSetState, SimState  (PGA-prefixed)
State written: None (fully read-only — friend lineups stored in session_state only)
"""

from __future__ import annotations

import hashlib as _hl
import io
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.pga_state import (  # noqa: E402
    get_slate_state,
    get_edge_state,
    get_lineup_state,
    set_lineup_state,
    get_sim_state,
    set_slate_state,
    set_edge_state,
    _pga_clear_published,
)
from yak_core.components import render_premium_lineup_card, render_lineup_cards_paged  # noqa: E402
from yak_core.config import (  # noqa: E402
    CONTEST_PRESETS,
    PGA_UI_CONTEST_LABELS, PGA_UI_CONTEST_MAP,
)
from yak_core.right_angle import (  # noqa: E402
    compute_pga_breakout_signals,
    generate_pga_slate_overview,
    apply_edge_adjustments,
    compute_breakout_candidates,
)
from yak_core.context import get_lab_analysis  # noqa: E402
from yak_core.display_format import normalise_ownership, standard_player_format  # noqa: E402
from yak_core.lineups import (  # noqa: E402
    build_multiple_lineups_with_exposure,
    build_showdown_lineups,
    to_dk_pga_upload_format,
    to_dk_showdown_upload_format,
)
from yak_core.calibration import apply_archetype, DFS_ARCHETYPES  # noqa: E402
from yak_core.edge import compute_edge_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Contest display helpers (PGA only)
# ---------------------------------------------------------------------------

_PGA_CONTEST_ORDER = [PGA_UI_CONTEST_MAP[k] for k in PGA_UI_CONTEST_LABELS]
_LABEL_SHORT = {v: k for k, v in PGA_UI_CONTEST_MAP.items()}

_CONTEST_TO_BUILD_MODE = {
    "PGA GPP": "ceiling", "PGA Cash": "floor", "PGA Showdown": "ceiling",
}
_BUILD_MODE_PROJ_COL = {"floor": "floor", "median": "proj", "ceiling": "proj"}

# ---------------------------------------------------------------------------
# Ricky flavor lines
# ---------------------------------------------------------------------------

_RICKY_LINES = [
    "Hoodie on. Cold brew in hand. Let's find some edges.",
    "Running quiet from the coffee shop. The angles don't lie.",
    "Process over picks. Edges over hype.",
    "Perpendicular to nonsense since day one.",
    "Low-key analytics. High-key results.",
    "The spreadsheet doesn't care about your feelings.",
    "Calm process, sharp lines. That's the Ricky way.",
    "Half-awake, fully locked in.",
]


def _ricky_quote() -> str:
    seed = st.session_state.get("_pga_rar_ricky_seed", "default")
    idx = int(_hl.md5(str(seed).encode()).hexdigest(), 16) % len(_RICKY_LINES)
    return _RICKY_LINES[idx]


# ---------------------------------------------------------------------------
# Admin pin gate (PGA persistence)
# ---------------------------------------------------------------------------

_ADMIN_PIN = st.secrets.get("ADMIN_PIN", "2018")


def _render_admin_clear(slate, lu_state, sim_state) -> None:
    """Pin-protected clear button using PGA persistence."""
    with st.expander("\u2699\ufe0f Admin", expanded=False):
        pin = st.text_input(
            "Enter PIN to clear", type="password",
            key="_pga_rar_pin", max_chars=4,
        )
        if pin and pin == _ADMIN_PIN:
            if st.button(
                "\U0001f5d1\ufe0f Clear All Published Data",
                key="_pga_rar_clear_pub",
                type="secondary",
            ):
                _pga_clear_published()
                lu_state.published_sets.clear()
                set_lineup_state(lu_state)
                slate.published = False
                slate.player_pool = None
                set_slate_state(slate)
                edge_obj = get_edge_state()
                edge_obj.ricky_edge_check = False
                edge_obj.edge_analysis_by_contest.clear()
                set_edge_state(edge_obj)
                st.rerun()
        elif pin:
            st.error("Wrong PIN.")


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------

def _status_strip(slate) -> None:
    """Compact one-line slate header."""
    parts = []
    if slate.sport:
        parts.append(f"**{slate.sport}**")
    if slate.slate_date:
        parts.append(slate.slate_date)
    if slate.site:
        parts.append(slate.site)
    if parts:
        st.caption(" \u00b7 ".join(parts))


# ---------------------------------------------------------------------------
# Tab 1 helpers — best lineup from Build & Publish
# ---------------------------------------------------------------------------

def _get_best_lineup(lu_state, sim_state, contest_label: str) -> tuple:
    """Return (lineup_rows_df, sim_metrics_dict, boom_bust_dict) for the #1 lineup."""
    pub = lu_state.published_sets.get(contest_label)
    if pub is not None:
        pub_df = pub.get("lineups_df", pd.DataFrame())
        boom_bust_df = pub.get("boom_bust_df")
    else:
        pub_df = lu_state.lineups.get(contest_label, pd.DataFrame())
        boom_bust_df = lu_state.get_boom_bust(contest_label) if hasattr(lu_state, "get_boom_bust") else None

    if pub_df is None or pub_df.empty:
        return None, None, None

    pipeline_df = (
        sim_state.pipeline_output.get(contest_label)
        or sim_state.pipeline_output.get("GPP_20")
    )

    best_idx = 0
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        if "boom_score" in boom_bust_df.columns:
            best_idx = int(boom_bust_df.sort_values("boom_score", ascending=False).iloc[0]["lineup_index"])
        else:
            best_idx = int(boom_bust_df.iloc[0]["lineup_index"])
    elif "lineup_index" in pub_df.columns:
        best_idx = int(pub_df["lineup_index"].min())

    lu_rows = pub_df[pub_df["lineup_index"] == best_idx] if "lineup_index" in pub_df.columns else pub_df

    sim_metrics = {}
    if pipeline_df is not None and not pipeline_df.empty and "lineup_index" in pipeline_df.columns:
        match = pipeline_df[pipeline_df["lineup_index"] == best_idx]
        if not match.empty:
            sim_metrics = match.iloc[0].to_dict()

    bb_row = None
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        bb_match = boom_bust_df[boom_bust_df["lineup_index"] == best_idx]
        if not bb_match.empty:
            bb_row = bb_match.iloc[0].to_dict()

    return lu_rows, sim_metrics or None, bb_row


# ===========================================================================
# TAB 1 — PGA EDGE ANALYSIS (Dashboard-style)
# ===========================================================================

_CARD_COLORS = {
    "core": "#f7931e",       # orange
    "leverage": "#a855f7",   # purple
    "value": "#4ade80",      # green
    "fade": "#ef4444",       # red
}


def _render_play_card(
    title: str,
    players: pd.DataFrame,
    color: str,
    badge_label: str,
    stat_format: str = "proj_salary",
) -> None:
    """Render a dashboard-style play card as HTML."""
    _badge_text = {"#4ade80": "#000", "#f7931e": "#000"}
    badge_fg = _badge_text.get(color, "#fff")

    rows_html = ""
    if players.empty:
        rows_html = (
            "<div style='padding:12px 0;font-size:12px;color:rgba(255,255,255,0.4);'>"
            "No players match this category for the current slate.</div>"
        )
    else:
        for _, r in players.iterrows():
            name = r.get("player_name", "")
            sal = float(r.get("salary", 0) or 0)
            proj = float(r.get("proj", 0) or 0)
            val = proj / (sal / 1000) if sal > 0 else 0

            if stat_format == "value":
                stat_str = f"${sal:,.0f} | {val:.2f} pts/$1K"
            elif stat_format == "proj_val":
                stat_str = f"{proj:.1f} pts | {val:.2f} val"
            else:
                stat_str = f"{proj:.1f} pts | ${sal:,.0f}"

            rows_html += (
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;padding:6px 0;"
                f"border-bottom:1px solid rgba(255,255,255,0.08);'>"
                f"<div>"
                f"<span style='font-weight:600;font-size:13px;'>{name}</span>"
                f"<span style='display:inline-block;padding:2px 6px;border-radius:4px;"
                f"font-size:9px;font-weight:bold;margin-left:6px;"
                f"background:{color};color:{badge_fg};'>{badge_label}</span>"
                f"</div>"
                f"<div style='text-align:right;font-size:12px;color:rgba(255,255,255,0.85);'>"
                f"{stat_str}</div>"
                f"</div>"
            )

    html = (
        f"<div style='background:rgba(255,255,255,0.06);border-radius:10px;"
        f"padding:12px;border-left:3px solid {color};margin-bottom:4px;'>"
        f"<h3 style='font-size:14px;margin-bottom:10px;color:{color};'>"
        f"{title}</h3>"
        f"{rows_html}"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PGA Info Cards — Course & Weather environment strip
# ---------------------------------------------------------------------------

def _render_pga_info_cards(pool: pd.DataFrame) -> None:
    """Compact course-environment strip above the edge analysis."""
    attrs = pool.attrs if hasattr(pool, "attrs") else {}

    cards: list[tuple[str, str, str]] = []  # (title, body_html, color)

    # 1. Course + Conditions
    course_name = attrs.get("course_name", "")
    course_city = attrs.get("course_city", "")
    fit_cols = {
        "driving_dist_adj": "Distance", "driving_acc_adj": "Accuracy",
        "approach_fit": "Approach", "short_game_fit": "Short Game",
    }
    demands = []
    for col, label in fit_cols.items():
        if col in pool.columns:
            v = pool[col].dropna().var()
            if v and v > 0:
                demands.append((label, v))
    demands.sort(key=lambda x: x[1], reverse=True)
    if course_name:
        loc = f" — {course_city}" if course_city else ""
        body = f"<b>{course_name}</b>{loc}"
        if demands:
            tags = " &middot; ".join(d[0] for d in demands[:3])
            body += f"<br><span style='color:rgba(255,255,255,0.5);font-size:11px;'>Key demands: {tags}</span>"
        cards.append(("\u26f3 Course", body, "#3b82f6"))

    # 2. Weather
    weather = attrs.get("weather", {})
    cur = weather.get("current", {})
    daily = weather.get("daily", [])
    if cur:
        temp = cur.get("temp_f", "")
        wind = cur.get("wind_mph", "")
        wdir = cur.get("wind_dir", "")
        cond = cur.get("conditions", "")
        body = f"{temp}\u00b0F &middot; {wind} mph {wdir}"
        if cond:
            body += f" &middot; {cond}"
        if daily:
            rows = []
            for d in daily[:4]:
                dt = d.get("date", "")[-5:]
                hi = d.get("high_f", "")
                dw = d.get("wind_mph", "")
                rain = d.get("precip_chance", 0)
                rain_flag = " \U0001f327\ufe0f" if int(rain or 0) >= 40 else ""
                rows.append(f"{dt}: {hi}\u00b0 / {dw}mph{rain_flag}")
            body += "<br><span style='color:rgba(255,255,255,0.45);font-size:11px;'>" + " &nbsp;|&nbsp; ".join(rows) + "</span>"
        si = weather.get("scoring_impact", "")
        if si:
            body += f"<br><span style='font-size:11px;color:#f59e0b;'>{si}</span>"
        cards.append(("\u2600\ufe0f Weather", body, "#f59e0b"))

    # 3. Wave split
    if "early_late_wave" in pool.columns:
        wave_str = pool["early_late_wave"].astype(str).str.strip().str.lower()
        n_early = wave_str.str.contains("early", na=False).sum()
        n_late = wave_str.str.contains("late", na=False).sum()
        if n_early > 0 or n_late > 0:
            body = f"Early: {n_early} &middot; Late: {n_late}"
            body += "<br><span style='color:rgba(255,255,255,0.45);font-size:11px;'>AM wave typically 0.15-0.30 strokes easier</span>"
            early_df = pool[wave_str.str.contains("early", na=False)]
            late_df = pool[wave_str.str.contains("late", na=False)]
            parts = []
            if not early_df.empty and "proj" in early_df.columns:
                names = ", ".join(early_df.nlargest(3, "proj")["player_name"].tolist())
                parts.append(f"<span style='color:#06b6d4;'>AM:</span> {names}")
            if not late_df.empty and "proj" in late_df.columns:
                names = ", ".join(late_df.nlargest(3, "proj")["player_name"].tolist())
                parts.append(f"<span style='color:#06b6d4;'>PM:</span> {names}")
            if parts:
                body += "<br><span style='font-size:11px;'>" + " &nbsp;|&nbsp; ".join(parts) + "</span>"
            cards.append(("\U0001f30a Waves", body, "#06b6d4"))

    # 4. Course history
    if "course_history" in pool.columns:
        ch = pool[["player_name", "course_history"]].dropna(subset=["course_history"])
        ch = ch[ch["course_history"].abs() >= 0.2]
        if not ch.empty:
            top = ch.nlargest(4, "course_history")
            items = []
            for _, r in top.iterrows():
                adj = float(r["course_history"])
                sign = "+" if adj >= 0 else ""
                items.append(f"{r['player_name']} <span style='color:#8b5cf6;'>{sign}{adj:.1f}</span>")
            cards.append(("\U0001f4ca History", " &middot; ".join(items), "#8b5cf6"))

    if not cards:
        return

    cols = st.columns(len(cards))
    for col, (title, body, color) in zip(cols, cards):
        with col:
            html = (
                f"<div style='background:rgba(255,255,255,0.05);border-radius:8px;"
                f"padding:10px 12px;border-left:3px solid {color};min-height:80px;'>"
                f"<div style='font-size:11px;font-weight:700;color:{color};"
                f"margin-bottom:6px;letter-spacing:0.5px;'>{title}</div>"
                f"<div style='font-size:12px;color:rgba(255,255,255,0.85);line-height:1.5;'>"
                f"{body}</div></div>"
            )
            st.markdown(html, unsafe_allow_html=True)
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 1 — PGA Edge Analysis
# ---------------------------------------------------------------------------

def _render_tab_analysis(slate, lu_state, sim_state) -> None:
    """PGA Edge Analysis: 2x2 dashboard boxes then top PGA lineups."""
    has_pool = slate.player_pool is not None and not slate.player_pool.empty
    if not has_pool:
        st.info("No PGA slate loaded yet. Load a PGA pool in The Lab first.")
        return

    pool = slate.player_pool.copy()
    _analysis = get_lab_analysis()
    if not _analysis["pool"].empty:
        pool = _analysis["pool"]

    # Course / Weather / Wave / History info cards
    _render_pga_info_cards(pool)

    # Compute PGA breakout signals
    signals_df = compute_pga_breakout_signals(pool)
    event_name = pool.attrs.get("event_name", "") if hasattr(pool, "attrs") else ""
    overview = generate_pga_slate_overview(pool, signals_df, event_name=event_name)

    # Slate overview bullets as styled callouts
    _bullet_colors = [
        ("\U0001f4aa", "#f7931e"),   # SG Total - orange
        ("\U0001f3af", "#4ade80"),   # Course Fits - green
        ("\U0001f4b0", "#3b82f6"),   # Value - blue
        ("\U0001f50d", "#a855f7"),   # Leverage - purple
    ]
    for i, bullet in enumerate(overview["bullets"]):
        color = _bullet_colors[i][1] if i < len(_bullet_colors) else "rgba(255,255,255,0.3)"
        _html_bullet = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', bullet)
        st.markdown(
            f"<div style='padding:6px 10px;margin-bottom:4px;border-radius:6px;"
            f"border-left:3px solid {color};background:rgba(255,255,255,0.04);"
            f"font-size:13px;'>{_html_bullet}</div>",
            unsafe_allow_html=True,
        )
    if overview["recommendation"]:
        rec = overview["recommendation"]
        if "thin" in rec.lower() or "balanced" in rec.lower():
            _rec_bg = "rgba(239,68,68,0.12)"
            _rec_border = "#ef4444"
        elif "strong" in rec.lower() or "5+" in rec:
            _rec_bg = "rgba(74,222,128,0.12)"
            _rec_border = "#4ade80"
        else:
            _rec_bg = "rgba(59,130,246,0.12)"
            _rec_border = "#3b82f6"
        st.markdown(
            f"<div style='padding:10px 14px;margin-top:6px;border-radius:8px;"
            f"background:{_rec_bg};border:1px solid {_rec_border};"
            f"font-size:13px;font-style:italic;color:rgba(255,255,255,0.9);'>"
            f"{rec}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Classify PGA players into 4 buckets
    sdf = signals_df.copy()

    def _safe_col(frame, name, default=0):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index)

    _sal = _safe_col(sdf, "salary")
    _proj = _safe_col(sdf, "proj")
    _own_col = "ownership" if "ownership" in sdf.columns else "own_pct"
    _own = normalise_ownership(_safe_col(sdf, _own_col))
    _edge_col = "pga_edge_composite" if "pga_edge_composite" in sdf.columns else "edge_composite"
    _edge = _safe_col(sdf, _edge_col)
    _val = np.where(_sal > 0, _proj / (_sal / 1000), 0)
    sdf["_sal"] = _sal
    sdf["_proj"] = _proj
    sdf["_own"] = _own
    sdf["_edge"] = _edge
    sdf["_val"] = _val

    # Core (Chalk): top projected, $8K+
    core = sdf[sdf["_sal"] >= 8000].nlargest(5, "_proj")
    _used = set(core["player_name"].tolist())

    # Leverage (GPP Gold): best edge, low ownership (<15%), not in core
    _lev_pool = sdf[(sdf["_own"] < 15) & (~sdf["player_name"].isin(_used))]
    leverage = _lev_pool.nlargest(5, "_edge")
    _used.update(leverage["player_name"].tolist())

    # Value (Salary Savers): best pts/$1K, under $7.5K
    _val_pool = sdf[(sdf["_sal"] < 7500) & (sdf["_sal"] > 0) & (~sdf["player_name"].isin(_used))]
    value = _val_pool.nlargest(5, "_val")
    _used.update(value["player_name"].tolist())

    # Fades
    _fade_pool = sdf[~sdf["player_name"].isin(_used)].copy()
    _fade_high_own = _fade_pool[_fade_pool["_own"] >= 10]
    if len(_fade_high_own) >= 3:
        fades = _fade_high_own.nsmallest(5, "_edge")
    else:
        _fade_sal = _fade_pool[_fade_pool["_sal"] >= 7000]
        fades = _fade_sal.nsmallest(5, "_edge") if not _fade_sal.empty else _fade_pool.nsmallest(5, "_edge")

    # 4-box layout (2x2)
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        _render_play_card(
            "CORE PLAYS (Chalk)", core, _CARD_COLORS["core"],
            "CHALK", stat_format="proj_salary",
        )
    with row1_c2:
        _render_play_card(
            "LEVERAGE PLAYS (GPP Gold)", leverage, _CARD_COLORS["leverage"],
            "UNDEROWNED", stat_format="proj_val",
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        _render_play_card(
            "VALUE PLAYS (Salary Savers)", value, _CARD_COLORS["value"],
            "VALUE", stat_format="value",
        )
    with row2_c2:
        _render_play_card(
            "FADE CANDIDATES", fades, _CARD_COLORS["fade"],
            "FADE", stat_format="proj_salary",
        )

    # Top PGA lineups from Build & Publish
    _pga_lineup_data = []
    for _pga_cl in _PGA_CONTEST_ORDER:
        lu_rows, sim_metrics, bb_row = _get_best_lineup(lu_state, sim_state, _pga_cl)
        if lu_rows is not None and not lu_rows.empty:
            _short = _LABEL_SHORT.get(_pga_cl, _pga_cl)
            _pga_lineup_data.append((_pga_cl, _short, lu_rows, sim_metrics, bb_row))

    if _pga_lineup_data:
        st.divider()
        for i in range(0, len(_pga_lineup_data), 2):
            chunk = _pga_lineup_data[i:i + 2]
            cols = st.columns(len(chunk))
            for col, (_pga_cl, _short, lu_rows, sim_metrics, bb_row) in zip(cols, chunk):
                with col:
                    render_premium_lineup_card(
                        lineup_rows=lu_rows,
                        sim_metrics=sim_metrics,
                        lineup_label=f"Top {_short}",
                        salary_cap=slate.salary_cap,
                        boom_bust_row=bb_row,
                        compact=True,
                    )

        _render_admin_clear(slate, lu_state, sim_state)


# ===========================================================================
# TAB 2 — OPTIMIZER (FantasyPros-style)
# ===========================================================================

def _build_pool_display(
    pool: pd.DataFrame,
    edge_df: Optional[pd.DataFrame],
    contest_label: str,
) -> pd.DataFrame:
    """Build the display DataFrame for the PGA player pool table."""
    df = pool.copy()

    # Merge edge metrics if available
    if edge_df is not None and not edge_df.empty:
        _edge_cols = ["player_name"]
        for c in ["edge_score", "edge_label", "smash_prob", "bust_prob",
                   "leverage", "own_pct"]:
            if c in edge_df.columns and c not in df.columns:
                _edge_cols.append(c)
        if len(_edge_cols) > 1:
            _sub = edge_df[_edge_cols].drop_duplicates(subset=["player_name"])
            df = df.merge(_sub, on="player_name", how="left")

    # Value column
    def _safe_col_opt(frame, name, default=0):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index)

    _sal = _safe_col_opt(df, "salary")
    _proj = _safe_col_opt(df, "proj")
    df["value"] = np.where(_sal > 0, _proj / (_sal / 1000.0), 0.0)

    # Normalise ownership
    _own_col_opt = "ownership" if "ownership" in df.columns else "own_pct"
    _own = _safe_col_opt(df, _own_col_opt)
    df["own_display"] = _own

    # Lock / Exclude booleans
    _prev_lock = st.session_state.get("_pga_rar_locked_players", set())
    _prev_excl = st.session_state.get("_pga_rar_excluded_players", set())
    df["Lock"] = df["player_name"].isin(_prev_lock)
    df["Exclude"] = df["player_name"].isin(_prev_excl)

    # PGA column selection
    cols = ["Lock", "Exclude", "player_name", "salary", "proj", "own_display",
            "edge_score", "value"]
    rename = {
        "player_name": "Player", "salary": "Salary",
        "proj": "Proj", "own_display": "Own%",
        "edge_score": "Edge", "value": "Value",
    }
    for c, label in [("sg_total", "SG Total"), ("course_fit", "Course Fit")]:
        if c in df.columns:
            cols.insert(-2, c)
            rename[c] = label

    cols = [c for c in cols if c in df.columns]
    display = df[cols].copy()
    display = display.rename(columns={k: v for k, v in rename.items() if k in display.columns})

    # Sort by Edge (GPP/Showdown) or Proj (Cash)
    is_cash = "cash" in contest_label.lower()
    if "Edge" in display.columns and not is_cash:
        display = display.sort_values("Edge", ascending=False, na_position="last")
    elif "Proj" in display.columns:
        display = display.sort_values("Proj", ascending=False, na_position="last")

    return display.reset_index(drop=True)


def _render_tab_optimizer(slate) -> None:
    """Tab 2: FantasyPros-style PGA optimizer with data_editor pool table."""

    if not slate.is_ready() or slate.player_pool is None or slate.player_pool.empty:
        st.info(
            "No PGA slate loaded yet. Ricky needs a slate in The Lab "
            "before you can build lineups."
        )
        return

    pool = slate.player_pool.copy()

    # 1. Contest type selector (PGA only)
    _ui_contest = st.radio(
        "Contest Type", PGA_UI_CONTEST_LABELS, horizontal=True,
        key="_pga_rar_opt_contest",
    )
    contest_label = PGA_UI_CONTEST_MAP[_ui_contest]
    preset = CONTEST_PRESETS.get(contest_label, {})

    # 2. Build settings row
    col_lu, col_exp, col_sal = st.columns(3)
    with col_lu:
        num_lineups = st.number_input(
            "# Lineups", min_value=1, max_value=150,
            value=int(preset.get("default_lineups", preset.get("num_lineups", 20))),
            key="_pga_rar_opt_num_lineups",
        )
    with col_exp:
        max_exp = st.slider(
            "Max Exposure", min_value=0.10, max_value=1.0, step=0.05,
            value=float(preset.get("default_max_exposure", preset.get("max_exposure", 0.50))),
            key="_pga_rar_opt_max_exp",
        )
    with col_sal:
        min_salary = st.number_input(
            "Min Salary Used", min_value=40000, max_value=50000, step=500,
            value=int(preset.get("min_salary", preset.get("min_salary_used", 46000))),
            key="_pga_rar_opt_min_salary",
        )

    # Slate status bar
    _n_players = len(pool)
    _cap_str = f"${slate.salary_cap:,} cap"
    _date_str = slate.slate_date or ""
    _parts = [p for p in [_date_str, f"{_n_players} players", _cap_str] if p]
    st.caption(f"\U0001f4cb {' \u00b7 '.join(_parts)}")

    # Compute edge metrics
    _edge_df = None
    try:
        _edge_df = compute_edge_metrics(
            pool, calibration_state=slate.calibration_state, sport="PGA",
        )
    except Exception:
        pass

    _breakout_df = None
    try:
        _breakout_df = compute_breakout_candidates(pool, top_n=15)
    except Exception:
        pass

    try:
        pool, _edge_overrides = apply_edge_adjustments(pool, edge_df=_edge_df, breakout_df=_breakout_df)
    except Exception:
        _edge_overrides = {}

    # 3. Player pool table
    st.divider()

    display_df = _build_pool_display(pool, _edge_df, contest_label)

    # Column configs for st.data_editor
    col_config: Dict[str, Any] = {
        "Lock": st.column_config.CheckboxColumn("\U0001f512", width="small", default=False),
        "Exclude": st.column_config.CheckboxColumn("\u2715", width="small", default=False),
    }
    if "Salary" in display_df.columns:
        col_config["Salary"] = st.column_config.NumberColumn("Salary", format="$%d")
    if "Proj" in display_df.columns:
        col_config["Proj"] = st.column_config.NumberColumn("Proj", format="%.1f")
    if "Own%" in display_df.columns:
        col_config["Own%"] = st.column_config.NumberColumn("Own%", format="%.1f%%")
    if "Edge" in display_df.columns:
        col_config["Edge"] = st.column_config.NumberColumn("Edge", format="%.2f")
    if "Value" in display_df.columns:
        col_config["Value"] = st.column_config.NumberColumn("Value", format="%.1fx")
    if "SG Total" in display_df.columns:
        col_config["SG Total"] = st.column_config.NumberColumn("SG Total", format="%.2f")
    if "Course Fit" in display_df.columns:
        col_config["Course Fit"] = st.column_config.NumberColumn("Fit", format="%.2f")
    if "Player" in display_df.columns:
        col_config["Player"] = st.column_config.TextColumn("Player", width="medium")

    edited_df = st.data_editor(
        display_df,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        key="_pga_rar_pool_editor",
        height=min(600, 40 + len(display_df) * 35),
    )

    # Extract lock / exclude selections
    _locked = set()
    _excluded = set()
    _name_col = "Player" if "Player" in edited_df.columns else "player_name"
    if _name_col in edited_df.columns:
        if "Lock" in edited_df.columns:
            _locked = set(edited_df.loc[edited_df["Lock"] == True, _name_col].tolist())
        if "Exclude" in edited_df.columns:
            _excluded = set(edited_df.loc[edited_df["Exclude"] == True, _name_col].tolist())
    st.session_state["_pga_rar_locked_players"] = _locked
    st.session_state["_pga_rar_excluded_players"] = _excluded

    # Lock/exclude summary
    _lock_excl_parts = []
    if _locked:
        _lock_excl_parts.append(f"\U0001f512 {len(_locked)} locked")
    if _excluded:
        _lock_excl_parts.append(f"\u2715 {len(_excluded)} excluded")
    _auto_excl = _edge_overrides.get("auto_exclude", []) if isinstance(_edge_overrides, dict) else []
    if _auto_excl:
        _lock_excl_parts.append(f"\u26a0 {len(_auto_excl)} auto-excluded (bust risk)")
    if _lock_excl_parts:
        st.caption(" \u00b7 ".join(_lock_excl_parts))

    # 4. Build button
    st.divider()

    archetype = preset.get("archetype", "Balanced")
    _merged_exclude = list(_excluded | set(_auto_excl))

    is_showdown = contest_label == "PGA Showdown" or slate.is_showdown

    _contest_type_map = {
        "PGA GPP": "gpp", "PGA Cash": "cash", "PGA Showdown": "gpp",
    }

    build_mode = _CONTEST_TO_BUILD_MODE.get(contest_label, "ceiling")
    proj_col = _BUILD_MODE_PROJ_COL.get(build_mode, "proj")
    if "cash" in contest_label.lower() and "floor" in pool.columns:
        proj_col = "floor"

    if st.button("\u26a1 Build Lineups", type="primary", key="_pga_rar_opt_build", use_container_width=True):
        _pool = pool.copy()
        if "player_id" not in _pool.columns:
            if "player_name" in _pool.columns:
                _pool["player_id"] = _pool["player_name"]
            elif "dk_player_id" in _pool.columns:
                _pool["player_id"] = _pool["dk_player_id"]

        _pga_preset = CONTEST_PRESETS.get(contest_label, {})
        cfg = {
            "NUM_LINEUPS": int(num_lineups),
            "SALARY_CAP": slate.salary_cap,
            "MAX_EXPOSURE": float(max_exp),
            "MIN_SALARY_USED": int(min_salary),
            "LOCK": list(_locked),
            "EXCLUDE": _merged_exclude,
            "PROJ_COL": proj_col,
            "CONTEST_TYPE": _contest_type_map.get(contest_label, "gpp"),
            "POS_SLOTS": _pga_preset.get("pos_slots", ["G"] * 6),
            "LINEUP_SIZE": _pga_preset.get("lineup_size", 6),
            "POS_CAPS": _pga_preset.get("pos_caps", {}),
            "GPP_MAX_PUNT_PLAYERS": _pga_preset.get("max_punt_players", 1),
            "GPP_MIN_MID_PLAYERS": _pga_preset.get("min_mid_salary_players", 2),
            "GPP_OWN_CAP": _pga_preset.get("own_cap", 5.0),
            "GPP_MIN_LOW_OWN_PLAYERS": _pga_preset.get("min_low_own_players", 1),
            "GPP_LOW_OWN_THRESHOLD": _pga_preset.get("low_own_threshold", 0.40),
            "GPP_FORCE_GAME_STACK": _pga_preset.get("force_game_stack", False),
        }

        # Per-player exposure caps from edge overrides
        if isinstance(_edge_overrides, dict):
            if _edge_overrides.get("max_exposure_players"):
                cfg["PLAYER_MAX_EXPOSURE"] = _edge_overrides["max_exposure_players"]
            if _edge_overrides.get("tier_player_names"):
                cfg["TIER_CONSTRAINTS"] = {
                    "tier_player_names": _edge_overrides["tier_player_names"],
                    "tier_min_players": _edge_overrides.get("tier_min_players", {}),
                    "tier_max_players": _edge_overrides.get("tier_max_players", {}),
                }

        try:
            with st.spinner(f"Building {num_lineups} {_ui_contest} lineups..."):
                if is_showdown:
                    lineups_df, expo_df = build_showdown_lineups(_pool, cfg)
                else:
                    opt_pool = apply_archetype(_pool.copy(), archetype)
                    lineups_df, expo_df = build_multiple_lineups_with_exposure(opt_pool, cfg)

            if lineups_df is not None and not lineups_df.empty:
                st.session_state["_pga_rar_friend_lineups"] = lineups_df
                st.session_state["_pga_rar_friend_expo"] = expo_df
                st.session_state["_pga_rar_friend_contest"] = contest_label
                st.session_state["_pga_rar_friend_is_showdown"] = is_showdown
                n_built = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else 1
                st.success(f"Built {n_built} lineups for **{_ui_contest}**.")
            else:
                st.error("Optimizer returned no lineups. Try different settings.")
        except Exception as exc:
            st.error(f"Optimizer error: {exc}")

    # 5. Lineup results
    friend_lineups = st.session_state.get("_pga_rar_friend_lineups")
    if friend_lineups is None or friend_lineups.empty:
        return

    st.divider()

    n_lu = len(friend_lineups["lineup_index"].unique()) if "lineup_index" in friend_lineups.columns else 0

    # Summary metrics
    if "proj" in friend_lineups.columns or "total_proj" in friend_lineups.columns:
        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        _sc1.metric("Lineups", n_lu)

        _total_sal_col = "total_salary" if "total_salary" in friend_lineups.columns else None
        if _total_sal_col:
            _lu_sals = friend_lineups.groupby("lineup_index")[_total_sal_col].first()
            _sc2.metric("Avg Salary", f"${_lu_sals.mean():,.0f}")
        elif "salary" in friend_lineups.columns:
            _lu_sals = friend_lineups.groupby("lineup_index")["salary"].sum()
            _sc2.metric("Avg Salary", f"${_lu_sals.mean():,.0f}")

        _total_proj_col = "total_proj" if "total_proj" in friend_lineups.columns else None
        if _total_proj_col:
            _lu_projs = friend_lineups.groupby("lineup_index")[_total_proj_col].first()
            _sc3.metric("Avg Proj", f"{_lu_projs.mean():.1f}")
            _sc4.metric("Top Lineup", f"{_lu_projs.max():.1f}")
        elif "proj" in friend_lineups.columns:
            _lu_projs = friend_lineups.groupby("lineup_index")["proj"].sum()
            _sc3.metric("Avg Proj", f"{_lu_projs.mean():.1f}")
            _sc4.metric("Top Lineup", f"{_lu_projs.max():.1f}")

    # Exposure table
    expo_df = st.session_state.get("_pga_rar_friend_expo")
    if expo_df is not None and not expo_df.empty:
        with st.expander("Player Exposures", expanded=False):
            _expo_fmt = standard_player_format(expo_df)
            st.dataframe(
                expo_df.style.format(_expo_fmt, na_rep=""),
                use_container_width=True, hide_index=True,
            )

    # Lineup cards (paged)
    render_lineup_cards_paged(
        lineups_df=friend_lineups,
        sim_results_df=None,
        salary_cap=slate.salary_cap,
        nav_key="_pga_rar_friend",
    )

    # DK CSV Export
    st.divider()
    _is_sd = st.session_state.get("_pga_rar_friend_is_showdown", False)
    try:
        if _is_sd:
            dk_csv = to_dk_showdown_upload_format(friend_lineups)
        else:
            dk_csv = to_dk_pga_upload_format(friend_lineups)

        if dk_csv is not None and not dk_csv.empty:
            buf = io.StringIO()
            dk_csv.to_csv(buf, index=False)
            st.download_button(
                label="\U0001f4e5 Download DK CSV",
                data=buf.getvalue(),
                file_name="ricky_pga_lineups.csv",
                mime="text/csv",
                key="_pga_rar_dk_export",
            )
    except Exception as exc:
        st.caption(f"CSV export unavailable: {exc}")


# ===========================================================================
# MAIN PAGE
# ===========================================================================

def main() -> None:
    st.title("\U0001f4d0 Right Angle Ricky")
    st.caption(_ricky_quote())

    slate = get_slate_state()
    edge = get_edge_state()
    lu_state = get_lineup_state()
    sim_state = get_sim_state()

    _status_strip(slate)

    # Empty state
    has_edge = bool(edge.edge_analysis_by_contest)
    has_lineups = any(
        (lu_state.lineups.get(c) is not None and not lu_state.lineups[c].empty)
        or c in lu_state.published_sets
        for c in _PGA_CONTEST_ORDER
    )
    has_pool = slate.player_pool is not None and not slate.player_pool.empty

    if not has_edge and not has_lineups and not has_pool:
        st.info(
            "No PGA pool loaded yet. Load a PGA pool in **The Lab** "
            "from DataGolf to get started."
        )
        return

    # Two tabs
    tab_analysis, tab_optimizer = st.tabs(
        ["\U0001f3af Ricky's Edge Analysis", "\U0001f527 Optimizer"]
    )

    with tab_analysis:
        _render_tab_analysis(slate, lu_state, sim_state)

    with tab_optimizer:
        _render_tab_optimizer(slate)


main()
