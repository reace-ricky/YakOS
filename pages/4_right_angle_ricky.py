"""Right Angle Ricky -- the public-facing page for friends.

Two tabs:

  Tab 1 – Ricky's Edge Analysis
    Dashboard-style slate overview: core plays, leverage plays,
    value plays, fades, then top GPP / Cash / Showdown lineup
    from Build & Publish.  No prop plays.

  Tab 2 – Optimizer
    FantasyPros-style optimizer with st.data_editor player pool
    table, inline Lock / Exclude checkboxes, contest-type-first
    flow, build settings, summary metrics, paged lineup cards,
    DK CSV export.

State read:  SlateState, RickyEdgeState, LineupSetState, SimState
State written: None (fully read-only — friend lineups stored in session_state only)
"""

from __future__ import annotations

import hashlib as _hl
import io
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import (  # noqa: E402
    get_slate_state,
    get_edge_state,
    get_lineup_state,
    get_sim_state,
)
from yak_core.components import render_premium_lineup_card, render_lineup_cards_paged  # noqa: E402
from yak_core.edge_metrics import (  # noqa: E402
    compute_ricky_confidence_for_contest,
    get_confidence_color,
)
from yak_core.config import (  # noqa: E402
    CONTEST_PRESETS, UI_CONTEST_LABELS, UI_CONTEST_MAP,
    PGA_UI_CONTEST_LABELS, PGA_UI_CONTEST_MAP,
)
from yak_core.ricky_signals import compute_ricky_signals, generate_slate_overview  # noqa: E402
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
    to_dk_upload_format,
    to_dk_showdown_upload_format,
)
from yak_core.calibration import apply_archetype, DFS_ARCHETYPES  # noqa: E402
from yak_core.edge import compute_edge_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Contest display helpers
# ---------------------------------------------------------------------------

_CONTEST_ORDER = [UI_CONTEST_MAP[k] for k in UI_CONTEST_LABELS]
_LABEL_SHORT = {v: k for k, v in UI_CONTEST_MAP.items()}

_CONTEST_TO_BUILD_MODE = {
    "GPP Main": "ceiling", "GPP Early": "ceiling", "GPP Late": "ceiling",
    "Cash Main": "floor", "Showdown": "ceiling", "PGA GPP": "ceiling",
}
_BUILD_MODE_PROJ_COL = {"floor": "floor", "median": "proj", "ceiling": "proj"}

# NBA positions for filter tabs
_NBA_POS_FILTERS = ["All", "PG", "SG", "SF", "PF", "C"]

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
    seed = st.session_state.get("_ricky_seed", "default")
    idx = int(_hl.md5(str(seed).encode()).hexdigest(), 16) % len(_RICKY_LINES)
    return _RICKY_LINES[idx]


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


def _confidence_pills(edge) -> None:
    """Render compact confidence pills across contests that have data."""
    available = [c for c in _CONTEST_ORDER if c in edge.edge_analysis_by_contest]
    if not available:
        return

    cols = st.columns(len(available))
    for col, contest_label in zip(cols, available, strict=True):
        payload = edge.edge_analysis_by_contest[contest_label]
        score = compute_ricky_confidence_for_contest(payload)
        color = get_confidence_color(score)
        short = _LABEL_SHORT.get(contest_label, contest_label)
        with col:
            if color == "green":
                st.success(f"**{short}** \u2014 {score:.0f}/100")
            elif color == "yellow":
                st.warning(f"**{short}** \u2014 {score:.0f}/100")
            else:
                st.markdown(
                    f"<div style='padding:0.75rem 1rem;border-radius:0.5rem;"
                    f"background:#3a1a1a;border:1px solid #6b3a3a;"
                    f"color:#c27a7a;font-size:0.9rem;'>"
                    f"<strong>{short}</strong> \u2014 {score:.0f}/100</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Tab 1 helpers — best lineup from Build & Publish
# ---------------------------------------------------------------------------

def _get_best_lineup(lu_state, sim_state, contest_label: str) -> tuple:
    """Return (lineup_rows_df, sim_metrics_dict, boom_bust_dict) for the #1 lineup."""
    # Check published sets first, then fall back to built lineups
    pub = lu_state.published_sets.get(contest_label)
    if pub is not None:
        pub_df = pub.get("lineups_df", pd.DataFrame())
        boom_bust_df = pub.get("boom_bust_df")
    else:
        # Fall back to built (unpublished) lineups
        pub_df = lu_state.lineups.get(contest_label, pd.DataFrame())
        boom_bust_df = lu_state.get_boom_bust(contest_label) if hasattr(lu_state, "get_boom_bust") else None

    if pub_df is None or pub_df.empty:
        return None, None, None

    pipeline_df = (
        sim_state.pipeline_output.get(contest_label)
        or sim_state.pipeline_output.get("GPP_20")
    )

    # Pick the best lineup — prefer highest boom_score
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
# TAB 1 — RICKY'S EDGE ANALYSIS (Dashboard-style)
# ===========================================================================

# ---------------------------------------------------------------------------
# HTML card renderer for the 4-box dashboard layout
# ---------------------------------------------------------------------------

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
    """Render a dashboard-style play card as HTML inside a Streamlit container.

    stat_format:
      'proj_salary' -> "48.9 pts | $9,400"
      'value'       -> "$4,400 | 5.95 pts/$1K"
      'proj_val'    -> "27.6 pts | 4.68 val"
    """
    rows_html = ""
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
            f"border-bottom:1px solid rgba(255,255,255,0.1);'>"
            f"<div>"
            f"<span style='font-weight:600;font-size:13px;'>{name}</span>"
            f"<span style='display:inline-block;padding:2px 6px;border-radius:4px;"
            f"font-size:9px;font-weight:bold;margin-left:6px;"
            f"background:{color};color:#fff;'>{badge_label}</span>"
            f"</div>"
            f"<div style='text-align:right;font-size:12px;'>{stat_str}</div>"
            f"</div>"
        )

    html = (
        f"<div style='background:rgba(255,255,255,0.07);border-radius:10px;"
        f"padding:12px;border-left:3px solid {color};'>"
        f"<h3 style='font-size:14px;margin-bottom:10px;color:{color};'>"
        f"{title}</h3>"
        f"{rows_html}"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_tab_analysis(slate, edge, lu_state, sim_state) -> None:
    """Tab 1: Dashboard-style — 4 play-type boxes (2x2) then
    top GPP/Cash/Showdown lineups as rows."""

    has_pool = slate.player_pool is not None and not slate.player_pool.empty
    if not has_pool:
        st.info("No player pool available. Load a slate in The Lab first.")
        return

    pool = slate.player_pool.copy()
    _analysis = get_lab_analysis()
    if not _analysis["pool"].empty:
        pool = _analysis["pool"]

    # ── Compute signals ───────────────────────────────────────────────
    signals_df = compute_ricky_signals(pool)
    contest_type = slate.contest_name or "GPP"
    overview = generate_slate_overview(pool, signals_df, contest_type=contest_type)

    # Slate overview bullets + recommendation
    for bullet in overview["bullets"]:
        st.markdown(f"- {bullet}")
    if overview["recommendation"]:
        st.info(overview["recommendation"])

    st.divider()

    # ── Classify players into 4 buckets ───────────────────────────────
    sdf = signals_df.copy()
    _sal = pd.to_numeric(sdf.get("salary", 0), errors="coerce").fillna(0)
    _proj = pd.to_numeric(sdf.get("proj", 0), errors="coerce").fillna(0)
    _own_col = "ownership" if "ownership" in sdf.columns else "own_pct"
    _own = normalise_ownership(
        pd.to_numeric(sdf.get(_own_col, 0), errors="coerce").fillna(0)
    )
    _edge = pd.to_numeric(sdf.get("edge_composite", 0), errors="coerce").fillna(0)
    _val = np.where(_sal > 0, _proj / (_sal / 1000), 0)
    sdf["_sal"] = _sal
    sdf["_proj"] = _proj
    sdf["_own"] = _own
    sdf["_edge"] = _edge
    sdf["_val"] = _val

    # Core (Chalk): top projected players, $7K+ salary
    core = sdf[sdf["_sal"] >= 7000].nlargest(5, "_proj")
    _used = set(core["player_name"].tolist())

    # Leverage (GPP Gold): best edge, low ownership (<15%), not in core
    _lev_pool = sdf[(sdf["_own"] < 15) & (~sdf["player_name"].isin(_used))]
    leverage = _lev_pool.nlargest(5, "_edge")
    _used.update(leverage["player_name"].tolist())

    # Value (Salary Savers): best pts/$1K, under $6.5K, not already used
    _val_pool = sdf[(sdf["_sal"] < 6500) & (sdf["_sal"] > 0) & (~sdf["player_name"].isin(_used))]
    value = _val_pool.nlargest(5, "_val")
    _used.update(value["player_name"].tolist())

    # Fades: high ownership (>15%), low edge, not already used
    _fade_pool = sdf[(sdf["_own"] >= 15) & (~sdf["player_name"].isin(_used))]
    fades = _fade_pool.nsmallest(5, "_edge")

    # ── 4-box layout (2x2) ────────────────────────────────────────────
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

    # ── Top lineups from Build & Publish (GPP / Cash / Showdown) ──────
    _lineup_data = []
    for contest_label in _CONTEST_ORDER:
        lu_rows, sim_metrics, bb_row = _get_best_lineup(lu_state, sim_state, contest_label)
        if lu_rows is not None and not lu_rows.empty:
            short = _LABEL_SHORT.get(contest_label, contest_label)
            _lineup_data.append((contest_label, short, lu_rows, sim_metrics, bb_row))

    if _lineup_data:
        st.divider()
        for contest_label, short, lu_rows, sim_metrics, bb_row in _lineup_data:
            st.markdown(f"**Top {short} Lineup**")
            render_premium_lineup_card(
                lineup_rows=lu_rows,
                sim_metrics=sim_metrics,
                lineup_label=f"#1 {short}",
                salary_cap=slate.salary_cap,
                boom_bust_row=bb_row,
                compact=True,
            )


# ---------------------------------------------------------------------------
# Tab 1 (PGA) — PGA Edge Analysis powered by breakout signals
# ---------------------------------------------------------------------------

def _render_tab_analysis_pga(slate, lu_state, sim_state) -> None:
    """Tab 1 for PGA: breakout signals + slate overview from DataGolf data."""
    has_pool = slate.player_pool is not None and not slate.player_pool.empty
    if not has_pool:
        st.info("No PGA slate loaded yet. Load a PGA pool in The Lab first.")
        return

    pool = slate.player_pool.copy()
    _analysis = get_lab_analysis()
    if not _analysis["pool"].empty:
        pool = _analysis["pool"]

    # Compute PGA breakout signals
    signals_df = compute_pga_breakout_signals(pool)
    event_name = pool.attrs.get("event_name", "") if hasattr(pool, "attrs") else ""
    overview = generate_pga_slate_overview(pool, signals_df, event_name=event_name)

    # Slate overview bullets
    for bullet in overview["bullets"]:
        st.markdown(f"- {bullet}")
    if overview["recommendation"]:
        st.info(overview["recommendation"])

    st.divider()

    # Top edges table
    top_edges = signals_df[signals_df["pga_edge_composite"] > 0].head(12)
    if not top_edges.empty:
        st.markdown("**Top PGA Edges**")
        display_df = pd.DataFrame()
        display_df["Player"] = top_edges["player_name"].values
        display_df["Salary"] = top_edges["salary"].values if "salary" in top_edges.columns else 0
        display_df["Proj"] = top_edges["proj"].values if "proj" in top_edges.columns else 0
        if "sg_total" in top_edges.columns:
            display_df["SG Total"] = top_edges["sg_total"].values
        if "course_fit" in top_edges.columns:
            display_df["Course Fit"] = top_edges["course_fit"].values
        own_col = "ownership" if "ownership" in top_edges.columns else "own_pct"
        if own_col in top_edges.columns:
            display_df["Own%"] = normalise_ownership(pd.Series(top_edges[own_col].values)).values
        display_df["Edge"] = (top_edges["pga_edge_composite"].values * 100).round(0).astype(int)
        display_df["Signals"] = top_edges["pga_signal_badges"].values

        _fmt = standard_player_format(display_df)
        if "SG Total" in display_df.columns:
            _fmt["SG Total"] = "{:+.2f}"
        if "Course Fit" in display_df.columns:
            _fmt["Course Fit"] = "{:+.2f}"
        st.dataframe(
            display_df.style.format(_fmt, na_rep=""),
            use_container_width=True, hide_index=True,
        )

    # Top PGA GPP lineup from Build & Publish
    lu_rows, sim_metrics, bb_row = _get_best_lineup(lu_state, sim_state, "PGA GPP")
    if lu_rows is not None and not lu_rows.empty:
        st.divider()
        st.markdown("**Top PGA GPP Lineup**")
        render_premium_lineup_card(
            lineup_rows=lu_rows,
            sim_metrics=sim_metrics,
            lineup_label="#1 PGA GPP",
            salary_cap=slate.salary_cap,
            boom_bust_row=bb_row,
            compact=True,
        )


# ===========================================================================
# TAB 2 — OPTIMIZER (FantasyPros-style)
# ===========================================================================

def _extract_games(pool: pd.DataFrame) -> list[str]:
    """Return sorted list of 'TEAM vs OPP' matchup strings."""
    opp_col = "opp" if "opp" in pool.columns else (
        "opponent" if "opponent" in pool.columns else None
    )
    if opp_col and "team" in pool.columns:
        teams = pool["team"].str.strip().str.upper().fillna("")
        opps = pool[opp_col].str.strip().str.upper().fillna("")
        pairs = {
            " vs ".join(sorted([t, o]))
            for t, o in zip(teams, opps)
            if t and o
        }
        return sorted(pairs)
    return []


def _filter_pool_by_games(pool: pd.DataFrame, selected_games: list[str]) -> pd.DataFrame:
    """Filter pool to only players in the selected games."""
    if not selected_games:
        return pool
    opp_col = "opp" if "opp" in pool.columns else (
        "opponent" if "opponent" in pool.columns else None
    )
    if not opp_col:
        return pool
    teams = pool["team"].str.strip().str.upper().fillna("")
    opps = pool[opp_col].str.strip().str.upper().fillna("")
    keys = pd.Series(
        [" vs ".join(sorted([t, o])) if t and o else t for t, o in zip(teams, opps)],
        index=pool.index,
    )
    return pool[keys.isin(selected_games)].reset_index(drop=True)


def _build_pool_display(
    pool: pd.DataFrame,
    edge_df: Optional[pd.DataFrame],
    sport: str,
    contest_label: str,
    pos_filter: str = "All",
) -> pd.DataFrame:
    """Build the display DataFrame for the player pool table."""
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

    # Compute Value column
    _sal = pd.to_numeric(df.get("salary", 0), errors="coerce").fillna(0)
    _proj = pd.to_numeric(df.get("proj", 0), errors="coerce").fillna(0)
    df["value"] = np.where(_sal > 0, _proj / (_sal / 1000.0), 0.0)

    # Normalise ownership display
    _own = pd.to_numeric(df.get("ownership", df.get("own_pct", 0)), errors="coerce").fillna(0)
    df["own_display"] = _own

    # Add Lock/Exclude boolean columns
    _prev_lock = st.session_state.get("_rar_locked_players", set())
    _prev_excl = st.session_state.get("_rar_excluded_players", set())
    df["Lock"] = df["player_name"].isin(_prev_lock)
    df["Exclude"] = df["player_name"].isin(_prev_excl)

    # Position filter (NBA only)
    if sport == "NBA" and pos_filter != "All" and "pos" in df.columns:
        df = df[df["pos"].str.contains(pos_filter, case=False, na=False)].copy()

    # Select and order columns based on sport + contest
    is_cash = "cash" in contest_label.lower()
    is_pga = sport == "PGA"

    if is_pga:
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
    elif is_cash:
        cols = ["Lock", "Exclude", "player_name", "team", "opp", "pos",
                "salary", "proj", "floor", "own_display", "value"]
        rename = {
            "player_name": "Player", "team": "Team", "opp": "Opp",
            "pos": "Pos", "salary": "Salary", "proj": "Proj",
            "floor": "Floor", "own_display": "Own%", "value": "Value",
        }
    else:
        # GPP / Showdown
        cols = ["Lock", "Exclude", "player_name", "team", "opp", "pos",
                "salary", "proj", "own_display", "edge_score", "value",
                "edge_label"]
        rename = {
            "player_name": "Player", "team": "Team", "opp": "Opp",
            "pos": "Pos", "salary": "Salary", "proj": "Proj",
            "own_display": "Own%", "edge_score": "Edge",
            "value": "Value", "edge_label": "Label",
        }

    cols = [c for c in cols if c in df.columns]
    display = df[cols].copy()
    display = display.rename(columns={k: v for k, v in rename.items() if k in display.columns})

    # Sort
    if "Edge" in display.columns and not is_cash:
        display = display.sort_values("Edge", ascending=False, na_position="last")
    elif "Proj" in display.columns:
        display = display.sort_values("Proj", ascending=False, na_position="last")

    return display.reset_index(drop=True)


def _render_tab_optimizer(slate) -> None:
    """Tab 2: FantasyPros-style optimizer with data_editor pool table."""

    if not slate.is_ready() or slate.player_pool is None or slate.player_pool.empty:
        st.info(
            "No slate loaded yet. Ricky needs to load a slate in The Lab "
            "before you can build lineups."
        )
        return

    pool = slate.player_pool.copy()
    sport = slate.sport or "NBA"

    # ── 1. Contest type selector ──────────────────────────────────────
    if sport == "PGA":
        _labels = PGA_UI_CONTEST_LABELS
        _label_map = PGA_UI_CONTEST_MAP
    else:
        _labels = UI_CONTEST_LABELS
        _label_map = UI_CONTEST_MAP

    _ui_contest = st.radio(
        "Contest Type", _labels, horizontal=True, key="_rar_opt_contest",
    )
    contest_label = _label_map[_ui_contest]
    preset = CONTEST_PRESETS.get(contest_label, {})

    # ── 2. Build settings row ─────────────────────────────────────────
    col_lu, col_exp, col_sal = st.columns(3)
    with col_lu:
        num_lineups = st.number_input(
            "# Lineups", min_value=1, max_value=150,
            value=int(preset.get("default_lineups", preset.get("num_lineups", 20))),
            key="_rar_opt_num_lineups",
        )
    with col_exp:
        max_exp = st.slider(
            "Max Exposure", min_value=0.10, max_value=1.0, step=0.05,
            value=float(preset.get("default_max_exposure", preset.get("max_exposure", 0.50))),
            key="_rar_opt_max_exp",
        )
    with col_sal:
        min_salary = st.number_input(
            "Min Salary Used", min_value=40000, max_value=50000, step=500,
            value=int(preset.get("min_salary", preset.get("min_salary_used", 46000))),
            key="_rar_opt_min_salary",
        )

    # Slate status bar
    _n_games = len(_extract_games(pool))
    _n_players = len(pool)
    _game_str = f"{_n_games} games" if _n_games else ""
    _cap_str = f"${slate.salary_cap:,} cap"
    _date_str = slate.slate_date or ""
    _parts = [p for p in [_date_str, _game_str, f"{_n_players} players", _cap_str] if p]
    st.caption(f"\U0001f4cb {' \u00b7 '.join(_parts)}")

    # ── Game filter (NBA only, expander) ──────────────────────────────
    all_games = _extract_games(pool)
    build_games: list[str] = []
    if all_games and sport == "NBA":
        _lab_games = slate.selected_games if hasattr(slate, "selected_games") else []
        _default_all = not _lab_games
        with st.expander(f"Games ({len(all_games)})", expanded=False):
            for _g in all_games:
                _default_on = _g in _lab_games if _lab_games else _default_all
                if st.checkbox(_g, value=_default_on, key=f"_rar_gf_{_g}"):
                    build_games.append(_g)
        if build_games and len(build_games) < len(all_games):
            pool = _filter_pool_by_games(pool, build_games)

    # ── Compute edge metrics ──────────────────────────────────────────
    _edge_df = None
    try:
        _edge_df = compute_edge_metrics(
            pool, calibration_state=slate.calibration_state, sport=sport,
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

    # ── 3. Player pool table ──────────────────────────────────────────
    st.divider()

    # Position filter tabs (NBA only)
    pos_filter = "All"
    if sport == "NBA":
        pos_filter = st.radio(
            "Position", _NBA_POS_FILTERS, horizontal=True,
            key="_rar_opt_pos_filter", label_visibility="collapsed",
        )

    display_df = _build_pool_display(pool, _edge_df, sport, contest_label, pos_filter)

    # Column configs for st.data_editor
    col_config: Dict[str, Any] = {
        "Lock": st.column_config.CheckboxColumn("\U0001f512", width="small", default=False),
        "Exclude": st.column_config.CheckboxColumn("\u2715", width="small", default=False),
    }
    if "Salary" in display_df.columns:
        col_config["Salary"] = st.column_config.NumberColumn("Salary", format="$%d")
    if "Proj" in display_df.columns:
        col_config["Proj"] = st.column_config.NumberColumn("Proj", format="%.1f")
    if "Floor" in display_df.columns:
        col_config["Floor"] = st.column_config.NumberColumn("Floor", format="%.1f")
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
    if "Label" in display_df.columns:
        col_config["Label"] = st.column_config.TextColumn("Label", width="medium")
    if "Player" in display_df.columns:
        col_config["Player"] = st.column_config.TextColumn("Player", width="medium")

    # Render editable table
    edited_df = st.data_editor(
        display_df,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        key="_rar_pool_editor",
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
    st.session_state["_rar_locked_players"] = _locked
    st.session_state["_rar_excluded_players"] = _excluded

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

    # ── 4. Build button ───────────────────────────────────────────────
    st.divider()

    archetype = preset.get("archetype", "Balanced")
    _merged_exclude = list(_excluded | set(_auto_excl))

    is_showdown = contest_label == "Showdown" or slate.is_showdown

    _contest_type_map = {
        "GPP Main": "gpp", "GPP Early": "gpp", "GPP Late": "gpp",
        "Cash Main": "cash", "Showdown": "showdown", "PGA GPP": "gpp",
    }

    build_mode = _CONTEST_TO_BUILD_MODE.get(contest_label, "ceiling")
    proj_col = _BUILD_MODE_PROJ_COL.get(build_mode, "proj")
    if "cash" in contest_label.lower() and "floor" in pool.columns:
        proj_col = "floor"

    if st.button("\u26a1 Build Lineups", type="primary", key="_rar_opt_build", use_container_width=True):
        _pool = pool.copy()
        if "player_id" not in _pool.columns:
            if "player_name" in _pool.columns:
                _pool["player_id"] = _pool["player_name"]
            elif "dk_player_id" in _pool.columns:
                _pool["player_id"] = _pool["dk_player_id"]

        cfg = {
            "NUM_LINEUPS": int(num_lineups),
            "SALARY_CAP": slate.salary_cap,
            "MAX_EXPOSURE": float(max_exp),
            "MIN_SALARY_USED": int(min_salary),
            "LOCK": list(_locked),
            "EXCLUDE": _merged_exclude,
            "PROJ_COL": proj_col,
            "CONTEST_TYPE": _contest_type_map.get(contest_label, "gpp"),
        }

        # Inject per-player exposure caps from edge overrides
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
                st.session_state["_rar_friend_lineups"] = lineups_df
                st.session_state["_rar_friend_expo"] = expo_df
                st.session_state["_rar_friend_contest"] = contest_label
                st.session_state["_rar_friend_is_showdown"] = is_showdown
                n_built = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else 1
                st.success(f"Built {n_built} lineups for **{_ui_contest}**.")
            else:
                st.error("Optimizer returned no lineups. Try different settings.")
        except Exception as exc:
            st.error(f"Optimizer error: {exc}")

    # ── 5. Lineup results ─────────────────────────────────────────────
    friend_lineups = st.session_state.get("_rar_friend_lineups")
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
    expo_df = st.session_state.get("_rar_friend_expo")
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
        nav_key="rar_friend",
    )

    # DK CSV Export
    st.divider()
    _is_sd = st.session_state.get("_rar_friend_is_showdown", False)
    try:
        if _is_sd:
            dk_csv = to_dk_showdown_upload_format(friend_lineups)
        else:
            dk_csv = to_dk_upload_format(friend_lineups)

        if dk_csv is not None and not dk_csv.empty:
            buf = io.StringIO()
            dk_csv.to_csv(buf, index=False)
            st.download_button(
                label="\U0001f4e5 Download DK CSV",
                data=buf.getvalue(),
                file_name="ricky_lineups.csv",
                mime="text/csv",
                key="_rar_dk_export",
            )
    except Exception as exc:
        st.caption(f"CSV export unavailable: {exc}")


# ===========================================================================
# MAIN PAGE
# ===========================================================================

def main() -> None:
    st.title("\U0001f4d0 Right Angle Ricky")
    st.caption(_ricky_quote())

    # ── Sport toggle at top ───────────────────────────────────────────
    slate = get_slate_state()
    _detected_sport = slate.sport if slate.sport else "NBA"
    sport = st.radio(
        "Sport",
        ["NBA", "PGA"],
        index=0 if _detected_sport == "NBA" else 1,
        horizontal=True,
        key="_rar_sport_toggle",
    )
    _is_pga = sport == "PGA"

    edge = get_edge_state()
    lu_state = get_lineup_state()
    sim_state = get_sim_state()

    _status_strip(slate)

    # Empty state
    has_edge = bool(edge.edge_analysis_by_contest)
    has_lineups = any(
        (lu_state.lineups.get(c) is not None and not lu_state.lineups[c].empty)
        or c in lu_state.published_sets
        for c in _CONTEST_ORDER
    ) or "PGA GPP" in lu_state.lineups
    has_pool = slate.player_pool is not None and not slate.player_pool.empty

    if not has_edge and not has_lineups and not has_pool:
        if _is_pga:
            st.info(
                "No PGA pool loaded yet. Select **PGA** in **The Lab** and "
                "load a pool from DataGolf."
            )
        else:
            st.info(
                "Ricky's got nothing to show yet. Load a slate in **The Lab**, "
                "approve in **Ricky's Edge Analysis**, and publish from **Build & Publish**."
            )
        return

    # Confidence strip (NBA only)
    if not _is_pga:
        _confidence_pills(edge)
        if has_edge:
            st.divider()

    # Two tabs
    tab_analysis, tab_optimizer = st.tabs(
        ["\U0001f3af Ricky's Edge Analysis", "\U0001f527 Optimizer"]
    )

    with tab_analysis:
        if _is_pga:
            _render_tab_analysis_pga(slate, lu_state, sim_state)
        else:
            _render_tab_analysis(slate, edge, lu_state, sim_state)

    with tab_optimizer:
        _render_tab_optimizer(slate)


main()
