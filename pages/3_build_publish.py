"""Lineup Optimizer – FantasyPros-style contest-driven optimizer.

Layout:
  1. Sport toggle (NBA / PGA)
  2. Slate status bar
  3. Contest type selector (drives everything downstream)
  4. Build settings row (# lineups, max exposure, min salary)
  5. Player pool table with inline Lock / Exclude checkboxes
  6. Build button → Lineup results + Export
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import (  # noqa: E402
    get_slate_state,
    get_edge_state,
    get_sim_state,
    get_lineup_state, set_lineup_state,
)
from yak_core.lineups import (  # noqa: E402
    build_multiple_lineups_with_exposure,
    to_dk_upload_format,
    build_showdown_lineups,
    to_dk_showdown_upload_format,
)
from yak_core.calibration import apply_archetype, DFS_ARCHETYPES  # noqa: E402
from yak_core.config import (  # noqa: E402
    CONTEST_PRESETS,
    UI_CONTEST_LABELS, UI_CONTEST_MAP,
    PGA_UI_CONTEST_LABELS, PGA_UI_CONTEST_MAP,
)
from yak_core.components import render_lineup_cards_paged  # noqa: E402
from yak_core.publishing import publish_edge_and_lineups  # noqa: E402
from yak_core.edge import compute_edge_metrics  # noqa: E402
from yak_core.lineup_scoring import compute_lineup_boom_bust, GRADE_COLORS as _GRADE_COLORS_HEX  # noqa: E402
from yak_core.right_angle import apply_edge_adjustments, compute_breakout_candidates  # noqa: E402
from yak_core.display_format import normalise_ownership, standard_player_format, standard_lineup_format  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────

_BUILD_MODE_COLS = {
    "floor": "floor",
    "median": "proj",
    "ceiling": "proj",
}
_CONTEST_TO_BUILD_MODE = {
    "GPP Main": "ceiling",
    "GPP Early": "ceiling",
    "GPP Late": "ceiling",
    "Cash Main": "floor",
    "Showdown": "ceiling",
    "PGA GPP": "ceiling",
}

# NBA positions for filter tabs
_NBA_POS_FILTERS = ["All", "PG", "SG", "SF", "PF", "C"]


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


def _apply_sim_learnings(pool: pd.DataFrame, sim) -> pd.DataFrame:
    """Apply non-destructive Sim Learnings boosts to effective_proj column."""
    pool = pool.copy()
    if "proj" not in pool.columns:
        return pool
    pool["effective_proj"] = pool["proj"].copy()
    for pname, learning in sim.sim_learnings.items():
        mask = pool.get("player_name", pd.Series(dtype=str)) == pname
        if mask.any():
            boost = float(learning.get("boost", 0))
            pool.loc[mask, "effective_proj"] = pool.loc[mask, "effective_proj"] * (1 + boost)
    return pool


# ── Pool table builder ─────────────────────────────────────────────

def _build_pool_display(
    pool: pd.DataFrame,
    edge_df: Optional[pd.DataFrame],
    sport: str,
    contest_label: str,
    pos_filter: str = "All",
) -> pd.DataFrame:
    """Build the display DataFrame for the player pool table.

    Returns a DataFrame with display-friendly columns and Lock/Excl booleans.
    """
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
    _prev_lock = st.session_state.get("_opt_locked_players", set())
    _prev_excl = st.session_state.get("_opt_excluded_players", set())
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
        # Add SG / course fit if available
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

    # Keep only columns that exist
    cols = [c for c in cols if c in df.columns]
    display = df[cols].copy()
    display = display.rename(columns={k: v for k, v in rename.items() if k in display.columns})

    # Sort by Edge desc (GPP), or Proj desc (Cash), or Proj desc (PGA)
    if "Edge" in display.columns and not is_cash:
        display = display.sort_values("Edge", ascending=False, na_position="last")
    elif "Proj" in display.columns:
        display = display.sort_values("Proj", ascending=False, na_position="last")

    return display.reset_index(drop=True)


# ── Lineup builder (same engine, cleaned up) ──────────────────────

def _build_lineups(
    pool: pd.DataFrame,
    num_lineups: int,
    max_exposure: float,
    min_salary: int,
    archetype: str,
    slate,
    lock_names: list,
    exclude_names: list,
    contest_label: str,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build lineups using the appropriate engine (Classic / Showdown)."""
    pool = pool.copy()

    if "effective_proj" in pool.columns:
        pool["raw_proj"] = pool["proj"].copy()
        pool["proj"] = pool["effective_proj"]

    if "player_id" not in pool.columns:
        if "player_name" in pool.columns:
            pool["player_id"] = pool["player_name"]
        elif "dk_player_id" in pool.columns:
            pool["player_id"] = pool["dk_player_id"]

    _contest_type_map = {
        "GPP Main": "gpp", "GPP Early": "gpp", "GPP Late": "gpp",
        "Cash Main": "cash", "Showdown": "showdown", "PGA GPP": "gpp",
    }

    proj_col = "proj"
    if "cash" in contest_label.lower() and "floor" in pool.columns:
        proj_col = "floor"

    cfg: Dict[str, Any] = {
        "NUM_LINEUPS": num_lineups,
        "SALARY_CAP": slate.salary_cap,
        "MAX_EXPOSURE": max_exposure,
        "MIN_SALARY_USED": min_salary,
        "LOCK": lock_names or [],
        "EXCLUDE": exclude_names or [],
        "PROJ_COL": proj_col,
        "CONTEST_TYPE": _contest_type_map.get(contest_label, "gpp"),
    }

    # Inject per-player exposure caps from edge overrides
    _eo = st.session_state.get("_edge_overrides", {})
    if _eo.get("max_exposure_players"):
        cfg["PLAYER_MAX_EXPOSURE"] = _eo["max_exposure_players"]
    if _eo.get("tier_player_names"):
        cfg["TIER_CONSTRAINTS"] = {
            "tier_player_names": _eo["tier_player_names"],
            "tier_min_players": _eo.get("tier_min_players", {}),
            "tier_max_players": _eo.get("tier_max_players", {}),
        }

    try:
        if slate.is_showdown:
            lineups_df, expo_df = build_showdown_lineups(pool, cfg)
        else:
            opt_pool = apply_archetype(pool.copy(), archetype)
            lineups_df, expo_df = build_multiple_lineups_with_exposure(opt_pool, cfg)
        return lineups_df, expo_df
    except Exception as exc:
        st.error(f"Optimizer error: {exc}")
        return None, None


# ── Late-swap suggestions ──────────────────────────────────────────

def _late_swap_suggestions(
    pool: pd.DataFrame,
    lineups_df: Optional[pd.DataFrame],
    injury_updates: list,
) -> list[dict]:
    """Generate late-swap candidates using pre-baked GTD rules."""
    suggestions: list[dict] = []
    if lineups_df is None or lineups_df.empty:
        return suggestions
    if not injury_updates:
        return suggestions

    player_pool_map: dict = {}
    if not pool.empty and "player_name" in pool.columns:
        for _, row in pool.iterrows():
            player_pool_map[str(row.get("player_name", ""))] = row.to_dict()

    for update in injury_updates:
        pname = str(update.get("player_name", ""))
        status = str(update.get("status", "")).upper()
        if not pname:
            continue
        in_lineups = False
        if "player_name" in lineups_df.columns:
            in_lineups = pname in lineups_df["player_name"].values
        if not in_lineups:
            continue

        if status in ("OUT", "IR", "O"):
            player_row = player_pool_map.get(pname, {})
            pos = player_row.get("pos", "")
            current_salary = float(player_row.get("salary", 0) or 0)
            in_lineup_players = set(lineups_df["player_name"].tolist()) if "player_name" in lineups_df.columns else set()
            candidates = [
                row for _, row in pool.iterrows()
                if (
                    str(row.get("pos", "")) == pos
                    and str(row.get("player_name", "")) != pname
                    and str(row.get("player_name", "")) not in in_lineup_players
                    and abs(float(row.get("salary", 0) or 0) - current_salary) <= 1500
                    and str(row.get("status", "")).upper() not in ("OUT", "IR", "O")
                )
            ]
            if candidates:
                best = max(candidates, key=lambda r: float(r.get("proj", 0) or 0))
                suggestions.append({
                    "action": "PIVOT", "out_player": pname,
                    "in_player": best.get("player_name", ""),
                    "pos": pos,
                    "salary_delta": int(float(best.get("salary", 0) or 0) - current_salary),
                    "reason": f"{pname} is {status}",
                })
            else:
                suggestions.append({
                    "action": "PIVOT", "out_player": pname,
                    "in_player": "(no replacement found)", "pos": pos,
                    "salary_delta": 0, "reason": f"{pname} is {status}",
                })
        elif status in ("GTD", "Q", "QUESTIONABLE", "DOUBTFUL", "LIMITED"):
            suggestions.append({
                "action": "REDUCE_EXPOSURE", "out_player": pname, "in_player": "",
                "pos": player_pool_map.get(pname, {}).get("pos", ""),
                "salary_delta": 0,
                "reason": f"{pname} is {status} — reduce exposure",
            })
    return suggestions


# =====================================================================
# MAIN PAGE
# =====================================================================

def main() -> None:
    st.set_page_config(layout="wide") if not hasattr(st, "_is_running_with_streamlit") else None

    # ── Load state ─────────────────────────────────────────────────
    slate = get_slate_state()
    sim = get_sim_state()
    lu_state = get_lineup_state()

    # ── 1. Header ──────────────────────────────────────────────────
    st.title("🎯 Lineup Optimizer")

    # ── 2. Sport toggle ────────────────────────────────────────────
    sport = getattr(slate, "sport", "NBA") or "NBA"
    _pool_has_pga = False
    if slate.player_pool is not None and not slate.player_pool.empty:
        _positions = set(slate.player_pool.get("pos", pd.Series()).dropna().unique())
        _pool_has_pga = _positions == {"G"} or sport == "PGA"

    if _pool_has_pga:
        sport = st.radio("Sport", ["NBA", "PGA"], index=1 if sport == "PGA" else 0,
                         horizontal=True, key="_opt_sport")
    else:
        sport = "NBA"

    # ── 3. Slate check ─────────────────────────────────────────────
    if not slate.is_ready():
        st.warning("No slate loaded. Go to **The Lab** and load a slate first.")
        st.stop()

    pool: pd.DataFrame = slate.player_pool.copy()
    pool = _apply_sim_learnings(pool, sim)

    # Slate status bar
    _n_games = len(_extract_games(pool))
    _n_players = len(pool)
    _game_str = f"{_n_games} games" if _n_games else ""
    _cap_str = f"${slate.salary_cap:,} cap"
    _date_str = slate.slate_date or ""
    _parts = [p for p in [_date_str, _game_str, f"{_n_players} players", _cap_str] if p]
    st.caption(f"📋 {' · '.join(_parts)}")

    # ── 4. Contest type selector ───────────────────────────────────
    if sport == "PGA":
        _labels = PGA_UI_CONTEST_LABELS
        _label_map = PGA_UI_CONTEST_MAP
    else:
        _labels = UI_CONTEST_LABELS
        _label_map = UI_CONTEST_MAP

    # Inherit from Lab if available
    _REVERSE_UI_MAP = {v: k for k, v in _label_map.items()}
    _lab_ui = _REVERSE_UI_MAP.get(slate.contest_name, _labels[0]) if slate.contest_name else _labels[0]
    _default_idx = _labels.index(_lab_ui) if _lab_ui in _labels else 0

    _ui_contest = st.radio(
        "Contest Type",
        _labels,
        index=_default_idx,
        horizontal=True,
        key="_opt_contest",
    )
    contest_label = _label_map[_ui_contest]
    preset = CONTEST_PRESETS.get(contest_label, {})

    # ── 5. Build settings row ──────────────────────────────────────
    col_lu, col_exp, col_sal = st.columns(3)
    with col_lu:
        num_lineups = st.number_input(
            "# Lineups", min_value=1, max_value=150,
            value=int(preset.get("default_lineups", preset.get("num_lineups", 20))),
            key="_opt_num_lineups",
        )
    with col_exp:
        max_exp = st.slider(
            "Max Exposure", min_value=0.10, max_value=1.0, step=0.05,
            value=float(preset.get("default_max_exposure", preset.get("max_exposure", 0.50))),
            key="_opt_max_exp",
        )
    with col_sal:
        min_salary = st.number_input(
            "Min Salary Used", min_value=40000, max_value=50000, step=500,
            value=int(preset.get("min_salary", preset.get("min_salary_used", 46000))),
            key="_opt_min_salary",
        )

    # ── Game filter (expander) ─────────────────────────────────────
    all_games = _extract_games(pool)
    build_games: list[str] = []
    if all_games and sport == "NBA":
        _lab_games = slate.selected_games if hasattr(slate, "selected_games") else []
        _default_all = not _lab_games
        with st.expander(f"Games ({len(all_games)})", expanded=False):
            for _g in all_games:
                _default_on = _g in _lab_games if _lab_games else _default_all
                if st.checkbox(_g, value=_default_on, key=f"_opt_gf_{_g}"):
                    build_games.append(_g)
        if build_games and len(build_games) < len(all_games):
            pool = _filter_pool_by_games(pool, build_games)

    # ── Compute edge metrics ───────────────────────────────────────
    _edge_df = getattr(slate, "edge_df", None)
    if _edge_df is None or (hasattr(_edge_df, "empty") and _edge_df.empty):
        try:
            _edge_df = compute_edge_metrics(
                pool, calibration_state=slate.calibration_state, sport=sport,
            )
        except Exception:
            _edge_df = None

    # Apply edge adjustments (breakout detection, tier classification)
    _breakout_df = None
    try:
        _breakout_df = compute_breakout_candidates(pool, top_n=15)
    except Exception:
        pass

    pool, _edge_overrides = apply_edge_adjustments(pool, edge_df=_edge_df, breakout_df=_breakout_df)
    st.session_state["_edge_overrides"] = _edge_overrides

    # ── 6. Player pool table ───────────────────────────────────────
    st.divider()

    # Position filter tabs (NBA only)
    pos_filter = "All"
    if sport == "NBA":
        pos_filter = st.radio(
            "Position",
            _NBA_POS_FILTERS,
            horizontal=True,
            key="_opt_pos_filter",
            label_visibility="collapsed",
        )

    display_df = _build_pool_display(pool, _edge_df, sport, contest_label, pos_filter)

    # Column configs for st.data_editor
    col_config: Dict[str, Any] = {
        "Lock": st.column_config.CheckboxColumn("🔒", width="small", default=False),
        "Exclude": st.column_config.CheckboxColumn("✕", width="small", default=False),
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
        key="_opt_pool_editor",
        height=min(600, 40 + len(display_df) * 35),
    )

    # Extract lock / exclude selections from edited table
    _locked = set()
    _excluded = set()
    _name_col = "Player" if "Player" in edited_df.columns else "player_name"
    if _name_col in edited_df.columns:
        if "Lock" in edited_df.columns:
            _locked = set(edited_df.loc[edited_df["Lock"] == True, _name_col].tolist())
        if "Exclude" in edited_df.columns:
            _excluded = set(edited_df.loc[edited_df["Exclude"] == True, _name_col].tolist())
    st.session_state["_opt_locked_players"] = _locked
    st.session_state["_opt_excluded_players"] = _excluded

    # Lock/exclude summary
    _lock_excl_parts = []
    if _locked:
        _lock_excl_parts.append(f"🔒 {len(_locked)} locked")
    if _excluded:
        _lock_excl_parts.append(f"✕ {len(_excluded)} excluded")
    _auto_excl = _edge_overrides.get("auto_exclude", [])
    if _auto_excl:
        _lock_excl_parts.append(f"⚠ {len(_auto_excl)} auto-excluded (bust risk)")
    if _lock_excl_parts:
        st.caption(" · ".join(_lock_excl_parts))

    # ── 7. Build button ────────────────────────────────────────────
    st.divider()

    archetype = preset.get("archetype", "Balanced")
    _merged_exclude = list(_excluded | set(_auto_excl))

    if st.button("⚡ Build Lineups", type="primary", key="_opt_build", use_container_width=True):
        with st.spinner(f"Building {num_lineups} {_ui_contest} lineups..."):
            lineups_df, expo_df = _build_lineups(
                pool,
                num_lineups=int(num_lineups),
                max_exposure=float(max_exp),
                min_salary=int(min_salary),
                archetype=str(archetype),
                slate=slate,
                lock_names=list(_locked),
                exclude_names=_merged_exclude,
                contest_label=contest_label,
            )
            if lineups_df is not None:
                lu_state.set_lineups(
                    contest_label, lineups_df,
                    {
                        "num_lineups": num_lineups,
                        "max_exposure": max_exp,
                        "min_salary": min_salary,
                        "archetype": archetype,
                    },
                )
                if expo_df is not None:
                    lu_state.exposures[contest_label] = expo_df

                # Boom/bust ranking
                player_results = sim.player_results
                if player_results is not None and not player_results.empty:
                    try:
                        bb_rankings = compute_lineup_boom_bust(
                            lineups_df=lineups_df,
                            sim_player_results=player_results,
                            contest_label=contest_label,
                        )
                        lu_state.set_boom_bust(contest_label, bb_rankings)
                    except Exception:
                        pass

                set_lineup_state(lu_state)
                st.success(f"Built {num_lineups} lineups for **{_ui_contest}**.")

    # ── 8. Lineup results ──────────────────────────────────────────
    built_labels = [lbl for lbl, df in lu_state.lineups.items() if df is not None and not df.empty]
    if not built_labels:
        return

    # Show results for the current contest type
    view_label = contest_label if contest_label in built_labels else built_labels[0]
    view_df = lu_state.lineups.get(view_label)

    if view_df is None or view_df.empty:
        return

    st.divider()

    n_lu = len(view_df["lineup_index"].unique()) if "lineup_index" in view_df.columns else 0

    # ── 8a. Summary metrics ────────────────────────────────────────
    _total_proj_col = "total_proj" if "total_proj" in view_df.columns else None
    _total_sal_col = "total_salary" if "total_salary" in view_df.columns else None

    if _total_proj_col or "proj" in view_df.columns:
        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        _sc1.metric("Lineups", n_lu)

        if _total_sal_col:
            _lu_sals = view_df.groupby("lineup_index")[_total_sal_col].first()
            _sc2.metric("Avg Salary", f"${_lu_sals.mean():,.0f}")
        elif "salary" in view_df.columns:
            _lu_sals = view_df.groupby("lineup_index")["salary"].sum()
            _sc2.metric("Avg Salary", f"${_lu_sals.mean():,.0f}")

        if _total_proj_col:
            _lu_projs = view_df.groupby("lineup_index")[_total_proj_col].first()
            _sc3.metric("Avg Proj", f"{_lu_projs.mean():.1f}")
            _sc4.metric("Top Lineup", f"{_lu_projs.max():.1f}")
        elif "proj" in view_df.columns:
            _lu_projs = view_df.groupby("lineup_index")["proj"].sum()
            _sc3.metric("Avg Proj", f"{_lu_projs.mean():.1f}")
            _sc4.metric("Top Lineup", f"{_lu_projs.max():.1f}")

    # ── 8b. Exposure table ─────────────────────────────────────────
    expo_df = lu_state.exposures.get(view_label)
    if expo_df is not None and not expo_df.empty:
        with st.expander("Player Exposures", expanded=False):
            _expo_fmt = standard_player_format(expo_df)
            st.dataframe(
                expo_df.style.format(_expo_fmt, na_rep=""),
                use_container_width=True, hide_index=True,
            )

    # ── 8c. Lineup cards (paged) ───────────────────────────────────
    pipeline_df = sim.pipeline_output.get(contest_label) or sim.pipeline_output.get("GPP_20")
    bb_df = lu_state.get_boom_bust(view_label)

    render_lineup_cards_paged(
        lineups_df=view_df,
        sim_results_df=pipeline_df,
        salary_cap=slate.salary_cap,
        nav_key=f"opt_lu_{view_label}",
        boom_bust_df=bb_df,
    )

    # ── 8d. Boom/Bust Rankings ─────────────────────────────────────
    if bb_df is not None and not bb_df.empty:
        with st.expander("Lineup Rankings (Boom/Bust)", expanded=False):
            _preset_mode = preset.get("tagging_mode", "ceiling")

            def _colour_grade(val: str) -> str:
                color = _GRADE_COLORS_HEX.get(str(val), "")
                return f"background-color:{color};color:#fff;font-weight:700;" if color else ""

            display_bb = bb_df.rename(columns={
                "lineup_index": "Lineup #", "total_proj": "Total Proj",
                "total_ceil": "Total Ceil", "total_floor": "Total Floor",
                "avg_smash_prob": "Avg Smash%", "avg_bust_prob": "Avg Bust%",
                "boom_score": "Boom Score", "bust_risk": "Bust Risk",
                "boom_bust_rank": "Rank", "lineup_grade": "Grade",
            }).copy()
            _bb_fmt = standard_lineup_format(display_bb)
            styled = display_bb.style.format(_bb_fmt, na_rep="").applymap(
                _colour_grade, subset=["Grade"]
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── 8e. Export & Publish ───────────────────────────────────────
    st.divider()
    col_csv, col_publish = st.columns(2)

    with col_csv:
        if st.button("📥 Prepare DK CSV", key="_opt_prep_csv"):
            try:
                if slate.is_showdown:
                    csv_df = to_dk_showdown_upload_format(view_df)
                else:
                    csv_df = to_dk_upload_format(view_df)
                csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
                fname = f"yakos_{_ui_contest.lower()}_{slate.slate_date}.csv"
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name=fname,
                    mime="text/csv",
                    key="_opt_download_csv",
                )
            except Exception as exc:
                st.error(f"CSV export failed: {exc}")

    with col_publish:
        _is_published = view_label in lu_state.published_sets
        _pub_label = "✅ Published" if _is_published else f"Publish to Edge Share"
        if st.button(_pub_label, type="primary", key="_opt_publish", disabled=_is_published):
            try:
                _ts = datetime.now(timezone.utc).isoformat()
                lu_state.publish(view_label, _ts)
                set_lineup_state(lu_state)
                _eff_edge_df = slate.edge_df
                if _eff_edge_df is None or _eff_edge_df.empty:
                    _eff_edge_df = compute_edge_metrics(
                        pool, calibration_state=slate.calibration_state, sport=sport,
                    )
                    slate.edge_df = _eff_edge_df
                    from yak_core.state import set_slate_state
                    set_slate_state(slate)
                payload = publish_edge_and_lineups(slate, view_df)
                st.session_state["_friends_payload"] = payload
                st.success(f"**{_ui_contest}** published to Edge Share")
            except Exception as exc:
                st.error(f"Publish failed: {exc}")

    # ── 9. Late Swap (collapsed) ───────────────────────────────────
    injury_updates = st.session_state.get("_hub_injury_updates", [])
    if injury_updates and built_labels:
        with st.expander("Late Swap Suggestions", expanded=False):
            swap_df = lu_state.lineups.get(view_label)
            suggestions = _late_swap_suggestions(pool, swap_df, injury_updates)
            if suggestions:
                st.warning(f"⚠ {len(suggestions)} swap suggestion(s):")
                st.dataframe(pd.DataFrame(suggestions), use_container_width=True, hide_index=True)
            else:
                st.info("No late-swap actions needed.")


main()
