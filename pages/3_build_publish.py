"""Build & Publish – YakOS Sprint 1 page.

Responsibilities (S1.5)
-----------------------
- Read roster rules directly from SlateState (Classic / Showdown).
- Support floor / median / ceiling build modes per contest type.
- Exposure management (min / max per player for MME).
- Simple contest selection advisor.
- Build lineups and support DK CSV export.
- "Publish to Edge Share" action per contest type.
- S1.7 late-swap suggestions per contest type using pre-baked GTD rules.

State read:  SlateState, RickyEdgeState (edge_check gate), SimState
State written: LineupSetState
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

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
from yak_core.config import CONTEST_PRESETS, CONTEST_PRESET_LABELS, UI_CONTEST_LABELS, UI_CONTEST_MAP  # noqa: E402
from yak_core.components import render_lineup_cards_paged  # noqa: E402
from yak_core.publishing import publish_edge_and_lineups  # noqa: E402
from yak_core.edge import compute_edge_metrics  # noqa: E402
from yak_core.lineup_scoring import compute_lineup_boom_bust, GRADE_COLORS as _GRADE_COLORS_HEX  # noqa: E402
from yak_core.right_angle import apply_edge_adjustments, compute_breakout_candidates  # noqa: E402
from yak_core.display_format import normalise_ownership, standard_player_format, standard_lineup_format  # noqa: E402


# ---------------------------------------------------------------------------
# Game extraction / filter (shared with The Lab)
# ---------------------------------------------------------------------------

def _extract_games_build(pool: pd.DataFrame) -> list[str]:
    """Return sorted list of 'TEAM vs OPP' matchup strings from the pool."""
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
    elif "team" in pool.columns:
        return sorted(pool["team"].dropna().str.strip().str.upper().unique().tolist())
    return []


def _filter_pool_by_games_build(pool: pd.DataFrame, selected_games: list[str]) -> pd.DataFrame:
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BUILD_MODE_COLS = {
    "floor": "floor",
    "median": "proj",
    "ceiling": "proj",  # GPP optimises on proj; ceil-upside handled by archetype
}
_CONTEST_TO_BUILD_MODE = {
    "GPP Main": "ceiling",
    "GPP Early": "ceiling",
    "GPP Late": "ceiling",
    "Cash Main": "floor",
    "Showdown": "ceiling",
}



def _apply_sim_learnings(pool: pd.DataFrame, sim: "SimState") -> pd.DataFrame:
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


def _get_proj_col(pool: pd.DataFrame, build_mode: str) -> str:
    """Return the best available projection column for the build mode."""
    desired = _BUILD_MODE_COLS.get(build_mode, "proj")
    if desired in pool.columns:
        return desired
    return "proj"


def _build_lineups(
    pool: pd.DataFrame,
    num_lineups: int,
    max_exposure: float,
    min_exposure: float,
    min_salary: int,
    proj_col: str,
    archetype: str,
    slate: "SlateState",
    lock_names: list,
    exclude_names: list,
    contest_label: str = "GPP Main",
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build lineups using the appropriate engine (Classic / Showdown)."""
    pool = pool.copy()
    # If edge-adjusted projections exist, promote them to 'proj' so the
    # archetype layer and optimizer both operate on the adjusted numbers.
    if "effective_proj" in pool.columns:
        pool["raw_proj"] = pool["proj"].copy()  # preserve original for reference
        pool["proj"] = pool["effective_proj"]
    # Ensure player_id column exists (pool from The Lab may only have player_name)
    if "player_id" not in pool.columns:
        if "player_name" in pool.columns:
            pool["player_id"] = pool["player_name"]
        elif "dk_player_id" in pool.columns:
            pool["player_id"] = pool["dk_player_id"]
    try:
        # Map UI contest label → CONTEST_TYPE for the optimizer
        _contest_type_map = {
            "GPP Main": "gpp", "GPP Early": "gpp", "GPP Late": "gpp",
            "Cash Main": "cash", "Showdown": "showdown",
            "PGA GPP": "gpp", "PGA Cash": "cash", "PGA Showdown": "gpp",
        }
        cfg = {
            "NUM_LINEUPS": num_lineups,
            "SALARY_CAP": slate.salary_cap,
            "MAX_EXPOSURE": max_exposure,
            "MIN_SALARY_USED": min_salary,
            "LOCK": lock_names or [],
            "EXCLUDE": exclude_names or [],
            "PROJ_COL": proj_col,
            "CONTEST_TYPE": _contest_type_map.get(contest_label, "gpp"),
        }
        # Inject PGA-specific optimizer settings from the preset
        if contest_label.startswith("PGA"):
            _pga_preset = CONTEST_PRESETS.get(contest_label, {})
            cfg["POS_SLOTS"] = _pga_preset.get("pos_slots", ["G"] * 6)
            cfg["LINEUP_SIZE"] = _pga_preset.get("lineup_size", 6)
            cfg["POS_CAPS"] = _pga_preset.get("pos_caps", {})
            cfg["GPP_MAX_PUNT_PLAYERS"] = _pga_preset.get("max_punt_players", 1)
            cfg["GPP_MIN_MID_PLAYERS"] = _pga_preset.get("min_mid_salary_players", 2)
            cfg["GPP_OWN_CAP"] = _pga_preset.get("own_cap", 5.0)
            cfg["GPP_MIN_LOW_OWN_PLAYERS"] = _pga_preset.get("min_low_own_players", 1)
            cfg["GPP_LOW_OWN_THRESHOLD"] = _pga_preset.get("low_own_threshold", 0.40)
            cfg["GPP_FORCE_GAME_STACK"] = _pga_preset.get("force_game_stack", False)
        # Inject per-player exposure caps from edge overrides
        _eo = st.session_state.get("_edge_overrides", {})
        if _eo.get("max_exposure_players"):
            cfg["PLAYER_MAX_EXPOSURE"] = _eo["max_exposure_players"]
        # Inject tier-based lineup composition constraints
        if _eo.get("tier_player_names"):
            cfg["TIER_CONSTRAINTS"] = {
                "tier_player_names": _eo["tier_player_names"],
                "tier_min_players": _eo.get("tier_min_players", {}),
                "tier_max_players": _eo.get("tier_max_players", {}),
            }
        _use_showdown = slate.is_showdown or contest_label == "Showdown"
        if _use_showdown:
            lineups_df, expo_df = build_showdown_lineups(pool, cfg)
        else:
            opt_pool = apply_archetype(pool.copy(), archetype)
            lineups_df, expo_df = build_multiple_lineups_with_exposure(opt_pool, cfg)
        return lineups_df, expo_df
    except Exception as exc:
        st.error(f"Optimizer error: {exc}")
        return None, None


def _late_swap_suggestions(
    pool: pd.DataFrame,
    lineups_df: Optional[pd.DataFrame],
    injury_updates: list,
) -> list[dict]:
    """Generate late-swap candidates using pre-baked GTD rules.

    Rules:
    - OUT → pivot to highest-value replacement at same position
    - Limited / GTD → reduce exposure suggestion
    """
    suggestions: list[dict] = []
    if lineups_df is None or lineups_df.empty:
        return suggestions
    if not injury_updates:
        return suggestions

    player_pool_map: dict = {}
    if not pool.empty and "player_name" in pool.columns:
        for _, row in pool.iterrows():
            pname = str(row.get("player_name", ""))
            player_pool_map[pname] = row.to_dict()

    for update in injury_updates:
        pname = str(update.get("player_name", ""))
        status = str(update.get("status", "")).upper()

        if not pname:
            continue

        # Check if player is in lineups
        in_lineups = False
        if "player_name" in lineups_df.columns:
            in_lineups = pname in lineups_df["player_name"].values
        if not in_lineups:
            continue

        if status in ("OUT", "IR", "O"):
            # Find replacement: same position, highest proj, not in lineups
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
                    "action": "PIVOT",
                    "out_player": pname,
                    "in_player": best.get("player_name", ""),
                    "pos": pos,
                    "salary_delta": int(float(best.get("salary", 0) or 0) - current_salary),
                    "reason": f"{pname} is {status}",
                })
            else:
                suggestions.append({
                    "action": "PIVOT",
                    "out_player": pname,
                    "in_player": "(no replacement found)",
                    "pos": pos,
                    "salary_delta": 0,
                    "reason": f"{pname} is {status}",
                })

        elif status in ("GTD", "Q", "QUESTIONABLE", "DOUBTFUL", "LIMITED"):
            suggestions.append({
                "action": "REDUCE_EXPOSURE",
                "out_player": pname,
                "in_player": "",
                "pos": player_pool_map.get(pname, {}).get("pos", ""),
                "salary_delta": 0,
                "reason": f"{pname} is {status} – reduce exposure",
            })

    return suggestions


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🏗️ Build & Publish")
    st.caption("Lock in the lineups. Ricky builds from edges, not gut feels.")

    slate = get_slate_state()
    edge = get_edge_state()
    sim = get_sim_state()
    lu_state = get_lineup_state()

    # ── Edge Check Gate ───────────────────────────────────────────────────
    if not edge.ricky_edge_check:
        st.error(
            "Edge Analysis not approved. "
            "Complete the approval on **Ricky's Edge Analysis** before building lineups."
        )
        st.stop()

    if not slate.is_ready():
        st.warning("No slate published. Go to **The Lab** and load a slate first.")
        st.stop()

    pool: pd.DataFrame = slate.player_pool.copy()
    pool = _apply_sim_learnings(pool, sim)

    # ── Apply Ricky's Edge adjustments to the pool ────────────────────
    _edge_df = getattr(slate, "edge_df", None)
    if _edge_df is None or (hasattr(_edge_df, "empty") and _edge_df.empty):
        try:
            _edge_df = compute_edge_metrics(pool, calibration_state=slate.calibration_state)
        except Exception:
            _edge_df = None

    # Auto-classify tiers on edge_df (same logic as ricky_edge.py)
    # Calibrated from 21-slate backtest (Feb 7 – Mar 5 2026, 3512 player-slates).
    if _edge_df is not None and not _edge_df.empty:
        if "auto_tier" not in _edge_df.columns:
            def _classify(row):
                sal = float(row.get("salary", 6000) or 6000)
                own = float(row.get("own_pct", 15) or 15)
                proj = float(row.get("proj", 15) or 15)
                val = proj / max(sal / 1000.0, 1.0)
                if sal >= 8000:
                    return "fade"
                if sal < 6000 and val >= 2.0 and own < 15:
                    return "core"
                if own < 12 and val >= 2.5 and sal < 7500:
                    return "leverage"
                if val >= 3.0:
                    return "value"
                return "neutral"
            _edge_df = _edge_df.copy()
            _edge_df["auto_tier"] = _edge_df.apply(_classify, axis=1)

    _breakout_df = None
    try:
        _breakout_df = compute_breakout_candidates(pool, top_n=15)
    except Exception:
        pass

    pool, _edge_overrides = apply_edge_adjustments(pool, edge_df=_edge_df, breakout_df=_breakout_df)

    # Store overrides in session for use during build
    st.session_state["_edge_overrides"] = _edge_overrides

    _n_adj = _edge_overrides.get("adjustments_applied", 0)
    _tier_names = _edge_overrides.get("tier_player_names", {})
    _has_tiers = bool(_tier_names)

    if _n_adj > 0 or _has_tiers:
        _n_excl = len(_edge_overrides.get("auto_exclude", []))
        _n_fade = len(_edge_overrides.get("max_exposure_players", {}))
        _n_core = len(_edge_overrides.get("min_exposure_players", []))
        _parts = []
        if _n_adj:
            _parts.append(f"{_n_adj} proj adjustments")
        if _has_tiers:
            tier_counts = {t: len(names) for t, names in _tier_names.items()}
            _parts.append(f"tiers: {', '.join(f'{t}={c}' for t, c in tier_counts.items())}")
            _tier_min = _edge_overrides.get("tier_min_players", {})
            _tier_max = _edge_overrides.get("tier_max_players", {})
            if _tier_min:
                for k, v in _tier_min.items():
                    _parts.append(f"min {v} {k.replace('_or_', '/')}")
            if _tier_max:
                for k, v in _tier_max.items():
                    _parts.append(f"max {v} {k}/lineup")
        if _n_fade:
            _parts.append(f"{_n_fade} fade caps")
        if _n_excl:
            _parts.append(f"{_n_excl} auto-excluded")
        st.caption(f"✅ Edge → Optimizer: {', '.join(_parts)}")

    # ─────────────────────────────────────────────────────────────────────
    # Section 1: Contest Selection Advisor
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("Contest Advisor")

    # Read RCI results from sim state (written by The Lab's RCI section)
    _rci_data = sim.contest_gauges  # RCI stores results here via set_rci_result
    _UI_ADVISOR_LABELS = [("GPP", "GPP Main"), ("Cash", "Cash Main"), ("Showdown", "Showdown")]
    _any_rci = any(_rci_data.get(label, {}).get("rci_score") for _, label in _UI_ADVISOR_LABELS)
    if _any_rci:
        advisor_rows = []
        for ui_label, label in _UI_ADVISOR_LABELS:
            preset = CONTEST_PRESETS.get(label, {})
            rci_entry = _rci_data.get(label, {})
            rci_score = float(rci_entry.get("rci_score", 0))
            rci_status = rci_entry.get("rci_status", "red")
            rec = (
                "✅ Strong" if rci_score >= 70
                else "✅ Playable" if rci_score >= 45
                else "⚠️ Thin" if rci_score >= 25
                else "❌ Not calibrated"
            )
            advisor_rows.append({
                "Contest": ui_label,
                "Build Mode": _CONTEST_TO_BUILD_MODE.get(label, "median"),
                "Default Lineups": preset.get("default_lineups", 1),
                "RCI": f"{int(rci_score)}/100",
                "Recommendation": rec,
            })
        st.dataframe(pd.DataFrame(advisor_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run edge analysis or sims in **The Lab** for contest recommendations.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 2: Build Controls
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("Build Config")

    # Auto-inherit contest type from Lab selection (mapped to UI labels)
    _REVERSE_UI_MAP = {v: k for k, v in UI_CONTEST_MAP.items()}
    _lab_ui = _REVERSE_UI_MAP.get(slate.contest_name, "GPP") if slate.contest_name else "GPP"
    _default_ui_idx = UI_CONTEST_LABELS.index(_lab_ui) if _lab_ui in UI_CONTEST_LABELS else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        _ui_contest = st.selectbox("Contest Type", UI_CONTEST_LABELS, index=_default_ui_idx, key="_bp_contest")
        contest_label = UI_CONTEST_MAP[_ui_contest]
        preset = CONTEST_PRESETS.get(contest_label, {})
    with col2:
        default_mode = _CONTEST_TO_BUILD_MODE.get(contest_label, "median")
        build_mode = st.selectbox(
            "Build Mode",
            ["floor", "median", "ceiling"],
            index=["floor", "median", "ceiling"].index(default_mode),
            key="_bp_build_mode",
        )
    with col3:
        archetype = st.selectbox(
            "Archetype",
            list(DFS_ARCHETYPES.keys()),
            index=list(DFS_ARCHETYPES.keys()).index(preset.get("archetype", "Balanced"))
            if preset.get("archetype", "Balanced") in DFS_ARCHETYPES else 0,
            key="_bp_archetype",
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        num_lineups = st.number_input(
            "# Lineups", min_value=1, max_value=150,
            value=int(preset.get("default_lineups", 1)),
            key="_bp_num_lineups",
        )
    with col5:
        max_exp = st.slider(
            "Max Exposure", min_value=0.1, max_value=1.0, step=0.05,
            value=float(preset.get("default_max_exposure", 0.5)),
            key="_bp_max_exp",
        )
    with col6:
        min_salary = st.number_input(
            "Min Salary Used", min_value=40000, max_value=50000, step=100,
            value=int(preset.get("min_salary", 48000)),
            key="_bp_min_salary",
        )

    # ── Game Selector — checkboxes in expander ─────────────────────
    all_games = _extract_games_build(pool)
    build_games: list[str] = []
    if all_games:
        _lab_games = slate.selected_games if hasattr(slate, "selected_games") else []
        # Default: all games selected unless Lab had a subset
        _default_all = not _lab_games
        with st.expander(f"Games ({len(all_games)})", expanded=False):
            for _g in all_games:
                _default_on = _g in _lab_games if _lab_games else _default_all
                if st.checkbox(_g, value=_default_on, key=f"_bp_gf_{_g}"):
                    build_games.append(_g)
        if build_games and len(build_games) < len(all_games):
            pool = _filter_pool_by_games_build(pool, build_games)

    # Lock/Exclude inline (no expander)
    player_names = sorted(pool["player_name"].dropna().tolist()) if "player_name" in pool.columns else []
    col_lock, col_excl = st.columns(2)
    with col_lock:
        lock_names = st.multiselect("Lock (in every lineup)", player_names, key="_bp_lock")
    with col_excl:
        exclude_names = st.multiselect("Exclude", player_names, key="_bp_exclude")

    _bp_game_note = f"  |  {len(build_games)} of {len(all_games)} games" if build_games else ""
    st.caption(
        f"**{len(pool)} players**  |  "
        f"Cap: ${slate.salary_cap:,}{_bp_game_note}"
    )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 3: Build Lineups
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("Build Lineups")

    proj_col = _get_proj_col(pool, build_mode)

    # Merge edge overrides into exclude list
    _eo = st.session_state.get("_edge_overrides", {})
    _auto_excl = _eo.get("auto_exclude", [])
    _merged_exclude = list(set(list(exclude_names) + _auto_excl))
    if _auto_excl:
        st.caption(f"Auto-excluded (bust risk): {', '.join(_auto_excl)}")

    if st.button("Build Lineups", type="primary", key="_bp_build"):
        with st.spinner(f"Building {num_lineups} {contest_label} lineups…"):
            lineups_df, expo_df = _build_lineups(
                pool,
                num_lineups=int(num_lineups),
                max_exposure=float(max_exp),
                min_exposure=0.0,
                min_salary=int(min_salary),
                proj_col=proj_col,
                archetype=str(archetype),
                slate=slate,
                lock_names=list(lock_names),
                exclude_names=_merged_exclude,
                contest_label=contest_label,
            )
            if lineups_df is not None:
                lu_state.set_lineups(
                    contest_label,
                    lineups_df,
                    {
                        "build_mode": build_mode,
                        "num_lineups": num_lineups,
                        "max_exposure": max_exp,
                        "min_salary": min_salary,
                        "archetype": archetype,
                        "proj_col": proj_col,
                    },
                )
                if expo_df is not None:
                    lu_state.exposures[contest_label] = expo_df
                # ── Boom/bust ranking ─────────────────────────────────────
                player_results = sim.player_results
                if player_results is not None and not player_results.empty:
                    bb_rankings = compute_lineup_boom_bust(
                        lineups_df=lineups_df,
                        sim_player_results=player_results,
                        contest_label=contest_label,
                    )
                    lu_state.set_boom_bust(contest_label, bb_rankings)
                set_lineup_state(lu_state)
                st.success(f"Built {num_lineups} lineups for **{contest_label}**.")

    # ── Built Lineups Summary (across all contest types) ──────────────────
    built_labels = [lbl for lbl, df in lu_state.lineups.items() if df is not None and not df.empty]
    if len(built_labels) > 1:
        st.divider()
        st.subheader("Lineup Summary")
        _summary_rows = []
        for _bl in built_labels:
            _bl_df = lu_state.lineups[_bl]
            _bl_n = len(_bl_df["lineup_index"].unique()) if "lineup_index" in _bl_df.columns else 0
            _bl_cfg = lu_state.build_configs.get(_bl, {})
            _bl_pub = "✅" if _bl in lu_state.published_sets else "—"
            _summary_rows.append({
                "Contest": _bl,
                "Lineups": _bl_n,
                "Mode": _bl_cfg.get("build_mode", "—"),
                "Archetype": _bl_cfg.get("archetype", "—"),
                "Published": _bl_pub,
            })
        st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True, hide_index=True)

        # Publish All button
        _unpublished = [lbl for lbl in built_labels if lbl not in lu_state.published_sets]
        if _unpublished:
            if st.button(f"Publish All ({len(built_labels)}) to Edge Share", type="primary", key="_bp_publish_all"):
                _pub_ts = datetime.now(timezone.utc).isoformat()
                for _pub_lbl in built_labels:
                    lu_state.publish(_pub_lbl, _pub_ts)
                set_lineup_state(lu_state)

                # Build friends payload from the last-published slate
                _eff_edge_df = slate.edge_df
                if _eff_edge_df is None or _eff_edge_df.empty:
                    try:
                        _eff_edge_df = compute_edge_metrics(pool, calibration_state=slate.calibration_state)
                        slate.edge_df = _eff_edge_df
                        from yak_core.state import set_slate_state  # noqa: PLC0415
                        set_slate_state(slate)
                    except Exception:
                        pass
                st.success(f"Published **{len(built_labels)}** lineup sets to Edge Share.")

    # ── View individual lineups ──────────────────────────────────────────
    if built_labels:
        # Default to the contest type that was just built (matches the selector above)
        _default_view_idx = built_labels.index(contest_label) if contest_label in built_labels else 0
        view_label = st.selectbox("View lineups for", built_labels, index=_default_view_idx, key="_bp_view_label")
        view_df = lu_state.lineups.get(view_label)

        if view_df is not None and not view_df.empty:
            n_lu = len(view_df["lineup_index"].unique()) if "lineup_index" in view_df.columns else 0
            st.caption(f"{n_lu} lineup(s)")

            # Pull pipeline metrics from SimState if available
            pipeline_df = sim.pipeline_output.get(contest_label) or sim.pipeline_output.get("GPP_20")

            # Pull boom/bust rankings for this label
            bb_df = lu_state.get_boom_bust(view_label)

            render_lineup_cards_paged(
                lineups_df=view_df,
                sim_results_df=pipeline_df,
                salary_cap=slate.salary_cap,
                nav_key=f"bp_lu_{view_label}",
                boom_bust_df=bb_df,
            )

            # Exposure view
            expo_df = lu_state.exposures.get(view_label)
            if expo_df is not None and not expo_df.empty:
                with st.expander("Player Exposures", expanded=False):
                    _expo_fmt = standard_player_format(expo_df)
                    st.dataframe(
                        expo_df.style.format(_expo_fmt, na_rep=""),
                        use_container_width=True, hide_index=True,
                    )

            # ── Boom/Bust Rankings ────────────────────────────────────────
            if bb_df is not None and not bb_df.empty:
                st.divider()
                st.subheader("🏆 Lineup Rankings (Boom/Bust)")

                # Contest-aware description
                _preset = CONTEST_PRESETS.get(view_label, {})
                _mode = _preset.get("tagging_mode", "ceiling")
                if _mode == "floor":
                    st.caption(
                        "Lineups ranked by **floor safety** — boom_score weights "
                        "floor (60%), projection (30%), and low bust risk (10%)."
                    )
                else:
                    st.caption(
                        "Lineups ranked by **ceiling upside** — boom_score weights "
                        "ceiling (50%), smash probability (30%), and low bust risk (20%)."
                    )

                # Grade colour map for styling
                def _colour_grade(val: str) -> str:
                    color = _GRADE_COLORS_HEX.get(str(val), "")
                    if color:
                        return f"background-color:{color};color:#fff;font-weight:700;"
                    return ""

                display_bb = bb_df.rename(columns={
                    "lineup_index": "Lineup #",
                    "total_proj": "Total Proj",
                    "total_ceil": "Total Ceil",
                    "total_floor": "Total Floor",
                    "avg_smash_prob": "Avg Smash%",
                    "avg_bust_prob": "Avg Bust%",
                    "boom_score": "Boom Score",
                    "bust_risk": "Bust Risk",
                    "boom_bust_rank": "Rank",
                    "lineup_grade": "Grade",
                }).copy()

                # Apply standard lineup formatting
                _bb_fmt = standard_lineup_format(display_bb)

                styled = display_bb.style.format(_bb_fmt, na_rep="").applymap(_colour_grade, subset=["Grade"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Summary line
                n_ab = int((bb_df["lineup_grade"].isin(["A", "B"])).sum())
                type_label = "safe floor for cash" if _mode == "floor" else "high ceiling for GPP"
                st.caption(f"**{n_ab} lineup(s)** graded A or B ({type_label}).")

            # ── Lineup Actuals (historical slates only) ───────────────
            _bp_has_actuals = (
                "actual_fp" in pool.columns
                and pool["actual_fp"].notna().any()
            )
            if _bp_has_actuals:
                st.divider()
                st.subheader("🎯 Lineup vs Actuals")
                st.caption("How did these lineups actually perform?")
                try:
                    from yak_core.sim_accuracy import score_lineup_set, summarize_lineup_accuracy  # noqa: PLC0415

                    _lu_verdicts = score_lineup_set(
                        lineups_df=view_df,
                        pool_df=pool,
                        pipeline_df=pipeline_df,
                    )
                    if not _lu_verdicts.empty:
                        _lu_summ = summarize_lineup_accuracy(_lu_verdicts)

                        _lsc1, _lsc2, _lsc3 = st.columns(3)
                        _lsc1.metric("Avg Actual", f"{_lu_summ.get('avg_actual', 0):.1f} FP")
                        _avg_err = _lu_summ.get('avg_error', 0)
                        _lsc2.metric("Avg Error", f"{_avg_err:+.1f} FP")
                        _lsc3.metric("MAE", f"{_lu_summ.get('mae', 0):.1f} FP")

                        # Show sim rating accuracy if available
                        _rat_acc = _lu_summ.get("rating_accuracy")
                        if _rat_acc is not None:
                            _a_avg = _lu_summ.get("a_avg_actual")
                            _d_avg = _lu_summ.get("d_avg_actual")
                            _parts = [f"Rating accuracy: {_rat_acc*100:.0f}%"]
                            if _a_avg is not None:
                                _parts.append(f"A-rated avg: {_a_avg:.1f}")
                            if _d_avg is not None:
                                _parts.append(f"D-rated avg: {_d_avg:.1f}")
                            st.caption(" · ".join(_parts))

                        # Verdicts table
                        _lv_show = [c for c in ["lineup_index", "total_proj", "total_actual",
                                                "lineup_error", "actual_grade"]
                                    if c in _lu_verdicts.columns]
                        # Add sim columns if present
                        for c in ["sim_rating", "sim_bucket", "rating_accurate"]:
                            if c in _lu_verdicts.columns:
                                _lv_show.append(c)

                        _lv_display = _lu_verdicts[_lv_show].rename(columns={
                            "lineup_index": "Lineup",
                            "total_proj": "Projected",
                            "total_actual": "Actual",
                            "lineup_error": "Diff",
                            "actual_grade": "Grade",
                            "sim_rating": "Sim Rating",
                            "sim_bucket": "Sim Grade",
                            "rating_accurate": "Grade Match",
                        })

                        _lv_fmt = standard_lineup_format(_lv_display)
                        if "Sim Rating" in _lv_display.columns:
                            _lv_fmt["Sim Rating"] = "{:.0f}"

                        def _color_lineup_diff(val):
                            try:
                                v = float(val)
                            except (ValueError, TypeError):
                                return ""
                            return "color: #4caf82" if v > 0 else "color: #e05c5c" if v < 0 else ""

                        try:
                            _lv_styled = _lv_display.style.format(_lv_fmt, na_rep="")
                            if "Diff" in _lv_display.columns:
                                _lv_styled = _lv_styled.applymap(_color_lineup_diff, subset=["Diff"])
                            st.dataframe(_lv_styled, use_container_width=True, hide_index=True)
                        except Exception:
                            st.dataframe(_lv_display, use_container_width=True, hide_index=True)
                    else:
                        st.info("Not enough actuals to score lineups (need 50%+ of players matched).")
                except Exception as _lv_exc:
                    st.warning(f"Lineup scoring failed: {_lv_exc}")

            st.divider()

            # ── Export & Publish ──────────────────────────────────────────
            st.subheader("Export & Publish")

            col_csv, col_publish = st.columns(2)

            with col_csv:
                if st.button("Prepare DK CSV", key="_bp_prep_csv"):
                    try:
                        if slate.is_showdown:
                            csv_df = to_dk_showdown_upload_format(view_df)
                        else:
                            csv_df = to_dk_upload_format(view_df)
                        csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
                        fname = f"yakos_{view_label.replace(' ', '_').lower()}_{slate.slate_date}.csv"
                        st.download_button(
                            label="Download DK CSV",
                            data=csv_bytes,
                            file_name=fname,
                            mime="text/csv",
                            key="_bp_download_csv",
                        )
                    except Exception as exc:
                        st.error(f"CSV export failed: {exc}")

            with col_publish:
                if st.button(f"Publish {view_label} to Edge Share", type="primary", key="_bp_publish"):
                    try:
                        # Publish lineup state
                        _ts = datetime.now(timezone.utc).isoformat()
                        lu_state.publish(view_label, _ts)
                        set_lineup_state(lu_state)

                        # Build friends payload
                        _eff_edge_df = slate.edge_df
                        if _eff_edge_df is None or _eff_edge_df.empty:
                            _eff_edge_df = compute_edge_metrics(
                                pool,
                                calibration_state=slate.calibration_state,
                            )
                            slate.edge_df = _eff_edge_df
                            from yak_core.state import set_slate_state  # noqa: PLC0415
                            set_slate_state(slate)
                        payload = publish_edge_and_lineups(slate, view_df)
                        st.session_state["_friends_payload"] = payload
                        st.success(f"**{view_label}** published to Edge Share")
                    except Exception as exc:
                        st.error(f"Publish failed: {exc}")

    else:
        st.info("Build lineups above to see results and export options.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 4: Late Swap Suggestions (S1.7)
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("Late Swap")
    st.caption("OUT → pivot to best replacement, GTD/Limited → reduce exposure.")

    injury_updates = st.session_state.get("_hub_injury_updates", [])
    if not injury_updates:
        st.info("No injury updates loaded. Use **The Lab → Injury / News Refresh** to fetch updates.")
    elif built_labels:
        swap_label = st.selectbox("Contest for swap suggestions", built_labels, key="_bp_swap_label")
        swap_df = lu_state.lineups.get(swap_label)
        suggestions = _late_swap_suggestions(pool, swap_df, injury_updates)

        if suggestions:
            st.warning(f"⚠️ {len(suggestions)} swap suggestion(s) for **{swap_label}**:")
            st.dataframe(pd.DataFrame(suggestions), use_container_width=True, hide_index=True)
        else:
            st.success("✅ No late-swap actions needed for this contest.")
    else:
        st.info("Build lineups first to generate late-swap suggestions.")


main()
