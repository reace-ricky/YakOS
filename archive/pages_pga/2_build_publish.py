"""Build & Publish – PGA standalone.

Responsibilities
----------------
- Read roster rules directly from SlateState (Classic / Showdown).
- Support floor / median / ceiling build modes per contest type.
- Exposure management (min / max per player for MME).
- Simple contest selection advisor.
- Build lineups and support DK CSV export.
- "Publish to Edge Share" action per contest type.

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

from yak_core.pga_state import (  # noqa: E402
    get_slate_state, set_slate_state,
    get_edge_state,
    get_sim_state,
    get_lineup_state, set_lineup_state,
    pga_publish,
    _pga_save_slate, _pga_save_edge,
)
from yak_core.lineups import (  # noqa: E402
    build_multiple_lineups_with_exposure,
    to_dk_pga_upload_format,
    build_showdown_lineups,
    to_dk_showdown_upload_format,
)
from yak_core.calibration import apply_archetype, DFS_ARCHETYPES  # noqa: E402
from yak_core.config import CONTEST_PRESETS, PGA_UI_CONTEST_LABELS, PGA_UI_CONTEST_MAP  # noqa: E402
from yak_core.components import render_lineup_cards_scrollable  # noqa: E402
from yak_core.publishing import publish_edge_and_lineups  # noqa: E402
from yak_core.edge import compute_edge_metrics  # noqa: E402
from yak_core.lineup_scoring import compute_lineup_boom_bust, GRADE_COLORS as _GRADE_COLORS_HEX  # noqa: E402
from yak_core.right_angle import apply_edge_adjustments, compute_breakout_candidates  # noqa: E402
from yak_core.display_format import normalise_ownership, standard_player_format, standard_lineup_format, TAG_COLORS, classify_player_tier  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BUILD_MODE_COLS = {
    "floor": "floor",
    "median": "proj",
    "ceiling": "proj",
}
_CONTEST_TO_BUILD_MODE = {
    "PGA GPP": "ceiling", "PGA Cash": "floor", "PGA Showdown": "ceiling",
}


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


def _get_proj_col(pool: pd.DataFrame, build_mode: str) -> str:
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
    slate,
    lock_names: list,
    exclude_names: list,
    contest_label: str = "PGA GPP",
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build lineups using the appropriate engine (Classic / Showdown)."""
    pool = pool.copy()
    # --- EXCLUDE: filter out excluded players before optimizer ---
    _excl = [n.strip() for n in (exclude_names or []) if n]
    if _excl and "player_name" in pool.columns:
        pool = pool[~pool["player_name"].isin(_excl)].reset_index(drop=True)
    # If edge-adjusted projections exist, promote them to 'proj'
    if "effective_proj" in pool.columns:
        pool["raw_proj"] = pool["proj"].copy()
        pool["proj"] = pool["effective_proj"]
    # Ensure player_id column exists
    if "player_id" not in pool.columns:
        if "player_name" in pool.columns:
            pool["player_id"] = pool["player_name"]
        elif "dk_player_id" in pool.columns:
            pool["player_id"] = pool["dk_player_id"]
    try:
        _contest_type_map = {
            "PGA GPP": "gpp", "PGA Cash": "cash", "PGA Showdown": "gpp",
        }
        _preset_cfg = dict(CONTEST_PRESETS.get(contest_label, {}))
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
        if _preset_cfg:
            if _preset_cfg.get("pos_slots"):
                cfg["POS_SLOTS"] = _preset_cfg["pos_slots"]
            if _preset_cfg.get("lineup_size"):
                cfg["LINEUP_SIZE"] = _preset_cfg["lineup_size"]
            if _preset_cfg.get("pos_caps"):
                cfg["POS_CAPS"] = _preset_cfg["pos_caps"]
            cfg["GPP_MAX_PUNT_PLAYERS"] = _preset_cfg.get("max_punt_players", _preset_cfg.get("GPP_MAX_PUNT_PLAYERS", 1))
            cfg["GPP_MIN_MID_PLAYERS"] = _preset_cfg.get("min_mid_salary_players", _preset_cfg.get("GPP_MIN_MID_PLAYERS", 2))
            cfg["GPP_OWN_CAP"] = _preset_cfg.get("own_cap", _preset_cfg.get("GPP_OWN_CAP", 5.0))
            cfg["GPP_MIN_LOW_OWN_PLAYERS"] = _preset_cfg.get("min_low_own_players", _preset_cfg.get("GPP_MIN_LOW_OWN_PLAYERS", 1))
            cfg["GPP_LOW_OWN_THRESHOLD"] = _preset_cfg.get("low_own_threshold", _preset_cfg.get("GPP_LOW_OWN_THRESHOLD", 0.40))
            cfg["GPP_FORCE_GAME_STACK"] = _preset_cfg.get("force_game_stack", _preset_cfg.get("GPP_FORCE_GAME_STACK", False))
        # Inject per-player exposure caps from edge overrides
        _eo = st.session_state.get("_pga_edge_overrides", {})
        if _eo.get("max_exposure_players"):
            cfg["PLAYER_MAX_EXPOSURE"] = _eo["max_exposure_players"]
        if _eo.get("tier_player_names"):
            cfg["TIER_CONSTRAINTS"] = {
                "tier_player_names": _eo["tier_player_names"],
                "tier_min_players": _eo.get("tier_min_players", {}),
                "tier_max_players": _eo.get("tier_max_players", {}),
            }
        _use_showdown = slate.is_showdown or contest_label == "PGA Showdown"
        if _use_showdown:
            lineups_df, expo_df = build_showdown_lineups(pool, cfg)
        else:
            opt_pool = apply_archetype(pool.copy(), archetype)
            lineups_df, expo_df = build_multiple_lineups_with_exposure(opt_pool, cfg)
        return lineups_df, expo_df
    except Exception as exc:
        st.error(f"Optimizer error: {exc}")
        return None, None


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
            "Complete the approval on **The Lab** before building lineups."
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

    if _edge_df is not None and not _edge_df.empty:
        if "auto_tier" not in _edge_df.columns:
            _edge_df = _edge_df.copy()
            _edge_df["auto_tier"] = _edge_df.apply(
                lambda row: classify_player_tier(
                    row.get("salary", 6000),
                    row.get("proj", 0),
                    row.get("own_pct", 15) if "own_pct" in row.index else row.get("ownership", 15),
                ),
                axis=1,
            )

    _breakout_df = None
    try:
        _breakout_df = compute_breakout_candidates(pool, top_n=15)
    except Exception:
        pass

    pool, _edge_overrides = apply_edge_adjustments(pool, edge_df=_edge_df, breakout_df=_breakout_df)

    # Store overrides in session for use during build (PGA-prefixed key)
    st.session_state["_pga_edge_overrides"] = _edge_overrides

    _n_adj = _edge_overrides.get("adjustments_applied", 0)
    _tier_names = _edge_overrides.get("tier_player_names", {})
    _has_tiers = bool(_tier_names)

    if _n_adj > 0 or _has_tiers:
        _n_excl = len(_edge_overrides.get("auto_exclude", []))
        _n_fade = len(_edge_overrides.get("max_exposure_players", {}))
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

    _edge_df = slate.edge_df
    _has_edge = _edge_df is not None and not _edge_df.empty
    _UI_ADVISOR_LABELS = [("GPP", "PGA GPP"), ("Cash", "PGA Cash"), ("Showdown", "PGA Showdown")]
    if _has_edge:
        _lev = _edge_df["leverage"].dropna()
        _own = (
            _edge_df["own_pct"] if "own_pct" in _edge_df.columns
            else _edge_df["ownership"] if "ownership" in _edge_df.columns
            else pd.Series(dtype=float)
        )
        _bust = _edge_df["bust_prob"] if "bust_prob" in _edge_df.columns else pd.Series(dtype=float)
        _smash = _edge_df["smash_prob"] if "smash_prob" in _edge_df.columns else pd.Series(dtype=float)
        _ceil_mag = _edge_df["ceil_magnitude"] if "ceil_magnitude" in _edge_df.columns else pd.Series(dtype=float)
        _edge_score = _edge_df["edge_score"] if "edge_score" in _edge_df.columns else pd.Series(dtype=float)

        _top8_idx = _edge_score.nlargest(8).index
        _avg_own_top8 = float(_own.reindex(_top8_idx).mean()) if not _own.empty else float("nan")

        _n_high_lev = int((_lev > 2.0).sum())
        _n_safe_floor = int((_bust < 0.25).sum()) if not _bust.empty else 0
        _avg_smash_top8 = float(_smash.reindex(_top8_idx).mean()) if not _smash.empty else 0.0
        _n_high_ceil = int((_ceil_mag > 0.6).sum()) if not _ceil_mag.empty else 0
        _avg_edge_top8 = float(_edge_score.reindex(_top8_idx).mean()) if not _edge_score.empty else 0.0

        _salary = _edge_df["salary"] if "salary" in _edge_df.columns else pd.Series(dtype=float)
        _sal_median = float(_salary.median()) if not _salary.empty else 0.0
        _ceil_median = float(_ceil_mag.median()) if not _ceil_mag.empty else 0.0
        _n_value = (
            int(((_salary < _sal_median) & (_ceil_mag > _ceil_median)).sum())
            if not _salary.empty and not _ceil_mag.empty else 0
        )

        advisor_rows = []
        for ui_label, label in _UI_ADVISOR_LABELS:
            preset = CONTEST_PRESETS.get(label, {})

            if label == "PGA GPP":
                _own_ok = not (pd.isna(_avg_own_top8))
                _own_str = f"{_avg_own_top8:.0f}% avg own" if _own_ok else "own N/A"
                if _n_high_lev >= 5 and (_own_ok and _avg_own_top8 < 20):
                    rec = f"✅ Strong — {_n_high_lev} leveraged, {_own_str}"
                elif _n_high_lev >= 3 or (_own_ok and _avg_own_top8 < 25):
                    rec = f"✅ Playable — {_n_high_lev} leveraged, {_own_str}"
                elif _n_high_lev >= 1:
                    rec = f"⚠️ Thin — {_n_high_lev} leveraged play(s), {_own_str}"
                else:
                    rec = "❌ Pass — no leverage edge detected"

            elif label == "PGA Cash":
                if _n_safe_floor >= 8 and _avg_smash_top8 > 0.45:
                    rec = f"✅ Strong — {_n_safe_floor} safe floors, {_avg_smash_top8:.2f} avg smash"
                elif _n_safe_floor >= 5 or _avg_smash_top8 > 0.35:
                    rec = f"✅ Playable — {_n_safe_floor} safe floors, {_avg_smash_top8:.2f} avg smash"
                elif _n_safe_floor >= 3:
                    rec = f"⚠️ Thin — {_n_safe_floor} safe-floor plays"
                else:
                    rec = "❌ Pass — floor concentration too low"

            else:  # PGA Showdown
                if _n_high_ceil >= 4 and _avg_edge_top8 > 0.5:
                    rec = f"✅ Strong — {_n_high_ceil} high-ceil plays, {_avg_edge_top8:.2f} avg edge"
                elif _n_high_ceil >= 2 or _avg_edge_top8 > 0.4:
                    rec = f"✅ Playable — {_n_high_ceil} high-ceil plays, {_avg_edge_top8:.2f} avg edge"
                elif _n_high_ceil >= 1:
                    rec = f"⚠️ Thin — {_n_high_ceil} high-ceil play(s)"
                else:
                    rec = "❌ Pass — ceiling concentration too low"

            advisor_rows.append({
                "Contest": ui_label,
                "Build Mode": _CONTEST_TO_BUILD_MODE.get(label, "median"),
                "Default Lineups": preset.get("default_lineups", 1),
                "Value Plays": _n_value,
                "Recommendation": rec,
            })
        st.dataframe(pd.DataFrame(advisor_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run edge analysis in **The Lab** for contest recommendations.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 2: Build Controls
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("Build Config")

    _REVERSE_UI_MAP = {v: k for k, v in PGA_UI_CONTEST_MAP.items()}
    _lab_ui = _REVERSE_UI_MAP.get(slate.contest_name, "GPP") if slate.contest_name else "GPP"
    if "_pga_bp_contest" not in st.session_state or st.session_state["_pga_bp_contest"] not in PGA_UI_CONTEST_LABELS:
        st.session_state["_pga_bp_contest"] = _lab_ui if _lab_ui in PGA_UI_CONTEST_LABELS else PGA_UI_CONTEST_LABELS[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        _ui_contest = st.selectbox("Contest Type", PGA_UI_CONTEST_LABELS, key="_pga_bp_contest")
        contest_label = PGA_UI_CONTEST_MAP[_ui_contest]
        preset = CONTEST_PRESETS.get(contest_label, {})
    with col2:
        default_mode = _CONTEST_TO_BUILD_MODE.get(contest_label, "median")
        if "_pga_bp_build_mode" not in st.session_state:
            st.session_state["_pga_bp_build_mode"] = default_mode
        build_mode = st.selectbox(
            "Build Mode",
            ["floor", "median", "ceiling"],
            key="_pga_bp_build_mode",
        )
    with col3:
        _default_archetype = preset.get("archetype", "Balanced")
        _arch_keys = list(DFS_ARCHETYPES.keys())
        if "_pga_bp_archetype" not in st.session_state or st.session_state["_pga_bp_archetype"] not in _arch_keys:
            st.session_state["_pga_bp_archetype"] = _default_archetype if _default_archetype in DFS_ARCHETYPES else _arch_keys[0]
        archetype = st.selectbox(
            "Archetype",
            _arch_keys,
            key="_pga_bp_archetype",
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        num_lineups = st.number_input(
            "# Lineups", min_value=1, max_value=150,
            value=int(preset.get("default_lineups", 1)),
            key="_pga_bp_num_lineups",
        )
    with col5:
        max_exp = st.slider(
            "Max Exposure", min_value=0.1, max_value=1.0, step=0.05,
            value=float(preset.get("default_max_exposure", 0.5)),
            key="_pga_bp_max_exp",
        )
    with col6:
        min_salary = st.number_input(
            "Min Salary Used", min_value=40000, max_value=50000, step=100,
            value=int(preset.get("min_salary", 48000)),
            key="_pga_bp_min_salary",
        )

    # Lock/Exclude inline
    player_names = sorted(pool["player_name"].dropna().tolist()) if "player_name" in pool.columns else []
    col_lock, col_excl = st.columns(2)
    with col_lock:
        lock_names = st.multiselect("Lock (in every lineup)", player_names, key="_pga_bp_lock")
    with col_excl:
        exclude_names = st.multiselect("Exclude", player_names, key="_pga_bp_exclude")

    _active_wave = st.session_state.get("_pga_wave_filter", "All Players")
    _wave_badge = f"  |  🌊 {_active_wave}" if _active_wave != "All Players" else ""
    st.caption(
        f"**{len(pool)} players**  |  "
        f"Cap: ${slate.salary_cap:,}"
        f"{_wave_badge}"
    )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 3: Build Lineups
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("Build Lineups")

    proj_col = _get_proj_col(pool, build_mode)

    # Merge edge overrides into exclude list
    _eo = st.session_state.get("_pga_edge_overrides", {})
    _auto_excl = _eo.get("auto_exclude", [])
    _merged_exclude = list(set(list(exclude_names) + _auto_excl))
    if _auto_excl:
        st.caption(f"Auto-excluded (bust risk): {', '.join(_auto_excl)}")

    if st.button("Build Lineups", type="primary", key="_pga_bp_build"):
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
            if st.button(f"Publish All ({len(built_labels)}) to Edge Share", type="primary", key="_pga_bp_publish_all"):
                _pub_ts = datetime.now(timezone.utc).isoformat()
                for _pub_lbl in built_labels:
                    pga_publish(lu_state, _pub_lbl, _pub_ts)
                set_lineup_state(lu_state)

                _eff_edge_df = slate.edge_df
                if _eff_edge_df is None or _eff_edge_df.empty:
                    try:
                        _eff_edge_df = compute_edge_metrics(pool, calibration_state=slate.calibration_state)
                        slate.edge_df = _eff_edge_df
                    except Exception:
                        pass
                slate.published = True
                slate.published_at = _pub_ts
                set_slate_state(slate)
                try:
                    _pga_save_slate(slate)
                    _pga_save_edge(edge)
                except Exception:
                    pass
                st.success(f"Published **{len(built_labels)}** lineup sets to Edge Share.")

    # ── View individual lineups ──────────────────────────────────────────
    if built_labels:
        if "_pga_bp_view_label" not in st.session_state or st.session_state["_pga_bp_view_label"] not in built_labels:
            st.session_state["_pga_bp_view_label"] = contest_label if contest_label in built_labels else built_labels[0]
        view_label = st.selectbox("View lineups for", built_labels, key="_pga_bp_view_label")
        view_df = lu_state.lineups.get(view_label)

        if view_df is not None and not view_df.empty:
            n_lu = len(view_df["lineup_index"].unique()) if "lineup_index" in view_df.columns else 0
            st.caption(f"{n_lu} lineup(s)")

            pipeline_df = sim.pipeline_output.get(view_label) or sim.pipeline_output.get("GPP_20")

            bb_df = lu_state.get_boom_bust(view_label)

            render_lineup_cards_scrollable(
                lineups_df=view_df,
                sim_results_df=pipeline_df,
                salary_cap=slate.salary_cap,
                nav_key=f"pga_bp_lu_{view_label}",
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

                _bb_fmt = standard_lineup_format(display_bb)

                styled = display_bb.style.format(_bb_fmt, na_rep="").applymap(_colour_grade, subset=["Grade"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

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

                        _lv_show = [c for c in ["lineup_index", "total_proj", "total_actual",
                                                "lineup_error", "actual_grade"]
                                    if c in _lu_verdicts.columns]
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

            _all_lu_indices = (
                sorted(view_df["lineup_index"].unique().tolist())
                if "lineup_index" in view_df.columns
                else []
            )
            _lu_options = [f"Lineup {i + 1}" for i in range(len(_all_lu_indices))]
            _label_to_idx = {lbl: _all_lu_indices[i] for i, lbl in enumerate(_lu_options)}
            _selected_lu_labels = st.multiselect(
                "Select lineups to export / publish",
                options=_lu_options,
                default=_lu_options,
                key=f"_pga_bp_lu_select_{view_label}",
            )
            _selected_indices = [_label_to_idx[lbl] for lbl in _selected_lu_labels]
            _n_selected = len(_selected_indices)
            _n_total = len(_all_lu_indices)
            st.caption(f"Publishing **{_n_selected} of {_n_total}** lineup(s)")

            if _selected_indices and "lineup_index" in view_df.columns:
                _publish_df = view_df[view_df["lineup_index"].isin(_selected_indices)].copy()
            else:
                _publish_df = pd.DataFrame()

            col_csv, col_publish = st.columns(2)

            with col_csv:
                if st.button("Prepare DK CSV", key="_pga_bp_prep_csv"):
                    if not _selected_indices:
                        st.warning("No lineups selected. Select at least one lineup to export.")
                    else:
                        try:
                            if slate.is_showdown:
                                csv_df = to_dk_showdown_upload_format(_publish_df)
                            else:
                                csv_df = to_dk_pga_upload_format(_publish_df)
                            csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
                            fname = f"yakos_pga_{view_label.replace(' ', '_').lower()}_{slate.slate_date}.csv"
                            st.download_button(
                                label="Download DK CSV",
                                data=csv_bytes,
                                file_name=fname,
                                mime="text/csv",
                                key="_pga_bp_download_csv",
                            )
                        except Exception as exc:
                            st.error(f"CSV export failed: {exc}")

            with col_publish:
                if st.button(f"Publish {view_label} to Edge Share", type="primary", key="_pga_bp_publish"):
                    if not _selected_indices:
                        st.warning("No lineups selected. Select at least one lineup to publish.")
                    else:
                        try:
                            _ts = datetime.now(timezone.utc).isoformat()
                            pga_publish(lu_state, view_label, _ts)
                            set_lineup_state(lu_state)

                            _eff_edge_df = slate.edge_df
                            if _eff_edge_df is None or _eff_edge_df.empty:
                                _eff_edge_df = compute_edge_metrics(
                                    pool,
                                    calibration_state=slate.calibration_state,
                                )
                                slate.edge_df = _eff_edge_df
                            slate.published = True
                            slate.published_at = _ts
                            set_slate_state(slate)
                            try:
                                _pga_save_slate(slate)
                                _pga_save_edge(edge)
                            except Exception:
                                pass
                            payload = publish_edge_and_lineups(slate, _publish_df)
                            st.session_state["_pga_friends_payload"] = payload
                            st.success(
                                f"**{view_label}** published to Edge Share "
                                f"({_n_selected} of {_n_total} lineup(s))"
                            )
                        except Exception as exc:
                            st.error(f"Publish failed: {exc}")

    else:
        st.info("Build lineups above to see results and export options.")


main()
