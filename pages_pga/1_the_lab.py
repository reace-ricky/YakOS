"""The Lab – PGA standalone.

Responsibilities
----------------
- **PGA Pool Loading** via DataGolf API (projections + SG + course fit).
- **Edge Metrics**: Smash / bust / leverage per player.
- **Calibration**: Backfill PGA and PGA Showdown calibration events.
- **Sim Sandbox**: Score sims against archived slates.
- **PGA Projections vs Actuals**: Per-event and per-salary-tier accuracy.

State read:  SlateState, RickyEdgeState, SimState  (PGA-prefixed)
State written: SlateState, SimState  (PGA-prefixed)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.pga_state import (  # noqa: E402
    get_slate_state, set_slate_state,
    get_edge_state, set_edge_state,
    get_sim_state, set_sim_state,
    get_lineup_state,
)
from yak_core.sims import prepare_sims_table  # noqa: E402
from yak_core.edge import compute_edge_metrics  # noqa: E402
from yak_core.publishing import build_ricky_lineups  # noqa: E402
from yak_core.config import (  # noqa: E402
    YAKOS_ROOT,
    CONTEST_PRESETS,
    PGA_UI_CONTEST_LABELS,
    PGA_UI_CONTEST_MAP,
    DK_PGA_LINEUP_SIZE,
    DK_PGA_POS_SLOTS,
    DK_PGA_SALARY_CAP,
)
from yak_core.right_angle import (  # noqa: E402
    compute_stack_scores,
    compute_value_scores,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_pool_by_wave(pool: pd.DataFrame, wave: str) -> pd.DataFrame:
    """Filter pool to early or late wave using DataGolf's early_late_wave column.

    Falls back to r1_teetime with an 11:30 cutoff if early_late_wave is missing.
    """
    if wave == "All Players" or pool.empty:
        return pool

    if "early_late_wave" in pool.columns and pool["early_late_wave"].notna().any():
        col = pool["early_late_wave"].astype(str).str.strip().str.lower()
        if wave == "Early Wave":
            mask = col.isin(["early", "e"])
        else:
            mask = col.isin(["late", "l"])
        filtered = pool[mask].reset_index(drop=True)
        if not filtered.empty:
            return filtered

    # Fallback: split on r1_teetime with 11:30 cutoff
    if "r1_teetime" in pool.columns and pool["r1_teetime"].notna().any():
        times = pd.to_datetime(pool["r1_teetime"], format="%H:%M", errors="coerce")
        cutoff = pd.to_datetime("11:30", format="%H:%M")
        if wave == "Early Wave":
            mask = times <= cutoff
        else:
            mask = times > cutoff
        valid = mask.notna()
        filtered = pool[mask & valid].reset_index(drop=True)
        if not filtered.empty:
            return filtered

    return pool


def _color_smash(val: float) -> str:
    if val >= 0.25:
        return "background-color: #1a472a; color: #90ee90"
    if val >= 0.10:
        return "background-color: #2d5a27; color: #c8f0c0"
    return ""


def _color_bust(val: float) -> str:
    if val >= 0.30:
        return "background-color: #6b1a1a; color: #f08080"
    if val >= 0.15:
        return "background-color: #4a1a1a; color: #f0c0c0"
    return ""


def _build_player_level_sim_results(pool: pd.DataFrame, variance: float) -> pd.DataFrame:
    """Build player-level sim results using the authoritative edge model."""
    from yak_core.edge import compute_edge_metrics as _cem  # noqa: PLC0415
    from yak_core.display_format import normalise_ownership  # noqa: PLC0415

    if pool.empty:
        return pd.DataFrame()
    df = pool.copy()

    if "sim_eligible" in df.columns:
        df = df[df["sim_eligible"].astype(bool)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    if "ownership" in df.columns:
        df["ownership"] = normalise_ownership(df["ownership"])

    edge_df = _cem(df, variance=variance, sport="PGA")

    keep_cols = [
        "player_name", "pos", "team", "salary", "proj", "floor",
        "ceil", "own_pct", "smash_prob", "bust_prob", "leverage",
    ]
    keep_cols = [c for c in keep_cols if c in edge_df.columns]
    result = edge_df[keep_cols].copy()

    if "leverage" in result.columns:
        result = result.sort_values("leverage", ascending=False, na_position="last")

    return result.reset_index(drop=True)


def _load_pga_pool(
    slate_date_str: str,
    contest_type_label: str,
    preset: dict,
    slate,
    sim,
    _contest_safe: str,
    status_container,
) -> Optional[pd.DataFrame]:
    """Load PGA player pool via DataGolf API."""
    from yak_core.datagolf import DataGolfClient
    from yak_core.pga_pool import build_pga_pool

    dg_key = (
        st.secrets.get("DATAGOLF_API_KEY")
        or os.environ.get("DATAGOLF_API_KEY")
        or "7e0b29081d2adaac7e3de0ed387c"
    )
    if not dg_key:
        status_container.error("DataGolf API key not configured.")
        return None

    try:
        status_container.write("Connecting to DataGolf API…")
        dg = DataGolfClient(api_key=dg_key)

        _dg_slate = preset.get("projection_slate", "main")
        status_container.write(f"Building PGA pool (projections + SG + course fit, slate={_dg_slate})…")
        pool = build_pga_pool(dg, site="draftkings", slate=_dg_slate)

        if pool.empty:
            status_container.error("DataGolf returned no players for the current event.")
            return None

        # Apply PGA calibration corrections
        _cal_sport = "PGA_SD" if _dg_slate == "showdown" else "PGA"
        try:
            from yak_core.calibration_feedback import get_correction_factors, apply_corrections
            _cf = get_correction_factors(sport=_cal_sport)
            if _cf.get("n_slates", 0) > 0:
                pool = apply_corrections(pool, _cf, sport=_cal_sport)
                status_container.write(f"📐 {_cal_sport} calibration applied ({_cf['n_slates']} event(s))")
        except Exception:
            pass

        event_name = pool.attrs.get("event_name", "PGA")
        course_name = pool.attrs.get("course_name", "")
        n_players = len(pool)
        status_container.write(
            f"✅ Pool built: {n_players} players — {event_name}"
            + (f" at {course_name}" if course_name else "")
        )

        # Ensure own_proj column exists
        if "own_proj" not in pool.columns:
            if "ownership" in pool.columns:
                pool["own_proj"] = pool["ownership"]
            elif "proj_own" in pool.columns:
                pool["own_proj"] = pool["proj_own"]
            else:
                pool["own_proj"] = 5.0

        if "sim_eligible" not in pool.columns:
            pool["sim_eligible"] = pool.get("status", "Active").apply(
                lambda s: str(s).strip().upper() not in {"WD", "OUT"}
            )

        # Update PGA slate state
        slate.sport = "PGA"
        slate.site = "DK"
        slate.slate_date = slate_date_str
        slate.contest_type = contest_type_label
        slate.contest_name = contest_type_label
        _is_sd = _dg_slate == "showdown"
        slate.is_showdown = _is_sd
        slate.roster_slots = DK_PGA_POS_SLOTS
        slate.salary_cap = DK_PGA_SALARY_CAP
        slate.player_pool = pool
        slate.published = True
        slate.published_at = datetime.now(timezone.utc).isoformat()
        set_slate_state(slate)

        # Cache in session state
        st.session_state[f"_pga_hub_pool_{slate_date_str}_{_contest_safe}"] = pool
        st.session_state[f"_pga_hub_rules_{slate_date_str}_{_contest_safe}"] = {
            "slots": DK_PGA_POS_SLOTS,
            "lineup_size": DK_PGA_LINEUP_SIZE,
            "salary_cap": DK_PGA_SALARY_CAP,
            "is_showdown": _is_sd,
        }

        return pool

    except Exception as exc:
        status_container.error(f"DataGolf load failed: {exc}")
        import traceback
        status_container.write(traceback.format_exc())
        return None


# =========================================================================
# Page content
# =========================================================================
st.header("🧪 The Lab — PGA")

slate = get_slate_state()
sim = get_sim_state()
edge = get_edge_state()

# =========================================================================
# SECTION 1: LOAD SLATE
# =========================================================================
st.subheader("📥 Load Slate")

col_date, col_contest = st.columns([1, 2])
with col_date:
    from zoneinfo import ZoneInfo
    _today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    slate_date = st.date_input("Date", value=pd.to_datetime(_today))
    slate_date_str = str(slate_date)
with col_contest:
    _ui_contest = st.selectbox("Contest Type", PGA_UI_CONTEST_LABELS)
    contest_type_label = PGA_UI_CONTEST_MAP[_ui_contest]
    preset = CONTEST_PRESETS[contest_type_label]

# Wave filter — target early/late tee-time waves (weather-impacted tournaments)
_WAVE_OPTIONS = ["All Players", "Early Wave", "Late Wave"]
if "_pga_wave_filter" not in st.session_state:
    st.session_state["_pga_wave_filter"] = "All Players"
_wave_selection = st.radio(
    "🌊 Wave Filter",
    _WAVE_OPTIONS,
    horizontal=True,
    key="_pga_wave_filter",
    help="Filter by tee-time wave. Early/Late come from DataGolf. Useful for weather-impacted tournaments.",
)

# Sim controls
col_nsims, _ = st.columns(2)
with col_nsims:
    n_sims = st.number_input(
        "MC Iterations", min_value=500, max_value=50000,
        step=500, value=int(sim.n_sims), key="_pga_lab_nsims",
    )
    if n_sims != sim.n_sims:
        sim.n_sims = int(n_sims)
        set_sim_state(sim)

# Contest-safe key for caching
_contest_safe = contest_type_label.lower().replace(" ", "_").replace("/", "-").replace("-", "_")

# Cache invalidation on date/contest change
_prev_date = st.session_state.get("_pga_hub_prev_date")
_prev_contest = st.session_state.get("_pga_hub_prev_contest")
_date_changed = _prev_date is not None and _prev_date != slate_date_str
_contest_changed = _prev_contest is not None and _prev_contest != contest_type_label
if _date_changed or _contest_changed:
    _stale_date = _prev_date if _prev_date is not None else slate_date_str
    _stale_contest = _prev_contest if _prev_contest is not None else contest_type_label
    _stale_safe = _stale_contest.lower().replace(" ", "_").replace("/", "-").replace("-", "_")
    for key in [
        f"_pga_hub_pool_{_stale_date}_{_stale_safe}",
        f"_pga_hub_rules_{_stale_date}_{_stale_safe}",
    ]:
        st.session_state.pop(key, None)
st.session_state["_pga_hub_prev_date"] = slate_date_str
st.session_state["_pga_hub_prev_contest"] = contest_type_label

# Wave change invalidation — force reload so the filter is applied from the full pool
_prev_wave = st.session_state.get("_pga_hub_prev_wave")
_wave_changed = _prev_wave is not None and _prev_wave != _wave_selection
if _wave_changed:
    _stale_safe_w = contest_type_label.lower().replace(" ", "_").replace("/", "-").replace("-", "_")
    for key in [
        f"_pga_hub_pool_{slate_date_str}_{_stale_safe_w}",
        f"_pga_hub_rules_{slate_date_str}_{_stale_safe_w}",
    ]:
        st.session_state.pop(key, None)
st.session_state["_pga_hub_prev_wave"] = _wave_selection

# ── PGA pool loading ─────────────────────────────────────────────────
_pool_loaded_key = f"_pga_hub_pool_{slate_date_str}_{_contest_safe}"
_already_loaded = st.session_state.get(_pool_loaded_key) is not None

st.caption("⛳ PGA pools are built from DataGolf (projections + strokes gained + course fit).")
if not _already_loaded:
    if st.button("🚀 Load PGA Pool", key="_pga_lab_load_go", type="primary"):
        with st.status("Ricky's loading the PGA pool…", expanded=True) as _load_status:
            pool_result = _load_pga_pool(
                slate_date_str=slate_date_str,
                contest_type_label=contest_type_label,
                preset=preset,
                slate=slate,
                sim=sim,
                _contest_safe=_contest_safe,
                status_container=_load_status,
            )
            if pool_result is not None:
                # Apply wave filter before edge metrics
                _wave = st.session_state.get("_pga_wave_filter", "All Players")
                if _wave != "All Players":
                    _pre_count = len(pool_result)
                    pool_result = _filter_pool_by_wave(pool_result, _wave)
                    _load_status.write(
                        f"🌊 {_wave}: {len(pool_result)} of {_pre_count} players"
                    )
                    # Update slate and cache with filtered pool
                    slate.player_pool = pool_result
                    set_slate_state(slate)
                    st.session_state[f"_pga_hub_pool_{slate_date_str}_{_contest_safe}"] = pool_result

                _load_status.write("Computing edge metrics…")
                try:
                    player_results = _build_player_level_sim_results(pool_result, sim.variance)
                    sim.player_results = player_results
                    _edge_df = compute_edge_metrics(
                        pool_result,
                        calibration_state=slate.calibration_state,
                        variance=sim.variance,
                        sport="PGA",
                    )
                    slate.edge_df = _edge_df
                    if "Edge" not in slate.active_layers:
                        slate.active_layers.append("Edge")
                    set_slate_state(slate)
                    set_sim_state(sim)
                    _load_status.write(f"✅ Pool loaded — {len(player_results)} players analyzed.")
                    # Auto-archive PGA slate snapshot
                    try:
                        from yak_core.slate_archive import archive_slate
                        archive_slate(
                            pool_result, slate_date_str,
                            contest_type=contest_type_label,
                            edge_df=_edge_df,
                        )
                        _load_status.write("💾 PGA slate archived for calibration.")
                    except Exception:
                        pass
                except Exception as _edge_exc:
                    _load_status.write(f"⚠️ Edge metrics failed: {_edge_exc}")
                _load_status.update(label="✅ PGA pool loaded", state="complete", expanded=False)
            else:
                _load_status.update(label="Load failed", state="error")
        st.rerun()
else:
    with st.status("✅ PGA pool loaded", state="complete", expanded=False):
        if st.button("🔄 Reload PGA Pool", key="_pga_lab_force_reload"):
            st.session_state.pop(_pool_loaded_key, None)
            st.rerun()

# ── Calibrate Past PGA Events ────────────────────────────────────────
with st.expander("📐 Calibrate Past Events", expanded=False):
    st.caption(
        "Backfill PGA calibration by comparing DataGolf projections "
        "to actual DK fantasy points from completed events."
    )
    _dg_key = (
        st.secrets.get("DATAGOLF_API_KEY")
        or os.environ.get("DATAGOLF_API_KEY")
        or "7e0b29081d2adaac7e3de0ed387c"
    )
    if not _dg_key:
        st.warning("DataGolf API key not configured.")
    else:
        from yak_core.datagolf import DataGolfClient as _DGC
        _cal_dg = _DGC(api_key=_dg_key)

        _evt_cache_key = "_pga_cal_event_list"
        if _evt_cache_key not in st.session_state:
            try:
                from yak_core.pga_calibration import get_pga_event_list
                st.session_state[_evt_cache_key] = get_pga_event_list(_cal_dg)
            except Exception as _evt_exc:
                st.error(f"Could not fetch event list: {_evt_exc}")
                st.session_state[_evt_cache_key] = pd.DataFrame()

        _evt_df = st.session_state[_evt_cache_key]
        if _evt_df.empty:
            st.info("No completed PGA events with DK data found.")
        else:
            try:
                from yak_core.calibration_feedback import _load_history as _lh
                _already_calibrated = set(_lh(sport="PGA").keys())
            except Exception:
                _already_calibrated = set()

            if _already_calibrated:
                _evt_df_filtered = _evt_df[
                    ~_evt_df["date"].astype(str).isin(_already_calibrated)
                ].reset_index(drop=True)
            else:
                _evt_df_filtered = _evt_df

            if _evt_df_filtered.empty:
                st.success("All available PGA events have been calibrated.")
            else:
                _evt_options = [
                    f"{row.get('event_name', 'Event')} ({row.get('date', '?')}) — ID {row['event_id']}"
                    for _, row in _evt_df_filtered.iterrows()
                ]
                _sel_idx = st.selectbox(
                    "Select event", range(len(_evt_options)),
                    format_func=lambda i: _evt_options[i],
                    key="_pga_cal_event_sel",
                )
                if st.button("Load & Calibrate", key="_pga_cal_go"):
                    _sel_row = _evt_df_filtered.iloc[_sel_idx]
                    _eid = int(_sel_row["event_id"])
                    _yr = int(_sel_row["calendar_year"])
                    _edate = str(_sel_row.get("date", ""))
                    with st.spinner(f"Calibrating event {_eid} ({_yr})…"):
                        try:
                            from yak_core.pga_calibration import calibrate_pga_event
                            _cal_result = calibrate_pga_event(
                                _cal_dg, event_id=_eid, year=_yr, slate_date=_edate,
                            )
                            if "error" in _cal_result:
                                st.warning(_cal_result["error"])
                            else:
                                _cal_mae = _cal_result.get("overall", {}).get("mae", "?")
                                _cal_n = _cal_result.get("n_players_calibrated", 0)
                                st.success(f"Calibrated {_cal_n} players — MAE: {_cal_mae}")
                                st.session_state.pop(_evt_cache_key, None)
                        except Exception as _cal_exc:
                            st.error(f"Calibration failed: {_cal_exc}")

            try:
                from yak_core.calibration_feedback import get_calibration_summary
                _pga_summary = get_calibration_summary(sport="PGA")
                if _pga_summary.get("n_slates", 0) > 0:
                    st.markdown(
                        f"**PGA calibration**: {_pga_summary['n_slates']} event(s), "
                        f"overall MAE {_pga_summary.get('overall_mae', '?'):.1f}"
                    )
            except Exception:
                pass

# ── Calibrate Past PGA Showdown Events ───────────────────────────────
with st.expander("📐 Calibrate Past Events (Showdown / Single-Round)", expanded=False):
    st.caption(
        "Backfill PGA **showdown** (single-round) calibration. "
        "Uses per-round projections and stores corrections under PGA_SD."
    )
    _sd_dg_key = (
        st.secrets.get("DATAGOLF_API_KEY")
        or os.environ.get("DATAGOLF_API_KEY")
        or "7e0b29081d2adaac7e3de0ed387c"
    )
    if not _sd_dg_key:
        st.warning("DataGolf API key not configured.")
    else:
        from yak_core.datagolf import DataGolfClient as _DGC_SD
        _sd_cal_dg = _DGC_SD(api_key=_sd_dg_key)

        _sd_evt_cache_key = "_pga_sd_cal_event_list"
        if _sd_evt_cache_key not in st.session_state:
            try:
                from yak_core.pga_calibration import get_pga_event_list
                st.session_state[_sd_evt_cache_key] = get_pga_event_list(_sd_cal_dg)
            except Exception as _sd_evt_exc:
                st.error(f"Could not fetch event list: {_sd_evt_exc}")
                st.session_state[_sd_evt_cache_key] = pd.DataFrame()

        _sd_evt_df = st.session_state[_sd_evt_cache_key]
        if _sd_evt_df.empty:
            st.info("No completed PGA events with DK data found.")
        else:
            try:
                from yak_core.calibration_feedback import _load_history as _lh_sd
                _sd_already_calibrated = set(_lh_sd(sport="PGA_SD").keys())
            except Exception:
                _sd_already_calibrated = set()

            if _sd_already_calibrated:
                _sd_evt_df_filtered = _sd_evt_df[
                    ~_sd_evt_df["date"].astype(str).isin(_sd_already_calibrated)
                ].reset_index(drop=True)
            else:
                _sd_evt_df_filtered = _sd_evt_df

            if _sd_evt_df_filtered.empty:
                st.success("All available PGA events have been calibrated for showdown.")
            else:
                _sd_evt_options = [
                    f"{row.get('event_name', 'Event')} ({row.get('date', '?')}) — ID {row['event_id']}"
                    for _, row in _sd_evt_df_filtered.iterrows()
                ]
                _sd_sel_idx = st.selectbox(
                    "Select event (Showdown)", range(len(_sd_evt_options)),
                    format_func=lambda i: _sd_evt_options[i],
                    key="_pga_sd_cal_event_sel",
                )
                if st.button("Load & Calibrate (Showdown)", key="_pga_sd_cal_go"):
                    _sd_sel_row = _sd_evt_df_filtered.iloc[_sd_sel_idx]
                    _sd_eid = int(_sd_sel_row["event_id"])
                    _sd_yr = int(_sd_sel_row["calendar_year"])
                    _sd_edate = str(_sd_sel_row.get("date", ""))
                    with st.spinner(f"Calibrating showdown event {_sd_eid} ({_sd_yr})…"):
                        try:
                            from yak_core.pga_calibration import calibrate_pga_showdown_event
                            _sd_cal_result = calibrate_pga_showdown_event(
                                _sd_cal_dg, event_id=_sd_eid, year=_sd_yr, slate_date=_sd_edate,
                            )
                            if "error" in _sd_cal_result:
                                st.warning(_sd_cal_result["error"])
                            else:
                                _sd_cal_mae = _sd_cal_result.get("overall", {}).get("mae", "?")
                                _sd_cal_n = _sd_cal_result.get("n_players_calibrated", 0)
                                st.success(f"Calibrated {_sd_cal_n} players (showdown) — MAE: {_sd_cal_mae}")
                                st.session_state.pop(_sd_evt_cache_key, None)
                        except Exception as _sd_cal_exc:
                            st.error(f"Showdown calibration failed: {_sd_cal_exc}")

            try:
                from yak_core.calibration_feedback import get_calibration_summary
                _pga_sd_summary = get_calibration_summary(sport="PGA_SD")
                if _pga_sd_summary.get("n_slates", 0) > 0:
                    st.markdown(
                        f"**PGA Showdown calibration**: {_pga_sd_summary['n_slates']} event(s), "
                        f"overall MAE {_pga_sd_summary.get('overall_mae', '?'):.1f}"
                    )
            except Exception:
                pass

st.divider()

# =========================================================================
# SECTION 2: SIMULATIONS
# =========================================================================
st.subheader("🎲 Simulations")

pool: pd.DataFrame = slate.player_pool if slate.player_pool is not None else pd.DataFrame()

# Show active wave filter status
if not pool.empty:
    _active_wave_sim = st.session_state.get("_pga_wave_filter", "All Players")
    if _active_wave_sim != "All Players":
        st.caption(f"🌊 **{_active_wave_sim}** active — {len(pool)} players in pool")

# Re-run edge metrics
if not pool.empty:
    if st.button("🔄 Re-run Edge Metrics", key="_pga_lab_run_sims"):
        with st.spinner("Recomputing edge metrics…"):
            try:
                player_results = _build_player_level_sim_results(pool, sim.variance)
                sim.player_results = player_results
                _edge_df = compute_edge_metrics(
                    pool,
                    calibration_state=slate.calibration_state,
                    variance=sim.variance,
                    sport="PGA",
                )
                slate.edge_df = _edge_df
                if "Edge" not in slate.active_layers:
                    slate.active_layers.append("Edge")
                set_slate_state(slate)
                set_sim_state(sim)
                st.success(f"Edge metrics updated — {len(player_results)} players analyzed.")
            except Exception as exc:
                st.error(f"Edge metrics failed: {exc}")
else:
    st.info("Load a PGA pool above first.")

# Sim results display
if sim.player_results is not None and not sim.player_results.empty:
    st.caption("Player-level smash / bust / leverage (sorted by leverage)")
    display_df = prepare_sims_table(sim.player_results)

    from yak_core.display_format import standard_player_format  # noqa: PLC0415
    _std_fmt = standard_player_format(display_df)

    def _style_row(row: pd.Series) -> list:
        styles = [""] * len(row)
        cols = list(row.index)
        if "smash_prob" in cols:
            idx = cols.index("smash_prob")
            styles[idx] = _color_smash(float(row["smash_prob"]))
        if "bust_prob" in cols:
            idx = cols.index("bust_prob")
            styles[idx] = _color_bust(float(row["bust_prob"]))
        return styles

    try:
        styled = display_df.style.apply(_style_row, axis=1).format(_std_fmt, na_rep="")
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Sim Sandbox ──────────────────────────────────────────────────────
with st.expander("🔬 Sim Sandbox", expanded=False):
    st.caption("Run PGA sims against archived actuals. See what's working, what's not, and apply fixes.")
    try:
        from yak_core.sim_sandbox import (
            run_sandbox, get_active_knobs, save_active_knobs,
            save_sandbox_run, get_sandbox_history,
        )

        _sb_knobs = get_active_knobs()
        st.markdown(
            f"**Active Knobs:** ceiling_boost = `{_sb_knobs['ceiling_boost']}` · "
            f"floor_dampen = `{_sb_knobs['floor_dampen']}`"
        )

        # PGA actuals backfill
        _pga_archives = [
            f for f in os.listdir(os.path.join(YAKOS_ROOT, "data", "slate_archive"))
            if f.startswith("pga_") or ("pga" in f.lower() and f.endswith(".parquet"))
        ] if os.path.isdir(os.path.join(YAKOS_ROOT, "data", "slate_archive")) else []
        _pga_need_actuals = []
        for _pf in _pga_archives:
            _pp = os.path.join(YAKOS_ROOT, "data", "slate_archive", _pf)
            if os.path.getsize(_pp) == 0:
                continue
            try:
                _pdf = pd.read_parquet(_pp)
                if "actual_fp" not in _pdf.columns or not _pdf["actual_fp"].notna().any():
                    _pga_need_actuals.append(_pf)
            except Exception:
                pass
        if _pga_need_actuals:
            st.caption(
                f"{len(_pga_need_actuals)} PGA archive(s) missing actuals. "
                "Backfill from DataGolf so the sandbox can score them."
            )
            if st.button("Backfill PGA Actuals", key="_pga_lab_sb_backfill"):
                _bf_dg_key = (
                    st.secrets.get("DATAGOLF_API_KEY")
                    or os.environ.get("DATAGOLF_API_KEY")
                    or "7e0b29081d2adaac7e3de0ed387c"
                )
                if _bf_dg_key:
                    from yak_core.datagolf import DataGolfClient as _BF_DGC
                    from yak_core.pga_calibration import (
                        get_pga_event_list as _bf_gel,
                        calibrate_pga_event as _bf_cal,
                    )
                    _bf_dg = _BF_DGC(api_key=_bf_dg_key)
                    with st.spinner("Fetching event list and backfilling actuals..."):
                        try:
                            _bf_events = _bf_gel(_bf_dg)
                            _bf_count = 0
                            for _, _ev in _bf_events.iterrows():
                                _ev_date = str(_ev.get("date", ""))
                                _matching = [f for f in _pga_need_actuals if _ev_date in f]
                                if _matching:
                                    _bf_cal(
                                        _bf_dg,
                                        event_id=int(_ev["event_id"]),
                                        year=int(_ev["calendar_year"]),
                                        slate_date=_ev_date,
                                    )
                                    _bf_count += 1
                            st.success(f"Backfilled actuals for {_bf_count} event(s). Run Sandbox now.")
                        except Exception as _bf_exc:
                            st.error(f"Backfill failed: {_bf_exc}")
                else:
                    st.warning("DataGolf API key not configured.")

        if st.button("Run Sandbox", key="_pga_lab_run_sandbox"):
            with st.spinner("Scoring PGA sims against archived slates..."):
                _sb_result = run_sandbox(knobs=_sb_knobs, sport="PGA")

            if "error" in _sb_result:
                st.error(_sb_result["error"])
            else:
                _sb_hist = get_sandbox_history()
                _sb_prev = _sb_hist[-1] if _sb_hist else None
                save_sandbox_run(_sb_result)
                st.session_state["_pga_sb_last_result"] = _sb_result
                st.session_state["_pga_sb_prev_result"] = _sb_prev

        # Display last result
        _sb_result = st.session_state.get("_pga_sb_last_result")
        _sb_prev = st.session_state.get("_pga_sb_prev_result")
        if _sb_result and "error" not in _sb_result:
            _sb_slates = [s["slate"] for s in _sb_result.get("per_slate", [])]
            if _sb_slates:
                import re as _re
                _date_parts = []
                for _sl in _sb_slates:
                    _m = _re.search(r"(\d{4}-\d{2}-\d{2})", _sl)
                    if _m:
                        _date_parts.append(_m.group(1))
                if _date_parts:
                    _date_parts.sort()
                    if _date_parts[0] == _date_parts[-1]:
                        st.caption(f"Slate: {_date_parts[0]}  ·  {len(_sb_slates)} slate(s) scored")
                    else:
                        st.caption(f"Slates: {_date_parts[0]} → {_date_parts[-1]}  ·  {len(_sb_slates)} slates scored")

            _k1, _k2, _k3, _k4, _k5 = st.columns(5)

            def _delta(key, fmt_pct=False, invert=False):
                if _sb_prev is None or key not in _sb_prev:
                    return None
                diff = _sb_result[key] - _sb_prev[key]
                if abs(diff) < 0.001:
                    return None
                if fmt_pct:
                    return f"{diff*100:+.0f}%"
                return f"{diff:+.1f}"

            _targets = {"avg_mae": 6.0, "avg_smash_precision": 0.25, "avg_bust_precision": 0.40, "avg_coverage": 0.80}

            def _help(key):
                t = _targets.get(key)
                if t is None:
                    return None
                if key == "avg_mae":
                    return f"target: {t:.0f} FP"
                return f"target: {t*100:.0f}%"

            _k1.metric("MAE", f"{_sb_result['avg_mae']:.1f} FP",
                       delta=_delta("avg_mae"), delta_color="inverse",
                       help=_help("avg_mae"))
            _k2.metric("Smash Prec", f"{_sb_result['avg_smash_precision']*100:.0f}%",
                       delta=_delta("avg_smash_precision", fmt_pct=True),
                       help=_help("avg_smash_precision"))
            _k3.metric("Bust Prec", f"{_sb_result['avg_bust_precision']*100:.0f}%",
                       delta=_delta("avg_bust_precision", fmt_pct=True),
                       help=_help("avg_bust_precision"))
            _k4.metric("Coverage", f"{_sb_result['avg_coverage']*100:.0f}%",
                       delta=_delta("avg_coverage", fmt_pct=True),
                       help=_help("avg_coverage"))
            _k5.metric("Slates", _sb_result['n_slates'])

            _sb_smashes = _sb_result.get("top_smashes", [])
            _sb_busts = _sb_result.get("worst_busts", [])
            if _sb_smashes or _sb_busts:
                _sbc1, _sbc2 = st.columns(2)
                with _sbc1:
                    if _sb_smashes:
                        st.markdown("**Top Smashes**")
                        _sm_df = pd.DataFrame(_sb_smashes)
                        _sm_show = [c for c in ["player", "salary", "proj", "actual", "diff"] if c in _sm_df.columns]
                        _sm_fmt = {"salary": "${:,.0f}", "proj": "{:.1f}", "actual": "{:.1f}", "diff": "{:+.1f}"}
                        st.dataframe(
                            _sm_df[_sm_show].style.format({k: v for k, v in _sm_fmt.items() if k in _sm_show}, na_rep=""),
                            use_container_width=True, hide_index=True,
                        )
                with _sbc2:
                    if _sb_busts:
                        st.markdown("**Worst Busts**")
                        _bu_df = pd.DataFrame(_sb_busts)
                        _bu_show = [c for c in ["player", "salary", "proj", "actual", "diff"] if c in _bu_df.columns]
                        _bu_fmt = {"salary": "${:,.0f}", "proj": "{:.1f}", "actual": "{:.1f}", "diff": "{:+.1f}"}
                        st.dataframe(
                            _bu_df[_bu_show].style.format({k: v for k, v in _bu_fmt.items() if k in _bu_show}, na_rep=""),
                            use_container_width=True, hide_index=True,
                        )

            _sb_breakouts = _sb_result.get("breakouts", [])
            if _sb_breakouts:
                st.markdown("**Breakouts** (beat ceiling or 5x+ value)")
                _bo_df = pd.DataFrame(_sb_breakouts)
                _bo_show = [c for c in ["player", "salary", "proj", "actual_fp", "ceil", "reasons"] if c in _bo_df.columns]
                _bo_fmt = {"salary": "${:,.0f}", "proj": "{:.1f}", "actual_fp": "{:.1f}", "ceil": "{:.1f}"}
                st.dataframe(
                    _bo_df[_bo_show].style.format({k: v for k, v in _bo_fmt.items() if k in _bo_show}, na_rep=""),
                    use_container_width=True, hide_index=True,
                )

            _sb_rec = _sb_result.get("recommendations", {})
            if _sb_rec:
                st.markdown("---")
                st.markdown("**Recommendations**")
                for _r in _sb_rec.get("reasons", []):
                    st.markdown(f"- {_r}")

                if _sb_rec.get("changed"):
                    _new = _sb_rec["recommended"]
                    st.markdown(
                        f"Suggested: ceiling_boost = `{_new['ceiling_boost']}` · "
                        f"floor_dampen = `{_new['floor_dampen']}`"
                    )
                    if st.button("Apply Recommended Knobs", key="_pga_lab_apply_sb_knobs"):
                        save_active_knobs(_new)
                        st.success(
                            f"Applied: ceiling_boost={_new['ceiling_boost']}, "
                            f"floor_dampen={_new['floor_dampen']}"
                        )
                else:
                    st.info("No knob changes recommended — current config is solid.")

            _sb_per_slate = _sb_result.get("per_slate", [])
            if _sb_per_slate:
                with st.expander("Per-Slate Breakdown", expanded=False):
                    _ps_df = pd.DataFrame(_sb_per_slate)
                    _ps_fmt = {"mae": "{:.1f}", "coverage": "{:.0%}", "smash_precision": "{:.0%}", "bust_precision": "{:.0%}"}
                    st.dataframe(
                        _ps_df.style.format({k: v for k, v in _ps_fmt.items() if k in _ps_df.columns}, na_rep=""),
                        use_container_width=True, hide_index=True,
                    )

    except Exception as _sb_exc:
        st.warning(f"Sim Sandbox unavailable: {_sb_exc}")

st.divider()

# =========================================================================
# SECTION 3: EDGE ANALYSIS
# =========================================================================
st.subheader("📊 Edge Analysis")

if not pool.empty and sim.player_results is not None and not sim.player_results.empty:
    pr = sim.player_results.copy()

    from yak_core.display_format import standard_player_format  # noqa: PLC0415

    if "ownership" in pr.columns and "own_pct" not in pr.columns:
        pr = pr.rename(columns={"ownership": "own_pct"})

    pos_edge = pr[pr["leverage"] > 1.2].nlargest(5, "leverage") if "leverage" in pr.columns else pd.DataFrame()
    neg_edge = pr[pr["leverage"] < 0.7].nsmallest(5, "leverage") if "leverage" in pr.columns else pd.DataFrame()

    ea_col1, ea_col2 = st.columns(2)
    with ea_col1:
        st.markdown("**Positive Leverage** — high smash, low owned")
        if not pos_edge.empty:
            _pe = pos_edge[[c for c in ["player_name", "salary", "own_pct", "smash_prob", "leverage"] if c in pos_edge.columns]].copy()
            _pe_fmt = standard_player_format(_pe)
            st.dataframe(_pe.style.format(_pe_fmt, na_rep=""), use_container_width=True, hide_index=True)
        else:
            st.caption("No high-leverage plays found.")

    with ea_col2:
        st.markdown("**Negative Leverage** — bust risk, over-owned")
        if not neg_edge.empty:
            _ne = neg_edge[[c for c in ["player_name", "salary", "own_pct", "bust_prob", "leverage"] if c in neg_edge.columns]].copy()
            _ne_fmt = standard_player_format(_ne)
            st.dataframe(_ne.style.format(_ne_fmt, na_rep=""), use_container_width=True, hide_index=True)
        else:
            st.caption("No over-owned bust risks found.")

    with st.expander("💰 Value Plays & Stacks", expanded=False):
        _MIN_VALUE_SALARY = 4000
        st.markdown(f"**Value Plays** (salary ≥ ${_MIN_VALUE_SALARY:,})")
        try:
            val_scores = compute_value_scores(pool)
            if not val_scores.empty:
                if "salary" in val_scores.columns:
                    val_scores = val_scores[
                        pd.to_numeric(val_scores["salary"], errors="coerce").fillna(0) >= _MIN_VALUE_SALARY
                    ]
                top_val = val_scores.nlargest(5, "value_score") if "value_score" in val_scores.columns else val_scores.head(5)
                show_cols = [c for c in ["player_name", "team", "salary", "proj", "value_score"] if c in top_val.columns]
                _vd = top_val[show_cols].copy()
                if "value_score" in _vd.columns:
                    _vd["value_score"] = _vd["value_score"].round(2)
                if "proj" in _vd.columns:
                    _vd["proj"] = _vd["proj"].round(1)
                st.dataframe(_vd, use_container_width=True, hide_index=True)
            else:
                st.caption("No value scores available.")
        except Exception as exc:
            st.caption(f"Value scores unavailable: {exc}")

        st.markdown("**Top Stacks**")
        try:
            stack_scores = compute_stack_scores(pool)
            if not stack_scores.empty:
                show_cols = [c for c in ["team", "stack_score"] if c in stack_scores.columns]
                st.dataframe(stack_scores[show_cols].head(6), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.caption(f"Stack scores unavailable: {exc}")
elif not pool.empty:
    st.info("Run sims first to populate edge analysis.")

st.divider()

# =========================================================================
# SECTION 4: PGA PROJECTIONS VS ACTUALS
# =========================================================================
st.subheader("📊 PGA Projections vs Actuals")
st.caption(
    "Compare DataGolf-based projections against actual DK fantasy points "
    "from calibrated events. Shows how the model performs by salary tier."
)
try:
    from yak_core.calibration_feedback import _load_history as _pva_load
    _pva_history = _pva_load(sport="PGA")
    if not _pva_history:
        st.info("No PGA calibration data yet. Calibrate past events above to see proj vs actuals.")
    else:
        _pva_dates = sorted(_pva_history.keys(), reverse=True)

        _pva_rows = []
        for _d in _pva_dates:
            _rec = _pva_history[_d]
            _ov = _rec.get("overall", {})
            _pva_rows.append({
                "Event Date": _d,
                "Players": _ov.get("n_players", 0),
                "Mean Error": _ov.get("mean_error", 0),
                "MAE": _ov.get("mae", 0),
                "RMSE": _ov.get("rmse", 0),
                "Correlation": _ov.get("correlation", 0),
            })
        _pva_df = pd.DataFrame(_pva_rows)

        _pk1, _pk2, _pk3, _pk4 = st.columns(4)
        _avg_mae = _pva_df["MAE"].mean()
        _avg_corr = _pva_df["Correlation"].mean()
        _avg_bias = _pva_df["Mean Error"].mean()
        _pk1.metric("Events", len(_pva_dates))
        _pk2.metric("Avg MAE", f"{_avg_mae:.1f} FP")
        _pk3.metric("Avg Bias", f"{_avg_bias:+.1f} FP",
                    help="Negative = model over-projects")
        _pk4.metric("Avg Corr", f"{_avg_corr:.3f}",
                    help="Projection-to-actual correlation (higher is better)")

        _pva_fmt = {
            "Mean Error": "{:+.1f}",
            "MAE": "{:.1f}",
            "RMSE": "{:.1f}",
            "Correlation": "{:.3f}",
        }
        st.dataframe(
            _pva_df.style.format(_pva_fmt),
            use_container_width=True, hide_index=True,
        )

        with st.expander("Accuracy by Salary Tier", expanded=False):
            _tier_accum = {}
            for _d in _pva_dates:
                for _tier, _stats in _pva_history[_d].get("by_salary_tier", {}).items():
                    if _tier not in _tier_accum:
                        _tier_accum[_tier] = {"errors": [], "maes": [], "n": 0}
                    _tier_accum[_tier]["errors"].append(_stats["mean_error"])
                    _tier_accum[_tier]["maes"].append(_stats["mae"])
                    _tier_accum[_tier]["n"] += _stats["n"]

            if _tier_accum:
                _tier_rows = []
                for _t, _a in sorted(_tier_accum.items()):
                    _tier_rows.append({
                        "Salary Tier": _t,
                        "Avg Bias": round(float(np.mean(_a["errors"])), 1),
                        "Avg MAE": round(float(np.mean(_a["maes"])), 1),
                        "Total Players": _a["n"],
                    })
                _tier_df = pd.DataFrame(_tier_rows)
                _tier_fmt = {"Avg Bias": "{:+.1f}", "Avg MAE": "{:.1f}"}
                st.dataframe(
                    _tier_df.style.format(_tier_fmt),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("No salary tier data available.")

except Exception as _pva_exc:
    st.warning(f"PGA Projections vs Actuals unavailable: {_pva_exc}")
