"""YakOS DFS Optimizer - Ricky's Slate Room + Optimizer + Calibration Lab."""

import hmac
import json
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# Default timezone — all "today" references use US/Eastern (EST/EDT).
_EST = ZoneInfo("America/New_York")


def _today_est():
    """Return today's date in US/Eastern (EST/EDT)."""
    return pd.Timestamp.now(tz=_EST).date()


# ── Persistent API-key helpers ──────────────────────────────────────────────
_API_CONFIG_PATH = Path(__file__).parent / "data" / "api_config.json"


def _load_persisted_api_key() -> str:
    """Return the Tank01 RapidAPI key saved in data/api_config.json, or ''."""
    if _API_CONFIG_PATH.exists():
        try:
            with open(_API_CONFIG_PATH, "r") as _f:
                return json.load(_f).get("rapidapi_key", "")
        except Exception:
            pass
    return ""


def _save_persisted_api_key(key: str) -> None:
    """Write the Tank01 RapidAPI key to data/api_config.json."""
    _API_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_API_CONFIG_PATH, "w") as _f:
        json.dump({"rapidapi_key": key}, _f)


# Make yak_core importable when running from the repo root (e.g. Streamlit Cloud)
_repo_root = str(Path(__file__).parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.lineups import build_multiple_lineups_with_exposure, to_dk_upload_format, build_showdown_lineups, to_dk_showdown_upload_format  # type: ignore
from yak_core.calibration import (  # type: ignore
    run_backtest_lineups,
    compute_calibration_metrics,
    identify_calibration_gaps,
    load_calibration_config,
    save_calibration_config,
    DFS_ARCHETYPES,
    DK_CONTEST_TYPES,
    DK_CONTEST_TYPE_MAP,
    apply_archetype,
    get_calibration_queue,
    action_queue_items,
    suggest_config_from_queue,
    build_approved_lineups,
    get_approved_lineups_by_archetype,
    compute_slate_kpis,
    BACKTEST_ARCHETYPES,
    run_archetype_backtest,
)
from yak_core.right_angle import (  # type: ignore
    ricky_annotate,
    detect_stack_alerts,
    detect_pace_environment,
    detect_high_value_plays,
    compute_stack_scores,
    compute_value_scores,
    compute_tiered_stack_alerts,
    compute_game_environment_cards,
)
from yak_core.sims import (  # type: ignore
    run_monte_carlo_for_lineups,
    simulate_live_updates,
    build_sim_player_accuracy_table,
    compute_player_anomaly_table,
    compute_sim_eligible,
    ContestType as _SimContestType,
)
from yak_core.live import (  # type: ignore
    fetch_live_opt_pool,
    fetch_injury_updates,
    fetch_actuals_from_api,
    NoGamesScheduledError,
    apply_manual_injury_overrides_to_pool,
)
from yak_core.multislate import (  # type: ignore
    parse_dk_contest_csv,
    discover_slates,
    run_multi_slate,
    compare_slates,
)
from yak_core.projections import (  # type: ignore
    salary_implied_proj,
    noisy_proj,
    yakos_fp_projection,
    yakos_minutes_projection,
    yakos_ownership_projection,
    yakos_ensemble,
)
from yak_core.scoring import calibration_kpi_summary, quality_color, _QUALITY_BG, _QUALITY_TEXT  # type: ignore
from yak_core.config import CONTEST_PRESETS, CONTEST_PRESET_LABELS, CONTEST_PRESET_ARCH_LABELS  # type: ignore
from yak_core.ownership import apply_ownership, apply_ownership_pipeline  # type: ignore
from yak_core.injury_cascade import apply_injury_cascade  # type: ignore
from yak_core.dvp import (  # type: ignore
    parse_dvp_upload,
    save_dvp_table,
    load_dvp_table,
    dvp_staleness_days,
    compute_league_averages,
    DVP_STALE_DAYS,
    DVP_DEFAULT_PATH as _DVP_DEFAULT_PATH,
)
from yak_core.alert_backtest import (  # type: ignore
    run_alert_backtest,
    score_stack_alerts,
    score_high_value_alerts,
    score_injury_cascade_alerts,
    score_game_environment_alerts,
    aggregate_alert_metrics,
    compute_overall_edge,
    tune_alert_thresholds,
    load_backtest,
    list_backtest_slates,
    DEFAULT_ALERT_THRESHOLDS,
)

# Map internal DK contest type string -> ContestType enum.
# Keys match values produced by DK_CONTEST_TYPE_MAP in yak_core/calibration.py.
# Used when passing contest_type to run_monte_carlo_for_lineups().
_INTERNAL_CT_TO_SIM_TYPE: dict = {
    "50/50": _SimContestType.CASH,
    "Single Entry": _SimContestType.SE_SMALL,
}


# -----------------------------
# Core helpers
# -----------------------------


def rename_rg_columns_to_yakos(df: pd.DataFrame) -> pd.DataFrame:
    """Map RotoGrinders NBA CSV columns to YakOS schema.

    Handles both the standard RG export (PLAYERID, PLAYER, SALARY, …) and
    the older friendly-header format (Name, Position, Salary, …).
    """
    col_map = {
        # Raw RG export headers
        "PLAYERID": "player_id",
        "PLAYER": "player_name",
        "SALARY": "salary",
        "POS": "pos",
        "TEAM": "team",
        "OPP": "opponent",
        "FPTS": "proj",
        "OWNERSHIP": "ownership",
        "POWN": "proj_own",
        "MINUTES": "minutes",
        "FLOOR": "floor",
        "CEIL": "ceil",
        "SIM85TH": "sim85",
        "INJURY": "status",
        # Friendly-header aliases (RG web export)
        "Name": "player_name",
        "Position": "pos",
        "Team": "team",
        "Opponent": "opponent",
        "Projection": "proj",
        "Ceiling": "ceil",
        "Floor": "floor",
    }
    out = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure player_id: fall back to player_name if not present
    if "player_id" not in out.columns:
        if "player_name" in out.columns:
            out["player_id"] = out["player_name"].astype(str)
        else:
            out["player_id"] = out.index.astype(str)

    # Ensure player_name
    if "player_name" not in out.columns:
        out["player_name"] = out["player_id"].astype(str)

    # Ensure required columns exist with defaults
    required_defaults = {"pos": "", "team": "", "opponent": "", "salary": 0, "proj": 0}
    for col, default in required_defaults.items():
        if col not in out.columns:
            out[col] = default

    # Standardize numeric dtypes
    out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
    for c in ["proj", "ceil", "floor", "sim85", "ownership", "minutes"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Clean ownership: strip trailing % (RG exports as "28.5%") then convert to float
    if "ownership" in out.columns:
        out["ownership"] = (
            out["ownership"].astype(str).str.replace("%", "", regex=False).pipe(pd.to_numeric, errors="coerce")
        )

    # Clean proj_own: strip trailing % (RG exports as "1.64%") then convert to float
    if "proj_own" in out.columns:
        out["proj_own"] = (
            out["proj_own"].astype(str).str.replace("%", "", regex=False).pipe(pd.to_numeric, errors="coerce")
        )
        # Mirror POWN into ext_own — this is the raw site ownership label (v1 external ownership)
        _ext_vals = out["proj_own"].copy()
        if _ext_vals.notna().any() and (_ext_vals > 0).any():
            out["ext_own"] = _ext_vals

    # Drop rows with missing salary or projection
    out = out.dropna(subset=["salary", "proj"])
    out = out[out["salary"] > 0]

    # Compute default sim_eligible based on status and projected minutes
    out = compute_sim_eligible(out)

    return out


def rename_rg_raw_to_yakos(df: pd.DataFrame) -> pd.DataFrame:
    """Alias kept for backwards compatibility; delegates to rename_rg_columns_to_yakos."""
    return rename_rg_columns_to_yakos(df)


def apply_slate_filters(
    pool: pd.DataFrame,
    slate_type: str,
    showdown_game: str | None,
) -> pd.DataFrame:
    if slate_type == "Classic":
        return pool

    if slate_type == "Showdown Captain" and showdown_game:
        home, away = showdown_game.split(" @ ")
        mask = ((pool["team"] == home) & (pool["opponent"] == away)) | (
            (pool["team"] == away) & (pool["opponent"] == home)
        )
        return pool[mask].copy()

    return pool


def get_showdown_games(pool: pd.DataFrame) -> list[str]:
    if pool.empty:
        return []
    games = set()
    for _, row in pool.iterrows():
        t, o = row.get("team"), row.get("opponent")
        if pd.isna(t) or pd.isna(o):
            continue
        if f"{t} @ {o}" not in games and f"{o} @ {t}" not in games:
            games.add(f"{t} @ {o}")
    return sorted(games)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# Percentage projection boost applied to high-smash players when
# "Apply sim learnings" is clicked.  Kept small (5 %) to avoid over-fitting
# to a single sim run.
_SIM_LEARNING_BOOST = 0.05

# DK NBA lineup score thresholds used in the sim vs actuals comparison table.
# Cash line: typical 50/50 / double-up cash threshold for an 8-man DK NBA slate.
# GPP line: approx 99th-percentile score required to finish in the top 1% of a GPP.
_SIM_CASH_LINE = 280.0
_SIM_GPP_LINE = 340.0

# Ordered list of projection style options used in the Optimizer selectbox.
_PROJ_STYLE_OPTIONS = ["proj", "floor", "ceil", "sim85"]

# Map internal contest type -> suggested default projection style.
# Internal types come from DK_CONTEST_TYPE_MAP in yak_core/calibration.py, e.g.:
#   "Double Up (50/50)" -> "50/50" -> "floor"  (cash/variance-minimizing)
#   "Tournament (GPP)"  -> "GPP"   -> "ceil"   (upside/ceiling-chasing)
# Any unlisted internal type falls back to "proj".
_CONTEST_PROJ_DEFAULTS: dict[str, str] = {
    "50/50": "floor",       # cash game — minimize variance
    "GPP": "ceil",          # tournament — maximize ceiling
    "MME": "ceil",          # multi-entry max — ceiling-driven
    "Captain": "ceil",      # showdown captain — high-upside picks
    "Single Entry": "proj", # single entry — balanced default
}


def _default_proj_style_for_contest(internal_contest: str) -> str:
    """Return the suggested default projection style for the given internal contest type."""
    return _CONTEST_PROJ_DEFAULTS.get(internal_contest, "proj")


@dataclass
class HistoricalSlateBundle:
    """Bundles together a player pool and its actuals for a single past slate date.

    Attributes:
        slate_date: ISO date string (e.g. "2026-03-01").
        pool_df: Cleaned player pool DataFrame including projection columns.
        actuals: DataFrame with columns ``player_name`` and ``actual_fp``
            (and optionally ``actual_minutes``).  ``None`` when actuals have not
            yet been fetched.
        proj_col: The projection column used by the optimizer / sim (usually
            ``"proj"``).
    """

    slate_date: str
    pool_df: pd.DataFrame
    actuals: Optional[pd.DataFrame] = None
    proj_col: str = "proj"

    def has_valid_actuals(self) -> bool:
        """Return True when actuals are present and non-empty."""
        return self.actuals is not None and not self.actuals.empty


def load_historical_actuals(slate_date_str: str) -> Optional[pd.DataFrame]:
    """Return actuals for *slate_date_str* from session state, or ``None``.

    Lookup order:
    1. ``st.session_state["historical_bundles"][slate_date_str].actuals``
       (populated when the user clicked *Fetch Pool from API* for a past date).
    2. ``st.session_state["actuals"][slate_date_str]`` (legacy dict).

    Returns ``None`` (not an empty DataFrame) when no actuals are available so
    callers can distinguish "not loaded" from "loaded but empty".
    """
    bundles: dict = st.session_state.get("historical_bundles", {})
    bundle = bundles.get(slate_date_str)
    if bundle is not None and bundle.has_valid_actuals():
        return bundle.actuals

    legacy: dict = st.session_state.get("actuals", {})
    acts = legacy.get(slate_date_str)
    if acts is not None and isinstance(acts, pd.DataFrame) and not acts.empty:
        return acts

    return None


def _slate_value_leader(pool_df: pd.DataFrame) -> str:
    """Return a formatted string naming the top value play (FP per $1K) in the pool."""
    if pool_df.empty:
        return "—"
    df = pool_df[pool_df["salary"] > 0].copy()
    df["_v"] = df["proj"] / (df["salary"] / 1000.0)
    top = df.nlargest(1, "_v")
    if top.empty:
        return "—"
    row = top.iloc[0]
    return f"{row['player_name']} ({row['_v']:.2f}x)"


def _apply_proj_fallback(pool: pd.DataFrame) -> pd.DataFrame:
    """Apply salary-implied projections if all proj values are zero or missing."""
    if "proj" not in pool.columns or pool["proj"].fillna(0).max() == 0:
        pool = pool.copy()
        pool["proj"] = noisy_proj(salary_implied_proj(pool["salary"]))
    return pool


# Default constants used by _apply_yakos_projections for b2b / blowout adjustments
_DEFAULT_B2B_DISCOUNT = 0.93
_DEFAULT_BLOWOUT_THRESHOLD = 15
_BLOWOUT_REDUCTION_STRONG = 0.90
_BLOWOUT_REDUCTION_MILD = 0.95


def _apply_yakos_projections(pool: pd.DataFrame, knobs: dict = None) -> pd.DataFrame:
    """Apply the YakOS projection engine to a player pool.

    Calls ``yakos_fp_projection`` per player (with whatever signals are
    available), blends the result with any existing Tank01/RG projections via
    ``yakos_ensemble``, and records the signal mix in a ``proj_source`` column.
    Also populates ``floor``, ``ceil``, ``proj_minutes``, and ``proj_own``
    when they are not already present.

    Parameters
    ----------
    pool : pd.DataFrame
        Player pool to project.
    knobs : dict, optional
        Calibration knobs from ``st.session_state["cal_knobs"]``.  Supported
        keys: ``ensemble_w_yakos``, ``ensemble_w_tank01``, ``ensemble_w_rg``
        (ensemble weights), ``b2b_discount`` (override default 0.93 b2b
        discount), ``blowout_threshold`` (spread threshold for blowout
        reduction).  When *None* or absent, existing defaults are used.
    """
    if pool is None or pool.empty:
        return pool

    knobs = knobs or {}

    pool = pool.copy()

    # Build ensemble weights dict from knobs (falls back to yakos_ensemble defaults)
    _ens_weights: dict = {}
    if knobs.get("ensemble_w_yakos") is not None:
        _ens_weights["yakos"] = float(knobs["ensemble_w_yakos"])
    if knobs.get("ensemble_w_tank01") is not None:
        _ens_weights["tank01"] = float(knobs["ensemble_w_tank01"])
    if knobs.get("ensemble_w_rg") is not None:
        _ens_weights["rg"] = float(knobs["ensemble_w_rg"])
    _ensemble_weights = _ens_weights if _ens_weights else None

    # b2b discount override: ratio to apply on top of the default multiplier
    _b2b_override = float(knobs["b2b_discount"]) if knobs.get("b2b_discount") is not None else None
    _blowout_threshold = int(knobs["blowout_threshold"]) if knobs.get("blowout_threshold") is not None else _DEFAULT_BLOWOUT_THRESHOLD

    proj_values: list = []
    floor_values: list = []
    ceil_values: list = []
    proj_minutes_values: list = []
    proj_own_values: list = []
    proj_source_values: list = []

    for _, row in pool.iterrows():
        salary = float(row.get("salary") or 0)

        # Tank01 projection already stored in "proj" (may be 0 if missing)
        _raw_tank01 = row.get("proj")
        tank01_proj = float(_raw_tank01) if _raw_tank01 is not None and pd.notna(_raw_tank01) and float(_raw_tank01) > 0 else None

        # RotoGrinders projection stored separately when available
        _raw_rg = row.get("proj_rg")
        rg_proj = float(_raw_rg) if _raw_rg is not None and pd.notna(_raw_rg) and float(_raw_rg) > 0 else None

        rg_ownership = row.get("ownership") if pd.notna(row.get("ownership", float("nan"))) else None

        player_features: dict = {"salary": salary}
        if tank01_proj is not None:
            player_features["tank01_proj"] = tank01_proj
        if rg_proj is not None:
            player_features["rg_proj"] = rg_proj
        if rg_ownership is not None:
            player_features["rg_ownership"] = rg_ownership

        # Include rolling/contextual columns if present
        for col in (
            "rolling_fp_5", "rolling_fp_10", "rolling_fp_20",
            "rolling_min_5", "rolling_min_10", "rolling_min_20",
            "dvp", "vegas_total", "spread", "b2b", "home", "rest_days",
        ):
            val = row.get(col)
            if val is not None and pd.notna(val):
                player_features[col] = val

        # FP projection
        fp_result = yakos_fp_projection(player_features)
        yakos_proj_val = fp_result["proj"]

        # Minutes projection (uses yakos_minutes_projection base calculation,
        # then applies b2b_discount knob scaling if provided)
        min_result = yakos_minutes_projection(player_features)
        proj_min = min_result["proj_minutes"]

        # Apply b2b_discount knob override: scale relative to the built-in default
        if _b2b_override is not None and player_features.get("b2b"):
            # yakos_minutes_projection already multiplied by _DEFAULT_B2B_DISCOUNT; scale to knob value
            proj_min = proj_min * (_b2b_override / _DEFAULT_B2B_DISCOUNT)

        # Apply blowout_threshold from knobs (overrides hardcoded threshold)
        # We re-apply the blowout logic using the knob threshold.  The base minutes
        # already have the default thresholds applied by yakos_minutes_projection, so
        # we only need to act when the knob threshold differs from the defaults.
        _abs_spread = 0.0
        try:
            _abs_spread = abs(float(player_features.get("spread", 0.0)))
        except (ValueError, TypeError):
            pass
        if _blowout_threshold != _DEFAULT_BLOWOUT_THRESHOLD and _abs_spread > 0:
            # Remove default blowout adjustment and reapply with knob threshold
            if _abs_spread >= _DEFAULT_BLOWOUT_THRESHOLD:
                proj_min /= _BLOWOUT_REDUCTION_STRONG  # undo default strong reduction
            elif _abs_spread >= (_DEFAULT_BLOWOUT_THRESHOLD - 5):
                proj_min /= _BLOWOUT_REDUCTION_MILD    # undo default mild reduction
            if _abs_spread >= _blowout_threshold:
                proj_min *= _BLOWOUT_REDUCTION_STRONG
            elif _abs_spread >= (_blowout_threshold - 5):
                proj_min *= _BLOWOUT_REDUCTION_MILD

        proj_min = round(max(0.0, proj_min), 1)

        # Ensemble blend: YakOS + Tank01 + RG (using knob weights when provided)
        blended = yakos_ensemble(yakos_proj_val, tank01_proj, rg_proj, weights=_ensemble_weights)
        final_proj = blended if blended > 0 else yakos_proj_val

        # Ownership projection (uses final blended proj as input)
        own_features = {**player_features, "proj": final_proj}
        own_result = yakos_ownership_projection(own_features)

        # Build proj_source label from which signal sources were available
        signal_sources: list = []
        if any(k in player_features for k in ("rolling_fp_5", "rolling_fp_10", "rolling_fp_20")):
            signal_sources.append("rolling")
        if tank01_proj is not None:
            signal_sources.append("tank01")
        if rg_proj is not None:
            signal_sources.append("rg")
        if not signal_sources:
            signal_sources.append("salary")

        proj_values.append(final_proj)
        floor_values.append(fp_result["floor"])
        ceil_values.append(fp_result["ceil"])
        proj_minutes_values.append(proj_min)
        proj_own_values.append(own_result["proj_own"])
        proj_source_values.append("+".join(signal_sources))

    pool["proj"] = proj_values
    # Only overwrite floor/ceil if they are absent or all-zero
    if "floor" not in pool.columns or pool["floor"].fillna(0).max() == 0:
        pool["floor"] = floor_values
    if "ceil" not in pool.columns or pool["ceil"].fillna(0).max() == 0:
        pool["ceil"] = ceil_values
    # proj_minutes: fill in when absent
    if "proj_minutes" not in pool.columns:
        pool["proj_minutes"] = proj_minutes_values
    # proj_own: fill in when absent or all-zero
    if "proj_own" not in pool.columns or pool["proj_own"].fillna(0).max() == 0:
        pool["proj_own"] = proj_own_values
    pool["proj_source"] = proj_source_values

    # --- Three-layer ownership pipeline ---
    # own_model: supervised GBM prediction; own_proj: blended with ext_own
    try:
        pool = apply_ownership_pipeline(pool)
    except Exception as _e:
        # Graceful fallback: own_proj = proj_own when pipeline fails
        import traceback; traceback.print_exc()
        if "own_proj" not in pool.columns:
            pool["own_proj"] = pool["proj_own"] if "proj_own" in pool.columns else 0.0

    return pool


def _refresh_injury_statuses(
    pool_df: pd.DataFrame, api_key: str
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch the latest Tank01 injury updates and apply them to *pool_df*.

    Returns the updated pool and a list of ``"Name: old → new"`` change strings.
    Safe to call unconditionally — silently no-ops when *api_key* is blank or
    the API call fails.
    """
    if not api_key:
        return pool_df, []
    from yak_core.sims import _INELIGIBLE_STATUSES as _INELIG_RF

    try:
        updates = fetch_injury_updates(
            _today_est().strftime("%Y%m%d"),
            {"RAPIDAPI_KEY": api_key},
        )
    except Exception:
        return pool_df, []

    pool = pool_df.copy()
    changes: list[str] = []
    if updates and "player_name" in pool.columns:
        for upd in updates:
            p_name = upd.get("player_name", "")
            new_status = upd.get("status", "")
            if not p_name or not new_status:
                continue
            mask = pool["player_name"] == p_name
            if not mask.any():
                continue
            old_status = (
                pool.loc[mask, "status"].iloc[0]
                if "status" in pool.columns
                else "Active"
            )
            if str(old_status) != str(new_status):
                pool.loc[mask, "status"] = new_status
                changes.append(f"{p_name}: {old_status} → {new_status}")

    # Apply manual injury overrides (always win over Tank01 status)
    pool = apply_manual_injury_overrides_to_pool(pool)

    # Drop any player whose status is ineligible so they never appear in sims
    if "status" in pool.columns:
        inelig = pool["status"].fillna("").str.upper().isin(_INELIG_RF)
        if inelig.any():
            for _p in pool.loc[inelig, "player_name"].tolist():
                changes.append(f"{_p}: removed (ineligible status)")
            pool = pool[~inelig].reset_index(drop=True)

    # Keep sim_player_pool_clean in sync
    pool = _add_injury_columns(pool)
    st.session_state["sim_player_pool_clean"] = pool[~pool["is_out"]].reset_index(drop=True) if "is_out" in pool.columns else pool

    return pool, changes


def _add_injury_columns(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``injury_status`` and ``is_out`` columns derived from ``status``.

    * ``injury_status`` = ``"Out"`` when status is OUT/IR/DND/etc.,
      ``"Day-To-Day"`` for GTD/Q/Questionable, ``"Healthy"`` otherwise.
    * ``is_out`` = ``True`` when ``injury_status == "Out"``.
    """
    from yak_core.sims import _INELIGIBLE_STATUSES as _INELIG_S
    _GTD_STATUSES = {"GTD", "Q", "QUESTIONABLE", "DAY-TO-DAY", "DTD", "PROBABLE"}
    pool = pool_df.copy()
    if "status" not in pool.columns:
        pool["injury_status"] = "Healthy"
        pool["is_out"] = False
        return pool
    norm = pool["status"].fillna("").astype(str).str.strip().str.upper()
    pool["injury_status"] = "Healthy"
    pool.loc[norm.isin(_GTD_STATUSES), "injury_status"] = "Day-To-Day"
    pool.loc[norm.isin(_INELIG_S), "injury_status"] = "Out"
    pool["is_out"] = pool["injury_status"].eq("Out")
    return pool


def _process_clean_pool(
    pool: pd.DataFrame,
) -> "tuple[pd.DataFrame, list, list]":
    """Apply the canonical post-load cleaning pipeline to a player pool.

    This is the single authoritative function that every pool loading path
    (Lab CSV upload, Lab API fetch) must call.  It ensures the injury cascade,
    ``injury_status``/``is_out`` columns, the OUT/IR hard-drop, and the
    ownership pipeline (``own_proj``) are applied identically everywhere so that
    the Slate Room, Optimizer, and Sim Module all read from a clean, consistent
    pool.

    Parameters
    ----------
    pool : pd.DataFrame
        Raw pool after column normalisation and YakOS projection application.

    Returns
    -------
    cleaned_pool : pd.DataFrame
        Pool with OUT/IR players removed, injury cascade bumps applied, and
        ``own_proj`` populated via :func:`~yak_core.ownership.apply_ownership_pipeline`.
    cascade : list of dict
        Cascade report from :func:`~yak_core.injury_cascade.apply_injury_cascade`.
    auto_excluded : list of dict
        ``[{player_name, status}, …]`` for every player that was hard-dropped.
    """
    from yak_core.sims import _INELIGIBLE_STATUSES as _INELIG_P

    # 1. Apply manual injury overrides (config/manual_injuries.csv) so that the
    #    cascade redistributes minutes from ANY out player, not just API ones.
    pool = apply_manual_injury_overrides_to_pool(pool)

    # 2. Injury cascade — OUT players must still be in the pool so their minutes
    #    can be redistributed to healthy teammates.
    pool, cascade = apply_injury_cascade(pool)

    # 3. Tag each player with injury_status / is_out columns.
    pool = _add_injury_columns(pool)

    # 4. Hard-drop ineligible players so they never reach the optimizer or sims.
    auto_excluded: list = []
    if "is_out" in pool.columns and "player_name" in pool.columns:
        auto_excluded = (
            pool.loc[pool["is_out"], ["player_name", "status"]].to_dict("records")
        )
        pool = pool[~pool["is_out"]].reset_index(drop=True)
    elif "status" in pool.columns and "player_name" in pool.columns:
        _inelig_mask = pool["status"].fillna("").str.upper().isin(_INELIG_P)
        auto_excluded = (
            pool.loc[_inelig_mask, ["player_name", "status"]].to_dict("records")
        )
        pool = pool[~_inelig_mask].reset_index(drop=True)

    # 5. Ownership pipeline: populates own_model + own_proj (ext_own blended).
    try:
        pool = apply_ownership_pipeline(pool)
    except Exception:
        if "own_proj" not in pool.columns:
            pool["own_proj"] = pool["proj_own"] if "proj_own" in pool.columns else 0.0

    return pool, cascade, auto_excluded


def ensure_session_state():
    if "lineups_df" not in st.session_state:
        st.session_state["lineups_df"] = None
    if "exposures_df" not in st.session_state:
        st.session_state["exposures_df"] = None
    if "current_lineup_index" not in st.session_state:
        st.session_state["current_lineup_index"] = 0
    if "pool_df" not in st.session_state:
        st.session_state["pool_df"] = None
    if "lab_lineups_df" not in st.session_state:
        st.session_state["lab_lineups_df"] = None
    if "lab_metrics" not in st.session_state:
        st.session_state["lab_metrics"] = None
    if "lab_contest_type" not in st.session_state:
        st.session_state["lab_contest_type"] = "GPP"
    # New state keys
    if "promoted_lineups" not in st.session_state:
        # list of dicts: {label, lineups_df, metadata}
        st.session_state["promoted_lineups"] = []
    if "approved_lineups" not in st.session_state:
        # list of ApprovedLineup objects (set by Calibration Lab)
        st.session_state["approved_lineups"] = []
    if "last_calibration_ts" not in st.session_state:
        st.session_state["last_calibration_ts"] = None
    if "cal_queue_df" not in st.session_state:
        st.session_state["cal_queue_df"] = None
    if "archetype" not in st.session_state:
        st.session_state["archetype"] = "Balanced"
    if "dk_contest_type" not in st.session_state:
        st.session_state["dk_contest_type"] = "Tournament (GPP)"
    if "prev_dk_contest_type" not in st.session_state:
        st.session_state["prev_dk_contest_type"] = st.session_state["dk_contest_type"]
    if "contest_preset" not in st.session_state:
        st.session_state["contest_preset"] = "20-Max GPP"
    if "sim_pool_df" not in st.session_state:
        st.session_state["sim_pool_df"] = None
    if "sim_pool_orig_df" not in st.session_state:
        st.session_state["sim_pool_orig_df"] = None
    if "sim_player_pool_clean" not in st.session_state:
        st.session_state["sim_player_pool_clean"] = None
    if "sim_lineups_df" not in st.session_state:
        st.session_state["sim_lineups_df"] = None
    if "sim_results_df" not in st.session_state:
        st.session_state["sim_results_df"] = None
    if "sim_actuals_df" not in st.session_state:
        st.session_state["sim_actuals_df"] = None
    if "sim_mode" not in st.session_state:
        st.session_state["sim_mode"] = "Live"
    if "sim_hist_date" not in st.session_state:
        st.session_state["sim_hist_date"] = None
    if "sim_custom_lineup" not in st.session_state:
        st.session_state["sim_custom_lineup"] = []
    if "sim_anomaly_df" not in st.session_state:
        st.session_state["sim_anomaly_df"] = None
    if "sim_anomaly_out_count" not in st.session_state:
        st.session_state["sim_anomaly_out_count"] = 0
    if "sim_lineup_scores" not in st.session_state:
        st.session_state["sim_lineup_scores"] = None
    if "ms_result" not in st.session_state:
        st.session_state["ms_result"] = None
    if "rapidapi_key" not in st.session_state:
        st.session_state["rapidapi_key"] = (
            os.environ.get("RAPIDAPI_KEY", "") or _load_persisted_api_key()
        )
    if "_pool_df_filename" not in st.session_state:
        st.session_state["_pool_df_filename"] = None
    if "stack_hit_log" not in st.session_state:
        # list of dicts: {slate_date, team, players, outcome, note}
        st.session_state["stack_hit_log"] = []
    if "other_root_causes" not in st.session_state:
        # list of dicts capturing queue rows marked review→Other
        st.session_state["other_root_causes"] = []
    if "bl_lineups_df" not in st.session_state:
        st.session_state["bl_lineups_df"] = None
    if "bl_metrics" not in st.session_state:
        st.session_state["bl_metrics"] = None
    if "bl_slate_date" not in st.session_state:
        st.session_state["bl_slate_date"] = None
    if "bl_slate_data" not in st.session_state:
        st.session_state["bl_slate_data"] = None
    if "bl_contest_type" not in st.session_state:
        st.session_state["bl_contest_type"] = None
    if "cmp_dk_df" not in st.session_state:
        st.session_state["cmp_dk_df"] = None
    if "calib_backtest_results" not in st.session_state:
        st.session_state["calib_backtest_results"] = None
    if "calib_drilldown_arch" not in st.session_state:
        st.session_state["calib_drilldown_arch"] = None
    if "calib_queue_arch" not in st.session_state:
        st.session_state["calib_queue_arch"] = None
    if "calib_queue_slate" not in st.session_state:
        st.session_state["calib_queue_slate"] = None
    if "cal_check_errors" not in st.session_state:
        st.session_state["cal_check_errors"] = None
    if "cal_check_errors_before" not in st.session_state:
        st.session_state["cal_check_errors_before"] = None
    if "actuals" not in st.session_state:
        # dict: slate_date_str → pd.DataFrame with columns player_name, actual_fp
        st.session_state["actuals"] = {}
    if "pool_date" not in st.session_state:
        # tracks the last date used when fetching the player pool
        st.session_state["pool_date"] = None
    if "auto_excluded_players" not in st.session_state:
        # list of dicts: {player_name, status} auto-excluded due to injury/status
        st.session_state["auto_excluded_players"] = []
    if "injury_cascade" not in st.session_state:
        # list of {out_player, team, out_proj_mins, beneficiaries: [...]}
        st.session_state["injury_cascade"] = []
    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = False
    if "sim_lu_nav" not in st.session_state:
        st.session_state["sim_lu_nav"] = 0
    if "dvp_table" not in st.session_state:
        # pd.DataFrame with columns Team, PG, SG, SF, PF, C — loaded from dvp_baseline.csv
        st.session_state["dvp_table"] = load_dvp_table(_DVP_DEFAULT_PATH)
    if "dvp_league_avgs" not in st.session_state:
        # dict: {"PG": float, ...} — league averages computed from dvp_table
        _dvp_init = st.session_state.get("dvp_table")
        st.session_state["dvp_league_avgs"] = (
            compute_league_averages(_dvp_init) if _dvp_init is not None else {}
        )
    if "_dvp_filename" not in st.session_state:
        st.session_state["_dvp_filename"] = None
    if "historical_bundles" not in st.session_state:
        # dict: slate_date_str → HistoricalSlateBundle
        # Populated when user clicks "Fetch Pool from API" for a past date.
        st.session_state["historical_bundles"] = {}


def run_optimizer(
    pool: pd.DataFrame,
    num_lineups: int,
    max_exposure: float,
    min_salary_used: int,
    proj_col: str = "proj",  # which column to use: 'proj', 'floor', 'ceil', or 'sim85'
    archetype: str = "Balanced",
    lock_names: list | None = None,
    exclude_names: list | None = None,
    bump_map: dict | None = None,
    slate_type: str = "Classic",
    max_pair_appearances: int = 0,
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    # Remap the chosen projection style to 'proj' so the optimizer always reads 'proj'.
    # We work on a copy to leave the caller's DataFrame unchanged.
    opt_pool = pool.copy()
    if proj_col != "proj":
        if proj_col in opt_pool.columns:
            # Overwrite 'proj' with the selected style column; original 'proj' is on the copy only.
            opt_pool["proj"] = pd.to_numeric(opt_pool[proj_col], errors="coerce").fillna(0)
        else:
            st.warning(f"Column '{proj_col}' not found; falling back to 'proj'.")

    # Apply DFS archetype adjustments
    opt_pool = apply_archetype(opt_pool, archetype)

    # Inject Edge Analysis scores so the LP objective can use them
    _ss_df = compute_stack_scores(opt_pool, top_n=len(opt_pool))
    if not _ss_df.empty:
        _team_score_map = _ss_df.set_index("team")["stack_score"].to_dict()
        opt_pool["stack_score"] = opt_pool["team"].map(_team_score_map).fillna(50.0)
    _vs_df = compute_value_scores(opt_pool, top_n=len(opt_pool), min_proj=0.0)
    if not _vs_df.empty and "player_name" in _vs_df.columns:
        _player_vscore_map = _vs_df.set_index("player_name")["value_score"].to_dict()
        opt_pool["value_score"] = opt_pool["player_name"].map(_player_vscore_map).fillna(50.0)

    cfg: Dict[str, Any] = {
        "SITE": "dk",
        "SPORT": "nba",
        "SLATE_TYPE": slate_type.lower().replace(" ", "_"),
        "NUM_LINEUPS": num_lineups,
        "MIN_SALARY_USED": min_salary_used,
        "MAX_EXPOSURE": max_exposure,
        "PROJ_COL": "proj",
        "SOLVER_TIME_LIMIT": 30,
        "LOCK": lock_names or [],
        "EXCLUDE": exclude_names or [],
        "BUMP": bump_map or {},
        "MAX_PAIR_APPEARANCES": max_pair_appearances,
    }

    progress_bar = st.progress(0, text="Optimizing lineups…")

    def _update_progress(done: int, total: int) -> None:
        pct = int(done / total * 100) if total > 0 else 0
        progress_bar.progress(pct, text=f"Solving lineup {done} of {total}…")

    try:
        if slate_type == "Showdown Captain":
            lineups_df, exposures_df = build_showdown_lineups(
                opt_pool, cfg, progress_callback=_update_progress
            )
        else:
            lineups_df, exposures_df = build_multiple_lineups_with_exposure(
                opt_pool, cfg, progress_callback=_update_progress
            )
    except Exception as e:
        progress_bar.empty()
        st.error(f"Optimizer error: {e}")
        return None, None

    progress_bar.empty()
    return lineups_df, exposures_df


def render_lineup_card(rows: pd.DataFrame, pool_df: "pd.DataFrame | None" = None) -> None:
    """Render a single lineup as a formatted card.

    Columns shown: Pos, Team, Player, Salary, Field%, Game, Points.
    Questionable players get a 🟡 indicator.  Footer shows totals.

    Parameters
    ----------
    rows :
        DataFrame slice for a single lineup (filtered by lineup_index).
    pool_df :
        Optional full player pool used to look up ``opponent`` / ``status``
        when those columns are absent from *rows*.
    """
    disp_rows = []
    for _, r in rows.iterrows():
        pos = str(r.get("slot", r.get("pos", "?")))
        team = str(r.get("team", ""))
        name = str(r.get("player_name", r.get("name", "")))
        salary = int(r.get("salary", 0))

        # Ownership: prefer own_proj (blended final), then proj_own, then ownership
        own_val = None
        for _ocol in ["own_proj", "proj_own", "ownership"]:
            _v = r.get(_ocol)
            if _v is not None and pd.notna(_v):
                try:
                    own_val = float(_v)
                    break
                except (ValueError, TypeError):
                    pass
        own_str = f"{own_val:.1f}%" if own_val is not None else "—"

        # Game: "team @ opponent"
        opp = str(r.get("opponent", r.get("opp", "")))
        if (not opp or opp in ("nan", "")) and pool_df is not None:
            _nc = "player_name" if "player_name" in pool_df.columns else "name"
            _m = pool_df[pool_df[_nc] == name]
            if not _m.empty:
                opp = str(_m.iloc[0].get("opponent", _m.iloc[0].get("opp", "")))
        game_str = f"{team} @ {opp}" if opp and opp not in ("nan", "") else team

        proj = float(r.get("proj", 0))

        # Injury flag (Q = yellow dot)
        status = str(r.get("status", "")).strip()
        if (not status or status in ("nan", "-", "")) and pool_df is not None:
            _nc = "player_name" if "player_name" in pool_df.columns else "name"
            _m = pool_df[pool_df[_nc] == name]
            if not _m.empty:
                status = str(_m.iloc[0].get("status", "")).strip()
        inj = ""
        if status.upper() == "Q":
            inj = " 🟡"

        disp_rows.append({
            "Pos": pos,
            "Team": team,
            "Player": name + inj,
            "Salary": f"${salary:,}",
            "Field%": own_str,
            "Game": game_str,
            "Points": round(proj, 2),
        })

    if disp_rows:
        st.dataframe(pd.DataFrame(disp_rows), use_container_width=True, hide_index=True)

    # Footer totals
    total_salary = int(pd.to_numeric(rows["salary"], errors="coerce").fillna(0).sum())
    proj_series = pd.to_numeric(rows["proj"], errors="coerce").fillna(0) if "proj" in rows.columns else pd.Series([0.0])
    total_proj = float(proj_series.sum())
    total_own = None
    for _ocol in ["own_proj", "proj_own", "ownership"]:
        if _ocol in rows.columns:
            _s = pd.to_numeric(rows[_ocol], errors="coerce")
            if _s.notna().any():
                total_own = float(_s.sum())
                break
    footer_parts = [f"**${total_salary:,}**"]
    if total_own is not None:
        footer_parts.append(f"**{total_own:.1f}%** field")
    footer_parts.append(f"**{total_proj:.2f} pts**")
    st.caption(" · ".join(footer_parts))


@st.cache_data
def load_historical_lineups() -> pd.DataFrame:
    """Load historical lineups CSV from repo data/ folder."""
    csv_path = Path(__file__).parent / "data" / "historical_lineups.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["slate_date"] = pd.to_datetime(df["slate_date"]).dt.date
        return df
    return pd.DataFrame()


def _diagnose_errors(queue_df: pd.DataFrame) -> dict:
    """Compute FP/Minutes/Ownership MAE and error counts from a calibration queue slice."""
    def _sc(df: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else pd.Series(0.0, index=df.index)

    fp_err = (_sc(queue_df, "actual") - _sc(queue_df, "proj")).abs()
    min_err = (_sc(queue_df, "actual_minutes") - _sc(queue_df, "proj_minutes")).abs()
    own_err = (_sc(queue_df, "own") - _sc(queue_df, "proj_own")).abs()

    fp_mae = float(fp_err.mean()) if len(fp_err) else 0.0
    min_mae = float(min_err.mean()) if len(min_err) else 0.0
    own_mae = float(own_err.mean()) if len(own_err) else 0.0

    fp_errors_n = int(fp_err.gt(6).sum())
    min_errors_n = int(min_err.gt(3).sum())
    own_errors_n = int(own_err.gt(3).sum())

    return {
        "fp_mae": fp_mae,
        "min_mae": min_mae,
        "own_mae": own_mae,
        "fp_errors": fp_errors_n,
        "min_errors": min_errors_n,
        "own_errors": own_errors_n,
        "n_players": max(len(queue_df), 1),
    }


def _generate_suggestions(errors: dict, current_knobs: dict) -> list:
    """Return a list of actionable knob suggestions based on error diagnosis."""
    suggestions = []

    if errors["min_errors"] > errors["fp_errors"]:
        suggestions.append({
            "knob": "blowout_threshold",
            "current": current_knobs.get("blowout_threshold", 15),
            "suggested": 20,
            "reason": (
                f"Minutes errors dominate ({errors['min_errors']} of {errors['n_players']} players) "
                "— loosening blowout threshold reduces false minute cuts"
            ),
        })
        cur_b2b = float(current_knobs.get("b2b_discount", 0.93))
        if cur_b2b > 0.85 and errors["min_mae"] > 3:
            suggestions.append({
                "knob": "b2b_discount",
                "current": cur_b2b,
                "suggested": round(max(cur_b2b - 0.05, 0.80), 2),
                "reason": (
                    f"Min MAE {errors['min_mae']:.1f} min — reducing B2B discount may improve minutes accuracy"
                ),
            })

    if errors["fp_mae"] > 10:
        suggestions.append({
            "knob": "ensemble_w_yakos",
            "current": current_knobs.get("ensemble_w_yakos", 0.40),
            "suggested": 0.65,
            "reason": (
                f"Large FP errors (MAE {errors['fp_mae']:.1f} pts) "
                "— increase YakOS model weight to lean more on trained projections"
            ),
        })
    elif errors["fp_mae"] > 6:
        cur_rg = float(current_knobs.get("ensemble_w_rg", 0.30))
        suggestions.append({
            "knob": "ensemble_w_rg",
            "current": cur_rg,
            "suggested": round(min(cur_rg + 0.10, 0.50), 2),
            "reason": (
                f"FP MAE {errors['fp_mae']:.1f} pts above target — boost RG consensus weight for more stable projections"
            ),
        })

    if errors["own_errors"] > errors["fp_errors"] and errors["own_errors"] > errors["min_errors"]:
        suggestions.append({
            "knob": "ensemble_w_rg",
            "current": current_knobs.get("ensemble_w_rg", 0.30),
            "suggested": round(min(float(current_knobs.get("ensemble_w_rg", 0.30)) + 0.10, 0.50), 2),
            "reason": (
                f"Ownership errors dominate ({errors['own_errors']} of {errors['n_players']} players) "
                "— increasing RG weight often improves ownership accuracy when RG data is fresh"
            ),
        })

    return suggestions


# -----------------------------
# Streamlit App Layout
# -----------------------------


st.set_page_config(
    page_title="YakOS DFS Optimizer",
    layout="wide",
)

ensure_session_state()

st.title("YakOS DFS Optimizer")

# ============================================================
# Sidebar: global knobs
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")

    sport = st.selectbox(
        "Sport",
        ["NBA", "PGA"],
        index=0,
        help="Select the sport. PGA support is a placeholder — NBA is fully wired.",
    )

    st.markdown("---")
    st.subheader("Lineup Builder")

    num_lineups_user = st.slider(
        "NUM_LINEUPS",
        min_value=1,
        max_value=300,
        value=5,
        step=1,
    )

    max_exposure_user = st.slider(
        "Max exposure per player",
        min_value=0.05,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help=(
            "Cap on how often one player can appear across all lineups. "
            "0.35 means a player can appear in at most 35% of your lineups."
        ),
    )

    min_salary_used_user = st.number_input(
        "MIN_SALARY_USED",
        min_value=0,
        max_value=50000,
        value=46500,
        step=500,
    )

    proj_style = st.selectbox(
        "Projection style",
        ["proj", "floor", "ceil", "sim85"],
        index=0,
        help=(
            "proj = base projection | floor = conservative / low-end | "
            "ceil = ceiling / best-case | sim85 = 85th-percentile simulation score"
        ),
    )

    st.markdown("---")
    st.subheader("🔑 API Settings")
    rapidapi_key_input = st.text_input(
        "Tank01 RapidAPI Key",
        value=st.session_state.get("rapidapi_key", ""),
        type="password",
        help=(
            "Required for 'Fetch from API' and live injury updates. "
            "Get a key at rapidapi.com/Tank01/api/tank01-fantasy-stats."
        ),
        key="sidebar_rapidapi_key",
    )
    if rapidapi_key_input:
        if rapidapi_key_input != st.session_state.get("rapidapi_key"):
            _save_persisted_api_key(rapidapi_key_input)
        st.session_state["rapidapi_key"] = rapidapi_key_input
        os.environ["RAPIDAPI_KEY"] = rapidapi_key_input

    # Sanity check: OUT players in sim pool should always be 0
    _sim_pool_check = st.session_state["sim_player_pool_clean"] if "sim_player_pool_clean" in st.session_state else None
    if _sim_pool_check is not None and not _sim_pool_check.empty:
        if "injury_status" in _sim_pool_check.columns:
            _out_count = int(_sim_pool_check["injury_status"].eq("Out").sum())
        elif "is_out" in _sim_pool_check.columns:
            _out_count = int(_sim_pool_check["is_out"].sum())
        else:
            _out_count = 0
        st.caption(f"OUT players in sim pool: {_out_count}")

tab_slate, tab_optimizer, tab_lab = st.tabs([
    "🏀 Ricky's Slate Room",
    "⚡ Optimizer",
    "🔒 Ricky's Lab",
])

# Keep backward-compat alias so all existing `with tab_calib:` blocks still work
tab_calib = tab_lab


# ============================================================
# Tab 1: 🏀 Ricky's Slate Room
# ============================================================
with tab_slate:
    st.subheader("🏀 Ricky's Slate Room")

    # ── 4.7 Last Updated Timestamp ───────────────────────────────────────
    _last_updated_ts = st.session_state.get("last_calibration_ts")
    if _last_updated_ts:
        st.caption(f"🕐 Last updated by Ricky: **{_last_updated_ts}**")

    if sport == "PGA":
        st.info("PGA support is coming soon. Please select NBA for now.")
    else:
        # ── Pool source indicator ────────────────────────────────────────
        # The Slate Room reads the pool published by 🔒 Ricky's Lab.
        # Load or fetch your player pool in the Lab tab to populate this view.
        pool_df = st.session_state.get("pool_df")
        approved_lineups = st.session_state.get("approved_lineups", [])
        last_cal_ts = st.session_state.get("last_calibration_ts")

        # Auto-Excluded Players panel (shown when pool is loaded)
        _auto_excl_list = st.session_state.get("auto_excluded_players", [])
        if _auto_excl_list:
            with st.expander(
                f"⚠️ Auto-Excluded Players ({len(_auto_excl_list)} — OUT/IR/Suspended)",
                expanded=False,
            ):
                st.markdown(
                    "These players were automatically excluded from sims and the optimizer "
                    "based on their injury/availability status. Use the **Manual Player "
                    "Eligibility Overrides** table in the Sim Module to re-enable any player."
                )
                _excl_df = pd.DataFrame(_auto_excl_list)
                st.dataframe(_excl_df, use_container_width=True, hide_index=True)

        if pool_df is None or pool_df.empty:
            st.info(
                "📋 **No player pool loaded.** Go to the **📡 Ricky's Lab** tab to upload a "
                "RotoGrinders CSV or fetch today's slate from the Tank01 API."
            )

        else:
            # ════════════════════════════════════════════════════════════
            # LAYER 1 — KPI Strip (slate-level, driven by approved lineups)
            # ════════════════════════════════════════════════════════════
            slate_kpis = compute_slate_kpis(approved_lineups, last_calibration_ts=last_cal_ts)

            _kpi_color_map = {"green": "#1b4332", "yellow": "#3b3b00", "red": "#4a0000"}
            _kpi_border_map = {"green": "#2d6a4f", "yellow": "#b5a300", "red": "#c0392b"}
            _kpi_text_map = {"green": "#52b788", "yellow": "#ffe66d", "red": "#e74c3c"}
            _kpi_c = slate_kpis["color"]
            _kpi_bg = _kpi_color_map.get(_kpi_c, "#1e1e1e")
            _kpi_border = _kpi_border_map.get(_kpi_c, "#555")
            _kpi_text = _kpi_text_map.get(_kpi_c, "#ccc")

            archetype_str = " · ".join(
                f"{k}: {v}" for k, v in slate_kpis["archetype_counts"].items()
            ) or "—"

            st.markdown(
                f"""
                <div style="
                    display:flex; gap:12px; flex-wrap:wrap; padding:10px 0 14px 0;
                    border-bottom: 1px solid #333; margin-bottom:14px;
                ">
                  <div style="background:{_kpi_bg};border:1px solid {_kpi_border};border-radius:8px;
                              padding:10px 18px;min-width:120px;text-align:center;">
                    <div style="font-size:11px;color:#aaa;letter-spacing:.5px;">SLATE EV (Ricky pool)</div>
                    <div style="font-size:20px;font-weight:700;color:{_kpi_text};">
                      {slate_kpis['slate_ev']:+.2f}
                    </div>
                  </div>
                  <div style="background:{_kpi_bg};border:1px solid {_kpi_border};border-radius:8px;
                              padding:10px 18px;min-width:140px;text-align:center;">
                    <div style="font-size:11px;color:#aaa;letter-spacing:.5px;">APPROVED LINEUPS</div>
                    <div style="font-size:20px;font-weight:700;color:{_kpi_text};">
                      {slate_kpis['approved_count']}
                    </div>
                    <div style="font-size:11px;color:#888;">{archetype_str}</div>
                  </div>
                  <div style="background:{_kpi_bg};border:1px solid {_kpi_border};border-radius:8px;
                              padding:10px 18px;min-width:130px;text-align:center;">
                    <div style="font-size:11px;color:#aaa;letter-spacing:.5px;">EXPOSURE RISK</div>
                    <div style="font-size:20px;font-weight:700;color:{_kpi_text};">
                      {slate_kpis['max_exposure']:.0%}
                    </div>
                    <div style="font-size:11px;color:#888;">max player exposure</div>
                  </div>
                  <div style="background:{_kpi_bg};border:1px solid {_kpi_border};border-radius:8px;
                              padding:10px 18px;min-width:130px;text-align:center;">
                    <div style="font-size:11px;color:#aaa;letter-spacing:.5px;">SIMMED HIT RATE</div>
                    <div style="font-size:20px;font-weight:700;color:{_kpi_text};">
                      {slate_kpis['simmed_hit_rate']:.0%}
                    </div>
                  </div>
                  <div style="background:#1e1e1e;border:1px solid #333;border-radius:8px;
                              padding:10px 18px;min-width:160px;text-align:center;">
                    <div style="font-size:11px;color:#aaa;letter-spacing:.5px;">LAST CALIBRATION</div>
                    <div style="font-size:13px;font-weight:600;color:#ccc;">
                      {slate_kpis['last_updated']}
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ════════════════════════════════════════════════════════════
            # LAYER 2 — Right Angle Ricky: Edge Analysis (data-driven)
            # ════════════════════════════════════════════════════════════
            st.markdown("### 📐 Right Angle Ricky — Edge Analysis")

            # Compute scored edge inputs (drive both UI and optimizer)
            _stack_scores_df = compute_stack_scores(pool_df, top_n=5)
            _value_scores_df = compute_value_scores(pool_df, top_n=20)
            _tiered_stack_alerts = compute_tiered_stack_alerts(pool_df)

            col_edge_l, col_edge_r = st.columns(2)

            with col_edge_l:
                # ── 4.2 Stack Alerts ──────────────────────────────────
                st.markdown("#### 🔥 Stack Alerts")
                if _tiered_stack_alerts:
                    for _sa in _tiered_stack_alerts:
                        _ou_part = f" | O/U {_sa['game_ou']:.1f}" if _sa['game_ou'] > 0 else ""
                        _sprd_part = f" | spread ±{_sa['spread']:.1f}" if _sa['spread'] > 0 else ""
                        st.markdown(
                            f"{_sa['tier_emoji']} **{_sa['team']}** — {_sa['tier']} "
                            f"({_sa['conditions_met']}/5 conditions) | "
                            f"implied {_sa['implied_total']:.1f}"
                            f"{_ou_part}{_sprd_part}"
                            f" | own {_sa['combined_ownership']:.0f}% | {_sa['key_players']}"
                        )
                        for _cond in _sa["conditions"]:
                            st.caption(f"  ✓ {_cond}")
                    st.markdown("")
                    _tiered_teams = [a["team"] for a in _tiered_stack_alerts]
                    with st.expander("📝 Log stack outcome", expanded=False):
                        log_slate_date = st.date_input(
                            "Slate date",
                            value=_today_est(),
                            key="stack_log_date",
                        )
                        log_stack_sel = st.selectbox(
                            "Which stack?",
                            _tiered_teams,
                            key="stack_log_sel",
                        )
                        log_note = st.text_input(
                            "Note (optional)",
                            placeholder="e.g. both went off, Jalen Duren 38 pts",
                            key="stack_log_note",
                        )
                        hit_col, miss_col = st.columns(2)
                        with hit_col:
                            if st.button("✅ Hit", key="stack_log_hit"):
                                st.session_state["stack_hit_log"].append({
                                    "slate_date": str(log_slate_date),
                                    "stack": log_stack_sel,
                                    "outcome": "Hit",
                                    "note": log_note,
                                })
                                st.success("Logged as ✅ Hit")
                        with miss_col:
                            if st.button("❌ Miss", key="stack_log_miss"):
                                st.session_state["stack_hit_log"].append({
                                    "slate_date": str(log_slate_date),
                                    "stack": log_stack_sel,
                                    "outcome": "Miss",
                                    "note": log_note,
                                })
                                st.warning("Logged as ❌ Miss")
                else:
                    st.info(
                        "No stacks meet the 3-condition threshold. "
                        "Conditions: implied total, O/U, spread ±7, correlation, ceiling/floor ratio."
                    )

                # ── 4.4 Game Environment Cards ────────────────────────
                st.markdown("#### ⚡ Game Environment")
                _game_cards = compute_game_environment_cards(pool_df)
                if _game_cards:
                    _vegas_avail = _game_cards[0]["vegas_available"] if _game_cards else False
                    if not _vegas_avail:
                        st.caption("⚠️ Vegas lines not loaded — using projection proxies.")
                    for _gc in _game_cards:
                        _flag_str = "  " + " ".join(_gc["flags"]) if _gc["flags"] else ""
                        _ou_display = f"{_gc['combined_ou']:.1f}" if _gc["combined_ou"] > 0 else "—"
                        _sprd_display = f"±{_gc['spread']:.1f}" if _gc["spread"] > 0 else "—"
                        st.markdown(
                            f"**{_gc['away']} @ {_gc['home']}** | "
                            f"O/U {_ou_display} | spread {_sprd_display} | "
                            f"pace {_gc['pace_rating']}{_flag_str}"
                        )
                        st.caption(
                            f"{_gc['away']} implied {_gc['away_implied']:.1f}  ·  "
                            f"{_gc['home']} implied {_gc['home_implied']:.1f}"
                        )
                else:
                    st.info("Upload a pool with opponent data for game environment analysis.")

            with col_edge_r:
                # ── 4.3 High-Value Plays ──────────────────────────────
                st.markdown("#### 💎 High-Value Plays")
                st.caption("Filter: adj proj ≥ 20 FP and proj minutes ≥ 20")

                # Build filtered value pool for 4.3
                _hvp_df = pool_df.copy()
                _hvp_df["proj"] = pd.to_numeric(_hvp_df.get("proj", 0), errors="coerce").fillna(0)
                _hvp_df["salary"] = pd.to_numeric(_hvp_df.get("salary", 0), errors="coerce").fillna(0)
                # Use adjusted_proj column when available
                _adj_col = "adjusted_proj" if "adjusted_proj" in _hvp_df.columns else "proj"
                _hvp_df["_adj_proj"] = pd.to_numeric(_hvp_df[_adj_col], errors="coerce").fillna(0)
                _mins_col = next((c for c in ["proj_minutes", "minutes"] if c in _hvp_df.columns), None)
                if _mins_col:
                    _hvp_df["_mins"] = pd.to_numeric(_hvp_df[_mins_col], errors="coerce").fillna(0)
                    _hvp_df = _hvp_df[(_hvp_df["_adj_proj"] >= 20) & (_hvp_df["_mins"] >= 20)]
                else:
                    _hvp_df = _hvp_df[_hvp_df["_adj_proj"] >= 20]

                if not _hvp_df.empty and _hvp_df["salary"].max() > 0:
                    _hvp_df["_value"] = _hvp_df["_adj_proj"] / (_hvp_df["salary"] / 1000.0)
                    _own_col_hvp = next(
                        (c for c in ["own_proj", "ownership", "proj_own", "ext_own"] if c in _hvp_df.columns),
                        None,
                    )
                    _median_own = 0.0
                    if _own_col_hvp:
                        _hvp_df[_own_col_hvp] = pd.to_numeric(_hvp_df[_own_col_hvp], errors="coerce").fillna(15.0)
                        _median_own = float(_hvp_df[_own_col_hvp].median())

                    def _edge_tag(row) -> str:
                        own = float(row[_own_col_hvp]) if _own_col_hvp else 15.0
                        if own < _median_own * 0.6:
                            return "Contrarian"
                        if own > _median_own * 1.4:
                            return "Chalk Value"
                        return "Leverage"

                    _hvp_df["_edge"] = _hvp_df.apply(_edge_tag, axis=1)

                    _tiers = {
                        "Spend-Up ($7K+)": _hvp_df[_hvp_df["salary"] >= 7000],
                        "Mid-Range ($5K–$7K)": _hvp_df[(_hvp_df["salary"] >= 5000) & (_hvp_df["salary"] < 7000)],
                        "Punt (<$5K)": _hvp_df[_hvp_df["salary"] < 5000],
                    }
                    for _tier_name, _tier_df in _tiers.items():
                        if _tier_df.empty:
                            continue
                        st.markdown(f"**{_tier_name}**")
                        for _, _vr in _tier_df.nlargest(5, "_value").iterrows():
                            _own_disp = ""
                            if _own_col_hvp:
                                _own_disp = f" | own {_vr[_own_col_hvp]:.0f}%"
                            _edge = _vr["_edge"]
                            _edge_emoji = {"Contrarian": "🟣", "Leverage": "🟡", "Chalk Value": "🔴"}.get(_edge, "⚪")
                            st.markdown(
                                f"- {_edge_emoji} **{_vr['player_name']}** ({_vr.get('team', '?')}) — "
                                f"${int(_vr['salary']):,} | adj proj {_vr['_adj_proj']:.1f} | "
                                f"value {_vr['_value']:.2f}x{_own_disp} | {_edge}"
                            )
                else:
                    st.info(
                        "No qualifying plays found. "
                        "Load a pool with players projecting ≥ 20 FP and ≥ 20 minutes."
                    )

            st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # LAYER 2b — Player Projections Table (4.5)
            # ════════════════════════════════════════════════════════════
            st.markdown("### 📋 Player Projections")
            _proj_disp_cols = [c for c in [
                "player_name", "pos", "team", "salary",
                "adjusted_proj", "proj", "original_proj", "injury_bump_fp",
                "floor", "ceil", "proj_minutes",
                "ext_own", "own_model", "own_proj", "proj_own",
                "status", "proj_source",
            ] if c in pool_df.columns]
            _proj_table = (
                pool_df[_proj_disp_cols]
                .copy()
                .sort_values(
                    "adjusted_proj" if "adjusted_proj" in pool_df.columns else "proj",
                    ascending=False,
                )
                .reset_index(drop=True)
            )
            # Compute value multiple
            if "salary" in _proj_table.columns:
                _vproj = (
                    _proj_table["adjusted_proj"]
                    if "adjusted_proj" in _proj_table.columns
                    else _proj_table["proj"]
                )
                _proj_table["value_x"] = (
                    _vproj / (_proj_table["salary"].replace(0, float("nan")) / 1000.0)
                ).round(2)

            with st.expander("📋 All Players — sorted by projection", expanded=True):
                # Ownership display toggle
                _own_toggle_opts = []
                if "ext_own" in _proj_table.columns and _proj_table["ext_own"].notna().any():
                    _own_toggle_opts.append("ext_own (raw site)")
                if "own_model" in _proj_table.columns:
                    _own_toggle_opts.append("own_model (GBM)")
                if "own_proj" in _proj_table.columns:
                    _own_toggle_opts.append("own_final (blended)")
                if "proj_own" in _proj_table.columns:
                    _own_toggle_opts.append("proj_own (legacy)")
                if _own_toggle_opts:
                    _own_view = st.radio(
                        "Field% view",
                        _own_toggle_opts,
                        index=min(2, len(_own_toggle_opts) - 1),
                        horizontal=True,
                        key="proj_table_own_toggle",
                    )
                _col_cfg: dict = {
                    "player_name": st.column_config.TextColumn("Player"),
                    "pos": st.column_config.TextColumn("Pos", width="small"),
                    "team": st.column_config.TextColumn("Team", width="small"),
                    "salary": st.column_config.NumberColumn("Salary", format="$%d"),
                    "adjusted_proj": st.column_config.NumberColumn("Adj Proj", format="%.2f"),
                    "proj": st.column_config.NumberColumn("Proj", format="%.2f"),
                    "original_proj": st.column_config.NumberColumn("Orig Proj", format="%.2f"),
                    "injury_bump_fp": st.column_config.NumberColumn("Inj Bump", format="%.2f"),
                    "floor": st.column_config.NumberColumn("Floor", format="%.2f"),
                    "ceil": st.column_config.NumberColumn("Ceil", format="%.2f"),
                    "proj_minutes": st.column_config.NumberColumn("Mins", format="%.1f"),
                    "ext_own": st.column_config.NumberColumn("Ext Own%", format="%.1f"),
                    "own_model": st.column_config.NumberColumn("Model Own%", format="%.1f"),
                    "own_proj": st.column_config.NumberColumn("Field%", format="%.1f"),
                    "proj_own": st.column_config.NumberColumn("Own % (legacy)", format="%.1f"),
                    "status": st.column_config.TextColumn("Status"),
                    "value_x": st.column_config.NumberColumn("Value x", format="%.2fx"),
                    "proj_source": st.column_config.TextColumn("Source"),
                }
                # Row highlighting: red for non-Active status, green for injury-bumped
                def _row_highlight(row):
                    styles = [""] * len(row)
                    status_val = str(row.get("status", "Active")).strip().upper()
                    bump_val = float(row.get("injury_bump_fp", 0) or 0)
                    if status_val not in ("ACTIVE", "", "NAN", "NONE"):
                        styles = ["background-color: #4a0000; color: #ff9999"] * len(row)
                    elif bump_val > 0:
                        styles = ["background-color: #003300; color: #90ee90"] * len(row)
                    return styles

                _styled_table = _proj_table.style.apply(_row_highlight, axis=1)
                st.dataframe(
                    _styled_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={k: v for k, v in _col_cfg.items() if k in _proj_table.columns},
                )

            # ════════════════════════════════════════════════════════════
            # LAYER 2c — Injury Cascade Report (4.1)
            # ════════════════════════════════════════════════════════════
            _cascade = st.session_state.get("injury_cascade", [])
            if _cascade:
                st.markdown("### 🚑 Injury Cascade Report")
                st.caption(
                    "Players marked OUT/IR with projected ≥ 20 min — "
                    "their minutes were redistributed to active teammates. "
                    "Projections above already include these bumps."
                )
                for _entry in _cascade:
                    _out = _entry["out_player"]
                    _team = _entry["team"]
                    _omins = _entry["out_proj_mins"]
                    _ofp = _entry.get("out_proj_fp", 0.0)
                    _benes = _entry["beneficiaries"]
                    with st.expander(
                        f"🚨 {_out} ({_team}) — OUT | was projected {_omins:.0f} min, {_ofp:.1f} FP",
                        expanded=True,
                    ):
                        if _benes:
                            for _b in _benes:
                                _sal_k = _b["salary"] / 1000.0
                                _val = _b["new_value_multiple"]
                                _sleeper = " 🔴 **Sleeper**" if _val >= 5.0 else ""
                                st.markdown(
                                    f"→ **{_b['name']}**: proj {_b['original_proj']:.1f} → "
                                    f"**{_b['adjusted_proj']:.1f} FP** | "
                                    f"${_b['salary']:,} | {_val:.1f}x value{_sleeper}"
                                )
                        else:
                            st.info("No eligible teammates found to redistribute minutes.")


            st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # LAYER 3 — Ricky's Approved Lineups (read-only from Lab) — 4.6
            # ════════════════════════════════════════════════════════════
            st.markdown("### 📥 Ricky's Approved Lineups")
            _last_pub_ts = st.session_state.get("last_calibration_ts")
            if _last_pub_ts:
                st.caption(f"Last published: **{_last_pub_ts}**")
            st.caption(
                "These lineups are published by Ricky from 🔒 Ricky's Lab after sims and backtests. "
                "Rerun to refresh."
            )

            _all_approved = st.session_state.get("approved_lineups", [])

            # Fall back to legacy promoted_lineups so existing workflows still show
            _legacy_promoted = st.session_state.get("promoted_lineups", [])

            if _all_approved:
                _by_arch = get_approved_lineups_by_archetype(_all_approved)
                _arch_order = ["GPP", "SE", "3-MAX", "50/50", "Showdown"]
                _arch_tabs_labels = [a for a in _arch_order if a in _by_arch] + [
                    a for a in _by_arch if a not in _arch_order
                ]

                if _arch_tabs_labels:
                    _arch_tab_widgets = st.tabs(_arch_tabs_labels)
                    for _atab, _alabel in zip(_arch_tab_widgets, _arch_tabs_labels):
                        with _atab:
                            _lu_list = _by_arch[_alabel]
                            for _idx, _alu in enumerate(_lu_list):
                                _late_badge = " 🕐 Late-swap set" if _alu.late_swap_window else ""
                                _tot_sal_hdr = int(sum(p.get("salary", 0) for p in _alu.players))
                                _hdr = (
                                    f"**#{_idx + 1}** · {_alu.site} · {_alu.slate} | "
                                    f"{_alabel} · ${_tot_sal_hdr:,} · "
                                    f"proj {_alu.proj_points:.1f} · ceil {_alu.sim_p90:.1f}"
                                    f"{_late_badge}"
                                )
                                with st.expander(_hdr, expanded=(_idx == 0)):
                                    if _alu.late_swap_window:
                                        st.info(f"🕐 Late-swap window: {_alu.late_swap_window}")
                                    _p_df = pd.DataFrame(_alu.players)
                                    if not _p_df.empty:
                                        _card_rows = []
                                        for _, _pr in _p_df.iterrows():
                                            _ov = _pr.get("ownership")
                                            try:
                                                _ov_flt = float(_ov) if _ov is not None and pd.notna(_ov) else None
                                            except (ValueError, TypeError):
                                                _ov_flt = None
                                            _card_rows.append({
                                                "Pos": str(_pr.get("pos", "?")),
                                                "Team": str(_pr.get("team", "")),
                                                "Player": str(_pr.get("name", "")),
                                                "Salary": f"${int(_pr.get('salary', 0)):,}",
                                                "Field%": f"{_ov_flt:.1f}%" if _ov_flt is not None else "—",
                                            })
                                        st.dataframe(pd.DataFrame(_card_rows), use_container_width=True, hide_index=True)
                                        _tot_sal = int(sum(p.get("salary", 0) for p in _alu.players))
                                        try:
                                            _tot_own = sum(float(p.get("ownership") or 0) for p in _alu.players)
                                        except (ValueError, TypeError):
                                            _tot_own = 0.0
                                        st.caption(
                                            f"**${_tot_sal:,}** salary · **{_tot_own:.1f}%** total field% · "
                                            f"**{_alu.proj_points:.2f} pts** proj · "
                                            f"**{_alu.sim_p90:.1f}** ceiling (p90)"
                                        )
                                    else:
                                        st.info("No player data.")
            elif _legacy_promoted:
                # Legacy display for lineups promoted via the old "Post to Slate Room" flow
                for i, entry in enumerate(_legacy_promoted):
                    label = entry.get("label", f"Promoted Set {i + 1}")
                    lu_df = entry.get("lineups_df")
                    meta = entry.get("metadata", {})
                    with st.expander(
                        f"**{label}** — {meta.get('contest_type', '')} | "
                        f"{len(lu_df['lineup_index'].unique()) if lu_df is not None else 0} lineups",
                        expanded=(i == 0),
                    ):
                        if lu_df is not None and not lu_df.empty:
                            annotated = ricky_annotate(lu_df)
                            unique_lu = sorted(annotated["lineup_index"].unique())
                            _nav_key = f"promoted_nav_{i}"
                            if _nav_key not in st.session_state:
                                st.session_state[_nav_key] = 0
                            _lp_pos = max(0, min(len(unique_lu) - 1, st.session_state[_nav_key]))
                            _lp_c1, _lp_c2, _lp_c3 = st.columns([1, 4, 1])
                            with _lp_c1:
                                if st.button("◀", key=f"leg_lu_prev_{i}", disabled=(_lp_pos == 0)):
                                    st.session_state[_nav_key] = _lp_pos - 1
                                    st.rerun()
                            with _lp_c2:
                                st.markdown(
                                    f"<div style='text-align:center'>Lineup {_lp_pos + 1} of {len(unique_lu)}</div>",
                                    unsafe_allow_html=True,
                                )
                            with _lp_c3:
                                if st.button("▶", key=f"leg_lu_next_{i}", disabled=(_lp_pos == len(unique_lu) - 1)):
                                    st.session_state[_nav_key] = _lp_pos + 1
                                    st.rerun()
                            lu_rows = annotated[annotated["lineup_index"] == unique_lu[_lp_pos]].copy()
                            conf = lu_rows["confidence"].iloc[0] if "confidence" in lu_rows.columns else "—"
                            tag = lu_rows["tag"].iloc[0] if "tag" in lu_rows.columns else "—"
                            st.markdown(
                                f"**Lineup {_lp_pos + 1} — ${int(lu_rows['salary'].sum()):,} | "
                                f"{lu_rows['proj'].sum():.2f} pts** · Conf: {conf} · Tag: {tag}"
                            )
                            render_lineup_card(lu_rows, pool_df=st.session_state.get("pool_df"))
                        else:
                            st.info("No lineups in this set.")
            else:
                st.info(
                    "No lineups published yet. "
                    "Run sims in **🔒 Ricky's Lab** and use "
                    "**✅ Publish to Slate Room** to surface high-confidence lineups here."
                )


# ============================================================
# Tab 2: ⚡ Optimizer
# ============================================================
with tab_optimizer:
    st.subheader("⚡ Optimizer")
    st.markdown(
        "Build lineups for today's slate. Pick a Contest Type and the optimizer "
        "will apply the right settings automatically."
    )

    if sport == "PGA":
        st.info("PGA support is coming soon. Please select NBA for now.")
    else:
        pool_df_opt = st.session_state.get("pool_df")
        if pool_df_opt is None:
            st.warning(
                "⏳ Waiting for Ricky to load today's slate. Check back soon."
            )
        else:
            # --- Contest Type picker ---
            st.markdown("### 1. Contest Type")
            _prev_preset = st.session_state.get("contest_preset", "20-Max GPP")
            contest_preset_sel = st.selectbox(
                "Contest Type",
                CONTEST_PRESET_LABELS,
                index=CONTEST_PRESET_LABELS.index(_prev_preset)
                if _prev_preset in CONTEST_PRESET_LABELS
                else 3,
                key="opt_contest_preset",
                help=(
                    "Cash Game = high-floor 50/50 | "
                    "Single Entry = balanced 1-lineup | "
                    "3-Max = small GPP | "
                    "20-Max GPP = tournament | "
                    "MME = mass multi-entry | "
                    "Showdown = single-game Captain mode"
                ),
            )
            st.session_state["contest_preset"] = contest_preset_sel
            _preset = CONTEST_PRESETS[contest_preset_sel]
            st.caption(f"📌 {_preset['description']}")

            # Derive all parameters from the selected preset
            slate_type_opt = _preset["slate_type"]
            archetype_sel = _preset["archetype"]
            internal_contest = _preset["internal_contest"]
            dk_contest_sel = contest_preset_sel  # label used in spinner / success messages (equals preset name)
            # Sync legacy session-state keys so downstream code still works
            st.session_state["archetype"] = archetype_sel
            st.session_state["dk_contest_type"] = contest_preset_sel

            # Showdown game picker — only shown when the Showdown preset is selected
            showdown_game_opt = None
            if slate_type_opt == "Showdown Captain":
                games = get_showdown_games(pool_df_opt)
                if games:
                    showdown_game_opt = st.selectbox("Showdown Game", games, key="opt_showdown")
                else:
                    st.warning("No games detected. Upload a valid pool.")

            # --- Optimizer controls ---
            st.markdown("---")
            st.markdown("### 2. Build Controls")
            ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(4)
            with ctrl_c1:
                num_lu_opt = st.slider(
                    "Lineups", 1, 300,
                    int(_preset["default_lineups"]),
                    key="opt_num_lu",
                )
            with ctrl_c2:
                max_exp_opt = st.slider(
                    "Max exposure", 0.05, 1.0,
                    float(_preset["default_max_exposure"]),
                    step=0.05, key="opt_exp",
                )
            with ctrl_c3:
                min_sal_opt = st.number_input(
                    "Min salary used", 0, 50000,
                    int(_preset["min_salary"]),
                    step=500, key="opt_sal",
                )
            with ctrl_c4:
                max_pair_opt = st.number_input(
                    "Max pair appearances",
                    min_value=0,
                    max_value=num_lu_opt,
                    value=0,
                    step=1,
                    key="opt_max_pair",
                    help=(
                        "Lineup diversity control: maximum number of lineups "
                        "in which any two players can appear together. "
                        "0 = disabled (default)."
                    ),
                )

            # Projection style driven by preset (still selectable for fine-tuning)
            _preset_proj_style = _preset["projection_style"]
            if _preset_proj_style not in _PROJ_STYLE_OPTIONS:
                _preset_proj_style = "proj"
            # Re-seed when contest preset changes
            if contest_preset_sel != st.session_state.get("prev_dk_contest_type"):
                st.session_state["opt_proj_style"] = _preset_proj_style
                st.session_state["prev_dk_contest_type"] = contest_preset_sel

            _cur_proj_style = st.session_state.get("opt_proj_style", _preset_proj_style)
            if _cur_proj_style not in _PROJ_STYLE_OPTIONS:
                _cur_proj_style = _preset_proj_style
            proj_style_opt = st.selectbox(
                "Projection style (ceiling/floor driver)",
                _PROJ_STYLE_OPTIONS,
                index=_PROJ_STYLE_OPTIONS.index(_cur_proj_style),
                key="opt_proj_style",
                help=(
                    "proj = base projection | floor = conservative / cash-game | "
                    "ceil = ceiling / GPP | sim85 = 85th-percentile sim output"
                ),
            )

            # --- Player Pool Overrides ---
            st.markdown("---")
            st.markdown("### 3. Player Pool Overrides")
            with st.expander("🔒 Lock / Exclude / Bump Players", expanded=False):
                override_c1, override_c2, override_c3 = st.columns(3)
                with override_c1:
                    lock_input = st.text_area(
                        "🔒 Lock players (one per line)",
                        placeholder="LeBron James\nSteph Curry",
                        height=120,
                        key="opt_lock_input",
                        help="These players will appear in every lineup.",
                    )
                with override_c2:
                    exclude_input = st.text_area(
                        "🚫 Exclude players (one per line)",
                        placeholder="Kevin Durant\nJa Morant",
                        height=120,
                        key="opt_exclude_input",
                        help="These players will be removed from the pool entirely.",
                    )
                with override_c3:
                    bump_input = st.text_area(
                        "⚡ Bump projections (Name | multiplier)",
                        placeholder="Nikola Jokic | 1.15\nAnthony Davis | 0.90",
                        height=120,
                        key="opt_bump_input",
                        help=(
                            "Multiply a player's projection by the given factor. "
                            "E.g. 1.15 = +15%, 0.90 = -10%."
                        ),
                    )

            # Parse override inputs
            _lock_names = [n.strip() for n in lock_input.splitlines() if n.strip()]
            _exclude_names = [n.strip() for n in exclude_input.splitlines() if n.strip()]
            _bump_map: dict = {}
            for line in bump_input.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 2 and parts[0] and parts[1]:
                    try:
                        _bump_map[parts[0]] = float(parts[1])
                    except ValueError:
                        pass

            if _lock_names:
                st.caption(f"🔒 Locked: {', '.join(_lock_names)}")
            if _exclude_names:
                st.caption(f"🚫 Excluded: {', '.join(_exclude_names)}")
            if _bump_map:
                bump_strs = [f"{n} ×{m:.2f}" for n, m in _bump_map.items()]
                st.caption(f"⚡ Bumped: {', '.join(bump_strs)}")

            # Apply slate filter
            pool_for_opt = apply_slate_filters(pool_df_opt, slate_type_opt, showdown_game_opt)

            st.markdown("---")
            if st.button("🚀 Build Lineups", type="primary", key="opt_build_btn"):
                with st.spinner(f"Optimizing ({dk_contest_sel} / {archetype_sel})..."):
                    # Refresh injury statuses from Tank01 before every lineup build
                    _opt_api_key = st.session_state.get("rapidapi_key", "")
                    _refreshed_opt, _opt_changes = _refresh_injury_statuses(
                        pool_df_opt, _opt_api_key
                    )
                    # Always build lineups from the refreshed pool
                    pool_for_opt = apply_slate_filters(
                        _refreshed_opt, slate_type_opt, showdown_game_opt
                    )
                    if _opt_changes:
                        st.session_state["pool_df"] = _refreshed_opt
                        st.info(
                            "**Injury refresh:** "
                            + " | ".join(_opt_changes[:8])
                            + (" …" if len(_opt_changes) > 8 else "")
                        )
                    lu_df, exp_df = run_optimizer(
                        pool_for_opt,
                        num_lineups=num_lu_opt,
                        max_exposure=max_exp_opt,
                        min_salary_used=min_sal_opt,
                        proj_col=proj_style_opt,
                        archetype=archetype_sel,
                        lock_names=_lock_names or None,
                        exclude_names=_exclude_names or None,
                        bump_map=_bump_map or None,
                        slate_type=slate_type_opt,
                        max_pair_appearances=int(max_pair_opt),
                    )
                    st.session_state["lineups_df"] = lu_df
                    st.session_state["exposures_df"] = exp_df
                    st.session_state["current_lineup_index"] = 0

            lineups_df = st.session_state.get("lineups_df")
            exposures_df = st.session_state.get("exposures_df")

            if lineups_df is not None and not lineups_df.empty:
                unique_lu = sorted(lineups_df["lineup_index"].unique())
                st.success(f"Generated {len(unique_lu)} lineups ({dk_contest_sel} / {archetype_sel})")

                # ── Ownership sanity check caption ────────────────────────
                _opt_pool_df = st.session_state.get("pool_df")
                if _opt_pool_df is not None and "own_proj" in _opt_pool_df.columns:
                    _own_s = _opt_pool_df["own_proj"].dropna()
                    if not _own_s.empty:
                        _own_src = (
                            "external (RG/FP)"
                            if "ext_own" in _opt_pool_df.columns and _opt_pool_df["ext_own"].notna().any()
                            else "internal model ⚠️"
                        )
                        st.caption(
                            f"Ownership source: {_own_src} | "
                            f"Min own: {_own_s.min():.1f}% | "
                            f"Max own: {_own_s.max():.1f}% | "
                            f"Median: {_own_s.median():.1f}%"
                        )

                # ── Arrow navigation ──────────────────────────────────────
                _lu_pos = max(0, min(len(unique_lu) - 1, st.session_state["current_lineup_index"]))
                _nav_c1, _nav_c2, _nav_c3 = st.columns([1, 4, 1])
                with _nav_c1:
                    if st.button("◀", key="opt_lu_prev", disabled=(_lu_pos == 0)):
                        st.session_state["current_lineup_index"] = _lu_pos - 1
                        st.rerun()
                with _nav_c2:
                    st.markdown(
                        f"<div style='text-align:center;font-size:1.05em'>"
                        f"Lineup {_lu_pos + 1} of {len(unique_lu)}</div>",
                        unsafe_allow_html=True,
                    )
                with _nav_c3:
                    if st.button("▶", key="opt_lu_next", disabled=(_lu_pos == len(unique_lu) - 1)):
                        st.session_state["current_lineup_index"] = _lu_pos + 1
                        st.rerun()

                cur_idx = unique_lu[_lu_pos]
                lu_idx = _lu_pos + 1
                rows = lineups_df[lineups_df["lineup_index"] == cur_idx].copy()
                st.markdown(
                    f"**Lineup {lu_idx} — ${int(rows['salary'].sum()):,} | "
                    f"{rows['proj'].sum():.2f} pts**"
                )
                render_lineup_card(rows, pool_df=st.session_state.get("pool_df"))

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "📥 Download lineups CSV",
                        data=to_csv_bytes(lineups_df),
                        file_name="yakos_lineups.csv",
                        mime="text/csv",
                        key="opt_dl_lu",
                    )
                with dl2:
                    if slate_type_opt == "Showdown Captain":
                        dk_upload_df = to_dk_showdown_upload_format(lineups_df)
                        dk_help = (
                            "DraftKings Showdown bulk entry format: one row per lineup "
                            "with CPT + 5 FLEX columns. "
                            "Fill in Entry ID and Contest info before uploading."
                        )
                    else:
                        dk_upload_df = to_dk_upload_format(lineups_df)
                        dk_help = (
                            "DraftKings bulk entry format: one row per lineup with "
                            "slot columns PG/SG/SF/PF/C/G/F/UTIL. "
                            "Fill in Entry ID and Contest info before uploading."
                        )
                    st.download_button(
                        "📥 Download DK upload CSV",
                        data=to_csv_bytes(dk_upload_df),
                        file_name="yakos_dk_upload.csv",
                        mime="text/csv",
                        key="opt_dl_dk",
                        help=dk_help,
                    )


# ============================================================
# Tab 3: 🔒 Ricky's Lab (admin-gated)
# ============================================================
with tab_lab:
    st.subheader("🔒 Ricky's Lab")

    # ── Admin authentication gate ────────────────────────────────────────────
    if not st.session_state.get("is_admin", False):
        st.info("🔒 Admin access required. Enter the password to continue.")
        _admin_pw_input = st.text_input(
            "Admin password",
            type="password",
            key="admin_pw_input",
        )
        if st.button("Unlock Lab", key="admin_unlock_btn"):
            _expected_pw = ""
            _secrets_error = False
            try:
                _expected_pw = st.secrets["admin_password"]
            except KeyError:
                _secrets_error = True
            except Exception:
                _secrets_error = True
            if _secrets_error:
                st.error("Admin password not configured. Set `admin_password` in .streamlit/secrets.toml.")
            elif _expected_pw and hmac.compare_digest(_admin_pw_input, _expected_pw):
                st.session_state["is_admin"] = True
                st.rerun()
            else:
                st.error("Admin access only.")
        st.stop()

    # ── 0. Player Pool / RG Projection Upload ─────────────────────────────
    st.markdown("### 📂 Load Player Pool")
    st.markdown(
        "Upload your player pool CSV here (RotoGrinders export or any compatible format). "
        "This powers the Slate Room, Optimizer, and Sims."
    )

    cal_upload_l, cal_upload_r = st.columns([2, 1])
    with cal_upload_l:
        rg_upload_cal = st.file_uploader(
            "Upload Player Pool CSV",
            type=["csv"],
            key="cal_rg_upload",
            help="RotoGrinders or compatible export — FPTS/SALARY/OWNERSHIP columns mapped automatically.",
        )
    with cal_upload_r:
        st.markdown("**— or fetch from API —**")
        fetch_slate_date_cal = st.date_input(
            "Slate date",
            value=_today_est(),
            key="cal_fetch_date",
        )
        if st.button(
            "🌐 Fetch Pool from API",
            key="cal_fetch_api_btn",
            help="Requires Tank01 RapidAPI key set in the sidebar.",
        ):
            api_key = st.session_state.get("rapidapi_key", "")
            if not api_key:
                st.error("Set your Tank01 RapidAPI key in the sidebar first.")
            else:
                with st.spinner("Fetching live DK pool from Tank01…"):
                    try:
                        live_pool = fetch_live_opt_pool(
                            str(fetch_slate_date_cal),
                            {"RAPIDAPI_KEY": api_key},
                        )
                        if "player_name" not in live_pool.columns and "name" in live_pool.columns:
                            live_pool = live_pool.rename(columns={"name": "player_name"})
                        live_pool = _apply_yakos_projections(live_pool, knobs=st.session_state.get("cal_knobs", {}))
                        # Canonical Lab pipeline: cascade → hard-drop OUT/IR → own_proj
                        live_pool, _cascade, _cal_auto_excl = _process_clean_pool(live_pool)
                        st.session_state["auto_excluded_players"] = _cal_auto_excl
                        st.session_state["injury_cascade"] = _cascade
                        st.session_state["pool_df"] = live_pool
                        st.session_state["sim_player_pool_clean"] = live_pool.copy()
                        st.session_state["pool_date"] = str(fetch_slate_date_cal)
                        # Auto-fetch actuals for past dates
                        _cal_date_str = str(fetch_slate_date_cal)
                        _cal_n_excl = len(_cal_auto_excl)
                        _cal_excl_note = f" {_cal_n_excl} excluded (OUT/IR)." if _cal_n_excl else ""
                        if fetch_slate_date_cal < _today_est():
                            try:
                                _acts = fetch_actuals_from_api(
                                    fetch_slate_date_cal.strftime("%Y%m%d"),
                                    {"RAPIDAPI_KEY": api_key},
                                )
                                st.session_state["actuals"][_cal_date_str] = _acts
                                st.session_state["sim_actuals_df"] = _acts
                                # Store as a HistoricalSlateBundle so pool + actuals
                                # travel together and calibration/sim can find them by date.
                                st.session_state["historical_bundles"][_cal_date_str] = HistoricalSlateBundle(
                                    slate_date=_cal_date_str,
                                    pool_df=live_pool.copy(),
                                    actuals=_acts,
                                    proj_col="proj",
                                )
                                st.success(
                                    f"✅ Loaded {len(live_pool)} players.{_cal_excl_note} "
                                    f"Actuals loaded for {_cal_date_str} ({len(_acts)} players)."
                                )
                            except NoGamesScheduledError:
                                # Store bundle with no actuals for a no-games date
                                st.session_state["historical_bundles"][_cal_date_str] = HistoricalSlateBundle(
                                    slate_date=_cal_date_str,
                                    pool_df=live_pool.copy(),
                                    actuals=None,
                                    proj_col="proj",
                                )
                                st.success(f"✅ Loaded {len(live_pool)} players.{_cal_excl_note}")
                                st.info(f"No games scheduled for {_cal_date_str}.")
                            except Exception as _ae:
                                # Store bundle with no actuals when API call fails
                                st.session_state["historical_bundles"][_cal_date_str] = HistoricalSlateBundle(
                                    slate_date=_cal_date_str,
                                    pool_df=live_pool.copy(),
                                    actuals=None,
                                    proj_col="proj",
                                )
                                st.success(f"✅ Loaded {len(live_pool)} players.{_cal_excl_note}")
                                st.warning(f"Actuals API error: {_ae}")
                        else:
                            st.success(
                                f"✅ Loaded {len(live_pool)} players.{_cal_excl_note} "
                                "Live slate — no actuals yet."
                            )
                    except Exception as _e:
                        st.error(f"API fetch failed: {_e}")

    if rg_upload_cal is not None:
        if st.session_state.get("_pool_df_filename") != rg_upload_cal.name:
            raw_df = pd.read_csv(rg_upload_cal)
            pool_df_cal = rename_rg_columns_to_yakos(raw_df)
            # Canonical Lab pipeline: cascade → hard-drop OUT/IR → own_proj.
            # CSV pools carry RG/API projections directly — no re-projection needed.
            pool_df_cal, _csv_cascade, _csv_excl = _process_clean_pool(pool_df_cal)
            st.session_state["injury_cascade"] = _csv_cascade
            if _csv_excl:
                st.session_state["auto_excluded_players"] = _csv_excl
            st.session_state["pool_df"] = pool_df_cal
            st.session_state["sim_player_pool_clean"] = pool_df_cal.copy()
            st.session_state["_pool_df_filename"] = rg_upload_cal.name
            st.success(f"✅ Pool loaded — {len(pool_df_cal)} players. Head to **🏀 Ricky's Slate Room** to review.")

    current_pool_df = st.session_state.get("pool_df")
    if current_pool_df is not None and not current_pool_df.empty:
        st.caption(f"Active pool: **{len(current_pool_df)} players** loaded.")
    else:
        st.info("No pool loaded yet. Upload a player pool CSV above to begin.")

    st.markdown("---")

    # ── Upload DvP Table ──────────────────────────────────────────────────────
    st.markdown("### 🛡️ Upload DvP Table")
    st.markdown(
        "Upload a FantasyPros Defense vs Position (DvP) CSV to inform edge analysis. "
        "Expected columns: `Team, PG_FPPG_Allowed, SG_FPPG_Allowed, SF_FPPG_Allowed, "
        "PF_FPPG_Allowed, C_FPPG_Allowed` (canonical and human-readable variants accepted)."
    )

    # Staleness banner
    _dvp_age = dvp_staleness_days(_DVP_DEFAULT_PATH)
    if _dvp_age is not None and _dvp_age > DVP_STALE_DAYS:
        st.warning(
            f"⚠️ DvP data is **{_dvp_age:.0f} days** old. "
            "Consider uploading a fresh FantasyPros export."
        )

    # Last-updated indicator
    _dvp_df_cur = st.session_state.get("dvp_table")
    if _dvp_age is not None:
        _dvp_updated_str = (
            pd.Timestamp.now() - pd.Timedelta(days=_dvp_age)
        ).strftime("%Y-%m-%d")
        st.caption(f"Last updated: **{_dvp_updated_str}**")
    elif _dvp_df_cur is None:
        st.caption("Last updated: _not yet uploaded_")

    dvp_upload_col, dvp_info_col = st.columns([2, 1])
    with dvp_upload_col:
        dvp_file = st.file_uploader(
            "Upload DvP CSV",
            type=["csv"],
            key="dvp_upload",
            help="FantasyPros DvP export — one row per NBA team.",
        )
    with dvp_info_col:
        if _dvp_df_cur is not None:
            st.markdown("**League averages (FPPG allowed)**")
            _avgs = st.session_state.get("dvp_league_avgs", {})
            for _pos in ["PG", "SG", "SF", "PF", "C"]:
                if _pos in _avgs:
                    st.markdown(f"- **{_pos}:** {_avgs[_pos]:.1f}")
        else:
            st.info("Upload a DvP table to see league averages.")

    if dvp_file is not None:
        if st.session_state.get("_dvp_filename") != dvp_file.name:
            try:
                _dvp_parsed = parse_dvp_upload(dvp_file)
                save_dvp_table(_dvp_parsed, _DVP_DEFAULT_PATH)
                _dvp_avgs = compute_league_averages(_dvp_parsed)
                st.session_state["dvp_table"] = _dvp_parsed
                st.session_state["dvp_league_avgs"] = _dvp_avgs
                st.session_state["_dvp_filename"] = dvp_file.name
                st.success(
                    f"✅ DvP table loaded — {len(_dvp_parsed)} teams, "
                    f"{len(_dvp_avgs)} positions. Saved to `data/dvp_baseline.csv`."
                )
                st.rerun()
            except Exception as _dvp_err:
                st.error(f"Failed to parse DvP CSV: {_dvp_err}")

    st.markdown("---")

    # Load historical data
    hist_df = load_historical_lineups()

    # ── Calibration KPI Strip ────────────────────────────────────────────────
    _kpis = calibration_kpi_summary(hist_df) if not hist_df.empty else {}

    if not _kpis:
        st.info(
            "No historical data found. Add `data/historical_lineups.csv` to populate these KPIs."
        )
    else:
        _strat = _kpis["strategy"]
        _ptsp = _kpis["points_player"]
        _mins_mae = _kpis["minutes"]["mae"] if "minutes" in _kpis else None
        _own_mae = _kpis["ownership"]["mae"] if "ownership" in _kpis else None
        _hit_rate = _strat["hit_rate"]

        _kpi_targets = {
            "pts": "target ≤ 6 pts",
            "mins": "target ≤ 3 min",
            "own": "target ≤ 3%",
            "hit": "target ≥ 70%",
        }

        def _kpi_card(label: str, value_str: str, metric_key: str, raw_value: float) -> str:
            qc = quality_color(metric_key, raw_value)
            bg = _QUALITY_BG[qc]
            color = _QUALITY_TEXT[qc]
            target_str = _kpi_targets.get(metric_key, "")
            return (
                f'<div style="border:1px solid #3a3a3a;border-radius:6px;'
                f'padding:10px 8px;text-align:center;background:{bg};">'
                f'<div style="font-size:0.72rem;text-transform:uppercase;'
                f'letter-spacing:0.06em;color:#aaa;margin-bottom:4px;">{label}</div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:{color};">{value_str}</div>'
                f'<div style="font-size:0.65rem;color:#888;margin-top:3px;">{target_str}</div>'
                f'</div>'
            )

        ks1, ks2, ks3, ks4 = st.columns(4)
        with ks1:
            st.markdown(
                _kpi_card(
                    "Pts MAE (player)",
                    f"{_ptsp['mae']:.2f} pts",
                    "pts",
                    _ptsp["mae"],
                ),
                unsafe_allow_html=True,
            )
        with ks2:
            if _mins_mae is not None:
                st.markdown(
                    _kpi_card("Min MAE (player)", f"{_mins_mae:.2f} min", "mins", _mins_mae),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    _kpi_card("Min MAE (player)", "N/A", "mins", 0.0),
                    unsafe_allow_html=True,
                )
        with ks3:
            if _own_mae is not None:
                st.markdown(
                    _kpi_card("Own MAE (player)", f"{_own_mae:.2f}%", "own", _own_mae),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    _kpi_card("Own MAE (player)", "N/A", "own", 0.0),
                    unsafe_allow_html=True,
                )
        with ks4:
            st.markdown(
                _kpi_card("Hit rate", f"{_hit_rate:.0%}", "hit", _hit_rate),
                unsafe_allow_html=True,
            )



    st.markdown("---")

    # ── Section A: 3-Step Calibration Workflow ───────────────────────────────
    st.markdown("### A. 📡 Calibration Check")

    # Populate dropdown from all dates where actuals have been fetched.
    # Merge both the legacy actuals dict and the historical_bundles dict
    # so dates fetched via "Fetch Pool from API" always appear.
    _loaded_actuals_dict = st.session_state.get("actuals", {})
    _bundle_dates = {
        d for d, b in st.session_state.get("historical_bundles", {}).items()
        if b.has_valid_actuals()
    }
    _actuals_dates = sorted(set(_loaded_actuals_dict.keys()) | _bundle_dates, reverse=True)

    _step1_col, _step1_override = st.columns([2, 1])
    with _step1_col:
        _run_check = st.button(
            "🔍 Run Calibration Check",
            type="primary",
            key="run_cal_check_btn",
            help="Diagnose projection errors for the most recent completed slate.",
        )
    with _step1_override:
        if _actuals_dates:
            _override_date = st.selectbox(
                "Override slate date",
                _actuals_dates,
                index=0,
                key="cal_check_date_override",
            )
        else:
            st.info(
                "No actuals loaded. Fetch a past-date pool above to enable calibration."
            )
            _override_date = None

    if _run_check:
        # Use load_historical_actuals to check both legacy dict and historical_bundles
        _check_acts = load_historical_actuals(_override_date) if _override_date else None
        if _override_date is None or _check_acts is None:
            st.warning(
                "No actuals for this date. Go to **Load Player Pool** and fetch a past date first."
            )
        else:
            # Prefer bundle pool (has aligned projections) over generic pool_df
            _bundle_for_check = st.session_state.get("historical_bundles", {}).get(_override_date)
            _cal_pool_check = (
                _bundle_for_check.pool_df
                if _bundle_for_check is not None and not _bundle_for_check.pool_df.empty
                else st.session_state.get("pool_df")
            )
            if _cal_pool_check is None or _cal_pool_check.empty:
                st.warning("No pool loaded. Fetch a player pool first.")
            else:
                # Debug: log bundle presence and actuals row count
                _bundle_note = (
                    f"Bundle found for {_override_date}: {len(_check_acts)} actuals rows."
                    if _bundle_for_check is not None
                    else f"Using legacy actuals dict for {_override_date}: {len(_check_acts)} rows."
                )
                st.caption(f"🔍 {_bundle_note}")
                # Build check slice from actuals + pool projections (pool provides proj column)
                _acts_df = _check_acts.copy()
                # Normalise to 'name' / 'actual' columns expected by _diagnose_errors
                if "player_name" in _acts_df.columns:
                    _acts_df = _acts_df.rename(columns={"player_name": "name"})
                if "actual_fp" in _acts_df.columns:
                    _acts_df = _acts_df.rename(columns={"actual_fp": "actual"})
                _check_slice = _acts_df.copy()

                # Merge pool projections/minutes/ownership into the slice
                if "proj" in _cal_pool_check.columns:
                    _pp = (
                        _cal_pool_check[["player_name", "proj"]]
                        .rename(columns={"player_name": "name", "proj": "_pp"})
                        .drop_duplicates("name")
                    )
                    _check_slice = _check_slice.merge(_pp, on="name", how="left")
                    _mask = _check_slice["_pp"].notna()
                    _check_slice.loc[_mask, "proj"] = _check_slice.loc[_mask, "_pp"]
                    _check_slice = _check_slice.drop(columns=["_pp"])
                if "minutes" in _cal_pool_check.columns:
                    _pm = (
                        _cal_pool_check[["player_name", "minutes"]]
                        .rename(columns={"player_name": "name", "minutes": "_pm"})
                        .drop_duplicates("name")
                    )
                    _check_slice = _check_slice.merge(_pm, on="name", how="left")
                    if "proj_minutes" not in _check_slice.columns:
                        _check_slice["proj_minutes"] = _check_slice["_pm"]
                    else:
                        _pmm = _check_slice["proj_minutes"].isna() & _check_slice["_pm"].notna()
                        _check_slice.loc[_pmm, "proj_minutes"] = _check_slice.loc[_pmm, "_pm"]
                    _check_slice = _check_slice.drop(columns=["_pm"])

                _errors = _diagnose_errors(_check_slice)
                st.session_state["cal_check_errors"] = _errors
                st.session_state["cal_check_errors_before"] = _errors.copy()

    _errors = st.session_state.get("cal_check_errors")

    if _errors is not None:
        # Traffic-light scorecard
        _SCORECARD_TARGETS = [
            ("FP MAE", _errors["fp_mae"], 6, "pts"),
            ("Min MAE", _errors["min_mae"], 3, "min"),
            ("Own MAE", _errors["own_mae"], 3, "%"),
        ]
        _sc_cols = st.columns(3)
        for _sc_col, (_metric, _val, _target, _unit) in zip(_sc_cols, _SCORECARD_TARGETS):
            if _val == 0.0:
                _status = "—"
                _card_bg = "#1e1e1e"
                _card_color = "#888"
            elif _val <= _target:
                _status = "✅"
                _card_bg = "#0d2d0d"
                _card_color = "#4caf50"
            elif _val <= _target * 1.5:
                _status = "⚠️"
                _card_bg = "#2d2200"
                _card_color = "#ff9800"
            else:
                _status = "❌"
                _card_bg = "#2d0d0d"
                _card_color = "#f44336"
            _sc_col.markdown(
                f'<div style="border:1px solid #3a3a3a;border-radius:6px;'
                f'padding:10px 8px;text-align:center;background:{_card_bg};">'
                f'<div style="font-size:0.72rem;text-transform:uppercase;'
                f'letter-spacing:0.06em;color:#aaa;margin-bottom:4px;">'
                f'{_status} {_metric}</div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:{_card_color};">'
                f'{_val:.2f} {_unit}</div>'
                f'<div style="font-size:0.65rem;color:#888;margin-top:3px;">'
                f'target ≤ {_target} {_unit}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Step 2: Suggested Fixes ──────────────────────────────────
        st.markdown("#### 💊 Suggested Fixes")
        _current_knobs = st.session_state.get("cal_knobs", {})
        _suggestions = _generate_suggestions(_errors, _current_knobs)

        if not _suggestions:
            st.success("✅ No urgent knob changes recommended — projection accuracy looks good!")
        else:
            for _s in _suggestions:
                _fix_col1, _fix_col2 = st.columns([4, 1])
                _fix_col1.markdown(
                    f"**`{_s['knob']}`**: `{_s['current']}` → `{_s['suggested']}`  \n"
                    f"_{_s['reason']}_"
                )
                if _fix_col2.button("Apply", key=f"apply_sug_{_s['knob']}"):
                    _knobs_updated = dict(st.session_state.get("cal_knobs", {}))
                    _knobs_updated[_s["knob"]] = _s["suggested"]
                    st.session_state["cal_knobs"] = _knobs_updated
                    st.rerun()

        # ── Step 3: Re-project & Compare ────────────────────────────
        st.markdown("")
        if st.button("🔄 Re-project & Compare", key="reproj_compare_btn", type="secondary"):
            _reproj_pool = st.session_state.get("pool_df")
            if _reproj_pool is None or _reproj_pool.empty:
                st.warning("No pool loaded. Load a player pool first.")
            else:
                _active_knobs = st.session_state.get("cal_knobs", {})
                _reproj_pool = _apply_yakos_projections(_reproj_pool, knobs=_active_knobs)
                st.session_state["pool_df"] = _reproj_pool
                # Re-compute errors using same actuals
                _check_date2 = st.session_state.get("cal_check_date_override", _actuals_dates[0] if _actuals_dates else None)
                _acts_df2_raw = load_historical_actuals(_check_date2) if _check_date2 else None
                if _check_date2 and _acts_df2_raw is not None:
                    _acts_df2 = _acts_df2_raw.copy()
                    if "player_name" in _acts_df2.columns:
                        _acts_df2 = _acts_df2.rename(columns={"player_name": "name"})
                    if "actual_fp" in _acts_df2.columns:
                        _acts_df2 = _acts_df2.rename(columns={"actual_fp": "actual"})
                    _check_slice2 = _acts_df2.copy()
                    if "proj" in _reproj_pool.columns:
                        _pp2 = (
                            _reproj_pool[["player_name", "proj"]]
                            .rename(columns={"player_name": "name", "proj": "_pp2"})
                            .drop_duplicates("name")
                        )
                        _check_slice2 = _check_slice2.merge(_pp2, on="name", how="left")
                        _m2 = _check_slice2["_pp2"].notna()
                        _check_slice2.loc[_m2, "proj"] = _check_slice2.loc[_m2, "_pp2"]
                        _check_slice2 = _check_slice2.drop(columns=["_pp2"])
                    _new_errors = _diagnose_errors(_check_slice2)
                    st.session_state["cal_check_errors"] = _new_errors

                    # Before/after comparison
                    _before = st.session_state.get("cal_check_errors_before", {})
                    st.markdown("##### 📊 Before vs After")
                    _ba_cols = st.columns(3)
                    for _ba_col, (_metric, _bkey, _unit) in zip(
                        _ba_cols,
                        [("FP MAE", "fp_mae", "pts"), ("Min MAE", "min_mae", "min"), ("Own MAE", "own_mae", "%")],
                    ):
                        _b_val = _before.get(_bkey, 0.0)
                        _a_val = _new_errors.get(_bkey, 0.0)
                        _delta = _a_val - _b_val
                        _delta_str = f"{_delta:+.2f} {_unit}"
                        _ba_col.metric(
                            _metric,
                            f"{_a_val:.2f} {_unit}",
                            delta=_delta_str,
                            delta_color="inverse",
                        )

    # ── Queue Details from historical CSV (if available) ────────────────────
    if not hist_df.empty:
        queue_df = get_calibration_queue(hist_df, prior_dates=3)
        if st.session_state.get("cal_queue_df") is None:
            st.session_state["cal_queue_df"] = queue_df
        cal_queue = st.session_state["cal_queue_df"]

        if cal_queue is not None and not cal_queue.empty:
            available_dates = sorted(cal_queue["slate_date"].unique(), reverse=True)
            with st.expander("📋 Player Details — Full Queue Table", expanded=False):
                queue_date_sel = st.selectbox(
                    "Queue slate date",
                    available_dates,
                    key="queue_date",
                )
                date_queue = cal_queue[cal_queue["slate_date"] == queue_date_sel]

                # Merge current pool projections into display
                _cal_pool = st.session_state.get("pool_df")
                _queue_display = date_queue.copy()
                if _cal_pool is not None and not _cal_pool.empty:
                    _pool_extra_cols = [
                        c for c in ["floor", "ceil"]
                        if c in _cal_pool.columns and c not in _queue_display.columns
                    ]
                    if "proj_own" not in _queue_display.columns and "ownership" in _cal_pool.columns:
                        _pool_extra_cols.append("ownership")
                    if _pool_extra_cols:
                        _pool_merge = (
                            _cal_pool[["player_name"] + _pool_extra_cols]
                            .rename(columns={"player_name": "name", "ownership": "proj_own"})
                            .groupby("name", as_index=False)
                            .first()
                        )
                        _queue_display = _queue_display.merge(_pool_merge, on="name", how="left")

                    if "proj" in _cal_pool.columns:
                        _pool_proj_df = (
                            _cal_pool[["player_name", "proj"]]
                            .rename(columns={"player_name": "name", "proj": "_pool_proj"})
                            .drop_duplicates("name")
                        )
                        _queue_display = _queue_display.merge(_pool_proj_df, on="name", how="left")
                        _pool_proj_mask = _queue_display["_pool_proj"].notna()
                        _queue_display.loc[_pool_proj_mask, "proj"] = _queue_display.loc[_pool_proj_mask, "_pool_proj"]
                        _queue_display = _queue_display.drop(columns=["_pool_proj"])

                    if "minutes" in _cal_pool.columns:
                        _pool_min_df = (
                            _cal_pool[["player_name", "minutes"]]
                            .rename(columns={"player_name": "name", "minutes": "_pool_minutes"})
                            .drop_duplicates("name")
                        )
                        _queue_display = _queue_display.merge(_pool_min_df, on="name", how="left")
                        if "proj_minutes" not in _queue_display.columns:
                            _queue_display["proj_minutes"] = _queue_display["_pool_minutes"]
                        else:
                            _pm_mask = (
                                _queue_display["proj_minutes"].isna()
                                & _queue_display["_pool_minutes"].notna()
                            )
                            _queue_display.loc[_pm_mask, "proj_minutes"] = _queue_display.loc[_pm_mask, "_pool_minutes"]
                        _queue_display = _queue_display.drop(columns=["_pool_minutes"])

                def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
                    return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else pd.Series(0.0, index=df.index)

                _qd = _queue_display.copy()
                _qd["pts_error"] = _safe_col(_qd, "actual") - _safe_col(_qd, "proj")
                _qd["min_error"] = _safe_col(_qd, "actual_minutes") - _safe_col(_qd, "proj_minutes")
                _qd["own_error"] = _safe_col(_qd, "own") - _safe_col(_qd, "proj_own")
                _qd["Flag"] = (
                    _qd["pts_error"].abs().gt(6)
                    | _qd["min_error"].abs().gt(3)
                    | _qd["own_error"].abs().gt(3)
                )

                _player_col = []
                for _, _r in _qd.iterrows():
                    parts = [str(_r.get("name", ""))]
                    tp = " / ".join(filter(None, [str(_r.get("team", "")), str(_r.get("pos", ""))]))
                    if tp:
                        parts.append(f"({tp})")
                    _player_col.append(" ".join(parts))
                _qd["Player"] = _player_col

                _focused_cols_map = {
                    "Player": "Player",
                    "salary": "Salary",
                    "proj": "Proj FP",
                    "actual": "Act FP",
                    "pts_error": "Error (pts)",
                    "proj_minutes": "Proj Mins",
                    "actual_minutes": "Act Mins",
                    "min_error": "Min Error",
                    "proj_own": "Proj Own %",
                    "own": "Act Own %",
                    "own_error": "Own Error",
                    "Flag": "Flag",
                }
                _focused_avail = [c for c in _focused_cols_map if c in _qd.columns or c == "Player"]
                _queue_focused = _qd[_focused_avail].rename(columns=_focused_cols_map)

                st.dataframe(
                    _queue_focused,
                    column_config={
                        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                        "Proj FP": st.column_config.NumberColumn("Proj FP", format="%.1f"),
                        "Act FP": st.column_config.NumberColumn("Act FP", format="%.1f"),
                        "Error (pts)": st.column_config.NumberColumn("Error (pts)", format="%+.1f"),
                        "Proj Mins": st.column_config.NumberColumn("Proj Mins", format="%.1f"),
                        "Act Mins": st.column_config.NumberColumn("Act Mins", format="%.1f"),
                        "Min Error": st.column_config.NumberColumn("Min Error", format="%+.1f"),
                        "Proj Own %": st.column_config.NumberColumn("Proj Own %", format="%.1f"),
                        "Act Own %": st.column_config.NumberColumn("Act Own %", format="%.1f"),
                        "Own Error": st.column_config.NumberColumn("Own Error", format="%+.1f"),
                        "Flag": st.column_config.CheckboxColumn("Flag", disabled=True),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

    # ---- Section B: Archetype Config Knobs ----
    st.markdown("---")
    st.markdown("### B. 🎛️ Archetype Config Knobs")

    with st.expander("Tune DFS Archetype Configurations", expanded=False):
        st.markdown(
            "Adjust the projection weights for each archetype. "
            "Changes take effect for the next optimizer run."
        )
        arch_sel_knobs = st.selectbox(
            "Archetype to tune",
            list(DFS_ARCHETYPES.keys()),
            key="knobs_arch_sel",
        )
        arch_cfg = DFS_ARCHETYPES[arch_sel_knobs]

        k1, k2 = st.columns(2)
        with k1:
            new_ceil_w = st.slider(
                "Ceiling weight", 0.0, 1.0, float(arch_cfg["ceil_weight"]), step=0.05,
                key="knob_ceil_w",
            )
            new_floor_w = st.slider(
                "Floor weight", 0.0, 1.0, float(arch_cfg["floor_weight"]), step=0.05,
                key="knob_floor_w",
            )
            new_proj_boost = st.slider(
                "Proj boost %", -0.20, 0.20, float(arch_cfg["proj_boost"]), step=0.01,
                key="knob_proj_boost",
            )
        with k2:
            new_own_fade = st.slider(
                "Ownership fade threshold (%)", 0.0, 50.0,
                float(arch_cfg["own_fade_threshold"]), step=1.0,
                key="knob_own_fade",
            )
            new_stack_bonus = st.slider(
                "Stack bonus (FP)", -5.0, 10.0, float(arch_cfg["stack_bonus"]), step=0.5,
                key="knob_stack_bonus",
            )
            new_value_thr = st.slider(
                "Min value threshold (FP/$1K)", 0.0, 7.0,
                float(arch_cfg["value_threshold"]), step=0.1,
                key="knob_value_thr",
            )

        if st.button("Apply knob changes to archetype", key="knobs_apply"):
            DFS_ARCHETYPES[arch_sel_knobs].update({
                "ceil_weight": new_ceil_w,
                "floor_weight": new_floor_w,
                "proj_boost": new_proj_boost,
                "own_fade_threshold": new_own_fade,
                "stack_bonus": new_stack_bonus,
                "value_threshold": new_value_thr,
            })
            st.success(f"Archetype '{arch_sel_knobs}' updated for this session.")

        st.markdown("---")
        st.markdown("**⚙️ Projection Engine Knobs**")
        st.markdown("These knobs control how YakOS blends projection sources and applies contextual adjustments.")

        _cal_knobs = st.session_state.get("cal_knobs", {})

        _pk1, _pk2 = st.columns(2)
        with _pk1:
            _ens_w_yakos = st.slider(
                "YakOS model weight (ensemble_w_yakos)",
                0.0, 1.0,
                float(_cal_knobs.get("ensemble_w_yakos", 0.40)),
                step=0.05,
                key="knob_ens_yakos",
                help="Weight of YakOS model projection in the ensemble blend.",
            )
            _ens_w_tank01 = st.slider(
                "Tank01 weight (ensemble_w_tank01)",
                0.0, 1.0,
                float(_cal_knobs.get("ensemble_w_tank01", 0.30)),
                step=0.05,
                key="knob_ens_tank01",
                help="Weight of Tank01 API projection in the ensemble blend.",
            )
            _ens_w_rg = st.slider(
                "RotoGrinders weight (ensemble_w_rg)",
                0.0, 1.0,
                float(_cal_knobs.get("ensemble_w_rg", 0.30)),
                step=0.05,
                key="knob_ens_rg",
                help="Weight of RotoGrinders projection in the ensemble blend.",
            )
            _ens_total = _ens_w_yakos + _ens_w_tank01 + _ens_w_rg
            if abs(_ens_total - 1.0) > 0.05:
                st.warning(f"⚠️ Ensemble weights sum to {_ens_total:.2f} — should be ~1.0.")
        with _pk2:
            _b2b_discount = st.slider(
                "B2B minutes discount (b2b_discount)",
                0.80, 1.00,
                float(_cal_knobs.get("b2b_discount", 0.93)),
                step=0.01,
                key="knob_b2b_discount",
                help="Multiplier applied to projected minutes for back-to-back games (default 0.93).",
            )
            _blowout_threshold = st.slider(
                "Blowout spread threshold (blowout_threshold)",
                8, 20,
                int(_cal_knobs.get("blowout_threshold", 15)),
                step=1,
                key="knob_blowout_threshold",
                help="Spread (pts) at which blowout minutes reduction kicks in (default 15).",
            )

        st.markdown("**🎯 Sim Calibration Knobs**")
        st.markdown("These knobs tune how the Monte Carlo sim classifies player outcomes (smash / bust) and scales upside / downside distributions.")
        _sk1, _sk2 = st.columns(2)
        with _sk1:
            _ceiling_boost = st.slider(
                "Ceiling boost (ceiling_boost)",
                0.5, 2.0,
                float(_cal_knobs.get("ceiling_boost", 1.0)),
                step=0.05,
                key="knob_ceiling_boost",
                help="Multiplier on upside outcomes above projection (>1 = more boom variance; default 1.0).",
            )
            _floor_dampen = st.slider(
                "Floor dampen (floor_dampen)",
                0.0, 1.5,
                float(_cal_knobs.get("floor_dampen", 1.0)),
                step=0.05,
                key="knob_floor_dampen",
                help="Multiplier on downside outcomes below projection (<1 = tighter floor; default 1.0).",
            )
        with _sk2:
            _smash_threshold = st.slider(
                "Smash threshold (smash_threshold)",
                1.0, 2.0,
                float(_cal_knobs.get("smash_threshold", 1.3)),
                step=0.05,
                key="knob_smash_threshold",
                help="Player smashes when outcome ≥ this × projection (default 1.3 = 30% over proj).",
            )
            _bust_threshold = st.slider(
                "Bust threshold (bust_threshold)",
                0.1, 0.9,
                float(_cal_knobs.get("bust_threshold", 0.5)),
                step=0.05,
                key="knob_bust_threshold",
                help="Player busts when outcome ≤ this × projection (default 0.5 = 50% of proj).",
            )

        if st.button("Save projection knobs", key="knobs_proj_save"):
            st.session_state["cal_knobs"] = {
                "ensemble_w_yakos": _ens_w_yakos,
                "ensemble_w_tank01": _ens_w_tank01,
                "ensemble_w_rg": _ens_w_rg,
                "b2b_discount": _b2b_discount,
                "blowout_threshold": _blowout_threshold,
                "ceiling_boost": _ceiling_boost,
                "floor_dampen": _floor_dampen,
                "smash_threshold": _smash_threshold,
                "bust_threshold": _bust_threshold,
            }
            st.success("Projection knobs saved to session.")

    # ── Re-project Pool with current knobs ──────────────────────────────────
    if st.button("🔄 Re-project Pool with Current Knobs", key="reproj_pool_btn"):
        _reproj_pool = st.session_state.get("pool_df")
        if _reproj_pool is None or _reproj_pool.empty:
            st.warning("No pool loaded. Load a player pool first.")
        else:
            _active_knobs = st.session_state.get("cal_knobs", {})
            _reproj_pool = _apply_yakos_projections(_reproj_pool, knobs=_active_knobs)
            st.session_state["pool_df"] = _reproj_pool
            st.success("Pool re-projected with updated knobs.")
            st.rerun()

    # ---- Section C: Sim Module ----
    st.markdown("---")
    st.markdown("### C. 🎲 Sim Module")

    # ── Mode selector ────────────────────────────────────────────────────────
    _sim_mode_opts = ["🔴 Live — Current Day Slate", "📅 Historical Date"]
    _sim_mode_sel = st.radio(
        "Slate mode",
        _sim_mode_opts,
        horizontal=True,
        key="sim_mode_radio",
        help="Live: runs sims on today's loaded pool. Historical: pick a past date and compare against real outcomes.",
    )
    _sim_is_historical = _sim_mode_sel.startswith("📅")

    if _sim_is_historical:
        _sim_hist_date = st.date_input(
            "Select historical slate date",
            value=st.session_state.get("sim_hist_date") or _today_est(),
            max_value=_today_est(),
            key="sim_hist_date_input",
            help="Choose a past slate date.  Load the matching pool above, then run sims and load actuals to compare.",
        )
        st.session_state["sim_hist_date"] = _sim_hist_date
        st.info(
            f"📅 Historical mode — slate date **{_sim_hist_date}**.  "
            "Make sure the player pool loaded above matches this date, then run sims below.  "
            "Load actuals in the **📊 Load Actuals** expander to compare sim predictions against real outcomes."
        )
    else:
        st.session_state["sim_hist_date"] = None
        st.caption("🔴 Live mode — using today's loaded player pool.")

    pool_for_sim = st.session_state.get("pool_df")
    # In historical mode, prefer the bundle's pool for the selected date so that
    # the projection column is always in sync with the actuals.
    if _sim_is_historical and _sim_hist_date:
        _sim_hist_date_str = str(_sim_hist_date)
        _sim_bundle = st.session_state.get("historical_bundles", {}).get(_sim_hist_date_str)
        if _sim_bundle is not None and not _sim_bundle.pool_df.empty:
            pool_for_sim = _sim_bundle.pool_df
    # Caption: show pool size and active knob summary
    _sim_knobs = st.session_state.get("cal_knobs", {})
    if pool_for_sim is not None and not pool_for_sim.empty:
        if _sim_knobs:
            _knob_summary = (
                f"w_yakos={_sim_knobs.get('ensemble_w_yakos', 0.40):.2f}, "
                f"w_tank01={_sim_knobs.get('ensemble_w_tank01', 0.30):.2f}, "
                f"w_rg={_sim_knobs.get('ensemble_w_rg', 0.30):.2f}, "
                f"b2b={_sim_knobs.get('b2b_discount', 0.93):.2f}, "
                f"blowout_thr={_sim_knobs.get('blowout_threshold', 15)}"
            )
        else:
            _knob_summary = "defaults"
        if "sim_eligible" in pool_for_sim.columns:
            _n_eligible = int(pool_for_sim["sim_eligible"].sum())
        else:
            _n_eligible = len(pool_for_sim)
        st.caption(
            f"Using projections from pool "
            f"(**{_n_eligible} sim-eligible players** out of {len(pool_for_sim)} loaded). "
            f"Knobs: {_knob_summary}."
        )
    if pool_for_sim is None:
        st.info(
            "Load a player pool using the **📂 Load Player Pool** section at the top of this tab to enable sims."
        )
    else:
        # Manual override (advanced) — compact collapsed section
        with st.expander("✏️ Manual Override (advanced)", expanded=False):
            st.markdown(
                "One per row: `PlayerName | STATUS | proj_adj | minutes_change`\n\n"
                "Example: `Zion Williamson | OUT | | ` or `Jayson Tatum | UPGRADED | | +4`"
            )
            news_text = st.text_area(
                "Manual updates",
                placeholder="Zion Williamson | OUT | |\nJayson Tatum | UPGRADED | | 4",
                height=100,
                key="sim_news_text",
            )

        news_updates = []
        if news_text.strip():
            for line in news_text.strip().splitlines():
                parts = [p.strip() for p in line.split("|")]
                if not parts[0]:
                    continue
                update = {"player_name": parts[0]}
                if len(parts) > 1 and parts[1]:
                    update["status"] = parts[1]
                if len(parts) > 2 and parts[2]:
                    try:
                        update["proj_adj"] = float(parts[2])
                    except ValueError:
                        pass
                if len(parts) > 3 and parts[3]:
                    try:
                        update["minutes_change"] = float(parts[3])
                    except ValueError:
                        pass
                news_updates.append(update)

            if news_updates:
                sim_pool = simulate_live_updates(pool_for_sim, news_updates)
                out_names_manual = [
                    u["player_name"] for u in news_updates
                    if u.get("status", "").upper() in {"OUT", "IR", "SUSPENDED", "G-LEAGUE"}
                ]
                # Drop OUT/IR players entirely rather than flagging sim_eligible so the
                # pool stays clean and consistent with the cascade-then-drop pattern.
                if out_names_manual and "player_name" in sim_pool.columns:
                    sim_pool = sim_pool[
                        ~sim_pool["player_name"].isin(out_names_manual)
                    ].reset_index(drop=True)
                st.session_state["sim_pool_df"] = sim_pool
                st.session_state["sim_pool_orig_df"] = sim_pool

        # Sim controls
        sim_col_l, sim_col_r = st.columns(2)
        with sim_col_l:
            sim_n_lu = st.slider("Lineups to sim", 1, 150, 20, key="sim_n_lu")
            sim_n_sims = st.slider("Sim iterations", 100, 2000, 500, step=100, key="sim_n_sims")
            sim_max_pair = st.number_input(
                "Max pair appearances",
                min_value=0,
                max_value=max(1, sim_n_lu),
                value=max(1, sim_n_lu // 4),
                step=1,
                key="sim_max_pair",
                help=(
                    "Lineup diversity: max lineups any two players can share. "
                    "Lower = more diverse builds. Default = 25% of lineup count."
                ),
            )
        with sim_col_r:
            _sim_prev_preset = st.session_state.get("sim_contest_preset", "20-Max GPP")
            sim_contest_preset_sel = st.selectbox(
                "Contest Type",
                CONTEST_PRESET_LABELS,
                index=CONTEST_PRESET_LABELS.index(_sim_prev_preset)
                if _sim_prev_preset in CONTEST_PRESET_LABELS
                else 3,
                key="sim_contest_preset",
                help=(
                    "Cash Game = high-floor 50/50 | "
                    "Single Entry = balanced 1-lineup | "
                    "3-Max = small GPP | "
                    "20-Max GPP = tournament | "
                    "MME = mass multi-entry | "
                    "Showdown = single-game Captain mode"
                ),
            )
            _sim_preset = CONTEST_PRESETS[sim_contest_preset_sel]
            st.caption(f"📌 {_sim_preset['description']}")
            sim_archetype = _sim_preset["archetype"]
            sim_dk_contest = sim_contest_preset_sel
            sim_vol = _sim_preset["volatility"]
            sim_min_sal = st.number_input(
                "Min salary", 0, 50000,
                int(_sim_preset["min_salary"]),
                step=500, key="sim_min_sal",
            )

        # Actuals upload — for historical slate calibration
        with st.expander("📊 Load Actuals (Historical Slate Calibration)", expanded=_sim_is_historical):
            # Determine the relevant date for actuals lookup.
            # In historical mode use the selected date; otherwise use the pool date.
            _hist_date_str = str(_sim_hist_date) if _sim_is_historical and _sim_hist_date else None
            _pool_date_str = st.session_state.get("pool_date", "")
            _actuals_lookup_date = _hist_date_str or _pool_date_str

            # --- Auto-populate sim_actuals_df from HistoricalSlateBundle when available ---
            if _actuals_lookup_date:
                _bundle_acts = load_historical_actuals(_actuals_lookup_date)
                _current_acts = st.session_state.get("sim_actuals_df")
                _bundle = st.session_state.get("historical_bundles", {}).get(_actuals_lookup_date)
                if _bundle_acts is not None and (_current_acts is None or _current_acts.empty):
                    # Auto-populate from bundle so user doesn't need a CSV upload
                    st.session_state["sim_actuals_df"] = _bundle_acts

            # Debug: show bundle presence and row count
            _bundle_present = _actuals_lookup_date in st.session_state.get("historical_bundles", {})
            _bundle_acts_count = 0
            if _bundle_present:
                _b = st.session_state["historical_bundles"][_actuals_lookup_date]
                _bundle_acts_count = len(_b.actuals) if _b.actuals is not None else 0
            st.caption(
                f"🔍 Bundle for **{_actuals_lookup_date or '—'}**: "
                f"{'✅ found' if _bundle_present else '❌ not found'}"
                + (f" · {_bundle_acts_count} actuals rows" if _bundle_present else "")
            )

            # Show status of already-loaded actuals (auto-loaded when pool was fetched)
            _loaded_actuals_status = st.session_state.get("sim_actuals_df")
            _has_actuals = _loaded_actuals_status is not None and not _loaded_actuals_status.empty
            if _has_actuals:
                st.info(
                    f"Actuals loaded for {_actuals_lookup_date}: "
                    f"**{len(_loaded_actuals_status)} players with box scores.** "
                    "Actuals are auto-loaded when you fetch a past-date pool."
                )
            else:
                st.info(
                    "No actuals loaded. Fetch a past-date pool via **'Fetch Pool from API'** "
                    "to auto-load actuals, or upload a CSV below."
                )

            # CSV upload is the fallback — only shown when no actuals are present
            if not _has_actuals:
                st.markdown("**📂 Upload actuals CSV** (fallback — RotoGrinders export or custom)")
            _actuals_upload = st.file_uploader(
                "Upload actuals CSV (fallback)",
                type=["csv"],
                key="sim_actuals_upload",
                help="RotoGrinders contest export or any CSV with player names and actual FP scored. Only needed when actuals are not auto-loaded from the API.",
                label_visibility="collapsed" if _has_actuals else "visible",
            )
            if _actuals_upload is not None:
                try:
                    _acts_raw = pd.read_csv(_actuals_upload)
                    # Normalise column names — apply only the first match found so
                    # we don't create duplicate target columns when, e.g., both
                    # "PLAYER" and "Player" happen to be present.
                    _col_renames: dict = {}
                    for _src in ("PLAYER", "Player"):
                        if _src in _acts_raw.columns and "player_name" not in _acts_raw.columns:
                            _col_renames[_src] = "player_name"
                            break
                    for _src in ("FPTS", "fpts"):
                        if _src in _acts_raw.columns and "actual_fp" not in _acts_raw.columns:
                            _col_renames[_src] = "actual_fp"
                            break
                    # Capture actual minutes when available (RG export: "MIN" or "Minutes")
                    for _src in ("MIN", "Minutes", "minutes", "actual_minutes"):
                        if _src in _acts_raw.columns and "actual_minutes" not in _acts_raw.columns:
                            _col_renames[_src] = "actual_minutes"
                            break
                    _acts_norm = _acts_raw.rename(columns=_col_renames)
                    # Accept 'actual' column directly too
                    if "actual" in _acts_norm.columns and "actual_fp" not in _acts_norm.columns:
                        _acts_norm = _acts_norm.rename(columns={"actual": "actual_fp"})
                    if "name" in _acts_norm.columns and "player_name" not in _acts_norm.columns:
                        _acts_norm = _acts_norm.rename(columns={"name": "player_name"})
                    _name_col = "player_name" if "player_name" in _acts_norm.columns else None
                    _fp_col = "actual_fp" if "actual_fp" in _acts_norm.columns else None
                    if _name_col and _fp_col:
                        _keep_cols = [_name_col, _fp_col]
                        if "actual_minutes" in _acts_norm.columns:
                            _keep_cols.append("actual_minutes")
                        _acts_clean = _acts_norm[_keep_cols].copy()
                        _acts_clean[_fp_col] = pd.to_numeric(_acts_clean[_fp_col], errors="coerce")
                        if "actual_minutes" in _acts_clean.columns:
                            _acts_clean["actual_minutes"] = pd.to_numeric(_acts_clean["actual_minutes"], errors="coerce")
                        _acts_clean = _acts_clean.dropna(subset=[_fp_col])
                        st.session_state["sim_actuals_df"] = _acts_clean
                        st.success(f"✅ Loaded actuals for {len(_acts_clean)} players.")
                    else:
                        st.error(
                            "Could not find required columns. Expected `PLAYER`/`name`/`player_name` "
                            "and `FPTS`/`actual`/`actual_fp`."
                        )
                except Exception as _e:
                    st.error(f"Error reading actuals file: {_e}")

            _loaded_actuals = st.session_state.get("sim_actuals_df")
            if _loaded_actuals is not None and not _loaded_actuals.empty:
                st.caption(f"Actuals loaded: **{len(_loaded_actuals)} players**")
                if _sim_is_historical:
                    # Historical sanity check: de-activate players with 0 actual FP in pool
                    _act_name_col = "player_name" if "player_name" in _loaded_actuals.columns else "name"
                    _act_fp_col = "actual_fp" if "actual_fp" in _loaded_actuals.columns else "actual"
                    if _act_name_col in _loaded_actuals.columns and _act_fp_col in _loaded_actuals.columns:
                        _zero_min_players = _loaded_actuals[
                            pd.to_numeric(_loaded_actuals[_act_fp_col], errors="coerce").fillna(0) <= 0
                        ][_act_name_col].tolist()
                        _cur_pool_raw = st.session_state.get("sim_pool_df")
                        _cur_pool = _cur_pool_raw if _cur_pool_raw is not None and not _cur_pool_raw.empty else pool_for_sim
                        if _cur_pool is not None and not _cur_pool.empty:
                            _pc_name_col = "player_name" if "player_name" in _cur_pool.columns else "name"
                            if _pc_name_col in _cur_pool.columns:
                                # Only de-activate players who are in the actuals with 0 FP
                                _pool_updated = _cur_pool.copy()
                                if "sim_eligible" not in _pool_updated.columns:
                                    _pool_updated["sim_eligible"] = True
                                _pool_updated.loc[
                                    _pool_updated[_pc_name_col].isin(_zero_min_players), "sim_eligible"
                                ] = False
                                st.session_state["sim_pool_df"] = _pool_updated
                                if _zero_min_players:
                                    st.info(
                                        f"🔍 Sanity check: dropped **{len(_zero_min_players)}** player(s) "
                                        f"with 0 actual FP from sim-eligible pool."
                                    )
                if st.button("Clear actuals", key="sim_clear_actuals_btn"):
                    st.session_state["sim_actuals_df"] = None
                    st.rerun()

        _active_sim_pool_raw = st.session_state.get("sim_pool_df")
        active_sim_pool = _active_sim_pool_raw if _active_sim_pool_raw is not None and not _active_sim_pool_raw.empty else pool_for_sim

        # ── Sim Player Filters ────────────────────────────────────────────
        with st.expander("🔧 Sim Player Filters", expanded=False):
            _sf_col_l, _sf_col_r = st.columns([1, 1])
            with _sf_col_l:
                _sf_excl_out = st.checkbox(
                    "Exclude OUT/IR/Suspended players",
                    value=st.session_state.get("sf_excl_out", True),
                    key="sf_excl_out",
                )
                _sf_excl_team = st.checkbox(
                    "Exclude players not on today's team list",
                    value=st.session_state.get("sf_excl_team", False),
                    key="sf_excl_team",
                )
            with _sf_col_r:
                _sf_min_min = st.slider(
                    "Min projected minutes",
                    min_value=0,
                    max_value=20,
                    value=st.session_state.get("sf_min_min", 4),
                    step=1,
                    key="sf_min_min",
                    help="Players projected for ≤ this many minutes are excluded from sims.",
                )

            # Derive today's team list from pool when team-filter is active
            _today_teams = None
            if _sf_excl_team and "team" in active_sim_pool.columns:
                _today_teams = active_sim_pool["team"].dropna().unique().tolist()

            # Determine which minutes column to use:
            # - Live slate  → proj_minutes (forward-looking)
            # - Historical  → actual_minutes when available from actuals (backward-looking)
            _actuals_for_mins = st.session_state.get("sim_actuals_df")
            _sf_minutes_col: Optional[str]
            if _sim_is_historical and _actuals_for_mins is not None and "actual_minutes" in _actuals_for_mins.columns:
                # Merge actual_minutes from actuals into active_sim_pool so compute_sim_eligible can use them
                _name_col_pool = "player_name" if "player_name" in active_sim_pool.columns else "name"
                _name_col_acts = "player_name" if "player_name" in _actuals_for_mins.columns else "name"
                _mins_lookup = (
                    _actuals_for_mins[[_name_col_acts, "actual_minutes"]]
                    .rename(columns={_name_col_acts: _name_col_pool})
                    .drop_duplicates(subset=[_name_col_pool])
                )
                active_sim_pool = active_sim_pool.drop(columns=["actual_minutes"], errors="ignore")
                active_sim_pool = active_sim_pool.merge(_mins_lookup, on=_name_col_pool, how="left")
                _sf_minutes_col = "actual_minutes"
                st.caption("📅 Historical mode — using **actual minutes** for sim eligibility filter.")
            else:
                _sf_minutes_col = "proj_minutes"

            # Recompute sim_eligible on the current active pool
            active_sim_pool = compute_sim_eligible(
                active_sim_pool,
                min_proj_minutes=float(_sf_min_min),
                exclude_out_ir=_sf_excl_out,
                today_teams=_today_teams,
                minutes_col=_sf_minutes_col,
            )
            # Persist updated eligibility back to session state
            _src_key = "sim_pool_df" if st.session_state.get("sim_pool_df") is not None else "pool_df"
            st.session_state[_src_key] = active_sim_pool

            _n_elig = int(active_sim_pool["sim_eligible"].sum())
            _n_total = len(active_sim_pool)
            st.caption(f"**Using {_n_elig} sim-eligible players out of {_n_total} loaded.**")
            # Sanity check: OUT/IR players should never be in the sim pool after cascade-then-drop.
            # This count should always be 0; if it isn't the load path bypassed the drop step.
            if "status" in active_sim_pool.columns:
                _out_count = int(
                    active_sim_pool["status"].fillna("").str.upper().isin({"OUT", "IR"}).sum()
                )
                st.caption(f"OUT players in sim pool: {_out_count}")

        # ── Manual Include / Exclude Table ───────────────────────────────
        with st.expander("📋 Manual Player Eligibility Overrides", expanded=False):
            st.markdown(
                "Edit the **Sim Eligible** checkbox to manually include or exclude players. "
                "Changes take effect immediately on the next sim run."
            )
            _me_filter_col, _me_search_col = st.columns([1, 2])
            with _me_filter_col:
                _me_show_non = st.checkbox(
                    "Show only non-eligible players",
                    value=False,
                    key="me_show_non_eligible",
                )
            with _me_search_col:
                _me_search = st.text_input(
                    "Search by name / team",
                    value="",
                    key="me_search_text",
                    placeholder="e.g. LeBron or LAL",
                )

            _me_pool = active_sim_pool.copy()
            _me_name_col = "player_name" if "player_name" in _me_pool.columns else (
                "name" if "name" in _me_pool.columns else None
            )
            if _me_name_col:
                _me_disp_cols = [_me_name_col]
                if "team" in _me_pool.columns:
                    _me_disp_cols.append("team")
                if "status" in _me_pool.columns:
                    _me_disp_cols.append("status")
                if "minutes" in _me_pool.columns:
                    _me_disp_cols.append("minutes")
                if "proj" in _me_pool.columns:
                    _me_disp_cols.append("proj")
                _me_disp_cols.append("sim_eligible")

                _me_view = _me_pool[_me_disp_cols].copy().reset_index(drop=True)
                if _me_show_non:
                    _me_view = _me_view[~_me_view["sim_eligible"]]
                if _me_search.strip():
                    _srch = _me_search.strip().lower()
                    _name_match = _me_view[_me_name_col].astype(str).str.lower().str.contains(_srch, na=False)
                    if "team" in _me_view.columns:
                        _team_match = _me_view["team"].astype(str).str.lower().str.contains(_srch, na=False)
                        _me_view = _me_view[_name_match | _team_match]
                    else:
                        _me_view = _me_view[_name_match]

                _edited = st.data_editor(
                    _me_view,
                    column_config={
                        "sim_eligible": st.column_config.CheckboxColumn("Sim Eligible", help="Uncheck to exclude from sims"),
                        _me_name_col: st.column_config.TextColumn("Player", disabled=True),
                    },
                    disabled=[c for c in _me_disp_cols if c != "sim_eligible"],
                    use_container_width=True,
                    height=300,
                    key="me_data_editor",
                )
                # Apply manual edits back to active_sim_pool
                if _edited is not None and not _edited.empty:
                    _edit_idx = _edited.index
                    active_sim_pool.loc[_edit_idx, "sim_eligible"] = _edited["sim_eligible"].values
                    _src_key2 = "sim_pool_df" if st.session_state.get("sim_pool_df") is not None else "pool_df"
                    st.session_state[_src_key2] = active_sim_pool

        if st.button("🎲 Run Sims", type="primary", key="sim_run_btn"):
            with st.spinner("Building lineups + running Monte Carlo sims..."):
                try:
                    # Auto-refresh injury statuses before running sims
                    if not _sim_is_historical:
                        _refresh_api_key = st.session_state.get("rapidapi_key", "")
                        active_sim_pool, _refresh_changes = _refresh_injury_statuses(
                            active_sim_pool, _refresh_api_key
                        )
                        if _refresh_changes:
                            _src_refresh = "sim_pool_df" if st.session_state.get("sim_pool_df") is not None else "pool_df"
                            st.session_state[_src_refresh] = active_sim_pool
                            st.info(
                                "**Status refresh:** "
                                + " | ".join(_refresh_changes[:8])
                                + (" …" if len(_refresh_changes) > 8 else "")
                            )
                    # Filter to sim-eligible players only
                    pool_for_sim_run = active_sim_pool[active_sim_pool["sim_eligible"]].copy() if "sim_eligible" in active_sim_pool.columns else active_sim_pool.copy()
                    # Hard guardrail: no OUT players allowed in the sim pool.
                    # Uses injury_status column when available; falls back to status.
                    if "injury_status" in pool_for_sim_run.columns:
                        _bad = pool_for_sim_run[pool_for_sim_run["injury_status"].eq("Out")]
                    elif "status" in pool_for_sim_run.columns:
                        _bad = pool_for_sim_run[
                            pool_for_sim_run["status"].fillna("").str.upper().isin({"OUT", "IR", "O"})
                        ]
                    else:
                        _bad = pd.DataFrame()
                    if not _bad.empty:
                        _bad_cols = [c for c in ["player_name", "player", "injury_status", "status"] if c in _bad.columns]
                        raise RuntimeError(
                            "OUT players still in sim pool: "
                            + str(_bad[_bad_cols].to_dict("records"))
                        )
                    _required_roster = 8
                    if len(pool_for_sim_run) < _required_roster + 5:
                        st.error(
                            f"Only {len(pool_for_sim_run)} sim-eligible players — need at least "
                            f"{_required_roster + 5} to build valid lineups. "
                            "Loosen your Sim Player Filters or check your pool."
                        )
                    else:
                        sim_lu_df, _ = run_optimizer(
                            pool_for_sim_run,
                            num_lineups=sim_n_lu,
                            max_exposure=0.4,
                            min_salary_used=sim_min_sal,
                            proj_col="proj",
                            archetype=sim_archetype,
                            max_pair_appearances=int(st.session_state.get("sim_max_pair", max(1, sim_n_lu // 4))),
                        )
                        if sim_lu_df is not None and not sim_lu_df.empty:
                            _sim_cal_knobs = st.session_state.get("cal_knobs", {})
                            # Map contest preset -> ContestType enum for dynamic thresholds
                            _internal_ct = _sim_preset["internal_contest"]
                            _sim_contest_type = _INTERNAL_CT_TO_SIM_TYPE.get(
                                _internal_ct, _SimContestType.GPP_LARGE
                            )
                            sim_res, _sim_lineup_scores = run_monte_carlo_for_lineups(
                                sim_lu_df, n_sims=sim_n_sims, volatility_mode=sim_vol,
                                contest_type=_sim_contest_type,
                                _return_scores=True,
                            )
                            # Annotate with Ricky confidence
                            annotated_sim = ricky_annotate(sim_lu_df, sim_res)
                            st.session_state["sim_lineups_df"] = annotated_sim
                            st.session_state["sim_results_df"] = sim_res
                            st.session_state["sim_lineup_scores"] = _sim_lineup_scores
                            # Ensure ownership is populated in the pool before anomaly calc.
                            # Use the same cleaned pool as sims; apply OUT filter as a hard guardrail.
                            _anomaly_base = st.session_state["sim_player_pool_clean"] if "sim_player_pool_clean" in st.session_state else None
                            if _anomaly_base is None or _anomaly_base.empty:
                                _anomaly_base = pool_for_sim_run
                            players_df = _anomaly_base.copy()
                            if "injury_status" in players_df.columns:
                                players_df = players_df[~players_df["injury_status"].eq("Out")].copy()
                                # Hard guardrail: verify the filter actually removed all OUT players
                                _post_filter_bad = players_df[players_df["injury_status"].eq("Out")]
                                if not _post_filter_bad.empty:
                                    raise RuntimeError(
                                        "OUT players in anomalies table: "
                                        + _post_filter_bad[["player_name", "injury_status"]].to_json(orient="records")
                                    )
                            st.session_state["sim_anomaly_out_count"] = int(
                                players_df["injury_status"].eq("Out").sum()
                                if "injury_status" in players_df.columns else 0
                            )
                            _anomaly_pool = apply_ownership(players_df)
                            # Compute per-player anomaly table using cal_knobs
                            _anomaly_df = compute_player_anomaly_table(
                                _anomaly_pool,
                                sim_lu_df,
                                n_sims=sim_n_sims,
                                cal_knobs=_sim_cal_knobs,
                            )
                            st.session_state["sim_anomaly_df"] = _anomaly_df
                            # Clear any previous custom lineup when sims are re-run
                            st.session_state["sim_custom_lineup"] = []
                            st.success(
                                f"Sims complete — {sim_n_lu} lineups × {sim_n_sims} iterations "
                                f"({len(pool_for_sim_run)} sim-eligible players)."
                            )
                        else:
                            st.error("Optimizer returned no lineups. Check pool and settings.")
                except Exception as e:
                    st.error(f"Sim error: {e}")

        sim_lu_df = st.session_state.get("sim_lineups_df")
        sim_res = st.session_state.get("sim_results_df")

        if sim_lu_df is not None and not sim_lu_df.empty and sim_res is not None and not sim_res.empty:
            st.markdown("#### Sim Results")
            sim_kpi = st.columns(4)
            sim_kpi[0].metric("Lineups", len(sim_res))
            sim_kpi[1].metric("Avg sim score", f"{sim_res['sim_mean'].mean():.1f}")
            sim_kpi[2].metric("Avg smash prob", f"{sim_res['smash_prob'].mean():.1%}")
            sim_kpi[3].metric("Avg bust prob", f"{sim_res['bust_prob'].mean():.1%}")

            with st.expander("Lineup-level sim metrics", expanded=True):
                st.markdown(
                    "**Column guide** — each row is one optimizer lineup simulated across "
                    "N iterations using per-player normal distributions:\n\n"
                    "| Column | What it means |\n"
                    "|--------|---------------|\n"
                    "| **Lineup #** | Lineup identifier (matches the Optimizer view) |\n"
                    "| **Avg Score** | Mean total FP across all sim iterations — best single-number estimate of lineup strength |\n"
                    "| **Std Dev** | Score variability. High = boom-or-bust GPP profile; low = steady cash-game floor |\n"
                    "| **Smash %** | Fraction of sims where this lineup scored ≥ contest smash threshold (p90 of full field) — varies naturally across lineups |\n"
                    "| **Bust %** | Fraction of sims where this lineup scored ≤ contest bust threshold (p30 of full field) — varies naturally across lineups |\n"
                    "| **Median Score** | Middle-of-distribution outcome (50th percentile) |\n"
                    "| **P85 (Upside)** | 85th-percentile score — what the lineup looks like on a good night |\n"
                    "| **P15 (Floor)** | 15th-percentile score — downside floor on a bad night |"
                )
                _sim_display = sim_res.sort_values("smash_prob", ascending=False).rename(columns={
                    "lineup_index": "Lineup #",
                    "sim_mean": "Avg Score",
                    "sim_std": "Std Dev",
                    "smash_prob": "Smash %",
                    "bust_prob": "Bust %",
                    "median_points": "Median Score",
                    "sim_p85": "P85 (Upside)",
                    "sim_p15": "P15 (Floor)",
                    "smash_threshold": "Smash Threshold",
                    "bust_threshold": "Bust Threshold",
                    "contest_type": "Contest Type",
                })
                st.dataframe(_sim_display, use_container_width=True, height=300)

            # ── Sim Lineup Browser ────────────────────────────────────────────
            st.markdown("#### 📋 Lineup Browser")
            st.caption("Browse generated lineups one at a time. Use ◀ ▶ to navigate.")
            _sim_unique_lu = sorted(sim_lu_df["lineup_index"].unique())
            _sim_lu_pos = max(0, min(len(_sim_unique_lu) - 1, st.session_state.get("sim_lu_nav", 0)))
            _slb_c1, _slb_c2, _slb_c3 = st.columns([1, 4, 1])
            with _slb_c1:
                if st.button("◀", key="sim_lu_prev", disabled=(_sim_lu_pos == 0)):
                    st.session_state["sim_lu_nav"] = _sim_lu_pos - 1
                    st.rerun()
            with _slb_c2:
                st.markdown(
                    f"<div style='text-align:center;font-size:1.05em'>"
                    f"Lineup {_sim_lu_pos + 1} of {len(_sim_unique_lu)}</div>",
                    unsafe_allow_html=True,
                )
            with _slb_c3:
                if st.button("▶", key="sim_lu_next", disabled=(_sim_lu_pos == len(_sim_unique_lu) - 1)):
                    st.session_state["sim_lu_nav"] = _sim_lu_pos + 1
                    st.rerun()

            _sim_cur_id = _sim_unique_lu[_sim_lu_pos]
            _sim_cur_rows = sim_lu_df[sim_lu_df["lineup_index"] == _sim_cur_id].copy()
            _sim_cur_res = sim_res[sim_res["lineup_index"] == _sim_cur_id]
            _sim_smash = (
                float(pd.to_numeric(_sim_cur_res["smash_prob"].iloc[0], errors="coerce") or 0)
                if not _sim_cur_res.empty else 0.0
            )
            st.markdown(
                f"**Lineup {_sim_lu_pos + 1} — ${int(_sim_cur_rows['salary'].sum()):,} | "
                f"{_sim_cur_rows['proj'].sum():.2f} pts** · Smash {_sim_smash:.1%}"
            )
            render_lineup_card(_sim_cur_rows, pool_df=active_sim_pool)

            # Per-lineup Publish to Slate Room button
            if st.button(
                f"✅ Publish Lineup {_sim_lu_pos + 1} to Slate Room",
                key=f"sim_pub_lu_{_sim_lu_pos}",
            ):
                if _sim_cur_res.empty:
                    st.warning("Sim results not available for this lineup — cannot publish.")
                else:
                    _pub_slate = getattr(
                        st.session_state.get("pool_df", pd.DataFrame()),
                        "attrs", {}
                    ).get("slate", f"NBA {_today_est()}")
                    _pub_arch = CONTEST_PRESET_ARCH_LABELS.get(sim_dk_contest, sim_dk_contest)
                    _pub_approved = build_approved_lineups(
                        lineups_df=_sim_cur_rows,
                        sim_results=_sim_cur_res,
                        contest_archetype=_pub_arch,
                        site="DK",
                        slate=_pub_slate,
                    )
                    st.session_state["approved_lineups"].extend(_pub_approved)
                    entry_pub = {
                        "label": f"Lineup {_sim_lu_pos + 1} — {sim_dk_contest} / {sim_archetype}",
                        "lineups_df": _sim_cur_rows,
                        "metadata": {"contest_type": sim_dk_contest, "archetype": sim_archetype, "n_lineups": 1},
                    }
                    st.session_state["promoted_lineups"].append(entry_pub)
                    st.success(f"Lineup {_sim_lu_pos + 1} published to Slate Room ✅")

            # ── Sim Anomaly Detection ─────────────────────────────────────────
            _sim_anomaly = st.session_state.get("sim_anomaly_df")
            if _sim_anomaly is not None and not _sim_anomaly.empty:
                high_leverage_count = int((_sim_anomaly["Flag"] == "🔥 HIGH LEVERAGE").sum())
                value_trap_count = int(_sim_anomaly["Value Trap"].sum())
                _top_play = _sim_anomaly.iloc[0] if len(_sim_anomaly) > 0 else None
                _summary_parts = [
                    f"Sim ran **{sim_n_sims}** iterations across **{sim_n_lu}** lineups.",
                    f"Found **{high_leverage_count}** high-leverage players, **{value_trap_count}** value traps.",
                ]
                if _top_play is not None:
                    _lev_val = _top_play["Leverage Score"]
                    _lev_str = f"{_lev_val:.2f}" if not (isinstance(_lev_val, float) and __import__("math").isnan(_lev_val)) else "N/A"
                    _summary_parts.append(
                        f"Top leverage play: **{_top_play['Player']}** "
                        f"({_top_play['Smash%']:.1f}% smash, {_top_play['Own%']:.1f}% owned, leverage {_lev_str}x)."
                    )
                st.info("  \n".join(_summary_parts))

                st.markdown("#### 🔍 Sim Anomalies — Leverage Spots")
                st.caption(
                    "Per-player simulation breakdown. "
                    "Smash%/Bust% are computed against the contest-level field threshold — "
                    "values vary naturally across players (not locked at 10%/30% by construction). "
                    "Leverage Score = Smash% / Own% (NaN when Own% < 0.1). "
                    "🔥 HIGH LEVERAGE = score ≥ 3 and Own% ≤ 15. ⚠️ Value Trap = bust rate > 40% despite high salary."
                )
                _anomaly_display = _sim_anomaly.copy()
                _anomaly_display["Value Trap"] = _anomaly_display["Value Trap"].apply(
                    lambda x: "⚠️ VALUE TRAP" if x else ""
                )
                # Allow click-to-filter: show high-leverage or value-trap filter buttons
                _af_col1, _af_col2, _af_col3 = st.columns([1, 1, 2])
                _show_hl = _af_col1.checkbox("🔥 Show only High Leverage", key="sim_filter_hl")
                _show_vt = _af_col2.checkbox("⚠️ Show only Value Traps", key="sim_filter_vt")
                if _show_hl:
                    _anomaly_display = _anomaly_display[_sim_anomaly["Flag"] == "🔥 HIGH LEVERAGE"]
                elif _show_vt:
                    _anomaly_display = _anomaly_display[_sim_anomaly["Value Trap"]]
                st.dataframe(_anomaly_display, use_container_width=True, height=350)
                _out_count = st.session_state.get("sim_anomaly_out_count", 0)
                st.caption(f"OUT players in anomalies source: {_out_count}")

            # ── Sim Diagnostics — Exposure & Eligibility ──────────────────────
            if sim_lu_df is not None and not sim_lu_df.empty:
                _diag_name_col = "player_name" if "player_name" in sim_lu_df.columns else (
                    "name" if "name" in sim_lu_df.columns else None
                )
                if _diag_name_col:
                    _diag_total_lu = sim_lu_df["lineup_index"].nunique()
                    _exposure_series = (
                        sim_lu_df.groupby(_diag_name_col)["lineup_index"]
                        .nunique()
                        .div(_diag_total_lu)
                        .mul(100)
                        .round(1)
                        .reset_index()
                        .rename(columns={"lineup_index": "Exposure %", _diag_name_col: "Player"})
                        .sort_values("Exposure %", ascending=False)
                    )

                    with st.expander("📊 Sim Diagnostics — Player Exposure", expanded=False):
                        st.caption("Top 20 players by exposure across all sim lineups.")
                        _top20_exp = _exposure_series.head(20)

                        # Bar chart of exposures
                        try:
                            import altair as alt
                            _exp_chart = (
                                alt.Chart(_top20_exp)
                                .mark_bar()
                                .encode(
                                    x=alt.X("Player:N", sort="-y", title=None),
                                    y=alt.Y("Exposure %:Q", title="Exposure %"),
                                    tooltip=["Player", "Exposure %"],
                                )
                                .properties(height=300)
                            )
                            st.altair_chart(_exp_chart, use_container_width=True)
                        except Exception:
                            st.bar_chart(_top20_exp.set_index("Player")["Exposure %"])

                        # Enrich with status and sim_eligible from pool
                        _diag_pool = active_sim_pool.copy()
                        _dp_name_col = "player_name" if "player_name" in _diag_pool.columns else "name"
                        _diag_extra_cols = [_dp_name_col]
                        for _ec in ("status", "minutes", "proj", "sim_eligible"):
                            if _ec in _diag_pool.columns:
                                _diag_extra_cols.append(_ec)
                        _diag_pool_sub = (
                            _diag_pool[_diag_extra_cols]
                            .drop_duplicates(subset=[_dp_name_col])
                            .rename(columns={_dp_name_col: "Player"})
                        )
                        _diag_merged = _exposure_series.merge(_diag_pool_sub, on="Player", how="left")

                        # Warn if ineligible/non-healthy players have exposure
                        _inelig_exposed = _diag_merged[
                            (~_diag_merged["sim_eligible"])
                            & (_diag_merged["Exposure %"] > 0)
                        ] if "sim_eligible" in _diag_merged.columns else pd.DataFrame()
                        _non_healthy = _diag_merged[
                            (~_diag_merged["status"].isin(["", "-", "P", "Q", "GTD"]))
                            & (_diag_merged["Exposure %"] > 0)
                        ] if "status" in _diag_merged.columns else pd.DataFrame()
                        if not _inelig_exposed.empty:
                            st.warning(
                                f"⚠️ {len(_inelig_exposed)} player(s) marked sim_eligible=False "
                                "still appear in sim lineups — re-run sims to apply latest filters."
                            )
                        if not _non_healthy.empty:
                            _bad_names = ", ".join(_non_healthy["Player"].head(5).tolist())
                            st.warning(
                                f"⚠️ Players with non-healthy status in sim lineups: **{_bad_names}**. "
                                "Consider tightening Sim Player Filters."
                            )

                        st.dataframe(_diag_merged, use_container_width=True, height=300)

            # ── Sim Diagnostics accordion ─────────────────────────────────────
            if sim_res is not None and not sim_res.empty:
                with st.expander("🔬 Sim Diagnostics", expanded=False):
                    _d_smash = float(sim_res["contest_smash_score"].iloc[0]) if "contest_smash_score" in sim_res.columns else float(sim_res["smash_threshold"].iloc[0])
                    _d_bust = float(sim_res["contest_bust_score"].iloc[0]) if "contest_bust_score" in sim_res.columns else float(sim_res["bust_threshold"].iloc[0])
                    _d_ct = sim_res["contest_type"].iloc[0] if "contest_type" in sim_res.columns else "—"

                    st.markdown("**Contest-Level Thresholds** (compare to historical cash lines during calibration)")
                    _th_col1, _th_col2, _th_col3 = st.columns(3)
                    _th_col1.metric("Contest Type", _d_ct)
                    _th_col2.metric("Smash Threshold (p90 field)", f"{_d_smash:.1f}")
                    _th_col3.metric("Bust Threshold (p30 field)", f"{_d_bust:.1f}")

                    # Sanity checks — sim_res["smash_pct"] is in [0,1] range
                    _avg_smash_pct = sim_res["smash_pct"].mean()
                    _avg_bust_pct = sim_res["bust_pct"].mean()
                    st.markdown("**Sanity Checks**")
                    _sc_col1, _sc_col2 = st.columns(2)
                    _sc_col1.metric(
                        "Avg Smash% across field",
                        f"{_avg_smash_pct:.1%}",
                        help="Expected ≈ 10% (p90 threshold means ~10% of field scores exceed it)",
                    )
                    _sc_col2.metric(
                        "Avg Bust% across field",
                        f"{_avg_bust_pct:.1%}",
                        help="Expected ≈ 30% (p30 threshold means ~30% of field scores fall below it)",
                    )
                    _smash_spread = sim_res["smash_pct"].std()
                    st.caption(
                        f"Smash% std dev across your {len(sim_res)} lineups: **{_smash_spread:.1%}** "
                        "(wider spread means the model differentiates well between lineups)."
                    )

                    # Histogram of sim scores for the currently-browsed lineup
                    _sim_scores_store = st.session_state.get("sim_lineup_scores")
                    if _sim_scores_store:
                        _diag_lu_ids = sorted(_sim_scores_store.keys())
                        _diag_lu_sel = st.selectbox(
                            "Select lineup for score histogram",
                            _diag_lu_ids,
                            format_func=lambda x: f"Lineup {x}",
                            key="diag_lineup_sel",
                        )
                        _diag_scores = _sim_scores_store.get(_diag_lu_sel)
                        if _diag_scores is not None and len(_diag_scores) > 0:
                            _hist_df = pd.DataFrame({"Score": _diag_scores})
                            try:
                                import altair as alt
                                _hist_base = alt.Chart(_hist_df).mark_bar(opacity=0.7).encode(
                                    x=alt.X("Score:Q", bin=alt.Bin(maxbins=40), title="Simulated Score"),
                                    y=alt.Y("count():Q", title="# Sims"),
                                    tooltip=["count():Q"],
                                ).properties(height=220, title=f"Lineup {_diag_lu_sel} — score distribution")
                                _smash_rule = alt.Chart(pd.DataFrame({"x": [_d_smash]})).mark_rule(
                                    color="#2ecc71", strokeWidth=2, strokeDash=[4, 2]
                                ).encode(x="x:Q")
                                _bust_rule = alt.Chart(pd.DataFrame({"x": [_d_bust]})).mark_rule(
                                    color="#e74c3c", strokeWidth=2, strokeDash=[4, 2]
                                ).encode(x="x:Q")
                                st.altair_chart(_hist_base + _smash_rule + _bust_rule, use_container_width=True)
                                st.caption(
                                    f"Green dashed line = contest smash threshold ({_d_smash:.1f}). "
                                    f"Red dashed line = contest bust threshold ({_d_bust:.1f})."
                                )
                            except Exception:
                                st.bar_chart(_hist_df["Score"].value_counts().sort_index())

            # ── Ownership Diagnostics ─────────────────────────────────────────
            with st.expander("📊 Ownership Diagnostics", expanded=False):
                st.markdown(
                    "Compare **predicted ownership** (own_proj) against **actual contest ownership** "
                    "after a slate completes.  Upload an actuals CSV with columns "
                    "`player_name` and `actual_own` (or `OWNERSHIP`)."
                )
                _own_diag_file = st.file_uploader(
                    "Upload actuals CSV (player_name, actual_own)",
                    type=["csv"],
                    key="own_diag_upload",
                )
                if _own_diag_file is not None:
                    try:
                        _own_act_df = pd.read_csv(_own_diag_file)
                        # Normalise column names
                        _own_col_map = {"PLAYER": "player_name", "OWNERSHIP": "actual_own"}
                        _own_act_df = _own_act_df.rename(
                            columns={k: v for k, v in _own_col_map.items() if k in _own_act_df.columns}
                        )
                        if "actual_own" in _own_act_df.columns:
                            _own_act_df["actual_own"] = (
                                _own_act_df["actual_own"].astype(str)
                                .str.replace("%", "", regex=False)
                                .pipe(pd.to_numeric, errors="coerce")
                            )
                        _own_pool_raw = st.session_state["sim_player_pool_clean"] if "sim_player_pool_clean" in st.session_state else None
                        _own_pool = _own_pool_raw if _own_pool_raw is not None and not _own_pool_raw.empty else pool_df
                        if _own_pool is not None and not _own_pool.empty:
                            _join_col = "player_name" if "player_name" in _own_pool.columns else "name"
                            _act_col = "player_name" if "player_name" in _own_act_df.columns else "name"
                            _merged_own = _own_pool.merge(
                                _own_act_df[[_act_col, "actual_own"]].rename(columns={_act_col: _join_col}),
                                on=_join_col,
                                how="inner",
                            )
                            from yak_core.ext_ownership import compute_ownership_diagnostics
                            _diag_result = compute_ownership_diagnostics(
                                _merged_own, actual_col="actual_own", pred_col="own_proj"
                            )
                            if "error" not in _diag_result:
                                _d1, _d2 = st.columns(2)
                                _d1.metric("Overall MAE", f"{_diag_result['overall_mae']:.2f}%")
                                _d2.metric(
                                    "Bias (pred − actual)",
                                    f"{_diag_result['overall_bias']:+.2f}%",
                                    help="Positive = over-predicting ownership",
                                )
                                # Bucket table
                                _bucket_rows = _diag_result.get("buckets", [])
                                if _bucket_rows:
                                    _bucket_df = pd.DataFrame(_bucket_rows)
                                    st.dataframe(_bucket_df, use_container_width=True, hide_index=True)
                                # Scatter plot (predicted vs actual)
                                if "own_proj" in _merged_own.columns:
                                    try:
                                        import altair as alt
                                        _scatter_df = _merged_own[["player_name", "own_proj", "actual_own"]].dropna()
                                        _scatter = alt.Chart(_scatter_df).mark_circle(size=60, opacity=0.7).encode(
                                            x=alt.X("actual_own:Q", title="Actual Own%"),
                                            y=alt.Y("own_proj:Q", title="Predicted (own_proj) %"),
                                            tooltip=["player_name:N", "own_proj:Q", "actual_own:Q"],
                                        ).properties(height=300, title="Predicted vs Actual Ownership")
                                        _diag_line = alt.Chart(
                                            pd.DataFrame({"x": [0, 80], "y": [0, 80]})
                                        ).mark_line(color="gray", strokeDash=[4, 2]).encode(
                                            x="x:Q", y="y:Q"
                                        )
                                        st.altair_chart(_scatter + _diag_line, use_container_width=True)
                                    except Exception:
                                        pass
                                # Download diagnostics CSV
                                _diag_csv = pd.DataFrame(_bucket_rows).to_csv(index=False)
                                st.download_button(
                                    "⬇️ Download Diagnostics CSV",
                                    _diag_csv,
                                    file_name="ownership_diagnostics.csv",
                                    mime="text/csv",
                                    key="own_diag_download",
                                )
                            else:
                                st.warning(f"Diagnostics error: {_diag_result['error']}")
                    except Exception as _own_diag_err:
                        st.error(f"Failed to process actuals: {_own_diag_err}")
                else:
                    # Show current ownership breakdown for active pool
                    _curr_pool_raw = st.session_state["sim_player_pool_clean"] if "sim_player_pool_clean" in st.session_state else None
                    _curr_pool = _curr_pool_raw if _curr_pool_raw is not None and not _curr_pool_raw.empty else pool_df
                    if _curr_pool is not None and not _curr_pool.empty:
                        # Ownership source sanity caption
                        _own_external_loaded = (
                            "ext_own" in _curr_pool.columns and _curr_pool["ext_own"].notna().any()
                            and (_curr_pool["ext_own"] > 0).any()
                        )
                        if "own_proj" in _curr_pool.columns:
                            _own_proj_s = _curr_pool["own_proj"].dropna()
                            if not _own_proj_s.empty:
                                st.caption(
                                    f"Ownership source: {'external RG/FP' if _own_external_loaded else 'internal model'} | "
                                    f"Min: {_own_proj_s.min():.1f}% | "
                                    f"Max: {_own_proj_s.max():.1f}% | "
                                    f"Median: {_own_proj_s.median():.1f}%"
                                )
                                if _own_proj_s.max() < 15:  # flag if max field% is suspiciously low
                                    st.warning(
                                        "⚠️ Max ownership is below 15%. "
                                        "The external ownership merge may be failing silently. "
                                        "Check that player names in your RG/FP CSV match the pool."
                                    )
                        # Merge diagnostics: show column list and name/POWN samples
                        with st.expander("🔍 Merge Diagnostics", expanded=False):
                            st.write("**Pool columns:**", _curr_pool.columns.tolist())
                            if "own_proj" in _curr_pool.columns:
                                st.write("**Sample own_proj (top 5):**", _curr_pool["own_proj"].head(5).tolist())
                            if "proj_own" in _curr_pool.columns:
                                st.write("**Sample proj_own (POWN) top 5:**", _curr_pool["proj_own"].head(5).tolist())
                            if "player_name" in _curr_pool.columns:
                                st.write("**Sample player names (pool):**", _curr_pool["player_name"].head(5).tolist())
                        _own_cols_avail = [c for c in ["ext_own", "own_model", "own_proj"] if c in _curr_pool.columns]
                        if _own_cols_avail:
                            st.markdown("**Current slate ownership breakdown (top 15 by own_proj):**")
                            _own_view_df = _curr_pool[["player_name"] + _own_cols_avail].copy() if "player_name" in _curr_pool.columns else _curr_pool[_own_cols_avail].copy()
                            _sort_col = "own_proj" if "own_proj" in _own_view_df.columns else _own_cols_avail[0]
                            _own_view_df = _own_view_df.sort_values(_sort_col, ascending=False).head(15).reset_index(drop=True)
                            _own_view_cfg = {
                                "ext_own": st.column_config.NumberColumn("Ext Own% (site)", format="%.1f"),
                                "own_model": st.column_config.NumberColumn("Model Own%", format="%.1f"),
                                "own_proj": st.column_config.NumberColumn("Field% (final)", format="%.1f"),
                            }
                            st.dataframe(_own_view_df, use_container_width=True, hide_index=True,
                                         column_config={k: v for k, v in _own_view_cfg.items() if k in _own_view_df.columns})

            # ── Custom Lineup Builder ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 🏗️ Custom Lineup Builder")
            st.markdown(
                "Build your own lineup from the **same player pool** the sim used.  "
                "Select one player per DraftKings roster slot, then save your lineup "
                "to compare it against the sim's recommendations."
            )

            _builder_pool = active_sim_pool.copy()
            # Build player label → row lookup
            _bp_name_col = "player_name" if "player_name" in _builder_pool.columns else (
                "name" if "name" in _builder_pool.columns else None
            )
            _bp_pos_col = "pos" if "pos" in _builder_pool.columns else None

            if _bp_name_col is None:
                st.warning("Player pool is missing a name column — cannot build lineup.")
            else:
                _bp_pool = _builder_pool.copy()
                _bp_pool["_label"] = _bp_pool[_bp_name_col].astype(str)
                if _bp_pos_col:
                    _bp_pool["_label"] = (
                        _bp_pool[_bp_name_col].astype(str)
                        + " ("
                        + _bp_pool[_bp_pos_col].astype(str)
                        + ", $"
                        + _bp_pool["salary"].astype(int).astype(str)
                        + ")"
                    )
                else:
                    _bp_pool["_label"] = (
                        _bp_pool[_bp_name_col].astype(str)
                        + " ($"
                        + _bp_pool["salary"].astype(int).astype(str)
                        + ")"
                    )

                # DK Classic slot → eligible positions
                _DK_SLOT_POS: dict = {
                    "PG": ["PG"],
                    "SG": ["SG"],
                    "SF": ["SF"],
                    "PF": ["PF"],
                    "C":  ["C"],
                    "G":  ["PG", "SG"],
                    "F":  ["SF", "PF"],
                    "UTIL": ["PG", "SG", "SF", "PF", "C"],
                }

                def _players_for_slot(slot: str) -> list:
                    """Return sorted player labels eligible for a DK Classic slot."""
                    eligible_pos = _DK_SLOT_POS.get(slot, [])
                    if _bp_pos_col and eligible_pos:
                        eligible_upper = {p.upper() for p in eligible_pos}
                        # Handle multi-position players (e.g. "SG/SF", "PF/C");
                        # split on "/" and check if any individual position matches.
                        mask = _bp_pool[_bp_pos_col].fillna("").str.upper().apply(
                            lambda pos: bool(set(pos.split("/")) & eligible_upper)
                        )
                        sub = _bp_pool[mask]
                    else:
                        sub = _bp_pool
                    return sorted(sub["_label"].tolist())

                # Render two columns of 4 slots each
                _slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
                _slot_cols = st.columns(2)
                _custom_selections: dict = {}

                # Helper: extract player name from a label like "Name (PG, $7800)"
                def _label_to_name(label: str | None) -> str | None:
                    if not label:
                        return None
                    return label.split(" (")[0]

                for _si, _slot in enumerate(_slots):
                    _col_idx = _si % 2
                    with _slot_cols[_col_idx]:
                        _opts = ["— select —"] + _players_for_slot(_slot)
                        # Pre-fill from saved custom lineup if available
                        _saved = st.session_state.get("sim_custom_lineup", [])
                        _default_idx = 0
                        if len(_saved) > _si and _saved[_si]:
                            # Find label matching saved name
                            _saved_name = _saved[_si]
                            for _oi, _o in enumerate(_opts):
                                if _label_to_name(_o) == _saved_name:
                                    _default_idx = _oi
                                    break
                        _sel = st.selectbox(
                            f"**{_slot}**",
                            options=_opts,
                            index=_default_idx,
                            key=f"sim_custom_slot_{_slot}",
                        )
                        _custom_selections[_slot] = _sel if _sel != "— select —" else None

                # Extract player names from selections
                _custom_names = [_label_to_name(_custom_selections[s]) for s in _slots]
                _filled = [n for n in _custom_names if n]

                # Salary + proj summary
                if _filled:
                    _custom_rows = []
                    for _n in _filled:
                        _row = _bp_pool[_bp_pool[_bp_name_col] == _n]
                        if not _row.empty:
                            _custom_rows.append(_row.iloc[0])
                    if _custom_rows:
                        _cdf = pd.DataFrame(_custom_rows)
                        _custom_sal = int(_cdf["salary"].sum()) if "salary" in _cdf.columns else 0
                        _custom_proj = float(_cdf["proj"].sum()) if "proj" in _cdf.columns else 0.0
                        _sal_color = "green" if _custom_sal <= 50000 else "red"
                        st.markdown(
                            f"💰 **Salary used:** <span style='color:{_sal_color}'>"
                            f"${_custom_sal:,} / $50,000</span>  &nbsp;&nbsp; "
                            f"📈 **Projected total:** **{_custom_proj:.1f} FP**",
                            unsafe_allow_html=True,
                        )

                _save_col, _clear_col = st.columns([1, 1])
                with _save_col:
                    if st.button("💾 Save Custom Lineup", key="sim_save_custom_btn"):
                        _missing = [_slots[i] for i, n in enumerate(_custom_names) if not n]
                        if _missing:
                            st.error(f"Please fill all 8 slots before saving. Missing: {', '.join(_missing)}")
                        else:
                            st.session_state["sim_custom_lineup"] = _custom_names
                            st.success("Custom lineup saved! See comparison below.")
                with _clear_col:
                    if st.button("🗑️ Clear Custom Lineup", key="sim_clear_custom_btn"):
                        st.session_state["sim_custom_lineup"] = []

            # ── Sim vs Custom Lineup Comparison ──────────────────────────────
            _saved_custom = st.session_state.get("sim_custom_lineup", [])
            if _saved_custom and len(_saved_custom) == 8:
                st.markdown("---")
                st.markdown("#### 📊 Sim vs Custom Lineup — Comparison")
                st.markdown(
                    "Side-by-side: the sim's **best lineup** (highest avg sim score) vs "
                    "**your custom lineup**.  Load actuals above to see real outcomes."
                )

                # Best sim lineup by sim_mean
                _best_sim_row = sim_res.sort_values("sim_mean", ascending=False).iloc[0]
                _best_sim_id = _best_sim_row["lineup_index"]
                _best_sim_mean = float(_best_sim_row["sim_mean"])
                _best_sim_players = sim_lu_df[sim_lu_df["lineup_index"] == _best_sim_id].copy()

                # Build custom lineup DataFrame from pool
                _cmp_custom_rows = []
                for _slot, _pname in zip(_slots, _saved_custom):
                    if _pname and _bp_name_col:
                        _row = active_sim_pool[active_sim_pool[_bp_name_col] == _pname]
                        if not _row.empty:
                            _r = _row.iloc[0].to_dict()
                            _r["slot"] = _slot
                            _cmp_custom_rows.append(_r)
                _cmp_custom_df = pd.DataFrame(_cmp_custom_rows) if _cmp_custom_rows else pd.DataFrame()

                # Actuals lookup helper
                _cmp_actuals = st.session_state.get("sim_actuals_df")

                def _lookup_actual(name: str) -> float | None:
                    if _cmp_actuals is None or _cmp_actuals.empty:
                        return None
                    _acts_name_col = "player_name" if "player_name" in _cmp_actuals.columns else "name"
                    _acts_fp_col = "actual_fp" if "actual_fp" in _cmp_actuals.columns else "actual"
                    _match = _cmp_actuals[_cmp_actuals[_acts_name_col] == name]
                    if not _match.empty and _acts_fp_col in _match.columns:
                        return float(_match.iloc[0][_acts_fp_col])
                    return None

                # Build display rows for sim lineup
                def _build_cmp_rows(players_df: pd.DataFrame, name_col: str) -> list:
                    rows = []
                    for _, r in players_df.iterrows():
                        pname = str(r.get(name_col, ""))
                        proj = float(r.get("proj", 0))
                        sal = int(r.get("salary", 0))
                        pos = str(r.get("pos", ""))
                        actual = _lookup_actual(pname)
                        rows.append({
                            "Player": pname,
                            "Pos": pos,
                            "Salary": sal,
                            "Proj FP": round(proj, 1),
                            "Actual FP": round(actual, 1) if actual is not None else "—",
                        })
                    return rows

                _cmp_col_sim, _cmp_col_custom = st.columns(2)

                with _cmp_col_sim:
                    st.markdown(
                        f"**🤖 Sim's Best Lineup** &nbsp; *(Lineup #{_best_sim_id}, "
                        f"avg {_best_sim_mean:.1f} FP)*"
                    )
                    if not _best_sim_players.empty:
                        _sim_name_col = "player_name" if "player_name" in _best_sim_players.columns else "name"
                        _sim_rows = _build_cmp_rows(_best_sim_players, _sim_name_col)
                        _sim_cmp_df = pd.DataFrame(_sim_rows)
                        _sim_proj_total = _sim_cmp_df["Proj FP"].sum()
                        _sim_actual_total: float | str = "—"
                        if _cmp_actuals is not None:
                            _sim_act_vals = [r["Actual FP"] for r in _sim_rows if r["Actual FP"] != "—"]
                            if _sim_act_vals:
                                _sim_actual_total = round(sum(_sim_act_vals), 1)
                        st.dataframe(_sim_cmp_df, use_container_width=True, hide_index=True)
                        st.markdown(
                            f"**Proj total:** {_sim_proj_total:.1f} FP  |  "
                            f"**Actual total:** {_sim_actual_total}"
                        )
                    else:
                        st.info("Sim lineup data unavailable.")

                with _cmp_col_custom:
                    st.markdown("**🧑 Your Custom Lineup**")
                    if not _cmp_custom_df.empty:
                        _custom_name_col = _bp_name_col or "player_name"
                        _cust_rows = _build_cmp_rows(_cmp_custom_df, _custom_name_col)
                        _cust_cmp_df = pd.DataFrame(_cust_rows)
                        _cust_proj_total = _cust_cmp_df["Proj FP"].sum()
                        _cust_actual_total: float | str = "—"
                        if _cmp_actuals is not None:
                            _cust_act_vals = [r["Actual FP"] for r in _cust_rows if r["Actual FP"] != "—"]
                            if _cust_act_vals:
                                _cust_actual_total = round(sum(_cust_act_vals), 1)
                        st.dataframe(_cust_cmp_df, use_container_width=True, hide_index=True)
                        st.markdown(
                            f"**Proj total:** {_cust_proj_total:.1f} FP  |  "
                            f"**Actual total:** {_cust_actual_total}"
                        )
                    else:
                        st.info("Custom lineup data unavailable.")

                # ── What the sim missed ───────────────────────────────────────
                if _cmp_actuals is not None and not _cmp_actuals.empty:
                    st.markdown("##### 🔍 What the Sim Missed")
                    st.caption(
                        "Players in the sim's best lineup compared to your custom lineup — "
                        "sorted by biggest projection error (|Proj − Actual|)."
                    )
                    _acc_result = build_sim_player_accuracy_table(active_sim_pool, _cmp_actuals)
                    if _acc_result["n_players"] > 0:
                        _acc_df = _acc_result["player_df"].copy()
                        # Tag which lineup each player appeared in
                        _sim_player_set = set(
                            _best_sim_players[
                                "player_name" if "player_name" in _best_sim_players.columns else "name"
                            ].tolist()
                        )
                        _custom_player_set = set(_saved_custom)
                        _acc_df["In Sim Lineup"] = _acc_df["name"].isin(_sim_player_set).map(
                            {True: "✅", False: ""}
                        )
                        _acc_df["In Custom Lineup"] = _acc_df["name"].isin(_custom_player_set).map(
                            {True: "✅", False: ""}
                        )
                        _miss_display = _acc_df.rename(columns={
                            "name": "Player",
                            "proj": "Proj FP",
                            "actual": "Actual FP",
                            "error": "Error (Proj−Act)",
                            "abs_error": "Abs Error",
                            "pct_error": "Err %",
                        }).sort_values("Abs Error", ascending=False)
                        for _c in ["Proj FP", "Actual FP", "Error (Proj−Act)", "Abs Error", "Err %"]:
                            if _c in _miss_display.columns:
                                _miss_display[_c] = _miss_display[_c].round(1)
                        st.dataframe(_miss_display, use_container_width=True, height=350)
                        st.download_button(
                            "📥 Download miss analysis CSV",
                            data=to_csv_bytes(_miss_display),
                            file_name="yakos_sim_miss_analysis.csv",
                            mime="text/csv",
                            key="sim_miss_dl",
                        )
                    else:
                        st.info("No player names matched between pool and actuals for miss analysis.")
                else:
                    st.info(
                        "Load actuals in the **📊 Load Actuals** expander above to see what the sim missed."
                    )

            # Apply learnings: boost projection of high-sim players for next run
            st.markdown("---")
            st.markdown("#### Apply Learnings to Live Slate Logic")
            if st.button("⚡ Apply sim learnings (boost high-smash players' projections)", key="sim_apply_btn"):
                # Players in high-smash lineups (smash_prob > 0.1) get a +5% boost
                top_lu_ids = sim_res[sim_res["smash_prob"] > 0.10]["lineup_index"].tolist()
                if top_lu_ids:
                    top_players = (
                        sim_lu_df[sim_lu_df["lineup_index"].isin(top_lu_ids)]
                        ["player_name"].value_counts()
                    )
                    boost_names = top_players[top_players >= 2].index.tolist()
                    if boost_names:
                        # Always boost from the original (pre-boost) pool to prevent
                        # compounding multiplier drift each time the button is clicked.
                        orig_pool_raw = st.session_state.get("sim_pool_orig_df")
                        orig_pool = orig_pool_raw if orig_pool_raw is not None and not orig_pool_raw.empty else pool_for_sim
                        updated_pool = orig_pool.copy()
                        mask = updated_pool["player_name"].isin(boost_names)
                        updated_pool.loc[mask, "proj"] = updated_pool.loc[mask, "proj"] * (1.0 + _SIM_LEARNING_BOOST)
                        st.session_state["sim_pool_df"] = updated_pool
                        st.success(
                            f"Boosted projections (+{_SIM_LEARNING_BOOST:.0%}) for {len(boost_names)} high-smash "
                            f"players: {', '.join(boost_names[:5])}"
                            + (" ..." if len(boost_names) > 5 else "")
                        )
                    else:
                        st.info("No players appeared in 2+ high-smash lineups; no boosts applied.")
                else:
                    st.info("No lineups with smash_prob > 10%. Lower threshold or run more lineups.")

            # Post to Ricky's Slate Room
            st.markdown("#### 📤 Publish to Slate Room")
            conf_threshold = st.slider(
                "Minimum confidence to promote", 40.0, 95.0, 65.0, step=5.0, key="sim_conf_thr"
            )
            if st.button("✅ Publish to Slate Room", type="primary", key="sim_post_btn"):
                if "confidence" in sim_lu_df.columns:
                    high_conf_ids = (
                        sim_lu_df.groupby("lineup_index")["confidence"]
                        .first()
                        .loc[lambda s: s >= conf_threshold]
                        .index.tolist()
                    )
                    promoted_df = sim_lu_df[sim_lu_df["lineup_index"].isin(high_conf_ids)].copy()
                else:
                    # Fall back to top-N by smash_prob
                    top_ids = sim_res.nlargest(5, "smash_prob")["lineup_index"].tolist()
                    promoted_df = sim_lu_df[sim_lu_df["lineup_index"].isin(top_ids)].copy()
                    high_conf_ids = top_ids

                if not promoted_df.empty:
                    entry = {
                        "label": f"Sims {sim_dk_contest} / {sim_archetype}",
                        "lineups_df": promoted_df,
                        "metadata": {
                            "contest_type": sim_dk_contest,
                            "archetype": sim_archetype,
                            "n_lineups": len(high_conf_ids),
                        },
                    }
                    st.session_state["promoted_lineups"].append(entry)

                    # Also write structured ApprovedLineup objects for the new Slate Room KPI strip
                    _sim_slate_label = getattr(
                        st.session_state.get("pool_df", pd.DataFrame()),
                        "attrs", {}
                    ).get("slate", f"NBA {_today_est()}")
                    _arch_label = CONTEST_PRESET_ARCH_LABELS.get(sim_dk_contest, sim_dk_contest)
                    _new_approved = build_approved_lineups(
                        lineups_df=promoted_df,
                        sim_results=sim_res,
                        contest_archetype=_arch_label,
                        site="DK",
                        slate=_sim_slate_label,
                    )
                    st.session_state["approved_lineups"].extend(_new_approved)
                    st.session_state["last_calibration_ts"] = pd.Timestamp.now(
                        tz=ZoneInfo("America/New_York")
                    ).strftime("%Y-%m-%d %H:%M ET")

                    st.success(
                        f"✅ Published {len(high_conf_ids)} high-confidence lineup(s) "
                        "to 🏀 Ricky's Slate Room!"
                    )
                else:
                    st.warning("No lineups met the confidence threshold.")

            # Download sim lineups in DK upload format
            st.markdown("#### 💾 Export Sim Lineups")
            sim_dl1, sim_dl2 = st.columns(2)
            with sim_dl1:
                st.download_button(
                    "Download sim lineups CSV",
                    data=to_csv_bytes(sim_lu_df),
                    file_name="yakos_sim_lineups.csv",
                    mime="text/csv",
                    key="sim_dl_lu",
                )
            with sim_dl2:
                sim_dk_upload_df = to_dk_upload_format(sim_lu_df)
                st.download_button(
                    "📥 Download DK upload CSV",
                    data=to_csv_bytes(sim_dk_upload_df),
                    file_name="yakos_sim_dk_upload.csv",
                    mime="text/csv",
                    key="sim_dl_dk",
                    help=(
                        "DraftKings bulk entry format: one row per lineup with "
                        "slot columns PG/SG/SF/PF/C/G/F/UTIL. "
                        "Fill in Entry ID and Contest info before uploading."
                    ),
                )

        # ---- Sim vs Actuals player accuracy table ----
        _sim_actuals = st.session_state.get("sim_actuals_df")
        if _sim_actuals is not None and not _sim_actuals.empty:
            st.markdown("#### 🎯 Sim vs Actuals — Player Accuracy")
            st.caption(
                "Compares the sim's input projections to actual DraftKings fantasy points scored. "
                "Load actuals above using the 📊 expander."
            )
            # Filter pool to only players who appeared in the generated sim lineups
            _sim_lu_for_acc = st.session_state.get("sim_lineups_df")
            if _sim_lu_for_acc is not None and not _sim_lu_for_acc.empty:
                _lu_acc_name_col = "player_name" if "player_name" in _sim_lu_for_acc.columns else "name"
                _lu_acc_players = set(_sim_lu_for_acc[_lu_acc_name_col].dropna().unique())
                _pool_acc_name_col = "player_name" if "player_name" in pool_for_sim.columns else "name"
                _acc_pool = pool_for_sim[pool_for_sim[_pool_acc_name_col].isin(_lu_acc_players)]
            else:
                _acc_pool = pool_for_sim
            _acc = build_sim_player_accuracy_table(_acc_pool, _sim_actuals)
            if _acc["n_players"] > 0:
                _acc_kpis = st.columns(5)
                _acc_kpis[0].metric("Players matched", _acc["n_players"])
                _acc_kpis[1].metric("MAE", f"{_acc['mae']:.1f} FP")
                _acc_kpis[2].metric("RMSE", f"{_acc['rmse']:.1f} FP")
                _acc_kpis[3].metric(
                    "Bias",
                    f"{_acc['bias']:+.1f} FP",
                    help="Positive = sim over-projected on average",
                )
                _acc_kpis[4].metric(
                    "Hit rate (±10 FP)",
                    f"{_acc['hit_rate']:.0f}%",
                    help="% of players where |proj − actual| ≤ 10 FP",
                )
                with st.expander("Player projection vs actuals table", expanded=True):
                    st.markdown(
                        "**Column guide**\n\n"
                        "| Column | Description |\n"
                        "|--------|-------------|\n"
                        "| **Name** | Player name |\n"
                        "| **Proj FP** | Pre-game sim projection |\n"
                        "| **Actual FP** | Real DraftKings fantasy points scored |\n"
                        "| **Error** | Proj − Actual (positive = over-projected) |\n"
                        "| **Abs Error** | |Error| — magnitude of miss |\n"
                        "| **Err %** | Error as % of actual score |\n"
                        "| **Error%** | abs(Proj − Actual) / Proj × 100 |\n"
                        "| **Outlier** | ✅ when Error% > 30% — biggest projection misses |"
                    )
                    _acc_display = _acc["player_df"].rename(columns={
                        "name": "Name",
                        "proj": "Proj FP",
                        "actual": "Actual FP",
                        "error": "Error",
                        "abs_error": "Abs Error",
                        "pct_error": "Err %",
                    })
                    _acc_display["Proj FP"] = _acc_display["Proj FP"].round(1)
                    _acc_display["Actual FP"] = _acc_display["Actual FP"].round(1)
                    _acc_display["Error"] = _acc_display["Error"].round(1)
                    _acc_display["Abs Error"] = _acc_display["Abs Error"].round(1)
                    _acc_display["Err %"] = _acc_display["Err %"].round(1)
                    # Error% = abs(proj - actual) / proj * 100 (projection-relative)
                    _acc_display["Error%"] = (
                        _acc_display["Abs Error"] / _acc_display["Proj FP"].replace(0, float("nan")) * 100.0
                    ).round(1)
                    _acc_display["Outlier"] = _acc_display["Error%"].apply(
                        lambda x: "✅" if pd.notna(x) and float(x) > 30.0 else ""
                    )
                    # Sort outliers to the top, then by Abs Error descending
                    _acc_display = _acc_display.sort_values(
                        ["Outlier", "Abs Error"], ascending=[False, False]
                    )
                    st.dataframe(
                        _acc_display,
                        use_container_width=True,
                        height=400,
                    )
                    st.download_button(
                        "📥 Download player accuracy CSV",
                        data=to_csv_bytes(_acc_display),
                        file_name="yakos_sim_player_accuracy.csv",
                        mime="text/csv",
                        key="sim_acc_dl",
                    )
            else:
                st.warning(
                    "No player names matched between the pool and the actuals file. "
                    "Check that both use the same player-name format."
                )

    st.markdown("---")
    # ---- Section D: Multi-Slate ----
    st.markdown("### D. Multi-Slate Comparison")

    with st.expander("Multi-Slate Comparison (expand to use)", expanded=False):
        st.markdown(
            "Discover available historical slates, batch-run the optimizer across "
            "multiple dates, and compare KPIs side-by-side."
        )

        slates_df = discover_slates()
        if slates_df.empty:
            st.info(
                "Multi-slate comparison requires historical parquet files. "
                "Run data collection notebooks to enable. "
                "(Files matching `tank_opt_pool_<date>.parquet` in `YAKOS_ROOT`.)"
            )
        else:
            st.write(f"**{len(slates_df)} slate(s) discovered:**")
            st.dataframe(slates_df[["slate_date", "filename", "size_kb"]], use_container_width=True)

            available_dates = slates_df["slate_date"].tolist()
            ms_dates = st.multiselect(
                "Select slate dates to compare",
                available_dates,
                default=available_dates[:min(3, len(available_dates))],
                key="ms_dates_sel",
            )

            ms_col1, ms_col2 = st.columns(2)
            with ms_col1:
                ms_num_lu = st.number_input(
                    "Lineups per slate",
                    min_value=1,
                    max_value=150,  # DK bulk-upload cap is 150 lineups per contest
                    value=20,
                    key="ms_num_lu",
                )
            with ms_col2:
                ms_max_exp = st.slider(
                    "Max exposure", min_value=0.1, max_value=1.0, value=0.6, step=0.05, key="ms_max_exp"
                )

            if st.button("▶ Run Multi-Slate", type="primary", key="ms_run_btn"):
                if not ms_dates:
                    st.warning("Select at least one slate date.")
                else:
                    with st.spinner(f"Running optimizer across {len(ms_dates)} slate(s)…"):
                        ms_result = run_multi_slate(
                            ms_dates,
                            base_cfg={"NUM_LINEUPS": ms_num_lu, "MAX_EXPOSURE": ms_max_exp},
                        )
                        st.session_state["ms_result"] = ms_result

            if st.session_state.get("ms_result"):
                ms_res = st.session_state["ms_result"]
                comparison = compare_slates(ms_res)
                sdf = comparison.get("summary_df", pd.DataFrame())

                if not sdf.empty:
                    st.markdown("#### Per-Slate Summary")
                    st.dataframe(sdf, use_container_width=True)

                if comparison.get("n_slates", 0) > 0:
                    st.markdown("#### Cross-Slate Trends")
                    trends = comparison.get("trends", {})
                    if trends:
                        trend_rows = []
                        for metric, stats in trends.items():
                            trend_rows.append({"metric": metric, **stats})
                        st.dataframe(pd.DataFrame(trend_rows), use_container_width=True)

                    cons = comparison.get("consistency", {})
                    if cons:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Slates OK", cons.get("n_slates_ok", 0))
                        c2.metric("Slates failed", cons.get("n_slates_fail", 0))
                        c3.metric(
                            "LU proj CV (%)",
                            f"{cons.get('lu_proj_cv', 0):.1f}",
                            help="Coefficient of variation of avg lineup proj across slates — lower = more consistent.",
                        )
                        if "avg_proj_error" in cons:
                            st.metric(
                                "Avg projection error (vs actuals)",
                                f"{cons['avg_proj_error']:+.1f} pts",
                            )

    st.markdown("---")
    st.caption("YakOS Calibration Lab — data-driven lineup refinement.")


# ============================================================
# Tab 4: 📡 Ricky's Calibration Lab
# ============================================================
with tab_calib:
    st.markdown("### E. 📡 Backtesting & Strategy Calibration")
    st.markdown(
        "Backtest any DFS build against past contests and see which contest archetypes "
        "need retuning — mirroring the BacktestIQ approach."
    )

    # Load historical data once for the whole tab
    _hist_df_bc = load_historical_lineups()

    st.markdown("---")

    # ── Backtest Controls ─────────────────────────────────────────────────────
    st.markdown("### Backtest Controls")

    _bc_col1, _bc_col2 = st.columns([1, 2])
    with _bc_col1:
        _bc_sport = st.selectbox(
            "Sport",
            ["NBA", "PGA"],
            key="bc_sport",
            help="PGA support is coming soon.",
        )
        _bc_site = st.selectbox(
            "Site",
            ["DraftKings (DK)", "FanDuel (FD)"],
            key="bc_site",
            help="FanDuel support is coming soon.",
        )

    with _bc_col2:
        if not _hist_df_bc.empty:
            _bc_min_date = min(_hist_df_bc["slate_date"])
            _bc_max_date = max(_hist_df_bc["slate_date"])
        else:
            _bc_min_date = _bc_max_date = _today_est()

        _bc_date_range = st.date_input(
            "Historical date range",
            value=(_bc_min_date, _bc_max_date),
            key="bc_date_range",
            help="Select the range of historical slates to backtest against.",
        )

    _bc_arch_options = list(BACKTEST_ARCHETYPES.keys())
    _bc_archetypes = st.multiselect(
        "Contest archetypes to test",
        _bc_arch_options,
        default=_bc_arch_options,
        key="bc_archetypes",
        help=(
            "Select which Ricky contest archetypes to backtest. "
            "Each maps to a specific contest type and default build config."
        ),
    )

    _bc_build_override = st.selectbox(
        "Build config (optimizer archetype)",
        ["— use archetype default —"] + list(DFS_ARCHETYPES.keys()),
        key="bc_build_override",
        help="Override the default DFS build config for all selected contest archetypes.",
    )
    _bc_build_config = (
        None if _bc_build_override == "— use archetype default —" else _bc_build_override
    )

    _bc_num_lineups = st.slider(
        "Lineups per contest",
        min_value=1,
        max_value=20,
        value=5,
        key="bc_num_lineups",
        help="Number of lineups to generate per (archetype, slate) pair.",
    )

    _bc_entry_fee = st.number_input(
        "Entry fee ($)",
        min_value=0.25,
        max_value=100.0,
        value=4.0,
        step=0.25,
        key="bc_entry_fee",
    )

    if st.button("▶ Run Backtest", type="primary", key="bc_run_btn"):
        if not _bc_archetypes:
            st.warning("Select at least one contest archetype.")
        elif _hist_df_bc.empty:
            st.warning(
                "No historical data found. Add `data/historical_lineups.csv` to enable backtesting."
            )
        else:
            # Filter by date range
            _start, _end = (
                (_bc_date_range[0], _bc_date_range[1])
                if isinstance(_bc_date_range, (tuple, list)) and len(_bc_date_range) == 2
                else (_bc_min_date, _bc_max_date)
            )
            _filtered_hist = _hist_df_bc[
                (_hist_df_bc["slate_date"] >= _start)
                & (_hist_df_bc["slate_date"] <= _end)
            ]
            if _filtered_hist.empty:
                st.warning("No historical data found in the selected date range.")
            else:
                with st.spinner(
                    f"Running backtest across {len(_bc_archetypes)} archetype(s) "
                    f"and {_filtered_hist['slate_date'].nunique()} slate(s)…"
                ):
                    try:
                        _bt_results = run_archetype_backtest(
                            hist_df=_filtered_hist,
                            archetypes=_bc_archetypes,
                            num_lineups=_bc_num_lineups,
                            entry_fee=_bc_entry_fee,
                            build_config_override=_bc_build_config,
                        )
                        st.session_state["calib_backtest_results"] = _bt_results
                        st.session_state["calib_drilldown_arch"] = None
                        st.session_state["calib_queue_arch"] = None
                        st.session_state["calib_queue_slate"] = None
                        _n_tested = _bt_results.get("global", {}).get("n_lineups", 0)
                        if _n_tested > 0:
                            st.success(
                                f"Backtest complete — {_n_tested} lineups tested across "
                                f"{_bt_results['global']['n_contests']} contest(s)."
                            )
                        else:
                            st.warning(
                                "Backtest ran but produced no results. "
                                "The historical data may not have enough players per slate "
                                "to run the optimizer (need ≥ 8 unique players)."
                            )
                    except Exception as _bt_err:
                        st.error(f"Backtest error: {_bt_err}")

    # ── Backtest Results ──────────────────────────────────────────────────────
    _bt = st.session_state.get("calib_backtest_results")
    if _bt and _bt.get("global") and _bt["global"].get("n_lineups", 0) > 0:
        st.markdown("---")
        st.markdown("### Backtest Results")

        # ── Global KPI strip ────────────────────────────────────────────────
        _g = _bt["global"]

        def _bt_roi_color(roi: float) -> str:
            if roi >= 10:
                return "good"
            elif roi >= 0:
                return "warn"
            return "bad"

        def _bt_cash_color(cash_rate: float) -> str:
            if cash_rate >= 60:
                return "good"
            elif cash_rate >= 45:
                return "warn"
            return "bad"

        def _bt_pct_color(avg_pct: float) -> str:
            if avg_pct <= 40:
                return "good"
            elif avg_pct <= 55:
                return "warn"
            return "bad"

        def _bt_kpi_card(label: str, value_str: str, color_key: str) -> str:
            bg = _QUALITY_BG[color_key]
            color = _QUALITY_TEXT[color_key]
            return (
                f'<div style="border:1px solid #3a3a3a;border-radius:6px;'
                f'padding:10px 8px;text-align:center;background:{bg};">'
                f'<div style="font-size:0.72rem;text-transform:uppercase;'
                f'letter-spacing:0.06em;color:#aaa;margin-bottom:4px;">{label}</div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:{color};">{value_str}</div>'
                f'</div>'
            )

        st.markdown(
            "**Global KPIs — all archetypes combined**  "
            f"({_g['n_lineups']} lineups · {_g['n_contests']} contest(s))"
        )
        _gk1, _gk2, _gk3, _gk4 = st.columns(4)
        with _gk1:
            st.markdown(
                _bt_kpi_card("Backtest ROI", f"{_g['roi']:+.1f}%", _bt_roi_color(_g["roi"])),
                unsafe_allow_html=True,
            )
        with _gk2:
            st.markdown(
                _bt_kpi_card("Cash Rate", f"{_g['cash_rate']:.1f}%", _bt_cash_color(_g["cash_rate"])),
                unsafe_allow_html=True,
            )
        with _gk3:
            st.markdown(
                _bt_kpi_card(
                    "Avg Finish %ile",
                    f"{_g['avg_percentile']:.1f}",
                    _bt_pct_color(_g["avg_percentile"]),
                ),
                unsafe_allow_html=True,
            )
        with _gk4:
            st.markdown(
                _bt_kpi_card("Best Finish %ile", f"{_g['best_finish']:.1f}", "good"),
                unsafe_allow_html=True,
            )
        st.caption(
            "ROI: green ≥ 10%, yellow 0–10%, red < 0.  "
            "Cash rate: green ≥ 60%, yellow 45–60%, red < 45%.  "
            "Avg %ile: green ≤ 40, yellow 40–55, red > 55 (lower = better finish)."
        )

        # ── Archetype summary table ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Archetype Summary")
        st.caption("Sorted by worst ROI first — problem archetypes float to the top.")

        _arch_rows = _bt.get("by_archetype", [])
        if _arch_rows:
            _arch_df = pd.DataFrame([
                {
                    "Archetype": r["archetype"],
                    "ROI (%)": r["roi"],
                    "Cash Rate (%)": r["cash_rate"],
                    "Avg Finish %ile": r["avg_percentile"],
                    "Best Finish %ile": r["best_finish"],
                    "Contests": r["n_contests"],
                    "Lineups": r["n_lineups"],
                }
                for r in _arch_rows
            ]).sort_values("ROI (%)", ascending=True)  # worst first

            # Row coloring: apply background via HTML table
            def _row_style(row: pd.Series) -> list:
                roi_c = _bt_roi_color(float(row["ROI (%)"]))
                cash_c = _bt_cash_color(float(row["Cash Rate (%)"]))
                # Use the worse of ROI vs cash_rate to pick row color
                _priority = {"bad": 0, "warn": 1, "good": 2}
                worst = min(roi_c, cash_c, key=lambda c: _priority[c])
                bg = _QUALITY_BG[worst]
                return [f"background-color:{bg}" for _ in row.index]

            _styled = _arch_df.style.apply(_row_style, axis=1).format({
                "ROI (%)": "{:+.1f}",
                "Cash Rate (%)": "{:.1f}",
                "Avg Finish %ile": "{:.1f}",
                "Best Finish %ile": "{:.1f}",
            })
            st.dataframe(_styled, use_container_width=True, hide_index=True)

            # Drilldown selector
            st.markdown("#### Drilldown by Archetype")
            _drill_options = ["— select to drill down —"] + [r["archetype"] for r in _arch_rows]
            _drill_sel = st.selectbox(
                "Select archetype to drill into",
                _drill_options,
                key="bc_drilldown_sel",
            )
            if _drill_sel != "— select to drill down —":
                st.session_state["calib_drilldown_arch"] = _drill_sel

            _drill_arch = st.session_state.get("calib_drilldown_arch")
            if _drill_arch and _drill_arch in [r["archetype"] for r in _arch_rows]:
                _drill_data = next(
                    (r for r in _arch_rows if r["archetype"] == _drill_arch), None
                )
                if _drill_data:
                    st.markdown(f"##### {_drill_arch} — Slate-Level Results")
                    _slate_rows = _drill_data.get("slate_results", [])
                    if _slate_rows:
                        _slate_df = pd.DataFrame(_slate_rows).rename(columns={
                            "slate_date": "Slate Date",
                            "contest": "Contest",
                            "roi": "ROI (%)",
                            "cash_rate": "Cash Rate (%)",
                            "avg_percentile": "Avg Finish %ile",
                            "best_finish": "Best Finish %ile",
                            "n_lineups": "Lineups",
                        })
                        st.dataframe(
                            _slate_df.style.format({
                                "ROI (%)": "{:+.1f}",
                                "Cash Rate (%)": "{:.1f}",
                                "Avg Finish %ile": "{:.1f}",
                                "Best Finish %ile": "{:.1f}",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )

                        # Link to Player Calibration Queue
                        st.markdown("**Open Player Calibration Queue for this archetype:**")
                        _slate_sel_options = [r["Slate Date"] for _, r in _slate_df.iterrows()]
                        _pq_slate_sel = st.selectbox(
                            "Select slate",
                            ["— pick a slate —"] + _slate_sel_options,
                            key="bc_pq_slate_sel",
                        )
                        if _pq_slate_sel != "— pick a slate —":
                            if st.button(
                                f"Open Player Queue: {_drill_arch} / {_pq_slate_sel}",
                                key="bc_open_pq_btn",
                            ):
                                st.session_state["calib_queue_arch"] = _drill_arch
                                st.session_state["calib_queue_slate"] = _pq_slate_sel
                    else:
                        st.info("No slate-level results for this archetype.")

        # ── Player Calibration Queue ─────────────────────────────────────────
        _pq_arch = st.session_state.get("calib_queue_arch")
        _pq_slate = st.session_state.get("calib_queue_slate")

        if _pq_arch and _pq_slate and not _hist_df_bc.empty:
            st.markdown("---")
            st.markdown(f"### Player Calibration Queue — {_pq_arch} / {_pq_slate}")
            st.caption(
                "Players from the selected archetype's builds on this slate, "
                "showing projection vs actual errors to guide recalibration."
            )

            # Filter historical data to this slate
            _pq_data = _hist_df_bc[_hist_df_bc["slate_date"].astype(str) == str(_pq_slate)].copy()

            if _pq_data.empty:
                st.info(f"No historical data found for slate {_pq_slate}.")
            else:
                _pq_qd = _pq_data.copy()
                _pq_qd["pts_error"] = (
                    pd.to_numeric(_pq_qd.get("actual", 0), errors="coerce").fillna(0)
                    - pd.to_numeric(_pq_qd.get("proj", 0), errors="coerce").fillna(0)
                )
                if "actual_minutes" in _pq_qd.columns and "proj_minutes" in _pq_qd.columns:
                    _pq_qd["min_error"] = (
                        pd.to_numeric(_pq_qd["actual_minutes"], errors="coerce").fillna(0)
                        - pd.to_numeric(_pq_qd["proj_minutes"], errors="coerce").fillna(0)
                    )
                else:
                    _pq_qd["min_error"] = 0.0
                if "own" in _pq_qd.columns and "proj_own" in _pq_qd.columns:
                    _pq_qd["own_error"] = (
                        pd.to_numeric(_pq_qd["own"], errors="coerce").fillna(0)
                        - pd.to_numeric(_pq_qd["proj_own"], errors="coerce").fillna(0)
                    )
                else:
                    _pq_qd["own_error"] = 0.0

                _pq_qd["Flag"] = (
                    _pq_qd["pts_error"].abs().gt(6)
                    | _pq_qd["min_error"].abs().gt(3)
                    | _pq_qd["own_error"].abs().gt(3)
                )

                _pq_player_col = []
                for _, _r in _pq_qd.iterrows():
                    _tp = " / ".join(filter(None, [str(_r.get("team", "")), str(_r.get("pos", ""))]))
                    _pq_player_col.append(
                        f"{_r.get('name', '')} ({_tp})" if _tp else str(_r.get("name", ""))
                    )
                _pq_qd["Player"] = _pq_player_col

                _pq_cols_map = {
                    "Player": "Player",
                    "salary": "Salary",
                    "proj": "Proj FP",
                    "actual": "Act FP",
                    "pts_error": "Error (pts)",
                    "proj_minutes": "Proj Mins",
                    "actual_minutes": "Act Mins",
                    "min_error": "Min Error",
                    "proj_own": "Proj Own %",
                    "own": "Act Own %",
                    "own_error": "Own Error",
                    "Flag": "Flag",
                }
                _pq_avail = [c for c in _pq_cols_map if c in _pq_qd.columns or c == "Player"]
                _pq_display = _pq_qd[_pq_avail].rename(columns=_pq_cols_map)

                st.dataframe(
                    _pq_display,
                    column_config={
                        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                        "Proj FP": st.column_config.NumberColumn("Proj FP", format="%.1f"),
                        "Act FP": st.column_config.NumberColumn("Act FP", format="%.1f"),
                        "Error (pts)": st.column_config.NumberColumn("Error (pts)", format="%+.1f"),
                        "Proj Mins": st.column_config.NumberColumn("Proj Mins", format="%.1f"),
                        "Act Mins": st.column_config.NumberColumn("Act Mins", format="%.1f"),
                        "Min Error": st.column_config.NumberColumn("Min Error", format="%+.1f"),
                        "Proj Own %": st.column_config.NumberColumn("Proj Own %", format="%.1f"),
                        "Act Own %": st.column_config.NumberColumn("Act Own %", format="%.1f"),
                        "Own Error": st.column_config.NumberColumn("Own Error", format="%+.1f"),
                        "Flag": st.column_config.CheckboxColumn("Flag", disabled=True),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                _pq_n_flagged = int(_pq_qd["Flag"].sum())
                if _pq_n_flagged > 0:
                    st.warning(
                        f"⚠️ {_pq_n_flagged} player(s) flagged with large projection errors. "
                        "Consider adjusting archetype config weights."
                    )

    elif _bt is not None and _bt.get("global") and _bt["global"].get("n_lineups", 0) == 0:
        st.info(
            "Backtest ran but no lineups were generated. "
            "Ensure `data/historical_lineups.csv` has at least 8 unique players per slate date."
        )

    st.markdown("---")
    st.caption("Ricky's Calibration Lab — BacktestIQ-style strategy calibration.")

    # ── Section F: 📊 Alert Validation Panel (Sprint 4B) ────────────────────
    st.markdown("### F. 📊 Alert Validation")

    _av_pool = st.session_state.get("pool_df")
    _av_actuals_in_pool = _av_pool is not None and not _av_pool.empty and "actual_fp" in _av_pool.columns

    with st.expander("📊 Alert Validation Panel", expanded=False):
        st.markdown(
            "Re-run the full alert engine on historical slates to measure hit rates, "
            "false positives, and overall edge vs field baseline.  "
            "Then optionally auto-tune thresholds and re-run."
        )

        # ── Slate selection ──────────────────────────────────────────────────
        _av_use_current = st.checkbox(
            "Use currently-loaded pool + actuals (requires actual_fp column)",
            value=_av_actuals_in_pool,
            key="av_use_current_pool",
        )

        _av_run = st.button("▶ Run Alert Backtest", key="av_run_btn")

        if _av_run:
            if _av_use_current and _av_actuals_in_pool:
                with st.spinner("Running alert backtest on current slate…"):
                    try:
                        _pool_for_bt = _av_pool.copy()
                        _actuals_for_bt = _pool_for_bt[["player_name", "actual_fp"]].copy()
                        if "actual_minutes" in _pool_for_bt.columns:
                            _actuals_for_bt["actual_minutes"] = _pool_for_bt["actual_minutes"].values
                        _slate_date_bt = str(
                            st.session_state.get("slate_date", pd.Timestamp.now().strftime("%Y-%m-%d"))
                        )
                        _bt_df = run_alert_backtest(
                            _slate_date_bt,
                            _pool_for_bt,
                            _actuals_for_bt,
                            persist=True,
                        )
                        st.session_state["alert_backtest_df"] = _bt_df
                        st.session_state["alert_backtest_slates"] = [_slate_date_bt]
                        st.success(f"✅ Backtest complete — {len(_bt_df)} alert records generated.")
                    except Exception as _av_exc:
                        st.error(f"Backtest failed: {_av_exc}")
            else:
                st.info(
                    "Load a player pool with an **actual_fp** column in 📂 Load Player Pool above, "
                    "then check the box to run the backtest on that slate."
                )

        # ── Show results ─────────────────────────────────────────────────────
        _av_bt_df = st.session_state.get("alert_backtest_df")

        if _av_bt_df is not None and not _av_bt_df.empty:
            _av_metrics = aggregate_alert_metrics([_av_bt_df])
            _av_edge = compute_overall_edge(_av_bt_df)

            st.markdown("---")
            # Overall edge score
            st.markdown("#### 🎯 Overall Blind-Follow Edge")
            _ecol1, _ecol2, _ecol3 = st.columns(3)
            _ecol1.metric("Flagged Hit Rate", f"{_av_edge['flagged_hit_rate']:.1%}")
            _ecol2.metric("Baseline Hit Rate", f"{_av_edge['baseline_hit_rate']:.1%}")
            _edge_delta = _av_edge["edge"]
            _ecol3.metric(
                "Edge",
                f"{_edge_delta:+.1%}",
                delta=f"{_edge_delta:+.1%}",
                delta_color="normal",
            )
            st.info(_av_edge["summary"])

            st.markdown("---")
            st.markdown("#### 📊 Per-Alert-Type Metrics")

            # Stack alerts
            _stk = _av_metrics["stack"]
            with st.expander(f"🔥 Stack Alerts — hit rate {_stk['hit_rate']:.1%} ({_stk['n_flagged']} flagged)", expanded=True):
                _sc1, _sc2, _sc3 = st.columns(3)
                _sc1.metric("Hit Rate", f"{_stk['hit_rate']:.1%}")
                _sc2.metric("Dud Rate", f"{_stk['dud_rate']:.1%}")
                _sc3.metric("False-Neg Rate", f"{_stk['false_neg_rate']:.1%}")
                if not _stk["per_slate"].empty:
                    st.dataframe(_stk["per_slate"], use_container_width=True, hide_index=True)
                if not _stk["examples_hit"].empty:
                    st.markdown("**Big Hits**")
                    st.dataframe(
                        _stk["examples_hit"][["entity_id", "actual_fp", "proj_total"]].rename(
                            columns={"entity_id": "team", "actual_fp": "actual_fp", "proj_total": "proj"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                if not _stk["examples_miss"].empty:
                    st.markdown("**Big Misses**")
                    st.dataframe(
                        _stk["examples_miss"][["entity_id", "actual_fp", "proj_total"]].rename(
                            columns={"entity_id": "team"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

            # High-value alerts
            _hv = _av_metrics["high_value"]
            with st.expander(f"💎 High-Value Alerts — hit rate {_hv['overall_hit_rate']:.1%} ({_hv['n_flagged']} flagged)", expanded=True):
                _hc1, _hc2 = st.columns(2)
                _hc1.metric("Overall Hit Rate", f"{_hv['overall_hit_rate']:.1%}")
                _hc2.metric("Flagged vs Unflagged Δ", f"{_hv['avg_delta_flagged'] - _hv['avg_delta_unflagged']:+.1f} FP")
                if not _hv["tier_detail"].empty:
                    st.markdown("**By Salary Tier**")
                    _tier_disp = _hv["tier_detail"].copy()
                    _tier_disp["hit_rate"] = _tier_disp["hit_rate"].apply(lambda x: f"{x:.1%}")
                    st.dataframe(_tier_disp, use_container_width=True, hide_index=True)
                if not _hv["examples_hit"].empty:
                    st.markdown("**Big Hits**")
                    _hv_cols = [c for c in ["entity_id", "actual_fp", "proj_total"] if c in _hv["examples_hit"].columns]
                    st.dataframe(_hv["examples_hit"][_hv_cols], use_container_width=True, hide_index=True)

            # Injury cascade alerts
            _cas = _av_metrics["injury_cascade"]
            with st.expander(f"🚑 Injury Cascade — {_cas['n_beneficiaries']} beneficiaries", expanded=False):
                _ic1, _ic2, _ic3 = st.columns(3)
                _ic1.metric("% Minutes Increased", f"{_cas['pct_minutes_increased']:.1%}")
                _ic2.metric("% FP Closer to Bumped", f"{_cas['pct_fp_closer_to_bumped']:.1%}")
                _ic3.metric("Mean Signed Error", f"{_cas['mean_signed_error']:+.1f} FP")
                if not _cas["per_slate"].empty:
                    st.dataframe(_cas["per_slate"], use_container_width=True, hide_index=True)

            # Game environment alerts
            _ge = _av_metrics["game_environment"]
            with st.expander(f"⚡ Game Environment — {_ge['n_shootout_flagged']} shootouts, {_ge['n_blowout_flagged']} blowouts", expanded=False):
                _gc1, _gc2, _gc3 = st.columns(3)
                _gc1.metric("Shootout Hit Rate", f"{_ge['shootout_hit_rate']:.1%}")
                _gc2.metric("Shootout Top-3 Rate", f"{_ge['shootout_top3_rate']:.1%}")
                _gc3.metric("Blowout Risk Hit Rate", f"{_ge['blowout_risk_hit_rate']:.1%}")

            # ── Auto-Tuning Block ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 🔧 Auto-Tuning Suggestions")
            _tuning = tune_alert_thresholds(_av_metrics)
            if _tuning["needs_tuning"]:
                st.warning(f"⚠️ {len(_tuning['changes'])} threshold adjustment(s) suggested:")
                for _ch in _tuning["changes"]:
                    st.markdown(f"- {_ch}")

                _av_tune_col1, _av_tune_col2 = st.columns(2)
                with _av_tune_col1:
                    st.markdown("**Current Thresholds**")
                    st.json(_tuning["current"])
                with _av_tune_col2:
                    st.markdown("**Proposed Thresholds**")
                    st.json(_tuning["proposed"])

                if st.button("🔄 Re-run with Tuned Thresholds", key="av_retune_btn"):
                    if _av_actuals_in_pool:
                        with st.spinner("Re-running backtest with tuned thresholds…"):
                            try:
                                _pool_for_bt2 = st.session_state["pool_df"].copy()
                                _actuals_for_bt2 = _pool_for_bt2[["player_name", "actual_fp"]].copy()
                                if "actual_minutes" in _pool_for_bt2.columns:
                                    _actuals_for_bt2["actual_minutes"] = _pool_for_bt2["actual_minutes"].values
                                _slate_date_bt2 = str(
                                    st.session_state.get("slate_date", pd.Timestamp.now().strftime("%Y-%m-%d"))
                                )
                                _bt_df_tuned = run_alert_backtest(
                                    _slate_date_bt2,
                                    _pool_for_bt2,
                                    _actuals_for_bt2,
                                    thresholds=_tuning["proposed"],
                                    persist=False,
                                )
                                _av_metrics_tuned = aggregate_alert_metrics([_bt_df_tuned])
                                _av_edge_tuned = compute_overall_edge(_bt_df_tuned)

                                st.markdown("##### Before vs After Tuning")
                                _bac1, _bac2 = st.columns(2)
                                with _bac1:
                                    st.markdown("**Before**")
                                    st.metric("Stack Hit Rate", f"{_av_metrics['stack']['hit_rate']:.1%}")
                                    st.metric("HV Hit Rate", f"{_av_metrics['high_value']['overall_hit_rate']:.1%}")
                                    st.metric("Edge", f"{_av_edge['edge']:+.1%}")
                                with _bac2:
                                    st.markdown("**After**")
                                    st.metric("Stack Hit Rate", f"{_av_metrics_tuned['stack']['hit_rate']:.1%}")
                                    st.metric("HV Hit Rate", f"{_av_metrics_tuned['high_value']['overall_hit_rate']:.1%}")
                                    st.metric("Edge", f"{_av_edge_tuned['edge']:+.1%}")
                            except Exception as _rt_exc:
                                st.error(f"Re-run failed: {_rt_exc}")
                    else:
                        st.info("Load a pool with actuals to re-run.")
            else:
                st.success("✅ All alert thresholds are performing well — no tuning required.")

        elif _av_bt_df is not None and _av_bt_df.empty:
            st.info("Backtest returned no records. Ensure the pool has opponent/team/proj columns.")
        else:
            st.info("Run the alert backtest above to see validation metrics here.")


# ── System Audit (PR #62) ──────────────────────────────────────────
# [x] All imports used
# [x] All session_state keys have readers and writers
# [x] No orphaned functions
# [x] 3-tab layout consistent (tab_slate, tab_optimizer, tab_lab); tab_calib aliased to tab_lab
# [x] Knobs propagate: slider → cal_knobs session_state → _apply_yakos_projections → yakos_ensemble
# [x] Both fetch call sites pass knobs (Slate Room ~line 764, Cal Lab ~line 1356)
# [x] Cal Lab section ordering verified: Load Pool → KPI Cards → A (Queue) → Error Diagnosis → B (Knobs) → Re-project Button → C (Sims) → D (Multi-Slate, hidden if no data) → E (Backtest)
# [x] Section D hidden gracefully when no parquets (collapsed expander + st.info message)
# ────────────────────────────────────────────────────────────────────

