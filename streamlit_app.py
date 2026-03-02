"""YakOS DFS Optimizer - Ricky's Slate Room + Optimizer + Calibration Lab."""

import json
import sys
import os
from typing import Dict, Any, Tuple
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
)
from yak_core.sims import (  # type: ignore
    run_monte_carlo_for_lineups,
    simulate_live_updates,
    build_sim_player_accuracy_table,
    compute_player_anomaly_table,
    SMASH_THRESHOLD as _SIM_SMASH_THRESHOLD,
    BUST_THRESHOLD as _SIM_BUST_THRESHOLD,
)
from yak_core.live import (  # type: ignore
    fetch_live_opt_pool,
    fetch_injury_updates,
    fetch_actuals_from_api,
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

    # Drop rows with missing salary or projection
    out = out.dropna(subset=["salary", "proj"])
    out = out[out["salary"] > 0]

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

# Map internal contest type → suggested default projection style.
# Internal types come from DK_CONTEST_TYPE_MAP in yak_core/calibration.py, e.g.:
#   "Double Up (50/50)" → "50/50" → "floor"  (cash/variance-minimizing)
#   "Tournament (GPP)"  → "GPP"   → "ceil"   (upside/ceiling-chasing)
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

    return pool


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
    if "sim_pool_df" not in st.session_state:
        st.session_state["sim_pool_df"] = None
    if "sim_pool_orig_df" not in st.session_state:
        st.session_state["sim_pool_orig_df"] = None
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


@st.cache_data
def load_historical_lineups() -> pd.DataFrame:
    """Load historical lineups CSV from repo data/ folder."""
    csv_path = Path(__file__).parent / "data" / "historical_lineups.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["slate_date"] = pd.to_datetime(df["slate_date"]).dt.date
        return df
    return pd.DataFrame()


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

tab_slate, tab_optimizer, tab_lab = st.tabs([
    "🏀 Ricky's Slate Room",
    "⚡ Optimizer",
    "📡 Calibration Lab",
])

# Keep backward-compat alias so all existing `with tab_calib:` blocks still work
tab_calib = tab_lab


# ============================================================
# Tab 1: 🏀 Ricky's Slate Room
# ============================================================
with tab_slate:
    st.subheader("🏀 Ricky's Slate Room")

    if sport == "PGA":
        st.info("PGA support is coming soon. Please select NBA for now.")
    else:
        # ── API fetch on the dashboard ───────────────────────────────────
        with st.expander("🌐 Fetch Player Pool from API", expanded=False):
            slate_fetch_col_l, slate_fetch_col_r = st.columns([1, 1])
            with slate_fetch_col_l:
                slate_fetch_date = st.date_input(
                    "Slate date",
                    value=_today_est(),
                    key="slate_fetch_date",
                )
            with slate_fetch_col_r:
                if st.button(
                    "🌐 Fetch Pool from API",
                    key="slate_fetch_api_btn",
                    help="Requires Tank01 RapidAPI key set in the sidebar.",
                ):
                    api_key = st.session_state.get("rapidapi_key", "")
                    if not api_key:
                        st.error("Set your Tank01 RapidAPI key in the sidebar first.")
                    else:
                        with st.spinner("Fetching live DK pool from Tank01…"):
                            try:
                                live_pool = fetch_live_opt_pool(
                                    str(slate_fetch_date),
                                    {"RAPIDAPI_KEY": api_key},
                                )
                                if "player_name" not in live_pool.columns and "name" in live_pool.columns:
                                    live_pool = live_pool.rename(columns={"name": "player_name"})
                                live_pool = _apply_yakos_projections(live_pool, knobs=st.session_state.get("cal_knobs", {}))
                                st.session_state["pool_df"] = live_pool
                                st.success(f"Loaded {len(live_pool)} players from API.")
                            except Exception as _e:
                                st.error(f"API fetch failed: {_e}")

        pool_df = st.session_state.get("pool_df")
        approved_lineups = st.session_state.get("approved_lineups", [])
        last_cal_ts = st.session_state.get("last_calibration_ts")

        if pool_df is None or pool_df.empty:
            st.info(
                "📋 **No player pool loaded.** Use the **Fetch Pool from API** button above "
                "to pull today's slate from Tank01."
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
            _value_scores_df = compute_value_scores(pool_df, top_n=8)

            col_edge_l, col_edge_r = st.columns(2)

            with col_edge_l:
                st.markdown("#### 🔥 Stack Alerts")
                if not _stack_scores_df.empty:
                    for _, _sr in _stack_scores_df.iterrows():
                        _lev_emoji = {"Low-owned CEIL": "🔵", "Moderate": "🟡", "Chalk": "🔴"}.get(
                            _sr["leverage_tag"], "⚪"
                        )
                        st.markdown(
                            f"- {_lev_emoji} **{_sr['team']}** — score {_sr['stack_score']:.0f} | "
                            f"proj {_sr['top_proj']:.1f} | ceil {_sr['top_ceil']:.1f} | "
                            f"{_sr['leverage_tag']} · {_sr['key_players']}"
                        )
                    st.markdown("")
                    with st.expander("📝 Log stack outcome", expanded=False):
                        log_slate_date = st.date_input(
                            "Slate date",
                            value=_today_est(),
                            key="stack_log_date",
                        )
                        _stack_opts = _stack_scores_df["team"].tolist()
                        log_stack_sel = st.selectbox(
                            "Which stack?",
                            _stack_opts,
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
                    st.info("No strong stack signals detected.")

                st.markdown("#### ⚡ Pace / Game Environment")
                pace_notes = detect_pace_environment(pool_df)
                if pace_notes:
                    for _pn in pace_notes:
                        st.markdown(f"- {_pn}")
                else:
                    st.info("Upload a pool with opponent data for game environment analysis.")

            with col_edge_r:
                st.markdown("#### 💎 High-Value Plays")
                if not _value_scores_df.empty:
                    for _, _vr in _value_scores_df.iterrows():
                        _own_emoji = {"Sneaky": "🟣", "Leverage": "🟡", "Chalk": "🔴"}.get(
                            _vr.get("ownership_tag", ""), "⚪"
                        )
                        st.markdown(
                            f"- {_own_emoji} **{_vr['player_name']}** ({_vr.get('team', '?')}) — "
                            f"${int(_vr['salary']):,} | proj {_vr['proj']:.1f} | "
                            f"value {_vr['value_eff']:.2f}x | {_vr.get('ownership_tag', '')}"
                        )
                else:
                    st.info("No stand-out value plays identified.")

            st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # LAYER 2b — Player Projections Table
            # ════════════════════════════════════════════════════════════
            st.markdown("### 📋 Player Projections")
            _proj_disp_cols = [c for c in [
                "player_name", "pos", "team", "salary",
                "proj", "floor", "ceil", "proj_minutes", "proj_own", "proj_source",
            ] if c in pool_df.columns]
            _proj_table = (
                pool_df[_proj_disp_cols]
                .sort_values("proj", ascending=False)
                .reset_index(drop=True)
            )
            with st.expander("📋 All Players — sorted by projection", expanded=True):
                _col_cfg: dict = {
                    "player_name": st.column_config.TextColumn("Player"),
                    "pos": st.column_config.TextColumn("Pos", width="small"),
                    "team": st.column_config.TextColumn("Team", width="small"),
                    "salary": st.column_config.NumberColumn("Salary", format="$%d"),
                    "proj": st.column_config.NumberColumn("Proj", format="%.2f"),
                    "floor": st.column_config.NumberColumn("Floor", format="%.2f"),
                    "ceil": st.column_config.NumberColumn("Ceil", format="%.2f"),
                    "proj_minutes": st.column_config.NumberColumn("Mins", format="%.1f"),
                    "proj_own": st.column_config.NumberColumn("Own %", format="%.1f"),
                    "proj_source": st.column_config.TextColumn("Source"),
                }
                st.dataframe(
                    _proj_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={k: v for k, v in _col_cfg.items() if k in _proj_table.columns},
                )

            st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # LAYER 3 — Ricky's Approved Lineups (read-only from Cal Lab)
            # ════════════════════════════════════════════════════════════
            st.markdown("### 📥 Ricky's Approved Lineups")
            st.caption(
                "These lineups come from Ricky's Calibration Lab after sims and backtests. "
                "Rerun Calibration to refresh."
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
                                _hdr = (
                                    f"**#{_idx + 1}** · {_alu.site} · {_alu.slate} | "
                                    f"ROI {_alu.sim_roi:.1%} · p90 {_alu.sim_p90:.1f} · "
                                    f"proj {_alu.proj_points:.1f}{_late_badge}"
                                )
                                with st.expander(_hdr, expanded=(_idx == 0)):
                                    if _alu.late_swap_window:
                                        st.info(f"🕐 Late-swap window: {_alu.late_swap_window}")
                                    _p_df = pd.DataFrame(_alu.players)
                                    if not _p_df.empty:
                                        st.dataframe(
                                            _p_df,
                                            use_container_width=True,
                                            hide_index=True,
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
                            sel_lu = st.selectbox(
                                "Select lineup",
                                range(1, len(unique_lu) + 1),
                                key=f"promoted_lu_{i}",
                            )
                            lu_rows = annotated[annotated["lineup_index"] == unique_lu[sel_lu - 1]].copy()
                            conf = lu_rows["confidence"].iloc[0] if "confidence" in lu_rows.columns else "—"
                            tag = lu_rows["tag"].iloc[0] if "tag" in lu_rows.columns else "—"
                            st.markdown(
                                f"**Confidence**: {conf} | **Tag**: {tag} | "
                                f"Salary: {int(lu_rows['salary'].sum()):,} | Proj: {lu_rows['proj'].sum():.1f}"
                            )
                            disp_cols = [c for c in ["slot", "team", "player_name", "salary", "proj", "confidence", "tag"] if c in lu_rows.columns]
                            st.dataframe(lu_rows[disp_cols], use_container_width=True, height=240)
                        else:
                            st.info("No lineups in this set.")
            else:
                st.info(
                    "No lineups approved yet. "
                    "Run sims in the **🔬 Calibration Lab** and use "
                    "**Post to Ricky's Slate Room** to surface high-confidence lineups here."
                )


# ============================================================
# Tab 2: ⚡ Optimizer
# ============================================================
with tab_optimizer:
    st.subheader("⚡ Optimizer")
    st.markdown(
        "Build lineups for today's slate. Choose your DraftKings contest type, "
        "DFS archetype, and projection style — then let the optimizer do the work."
    )

    if sport == "PGA":
        st.info("PGA support is coming soon. Please select NBA for now.")
    else:
        pool_df_opt = st.session_state.get("pool_df")
        if pool_df_opt is None:
            st.warning(
                "Load a player pool first — fetch from the Tank01 API in **🏀 Ricky's Slate Room**, "
                "or upload a CSV in the **🔬 Calibration Lab**."
            )
        else:
            # --- Contest & slate controls ---
            st.markdown("### 1. Contest & Slate Context")
            col_opt_l, col_opt_r = st.columns(2)

            with col_opt_l:
                slate_type_opt = st.selectbox(
                    "Slate Type",
                    ["Classic", "Showdown Captain"],
                    index=0,
                    key="opt_slate_type",
                )

                dk_contest_sel = st.selectbox(
                    "DraftKings Contest Type",
                    DK_CONTEST_TYPES,
                    index=DK_CONTEST_TYPES.index(st.session_state["dk_contest_type"])
                    if st.session_state["dk_contest_type"] in DK_CONTEST_TYPES
                    else 0,
                    key="opt_dk_contest",
                    help="Mirror of the contest types available in the DraftKings lobby.",
                )
                st.session_state["dk_contest_type"] = dk_contest_sel
                internal_contest = DK_CONTEST_TYPE_MAP.get(dk_contest_sel, "GPP")

            showdown_game_opt = None
            with col_opt_r:
                if slate_type_opt == "Showdown Captain":
                    games = get_showdown_games(pool_df_opt)
                    if games:
                        showdown_game_opt = st.selectbox("Showdown Game", games, key="opt_showdown")
                    else:
                        st.warning("No games detected. Upload a valid pool.")

                archetype_sel = st.selectbox(
                    "DFS Archetype",
                    list(DFS_ARCHETYPES.keys()),
                    index=list(DFS_ARCHETYPES.keys()).index(st.session_state["archetype"])
                    if st.session_state["archetype"] in DFS_ARCHETYPES
                    else 0,
                    key="opt_archetype",
                    help=(
                        "Ceiling Hunter = max upside (GPP) | "
                        "Floor Lock = cash game | "
                        "Contrarian = low ownership | "
                        "Stacker = team correlation | "
                        "Balanced = default"
                    ),
                )
                st.session_state["archetype"] = archetype_sel
                arch_desc = DFS_ARCHETYPES[archetype_sel]["description"]
                st.caption(f"📌 {arch_desc}")

            # --- Optimizer controls ---
            st.markdown("---")
            st.markdown("### 2. Build Controls")
            ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(4)
            with ctrl_c1:
                num_lu_opt = st.slider("Lineups", 1, 300, num_lineups_user, key="opt_num_lu")
            with ctrl_c2:
                max_exp_opt = st.slider("Max exposure", 0.05, 1.0, max_exposure_user, step=0.05, key="opt_exp")
            with ctrl_c3:
                min_sal_opt = st.number_input("Min salary used", 0, 50000, min_salary_used_user, step=500, key="opt_sal")
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

            # Auto-set projection style when contest type changes
            if dk_contest_sel != st.session_state.get("prev_dk_contest_type"):
                st.session_state["opt_proj_style"] = _default_proj_style_for_contest(internal_contest)
                st.session_state["prev_dk_contest_type"] = dk_contest_sel

            _cur_proj_style = st.session_state.get("opt_proj_style", "proj")
            if _cur_proj_style not in _PROJ_STYLE_OPTIONS:
                _cur_proj_style = "proj"
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

                lu_idx = st.number_input("Lineup #", 1, len(unique_lu), 1, step=1, key="opt_lu_idx")
                cur_idx = unique_lu[lu_idx - 1]
                rows = lineups_df[lineups_df["lineup_index"] == cur_idx].copy()
                st.markdown(
                    f"**Lineup {lu_idx}** — Salary: {int(rows['salary'].sum()):,} | "
                    f"Proj: {rows['proj'].sum():.2f}"
                )
                disp_cols = [c for c in ["slot", "team", "player_name", "salary", "proj"] if c in rows.columns]
                lu_display = rows[disp_cols].copy()
                if "salary" in lu_display.columns:
                    lu_display["salary"] = lu_display["salary"].astype(int)
                st.dataframe(lu_display, use_container_width=True, height=260)

                if exposures_df is not None and not exposures_df.empty:
                    with st.expander("Player Exposures", expanded=False):
                        st.dataframe(exposures_df, use_container_width=True, height=400)

                dl1, dl2, dl3 = st.columns(3)
                with dl1:
                    st.download_button(
                        "Download lineups CSV",
                        data=to_csv_bytes(lineups_df),
                        file_name="yakos_lineups.csv",
                        mime="text/csv",
                        key="opt_dl_lu",
                    )
                with dl2:
                    if exposures_df is not None and not exposures_df.empty:
                        st.download_button(
                            "Download exposures CSV",
                            data=to_csv_bytes(exposures_df),
                            file_name="yakos_exposures.csv",
                            mime="text/csv",
                            key="opt_dl_exp",
                        )
                with dl3:
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
# Tab 3: 🔬 Calibration Lab
# ============================================================
with tab_lab:
    st.subheader("📡 Calibration Lab")

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
                        st.session_state["pool_df"] = live_pool
                        st.success(f"Loaded {len(live_pool)} players from API.")
                    except Exception as _e:
                        st.error(f"API fetch failed: {_e}")

    if rg_upload_cal is not None:
        if st.session_state.get("_pool_df_filename") != rg_upload_cal.name:
            raw_df = pd.read_csv(rg_upload_cal)
            pool_df_cal = rename_rg_columns_to_yakos(raw_df)
            st.session_state["pool_df"] = pool_df_cal
            st.session_state["_pool_df_filename"] = rg_upload_cal.name
            st.success(f"✅ Pool loaded — {len(pool_df_cal)} players. Head to **🏀 Ricky's Slate Room** to review.")

    current_pool_df = st.session_state.get("pool_df")
    if current_pool_df is not None and not current_pool_df.empty:
        st.caption(f"Active pool: **{len(current_pool_df)} players** loaded.")
    else:
        st.info("No pool loaded yet. Upload a player pool CSV above to begin.")

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

    # ── Section A: Calibration Queue ─────────────────────────────────────
    st.markdown("### A. Calibration Queue — Prior-Day Lineups")

    if hist_df.empty:
        st.warning(
            "No historical data found. Add `data/historical_lineups.csv` to the repo "
            "with columns: slate_date, contest_name, pos, team, name, "
            "salary, proj, proj_own, own, actual."
        )
    else:
        queue_df = get_calibration_queue(hist_df, prior_dates=3)
        if st.session_state.get("cal_queue_df") is None:
            st.session_state["cal_queue_df"] = queue_df

        cal_queue = st.session_state["cal_queue_df"]

        if cal_queue is not None and not cal_queue.empty:
            available_dates = sorted(cal_queue["slate_date"].unique(), reverse=True)
            queue_date_sel = st.selectbox("Queue slate date", available_dates, key="queue_date")
            date_queue = cal_queue[cal_queue["slate_date"] == queue_date_sel]

            # KPIs for the queue date
            kq_cols = st.columns(6)
            n_players = len(date_queue)
            avg_proj = date_queue["proj"].mean() if "proj" in date_queue.columns else 0
            avg_proj_own = date_queue["proj_own"].mean() if "proj_own" in date_queue.columns else 0
            avg_actual = date_queue["actual"].mean() if "actual" in date_queue.columns else 0
            avg_own = date_queue["own"].mean() if "own" in date_queue.columns else 0
            n_reviewed = (date_queue["queue_status"] == "reviewed").sum() if "queue_status" in date_queue.columns else 0
            kq_cols[0].metric("Players in queue", n_players)
            kq_cols[1].metric("Avg Projected", f"{avg_proj:.1f}")
            kq_cols[2].metric("Avg Proj Own %", f"{avg_proj_own:.1f}%")
            kq_cols[3].metric("Avg actual score", f"{avg_actual:.1f}")
            kq_cols[4].metric("Avg ownership", f"{avg_own:.1f}%")
            kq_cols[5].metric("Reviewed", n_reviewed)

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

                # Overwrite stale historical projections with real pool projections
                # (pool contains RG FPTS — real consensus projections, not salary proxy)
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

                # Populate proj_minutes from pool's MINUTES column when not yet present
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

            # Compute error columns
            def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
                return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else pd.Series(0.0, index=df.index)

            _qd = _queue_display.copy()
            _qd["pts_error"] = _safe_col(_qd, "actual") - _safe_col(_qd, "proj")
            _qd["min_error"] = _safe_col(_qd, "actual_minutes") - _safe_col(_qd, "proj_minutes")
            _qd["own_error"] = _safe_col(_qd, "own") - _safe_col(_qd, "proj_own")
            # Flag: any error exceeds KPI thresholds (pts >6, mins >3, own >3)
            _qd["Flag"] = (
                _qd["pts_error"].abs().gt(6)
                | _qd["min_error"].abs().gt(3)
                | _qd["own_error"].abs().gt(3)
            )

            # Build player label combining team + pos
            _player_col = []
            for _, _r in _qd.iterrows():
                parts = [str(_r.get("name", ""))]
                tp = " / ".join(filter(None, [str(_r.get("team", "")), str(_r.get("pos", ""))]))
                if tp:
                    parts.append(f"({tp})")
                _player_col.append(" ".join(parts))
            _qd["Player"] = _player_col

            # Focused visible columns only
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

            # ── Calibration Queue — read-only accuracy dashboard ──
            st.markdown("#### Calibration Queue")
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

            _n_flagged = int(_queue_focused["Flag"].sum()) if "Flag" in _queue_focused.columns else 0
            if _n_flagged > 0:
                st.warning(
                    f"⚠️ {_n_flagged} player(s) flagged with large projection errors "
                    "(pts >6, mins >3, or own% >3). Review archetype config knobs below."
                )

                # ── Error Diagnosis ──────────────────────────────────────────
                _qd_diag = _qd.copy()
                _pts_errors = _qd_diag[_qd_diag["pts_error"].abs().gt(6)] if "pts_error" in _qd_diag.columns else _qd_diag.iloc[:0]
                _min_errors = _qd_diag[_qd_diag["min_error"].abs().gt(3)] if "min_error" in _qd_diag.columns else _qd_diag.iloc[:0]
                _own_errors = _qd_diag[_qd_diag["own_error"].abs().gt(3)] if "own_error" in _qd_diag.columns else _qd_diag.iloc[:0]

                st.markdown("#### 🔍 Error Diagnosis")
                _diag_col1, _diag_col2, _diag_col3 = st.columns(3)
                _diag_col1.metric("FP Errors", len(_pts_errors), help="Players with >6pt projection error")
                _diag_col2.metric("Minutes Errors", len(_min_errors), help="Players with >3min projection error")
                _diag_col3.metric("Ownership Errors", len(_own_errors), help="Players with >3% ownership error")

                _err_counts = {
                    "pts": len(_pts_errors),
                    "min": len(_min_errors),
                    "own": len(_own_errors),
                }
                _dominant = max(_err_counts, key=_err_counts.get)
                if _err_counts[_dominant] > 0:
                    if _dominant == "pts":
                        st.info(
                            "💡 FP errors dominate. Try adjusting ensemble weights below "
                            "(reduce YakOS weight if model is biased, increase Tank01/RG weight for consensus)."
                        )
                    elif _dominant == "min":
                        st.info(
                            "💡 Minutes errors dominate. Check b2b discount and blowout threshold knobs below."
                        )
                    else:
                        st.info(
                            "💡 Ownership errors dominate. This typically means RG ownership data was stale. "
                            "Consider increasing RG ensemble weight if RG data is fresh."
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
        st.caption(f"Using projections from pool ({len(pool_for_sim)} players). Knobs: {_knob_summary}.")
    if pool_for_sim is None:
        st.info(
            "Load a player pool using the **📂 Load Player Pool** section at the top of this tab to enable sims."
        )
    else:
        # News / lineup updates
        with st.expander("📰 Live News & Lineup Updates", expanded=False):
            # --- API fetch ---
            api_news_col, manual_news_col = st.columns([1, 2])
            with api_news_col:
                st.markdown("**🌐 Fetch from API**")
                _news_default_date = st.session_state["sim_hist_date"] if _sim_is_historical else _today_est()
                news_slate_date = st.date_input(
                    "Slate date",
                    value=_news_default_date,
                    key="sim_news_date",
                )
                if st.button(
                    "Fetch Live Injury Updates",
                    key="sim_fetch_injuries_btn",
                    help="Pulls today's injury/status list from Tank01 API.",
                ):
                    api_key = st.session_state.get("rapidapi_key", "")
                    if not api_key:
                        st.error("Set your Tank01 RapidAPI key in the sidebar first.")
                    else:
                        with st.spinner("Fetching injury updates…"):
                            try:
                                api_updates = fetch_injury_updates(
                                    news_slate_date.strftime("%Y%m%d"),
                                    {"RAPIDAPI_KEY": api_key},
                                )
                                if api_updates:
                                    sim_pool_api = simulate_live_updates(
                                        pool_for_sim, api_updates
                                    )
                                    st.session_state["sim_pool_df"] = sim_pool_api
                                    st.session_state["sim_pool_orig_df"] = sim_pool_api
                                    st.success(
                                        f"Applied {len(api_updates)} injury update(s) "
                                        "from API."
                                    )
                                    out_players = [
                                        u["player_name"]
                                        for u in api_updates
                                        if u.get("status") == "OUT"
                                    ]
                                    if out_players:
                                        st.warning(
                                            "**OUT**: "
                                            + ", ".join(out_players[:10])
                                            + (" …" if len(out_players) > 10 else "")
                                        )
                                else:
                                    st.info("No injury updates found for this date.")
                            except Exception as _e:
                                st.error(f"Injury API error: {_e}")

            with manual_news_col:
                st.markdown("**✏️ Manual Updates**")
                st.markdown(
                    "One per row: `PlayerName | STATUS | proj_adj | minutes_change`\n\n"
                    "Example: `Zion Williamson | OUT | | ` or `Jayson Tatum | UPGRADED | | +4`"
                )
                news_text = st.text_area(
                    "News updates",
                    placeholder="Zion Williamson | OUT | |\nJayson Tatum | UPGRADED | | 4",
                    height=120,
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
                    st.session_state["sim_pool_df"] = sim_pool
                    st.session_state["sim_pool_orig_df"] = sim_pool
                    changed = []
                    st.info(f"Applied {len(news_updates)} update(s) to player pool.")
                    for u in news_updates:
                        orig = pool_for_sim[pool_for_sim["player_name"] == u["player_name"]]
                        upd = sim_pool[sim_pool["player_name"] == u["player_name"]]
                        if not orig.empty and not upd.empty:
                            orig_p = orig["proj"].iloc[0]
                            upd_p = upd["proj"].iloc[0]
                            changed.append(f"{u['player_name']}: {orig_p:.1f} → {upd_p:.1f}")
                    if changed:
                        st.write("**Projection changes:**")
                        for c in changed:
                            st.markdown(f"- {c}")

        # Sim controls
        sim_col_l, sim_col_r = st.columns(2)
        with sim_col_l:
            sim_n_lu = st.slider("Lineups to sim", 1, 150, 20, key="sim_n_lu")
            sim_vol = st.selectbox("Volatility mode", ["standard", "low", "high"], key="sim_vol")
            sim_n_sims = st.slider("Sim iterations", 100, 2000, 500, step=100, key="sim_n_sims")
        with sim_col_r:
            sim_archetype = st.selectbox(
                "DFS Archetype", list(DFS_ARCHETYPES.keys()), key="sim_archetype"
            )
            sim_dk_contest = st.selectbox(
                "DraftKings Contest Type", DK_CONTEST_TYPES, key="sim_dk_contest"
            )
            sim_min_sal = st.number_input("Min salary", 0, 50000, 46500, step=500, key="sim_min_sal")

        # Actuals upload — for historical slate calibration
        with st.expander("📊 Load Actuals (Historical Slate Calibration)", expanded=_sim_is_historical):
            st.markdown(
                "Load actual DraftKings fantasy points to compare sim projections against real "
                "outcomes.  Choose **API** (fastest — pulls directly from Tank01) or **CSV** "
                "(manual upload of a RotoGrinders export)."
            )
            _acts_tab_api, _acts_tab_csv = st.tabs(["🌐 Fetch from API", "📂 Upload CSV"])

            with _acts_tab_api:
                st.markdown(
                    "Fetch actual player DK scores for a completed slate directly from the "
                    "Tank01 API.  Requires your RapidAPI key set in the sidebar."
                )
                _api_acts_default = st.session_state["sim_hist_date"] if _sim_is_historical else _today_est()
                _api_acts_date = st.date_input(
                    "Slate date (past game day)",
                    value=_api_acts_default,
                    key="sim_actuals_api_date",
                    help="Choose the game day you want actuals for.",
                )
                if st.button("Fetch Actuals from API", key="sim_fetch_actuals_btn"):
                    _api_key = st.session_state.get("rapidapi_key", "")
                    if not _api_key:
                        st.error("Set your Tank01 RapidAPI key in the sidebar first.")
                    else:
                        with st.spinner("Fetching actuals from Tank01…"):
                            try:
                                _api_acts_df = fetch_actuals_from_api(
                                    _api_acts_date.strftime("%Y%m%d"),
                                    {"RAPIDAPI_KEY": _api_key},
                                )
                                st.session_state["sim_actuals_df"] = _api_acts_df
                                st.success(
                                    f"✅ Loaded actuals for {len(_api_acts_df)} players "
                                    f"({_api_acts_date})."
                                )
                            except Exception as _api_err:
                                st.error(f"API actuals fetch failed: {_api_err}")

            with _acts_tab_csv:
                st.markdown(
                    "Upload an actuals CSV — RotoGrinders contest export (`FPTS` / `PLAYER` "
                    "columns) or any CSV with `name`/`player_name` and `actual`/`actual_fp`."
                )
                _actuals_upload = st.file_uploader(
                    "Upload actuals CSV",
                    type=["csv"],
                    key="sim_actuals_upload",
                    help="RotoGrinders contest export or any CSV with player names and actual FP scored.",
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
                        _acts_norm = _acts_raw.rename(columns=_col_renames)
                        # Accept 'actual' column directly too
                        if "actual" in _acts_norm.columns and "actual_fp" not in _acts_norm.columns:
                            _acts_norm = _acts_norm.rename(columns={"actual": "actual_fp"})
                        if "name" in _acts_norm.columns and "player_name" not in _acts_norm.columns:
                            _acts_norm = _acts_norm.rename(columns={"name": "player_name"})
                        _name_col = "player_name" if "player_name" in _acts_norm.columns else None
                        _fp_col = "actual_fp" if "actual_fp" in _acts_norm.columns else None
                        if _name_col and _fp_col:
                            _acts_clean = _acts_norm[[_name_col, _fp_col]].copy()
                            _acts_clean[_fp_col] = pd.to_numeric(_acts_clean[_fp_col], errors="coerce")
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
                if st.button("Clear actuals", key="sim_clear_actuals_btn"):
                    st.session_state["sim_actuals_df"] = None
                    st.rerun()

        active_sim_pool = st.session_state.get("sim_pool_df") or pool_for_sim

        if st.button("🎲 Run Sims", type="primary", key="sim_run_btn"):
            with st.spinner("Building lineups + running Monte Carlo sims..."):
                try:
                    sim_lu_df, _ = run_optimizer(
                        active_sim_pool,
                        num_lineups=sim_n_lu,
                        max_exposure=0.4,
                        min_salary_used=sim_min_sal,
                        proj_col="proj",
                        archetype=sim_archetype,
                    )
                    if sim_lu_df is not None and not sim_lu_df.empty:
                        _sim_cal_knobs = st.session_state.get("cal_knobs", {})
                        sim_res = run_monte_carlo_for_lineups(
                            sim_lu_df, n_sims=sim_n_sims, volatility_mode=sim_vol
                        )
                        # Annotate with Ricky confidence
                        annotated_sim = ricky_annotate(sim_lu_df, sim_res)
                        st.session_state["sim_lineups_df"] = annotated_sim
                        st.session_state["sim_results_df"] = sim_res
                        # Compute per-player anomaly table using cal_knobs
                        _anomaly_df = compute_player_anomaly_table(
                            active_sim_pool,
                            sim_lu_df,
                            n_sims=sim_n_sims,
                            cal_knobs=_sim_cal_knobs,
                        )
                        st.session_state["sim_anomaly_df"] = _anomaly_df
                        # Clear any previous custom lineup when sims are re-run
                        st.session_state["sim_custom_lineup"] = []
                        st.success(
                            f"Sims complete — {sim_n_lu} lineups × {sim_n_sims} iterations."
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
                    f"| **Smash %** | Probability of scoring **≥ {_SIM_SMASH_THRESHOLD:.0f} FP** — the GPP \"smash\" threshold. Higher is better for tournaments |\n"
                    f"| **Bust %** | Probability of scoring **≤ {_SIM_BUST_THRESHOLD:.0f} FP** — likely cash-game miss. Lower is better |\n"
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
                })
                st.dataframe(_sim_display, use_container_width=True, height=300)

            # ── Sim Anomaly Detection ─────────────────────────────────────────
            _sim_anomaly = st.session_state.get("sim_anomaly_df")
            if _sim_anomaly is not None and not _sim_anomaly.empty:
                _high_lev = _sim_anomaly[_sim_anomaly["Flag"] == "🔥 HIGH LEVERAGE"]
                _val_traps = _sim_anomaly[_sim_anomaly["Value Trap"]]
                _top_play = _sim_anomaly.iloc[0] if len(_sim_anomaly) > 0 else None
                _summary_parts = [
                    f"Sim ran **{sim_n_sims}** iterations across **{sim_n_lu}** lineups.",
                    f"Found **{len(_high_lev)}** high-leverage player(s) (smash rate > own%).",
                    f"Found **{len(_val_traps)}** value trap(s) (bust rate > 40% despite high salary).",
                ]
                if _top_play is not None:
                    _summary_parts.append(
                        f"Top leverage play: **{_top_play['Player']}** "
                        f"({_top_play['Smash%']:.1f}% smash, {_top_play['Own%']:.1f}% owned)."
                    )
                st.info("  \n".join(_summary_parts))

                st.markdown("#### 🔍 Sim Anomalies — Leverage Spots")
                st.caption(
                    "Per-player simulation breakdown. "
                    "Leverage Score = Smash% / Own% — higher means more upside relative to expected ownership. "
                    "🔥 HIGH LEVERAGE = score > 3.0. ⚠️ Value Trap = busts frequently despite high salary."
                )
                _anomaly_display = _sim_anomaly.copy()
                _anomaly_display["Value Trap"] = _anomaly_display["Value Trap"].apply(
                    lambda x: "⚠️ VALUE TRAP" if x else ""
                )
                st.dataframe(_anomaly_display, use_container_width=True, height=350)

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
                        orig_pool = st.session_state.get("sim_pool_orig_df") or pool_for_sim
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
            st.markdown("#### 📤 Post to Ricky's Slate Room")
            conf_threshold = st.slider(
                "Minimum confidence to promote", 40.0, 95.0, 65.0, step=5.0, key="sim_conf_thr"
            )
            if st.button("Post high-confidence lineups → Ricky's Slate Room", type="primary", key="sim_post_btn"):
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
                    _arch_label = {
                        "Tournament (GPP)": "GPP",
                        "Single Entry": "SE",
                        "Max Entry (MME)": "3-MAX",
                        "50/50": "50/50",
                        "Double Up": "50/50",
                        "Showdown Captain": "Showdown",
                    }.get(sim_dk_contest, sim_dk_contest)
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
                        f"Promoted {len(high_conf_ids)} high-confidence lineup(s) "
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

