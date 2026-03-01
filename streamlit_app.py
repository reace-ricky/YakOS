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


# Make yak_core importable (works when yak_core is in the repo / installed)
if "yak_core" not in sys.modules:
    pass

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
    SMASH_THRESHOLD as _SIM_SMASH_THRESHOLD,
    BUST_THRESHOLD as _SIM_BUST_THRESHOLD,
)
from yak_core.live import (  # type: ignore
    fetch_live_opt_pool,
    fetch_injury_updates,
)
from yak_core.multislate import (  # type: ignore
    parse_dk_contest_csv,
    discover_slates,
    run_multi_slate,
    compare_slates,
)
from yak_core.projections import salary_implied_proj, noisy_proj  # type: ignore
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


@st.cache_data
def load_rg_pool(filename: str) -> pd.DataFrame:
    """Load a raw RG projection CSV from repo data/ folder."""
    csv_path = Path(__file__).parent / "data" / filename
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return rename_rg_raw_to_yakos(df)
    return pd.DataFrame()


# Map slate dates to RG pool files in data/
RG_POOL_FILES = {
    "2026-02-27": "NBADK20260227.csv",
}


# -----------------------------
# Streamlit App Layout
# -----------------------------


st.set_page_config(
    page_title="YakOS DFS Optimizer",
    layout="wide",
)

ensure_session_state()

# ── Auto-load sample pool on first run so the dashboard shows projections ──
if st.session_state.get("pool_df") is None:
    _latest_sample = sorted(RG_POOL_FILES.keys())[-1] if RG_POOL_FILES else None
    if _latest_sample:
        _sample_df = load_rg_pool(RG_POOL_FILES[_latest_sample])
        if not _sample_df.empty:
            st.session_state["pool_df"] = _apply_proj_fallback(_sample_df)

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

tab_slate, tab_optimizer, tab_lab, tab_calib = st.tabs([
    "🏀 Ricky's Slate Room",
    "⚡ Optimizer",
    "🔬 Calibration Lab",
    "📡 Ricky's Calibration Lab",
])


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
                                live_pool = _apply_proj_fallback(live_pool)
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
                "or upload your RotoGrinders projection sheet in the **🔬 Calibration Lab** tab."
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
                "Load a player pool in the **🔬 Calibration Lab** first — "
                "upload an RG projection sheet there to get started."
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
    st.subheader("🔬 Calibration Lab")

    # ── 0. Player Pool / RG Projection Upload ─────────────────────────────
    st.markdown("### 📂 Load Player Pool")
    st.markdown(
        "Upload your RotoGrinders projection sheet here. "
        "This is the primary way to load a pool — it powers the Slate Room, Optimizer, and Sims."
    )

    cal_upload_l, cal_upload_r = st.columns([2, 1])
    with cal_upload_l:
        rg_upload_cal = st.file_uploader(
            "Upload RotoGrinders NBA CSV",
            type=["csv"],
            key="cal_rg_upload",
            help="RG projection export — FPTS, SALARY, OWNERSHIP columns will be mapped automatically.",
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
                        live_pool = _apply_proj_fallback(live_pool)
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
        st.info("No pool loaded yet. Upload an RG CSV above to begin.")

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

        def _kpi_card(label: str, value_str: str, metric_key: str, raw_value: float) -> str:
            qc = quality_color(metric_key, raw_value)
            bg = _QUALITY_BG[qc]
            color = _QUALITY_TEXT[qc]
            return (
                f'<div style="border:1px solid #3a3a3a;border-radius:6px;'
                f'padding:10px 8px;text-align:center;background:{bg};">'
                f'<div style="font-size:0.72rem;text-transform:uppercase;'
                f'letter-spacing:0.06em;color:#aaa;margin-bottom:4px;">{label}</div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:{color};">{value_str}</div>'
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

        st.caption(
            "Lower MAE is better; higher hit rate is better.  "
            "Colored cards: green = in line; yellow = borderline; red = needs calibration."
        )

        # ── Advanced breakdown (collapsed) ───────────────────────────────────
        with st.expander("🔬 Advanced breakdown", expanded=False):
            _ptsl = _kpis["points_lineup"]
            st.markdown("**Points — lineup level**")
            _dc1, _dc2, _dc3, _dc4 = st.columns(4)
            _dc1.metric("Mean Error (pts)", f"{_ptsl['mean_error']:+.2f}", help="avg(actual − proj) per lineup")
            _dc2.metric("Std Dev (pts)", f"{_ptsl['std_error']:.2f}")
            _dc3.metric("RMSE (pts)", f"{_ptsl['rmse']:.2f}")
            _r2_val = _ptsl["r_squared"]
            _r_val = (_r2_val ** 0.5) if _r2_val >= 0 else float("nan")
            _dc4.metric("R² (lineup)", f"{_r2_val:.3f}", help="Correlation between proj and actual lineup scores")
            _lu_df = _ptsl["df"][["lineup_id", "proj", "actual", "error"]].copy()
            _lu_df = _lu_df.rename(columns={"proj": "Projected", "actual": "Actual"})
            st.scatter_chart(_lu_df, x="Projected", y="Actual", height=240)
            st.caption(f"r = {_r_val:.3f}  R² = {_r2_val:.3f}  — diagonal = perfect calibration.")

            st.markdown("**Points — player level**")
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            _pc1.metric("Mean Error (player)", f"{_ptsp['mean_error']:+.2f}")
            _pc2.metric("MAE (player)", f"{_ptsp['mae']:.2f}")
            _pc3.metric("R² (player)", f"{_ptsp['r_squared']:.3f}")
            _pc4.metric("Avg Actual (player)", f"{_ptsp['df']['actual'].mean():.1f}")

            if "points_salary" in _kpis:
                st.markdown("**Error by salary bracket**")
                _sal_df = _kpis["points_salary"]["df"].copy()
                _sal_df.columns = [str(c) for c in _sal_df.columns]
                st.dataframe(
                    _sal_df.rename(columns={
                        "salary_bracket": "Salary Bracket",
                        "avg_proj": "Avg Projected",
                        "avg_actual": "Avg Actual",
                        "mean_error": "Mean Error",
                        "mae": "MAE",
                        "count": "Players",
                    }),
                    use_container_width=True,
                )

            if "minutes" in _kpis:
                st.markdown("**Minutes accuracy**")
                _mins = _kpis["minutes"]
                _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                _mc1.metric("Mean Error (mins)", f"{_mins['mean_error']:+.2f}")
                _mc2.metric("MAE (mins)", f"{_mins['mae']:.2f}")
                _mc3.metric("Pts err (>5 min miss)", f"{_mins['avg_pts_err_large_min_miss']:+.2f}")
                _mc4.metric("Pts err (≤5 min miss)", f"{_mins['avg_pts_err_small_min_miss']:+.2f}")

            if "ownership" in _kpis:
                st.markdown("**Ownership accuracy**")
                _own = _kpis["ownership"]
                _oc1, _oc2, _oc3 = st.columns(3)
                _oc1.metric("Mean Error (own %)", f"{_own['mean_error']:+.2f}%")
                _oc2.metric("MAE (own %)", f"{_own['mae']:.2f}%")
                _oc3.metric("Players Tracked", f"{int(_kpis['ownership']['bucket_df']['count'].sum())}")
                _bkt = _kpis["ownership"]["bucket_df"].copy()
                st.dataframe(
                    _bkt.rename(columns={
                        "bucket": "Proj Own Bucket",
                        "avg_proj_own": "Avg Proj Own%",
                        "avg_actual_own": "Avg Actual Own%",
                        "mean_error": "Mean Error",
                        "mae": "MAE",
                        "count": "Players",
                    }),
                    use_container_width=True,
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

            # ── Review & Action — check rows, pick a bubble, hit Apply ──
            st.markdown("#### Review & Action")
            queue_edit_df = _queue_focused.copy()
            queue_edit_df.insert(0, "✓", False)
            edited_queue = st.data_editor(
                queue_edit_df,
                column_config={
                    "✓": st.column_config.CheckboxColumn("✓", default=False, width="small"),
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
                disabled=[c for c in queue_edit_df.columns if c != "✓"],
                use_container_width=True,
                key=f"queue_editor_{queue_date_sel}",
            )

            n_selected = int(edited_queue["✓"].sum())
            action_choice = st.radio(
                "Action",
                ["pass", "review"],
                horizontal=True,
                key="queue_action",
            )

            root_cause = None
            if action_choice == "review":
                root_cause = st.selectbox(
                    "Root Cause",
                    ["proj pts", "proj mins", "proj own %", "Other"],
                    key="queue_root_cause",
                )

            if st.button(
                f"Apply to {n_selected} selected" if n_selected else "Apply (select rows above)",
                key="queue_apply_btn",
                disabled=(n_selected == 0),
            ):
                sel_row_ids = date_queue.index[edited_queue["✓"].values].tolist()
                updated_q = action_queue_items(
                    st.session_state["cal_queue_df"], sel_row_ids, action_choice, id_col="row_id"
                )
                st.session_state["cal_queue_df"] = updated_q
                if action_choice == "review" and root_cause == "Other":
                    sel_rows = date_queue.loc[sel_row_ids]
                    for _, row in sel_rows.iterrows():
                        def _to_float(val):
                            try:
                                v = float(val)
                                return v if not pd.isna(v) else 0.0
                            except (TypeError, ValueError):
                                return 0.0
                        st.session_state["other_root_causes"].append({
                            "slate_date": str(row.get("slate_date", queue_date_sel)),
                            "name": str(row.get("name", "")),
                            "pos": str(row.get("pos", "")),
                            "proj": _to_float(row.get("proj")),
                            "actual": _to_float(row.get("actual")),
                            "own": _to_float(row.get("own")),
                        })
                st.success(f"Marked {n_selected} row(s) as '{action_choice}'.")

            with st.expander("🔍 Other Root Causes", expanded=False):
                other_causes = st.session_state.get("other_root_causes", [])
                if other_causes:
                    st.caption(
                        f"{len(other_causes)} item(s) flagged as 'Other'. "
                        "Watch for emerging trends across these entries."
                    )
                    st.dataframe(pd.DataFrame(other_causes), use_container_width=True)
                    if st.button("Clear 'Other' causes", key="clear_other_causes"):
                        st.session_state["other_root_causes"] = []
                        st.rerun()
                else:
                    st.info("No 'Other' root causes logged yet.")

    # ---- Stack Hit Log ----
    st.markdown("---")
    st.markdown("### 📊 Stack Hit Log")
    st.markdown("Track which stack calls hit so you can calibrate signal accuracy over time.")

    hit_log = st.session_state.get("stack_hit_log", [])
    if hit_log:
        hit_df = pd.DataFrame(hit_log)
        total = len(hit_df)
        n_hits = (hit_df["outcome"] == "Hit").sum()
        hit_rate = n_hits / total

        log_kpi1, log_kpi2, log_kpi3 = st.columns(3)
        log_kpi1.metric("Total Logged", total)
        log_kpi2.metric("Hits", n_hits)
        log_kpi3.metric("Hit Rate", f"{hit_rate:.0%}")

        st.dataframe(hit_df, use_container_width=True, height=220)

        if st.button("🗑️ Clear log", key="stack_log_clear"):
            st.session_state["stack_hit_log"] = []
            st.rerun()
    else:
        st.info(
            "No stack outcomes logged yet. "
            "Use the **📝 Log stack outcome** control in **🏀 Ricky's Slate Room** "
            "after each slate to build a hit-rate history."
        )

    # ---- Consolidated: Build Best Lineup + Compare vs Contest ----
    st.markdown("---")
    with st.expander("🏗️ Build Best Lineup for a Slate", expanded=False):
        st.markdown(
            "Select a past slate, apply YakOS projections to the RG pool, "
            "and generate our best optimizer lineup to use as a calibration baseline."
        )
        _bl_dates_set = set(RG_POOL_FILES.keys())
        if not hist_df.empty:
            _bl_dates_set |= {str(d) for d in hist_df["slate_date"].unique()}
        _bl_available = sorted(_bl_dates_set, reverse=True)
        if not _bl_available:
            st.info(
                "No slate data available. Add an entry to `RG_POOL_FILES` or load "
                "`data/historical_lineups.csv` to enable this feature."
            )
        else:
            _bl_date = st.selectbox("Slate date", _bl_available, key="bl_date_sel")
            _bl_slate_data = (
                hist_df[hist_df["slate_date"] == _bl_date].copy()
                if not hist_df.empty else pd.DataFrame()
            )

            if not _bl_slate_data.empty:
                st.markdown("#### Historical Slate Summary")
                _bl_kpis = st.columns(4)
                _bl_n_lu = _bl_slate_data["lineup_id"].nunique()
                _bl_avg_proj = (
                    _bl_slate_data.groupby("lineup_id")["proj"].sum().mean()
                    if "proj" in _bl_slate_data.columns else 0
                )
                _bl_avg_actual = (
                    _bl_slate_data.groupby("lineup_id")["actual"].sum().mean()
                    if "actual" in _bl_slate_data.columns else 0
                )
                _bl_err = _bl_avg_actual - _bl_avg_proj if _bl_avg_actual else 0
                _bl_kpis[0].metric("Lineups", _bl_n_lu)
                _bl_kpis[1].metric("Avg Projected", f"{_bl_avg_proj:.1f}")
                _bl_kpis[2].metric("Avg Actual", f"{_bl_avg_actual:.1f}")
                _bl_kpis[3].metric("Avg Error", f"{_bl_err:+.1f}")

                if "actual" in _bl_slate_data.columns:
                    _bl_agg = {"actual": "mean", "salary": "first", "own": "mean"}
                    if "proj" in _bl_slate_data.columns:
                        _bl_agg["proj"] = "mean"
                    _bl_player_agg = _bl_slate_data.groupby("name").agg(_bl_agg).reset_index()
                    if "proj" in _bl_player_agg.columns:
                        _bl_player_agg["error"] = _bl_player_agg["actual"] - _bl_player_agg["proj"]
                        _bl_player_agg["abs_error"] = _bl_player_agg["error"].abs()
                        _bl_player_agg = _bl_player_agg.sort_values("abs_error", ascending=False)
                    with st.expander("Player accuracy (historical entries)", expanded=False):
                        st.dataframe(_bl_player_agg, use_container_width=True, height=300)

            _bl_rg_file = RG_POOL_FILES.get(str(_bl_date))
            if not _bl_rg_file:
                st.info(
                    f"No RG pool file configured for {_bl_date}. "
                    "Add an entry to `RG_POOL_FILES` to enable lineup generation."
                )
            else:
                _bl_rg_pool = load_rg_pool(_bl_rg_file)
                if _bl_rg_pool.empty:
                    st.warning("RG pool file loaded but contains no valid players.")
                else:
                    _bl_col_l, _bl_col_r = st.columns(2)
                    with _bl_col_l:
                        _bl_num_lu = st.slider("Lineups to generate", 1, 150, 10, key="bl_num_lu")
                        _bl_min_sal = st.number_input(
                            "Min salary", 0, 50000, 46500, step=500, key="bl_min_sal"
                        )
                    with _bl_col_r:
                        _bl_max_exp = st.slider(
                            "Max exposure", 0.05, 1.0, 0.35, step=0.05, key="bl_max_exp"
                        )
                        _bl_dk_contest = st.selectbox(
                            "DraftKings Contest Type",
                            DK_CONTEST_TYPES,
                            key="bl_contest_sel",
                        )
                        _bl_archetype = st.selectbox(
                            "DFS Archetype",
                            list(DFS_ARCHETYPES.keys()),
                            key="bl_archetype_sel",
                        )

                    if st.button("▶ Build Best Lineup", type="primary", key="bl_run_btn"):
                        with st.spinner("Generating lineup..."):
                            _bl_cal_config = load_calibration_config()
                            _bl_internal_contest = DK_CONTEST_TYPE_MAP.get(_bl_dk_contest, "GPP")
                            try:
                                _bl_pool_arched = apply_archetype(_bl_rg_pool, _bl_archetype)
                                _bl_lu_df, _ = run_backtest_lineups(
                                    pool=_bl_pool_arched,
                                    num_lineups=_bl_num_lu,
                                    max_exposure=_bl_max_exp,
                                    min_salary_used=_bl_min_sal,
                                    contest_type=_bl_internal_contest,
                                    config=_bl_cal_config,
                                )
                                if (
                                    not _bl_slate_data.empty
                                    and "actual" in _bl_slate_data.columns
                                ):
                                    _bl_actuals_lkp = (
                                        _bl_slate_data[["name", "actual"]]
                                        .rename(columns={"name": "player_name"})
                                        .drop_duplicates("player_name")
                                    )
                                    _bl_metrics = compute_calibration_metrics(
                                        _bl_lu_df, _bl_actuals_lkp
                                    )
                                else:
                                    _bl_metrics = None
                                st.session_state["bl_lineups_df"] = _bl_lu_df
                                st.session_state["bl_metrics"] = _bl_metrics
                                st.session_state["bl_slate_date"] = _bl_date
                                st.session_state["bl_slate_data"] = _bl_slate_data
                                st.session_state["bl_contest_type"] = _bl_dk_contest
                                st.success(
                                    f"Generated {_bl_lu_df['lineup_index'].nunique()} "
                                    f"lineup(s) for {_bl_date}."
                                )
                            except Exception as e:
                                st.error(f"Optimizer error: {e}")

                    if st.session_state.get("bl_lineups_df") is not None:
                        _bl_lu_df = st.session_state["bl_lineups_df"]
                        _bl_metrics = st.session_state["bl_metrics"]
                        _bl_used_date = st.session_state.get("bl_slate_date", _bl_date)
                        _bl_used_contest = st.session_state.get(
                            "bl_contest_type", _bl_dk_contest
                        )

                        st.markdown("#### Best Lineup")
                        _bl_lu_scores = _bl_lu_df.groupby("lineup_index")["proj"].sum()
                        _bl_lu_scores = _bl_lu_scores.dropna()
                        if _bl_lu_scores.empty:
                            st.warning("No lineups generated.")
                        else:
                            _bl_top_idx = int(_bl_lu_scores.idxmax())
                            _bl_top_rows = _bl_lu_df[
                                _bl_lu_df["lineup_index"] == _bl_top_idx
                            ].copy()

                            # Merge actuals from saved slate data
                            _bl_sd = st.session_state.get("bl_slate_data", pd.DataFrame())
                            if not _bl_sd.empty:
                                _bl_act = (
                                    _bl_sd[
                                        [c for c in ["name", "actual", "own"] if c in _bl_sd.columns]
                                    ]
                                    .drop_duplicates("name")
                                    .rename(
                                        columns={
                                            "name": "player_name",
                                            "actual": "actual_fp",
                                            "own": "actual_own",
                                        }
                                    )
                                )
                                _bl_top_rows = _bl_top_rows.merge(
                                    _bl_act, on="player_name", how="left"
                                )

                            _bl_ticket_cols = [
                                c for c in [
                                    "slot", "player_name", "pos", "salary", "proj",
                                    "proj_own", "actual_own", "actual_fp",
                                ]
                                if c in _bl_top_rows.columns
                            ]
                            _bl_col_labels = {
                                "slot": "Slot", "player_name": "Player",
                                "pos": "Position", "salary": "Salary",
                                "proj": "Proj FP", "proj_own": "Proj Own%",
                                "actual_own": "Actual Own%", "actual_fp": "Actual FP",
                            }
                            st.dataframe(
                                _bl_top_rows[_bl_ticket_cols]
                                .rename(columns=_bl_col_labels)
                                .reset_index(drop=True),
                                use_container_width=True,
                            )
                            _bl_tot_cols = st.columns(3)
                            _bl_tot_cols[0].metric(
                                "Total Salary", f"${_bl_top_rows['salary'].sum():,.0f}"
                            )
                            _bl_tot_cols[1].metric(
                                "Total Proj FP", f"{_bl_top_rows['proj'].sum():.1f}"
                            )
                            if "actual_fp" in _bl_top_rows.columns:
                                _bl_tot_cols[2].metric(
                                    "Total Actual FP",
                                    f"{_bl_top_rows['actual_fp'].fillna(0).sum():.1f}",
                                )

                            if _bl_metrics is not None:
                                _bl_ll = _bl_metrics["lineup_level"]
                                _bl_kpi_bt = st.columns(4)
                                _bl_kpi_bt[0].metric("Lineups", len(_bl_ll["df"]))
                                _bl_kpi_bt[1].metric(
                                    "Avg Projected", f"{_bl_ll['avg_proj']:.1f}"
                                )
                                _bl_kpi_bt[2].metric(
                                    "Avg Actual", f"{_bl_ll['avg_actual']:.1f}"
                                )
                                _bl_kpi_bt[3].metric(
                                    "Avg Error", f"{_bl_ll['avg_error']:+.1f}"
                                )

                                _bl_internal_used = DK_CONTEST_TYPE_MAP.get(
                                    _bl_used_contest, "GPP"
                                )
                                _bl_insights = identify_calibration_gaps(
                                    _bl_metrics, _bl_internal_used
                                )
                                if _bl_insights["findings"]:
                                    st.markdown("##### Calibration Insights")
                                    for _bl_finding in _bl_insights["findings"]:
                                        st.markdown(f"- {_bl_finding}")

                            st.download_button(
                                "Download all lineups CSV",
                                data=to_csv_bytes(_bl_lu_df),
                                file_name=f"yakos_best_lineups_{_bl_used_date}.csv",
                                mime="text/csv",
                                key="bl_download",
                            )

    with st.expander("📊 Compare vs. Contest Lineup", expanded=False):
        st.markdown(
            "Upload a DraftKings contest CSV to import real ownership, then select a "
            "top lineup from that slate and compare it side-by-side with the YakOS "
            "best lineup above as a calibration tool."
        )
        _cmp_upload = st.file_uploader(
            "Upload DraftKings contest CSV",
            type=["csv"],
            key="cmp_dk_contest_csv",
            help=(
                "Export from DraftKings: My Contests → contest page → Export CSV. "
                "Accepted formats: contest results with a 'Lineup' column, or "
                "a simple Name / Ownership % file."
            ),
        )
        if _cmp_upload is not None:
            try:
                _cmp_dk_df = parse_dk_contest_csv(_cmp_upload)
                st.session_state["cmp_dk_df"] = _cmp_dk_df
                st.success(f"Parsed {len(_cmp_dk_df)} players from DK contest CSV.")
            except Exception as e:
                st.error(f"Failed to parse DK contest CSV: {e}")

        _cmp_dk_df = st.session_state.get("cmp_dk_df")
        if _cmp_dk_df is not None and not _cmp_dk_df.empty:
            with st.expander("Contest player pool", expanded=False):
                _cmp_disp_cols = [
                    c for c in ["player_name", "pos", "team", "salary", "actual_fp", "ownership"]
                    if c in _cmp_dk_df.columns
                ]
                _cmp_sorted = (
                    _cmp_dk_df[_cmp_disp_cols].sort_values("actual_fp", ascending=False)
                    if "actual_fp" in _cmp_dk_df.columns
                    else _cmp_dk_df[_cmp_disp_cols]
                )
                st.dataframe(_cmp_sorted, use_container_width=True, height=250)

            # User selects players to form their "contest lineup"
            st.markdown("**Select your top contest lineup (up to 8 players):**")
            _cmp_names = (
                sorted(_cmp_dk_df["player_name"].dropna().unique().tolist())
                if "player_name" in _cmp_dk_df.columns
                else []
            )
            _cmp_selected = st.multiselect(
                "Pick players for your contest lineup",
                _cmp_names,
                max_selections=8,
                key="cmp_player_sel",
            )

            # Side-by-side comparison with YakOS best lineup
            _cmp_bl_lu = st.session_state.get("bl_lineups_df")
            if _cmp_bl_lu is not None and _cmp_selected:
                st.markdown("#### Side-by-Side Comparison")
                _cmp_lu_scores = _cmp_bl_lu.groupby("lineup_index")["proj"].sum().dropna()
                if _cmp_lu_scores.empty:
                    st.warning("YakOS lineup data is empty — rebuild using the expander above.")
                else:
                    _cmp_top_idx = int(_cmp_lu_scores.idxmax())
                    _cmp_yakos_lu = _cmp_bl_lu[
                        _cmp_bl_lu["lineup_index"] == _cmp_top_idx
                    ][["player_name", "pos", "salary", "proj"]].copy()

                    _cmp_contest_lu = (
                        _cmp_dk_df[_cmp_dk_df["player_name"].isin(_cmp_selected)][
                            [c for c in ["player_name", "pos", "salary", "actual_fp", "ownership"]
                             if c in _cmp_dk_df.columns]
                        ].copy()
                        if "player_name" in _cmp_dk_df.columns
                        else pd.DataFrame()
                    )

                    _cmp_col1, _cmp_col2 = st.columns(2)
                    with _cmp_col1:
                        st.markdown("**🤖 YakOS Best Lineup**")
                        _cmp_yk_cols = [
                            c for c in ["player_name", "pos", "salary", "proj"]
                            if c in _cmp_yakos_lu.columns
                        ]
                        st.dataframe(
                            _cmp_yakos_lu[_cmp_yk_cols].reset_index(drop=True),
                            use_container_width=True,
                        )
                        st.metric("Total Proj FP", f"{_cmp_yakos_lu['proj'].sum():.1f}")
                        st.metric("Total Salary", f"${_cmp_yakos_lu['salary'].sum():,.0f}")
                    with _cmp_col2:
                        st.markdown("**🎯 My Contest Pick**")
                        _cmp_ct_disp = [
                            c for c in ["player_name", "pos", "salary", "actual_fp", "ownership"]
                            if c in _cmp_contest_lu.columns
                        ]
                        st.dataframe(
                            _cmp_contest_lu[_cmp_ct_disp].reset_index(drop=True),
                            use_container_width=True,
                        )
                        if "actual_fp" in _cmp_contest_lu.columns:
                            st.metric(
                                "Total Actual FP",
                                f"{_cmp_contest_lu['actual_fp'].fillna(0).sum():.1f}",
                            )
                        if "salary" in _cmp_contest_lu.columns:
                            st.metric(
                                "Total Salary",
                                f"${_cmp_contest_lu['salary'].fillna(0).sum():,.0f}",
                            )
            elif _cmp_selected and _cmp_bl_lu is None:
                st.info(
                    "Build a YakOS lineup first using "
                    "**🏗️ Build Best Lineup for a Slate** above."
                )
            elif not _cmp_selected:
                st.info(
                    "Select up to 8 players above to build your contest lineup for comparison."
                )

            # Merge real ownership into live pool
            _cmp_live_pool = st.session_state.get("pool_df")
            if (
                _cmp_live_pool is not None
                and not _cmp_live_pool.empty
                and "ownership" in _cmp_dk_df.columns
            ):
                if st.button("Merge real ownership → Live Pool", key="cmp_merge_own_btn"):
                    _cmp_merged_pool = _cmp_live_pool.copy()
                    _own_lkp = dict(
                        zip(_cmp_dk_df["player_name"], _cmp_dk_df["ownership"])
                    )
                    _cmp_merged_pool["ownership"] = (
                        _cmp_merged_pool["player_name"]
                        .map(_own_lkp)
                        .fillna(_cmp_merged_pool.get("ownership", 0))
                    )
                    st.session_state["pool_df"] = _cmp_merged_pool
                    _cmp_matched = _cmp_merged_pool["player_name"].isin(_own_lkp).sum()
                    st.success(
                        f"Merged real ownership for {_cmp_matched}/{len(_cmp_merged_pool)} "
                        "players. Pool updated — head to the Optimizer to rebuild lineups."
                    )
            elif _cmp_live_pool is None:
                st.info(
                    "Load a player pool using the **📂 Load Player Pool** section "
                    "above to merge ownership."
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

    # ---- Section C: Sim Module ----
    st.markdown("---")
    st.markdown("### C. 🎲 Sim Module — Live Player Pool")
    st.markdown(
        "Run Monte Carlo sims on the live player pool. "
        "Apply news / injury updates, then promote high-confidence lineups to Ricky's Slate Room."
    )

    pool_for_sim = st.session_state.get("pool_df")
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
                news_slate_date = st.date_input(
                    "Slate date",
                    value=_today_est(),
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
                        sim_res = run_monte_carlo_for_lineups(
                            sim_lu_df, n_sims=sim_n_sims, volatility_mode=sim_vol
                        )
                        # Annotate with Ricky confidence
                        annotated_sim = ricky_annotate(sim_lu_df, sim_res)
                        st.session_state["sim_lineups_df"] = annotated_sim
                        st.session_state["sim_results_df"] = sim_res
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

            # Apply learnings: boost projection of high-sim players for next run
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

    st.markdown("---")
    # ---- Section D: Multi-Slate ----
    st.markdown("### D. Multi-Slate Comparison")
    st.markdown(
        "Discover available historical slates, batch-run the optimizer across "
        "multiple dates, and compare KPIs side-by-side."
    )

    slates_df = discover_slates()
    if slates_df.empty:
        st.info(
            "No parquet pool files found in the YakOS root directory.  "
            "Multi-slate comparison requires `tank_opt_pool_<date>.parquet` files.  "
            "Set the `YAKOS_ROOT` environment variable to the folder that contains them."
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
    st.markdown("## 📡 Ricky's Calibration Lab – Backtesting & Calibration")
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

