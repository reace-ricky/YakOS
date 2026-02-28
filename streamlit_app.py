"""YakOS DFS Optimizer - Ricky's Slate Room + Optimizer + Calibration Lab."""

import sys
import os
from typing import Dict, Any, Tuple
from pathlib import Path

import pandas as pd
import streamlit as st

# Make yak_core importable (works when yak_core is in the repo / installed)
if "yak_core" not in sys.modules:
    pass

from yak_core.lineups import build_multiple_lineups_with_exposure  # type: ignore
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
)
from yak_core.right_angle import (  # type: ignore
    ricky_annotate,
    detect_stack_alerts,
    detect_pace_environment,
    detect_high_value_plays,
)
from yak_core.sims import (  # type: ignore
    run_monte_carlo_for_lineups,
    simulate_live_updates,
)
from yak_core.live import (  # type: ignore
    fetch_live_opt_pool,
    fetch_injury_updates,
)


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
    for c in ["proj", "ceil", "floor", "sim85", "ownership"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Clean ownership: strip trailing % (RG exports as "28.5%") then convert to float
    if "ownership" in out.columns:
        out["ownership"] = (
            out["ownership"].astype(str).str.replace("%", "", regex=False).pipe(pd.to_numeric, errors="coerce")
        )

    # Drop rows with missing salary or projection
    out = out.dropna(subset=["salary", "proj"])
    out = out[out["salary"] > 0]

    # Handle multi-position: take first position for optimizer
    out["pos"] = out["pos"].astype(str).str.split("/").str[0]

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
    if "sim_lineups_df" not in st.session_state:
        st.session_state["sim_lineups_df"] = None
    if "sim_results_df" not in st.session_state:
        st.session_state["sim_results_df"] = None
    if "rapidapi_key" not in st.session_state:
        st.session_state["rapidapi_key"] = os.environ.get("RAPIDAPI_KEY", "")


def run_optimizer(
    pool: pd.DataFrame,
    num_lineups: int,
    max_exposure: float,
    min_salary_used: int,
    proj_col: str = "proj",  # which column to use: 'proj', 'floor', 'ceil', or 'sim85'
    archetype: str = "Balanced",
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

    cfg: Dict[str, Any] = {
        "SITE": "dk",
        "SPORT": "nba",
        "SLATE_TYPE": "classic",
        "NUM_LINEUPS": num_lineups,
        "MIN_SALARY_USED": min_salary_used,
        "MAX_EXPOSURE": max_exposure,
        "PROJ_COL": "proj",
        "SOLVER_TIME_LIMIT": 30,
    }

    progress_bar = st.progress(0, text="Optimizing lineups…")

    def _update_progress(done: int, total: int) -> None:
        pct = int(done / total * 100) if total > 0 else 0
        progress_bar.progress(pct, text=f"Solving lineup {done} of {total}…")

    try:
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
        st.session_state["rapidapi_key"] = rapidapi_key_input
        os.environ["RAPIDAPI_KEY"] = rapidapi_key_input

tab_slate, tab_optimizer, tab_lab = st.tabs([
    "🏀 Ricky's Slate Room",
    "⚡ Optimizer",
    "🔬 Calibration Lab",
])


# ============================================================
# Tab 1: 🏀 Ricky's Slate Room
# ============================================================
with tab_slate:
    st.subheader("🏀 Ricky's Slate Room")

    # --- Pool upload ---
    st.markdown("### 1. Load Today's Player Pool")

    if sport == "PGA":
        st.info("PGA support is coming soon. Please select NBA for now.")
    else:
        load_col_l, load_col_r = st.columns([2, 1])

        with load_col_l:
            uploaded_file = st.file_uploader(
                "Upload RotoGrinders (or combined) NBA CSV  *(optional)*",
                type=["csv"],
                help=(
                    "RG projections are optional — upload to calibrate projections. "
                    "If not uploaded, use 'Fetch from API' or internal YakOS projections."
                ),
                key="front_rg_upload",
            )

        with load_col_r:
            st.markdown("**— or —**")
            fetch_date_input = st.date_input(
                "Slate date (for API fetch)",
                value=pd.Timestamp.today().date(),
                key="slate_fetch_date",
            )
            if st.button(
                "🌐 Fetch Pool from API",
                key="fetch_api_btn",
                help="Requires Tank01 RapidAPI key set in the sidebar.",
            ):
                api_key = st.session_state.get("rapidapi_key", "")
                if not api_key:
                    st.error("Set your Tank01 RapidAPI key in the sidebar first.")
                else:
                    with st.spinner("Fetching live DK pool from Tank01…"):
                        try:
                            live_pool = fetch_live_opt_pool(
                                str(fetch_date_input),
                                {"RAPIDAPI_KEY": api_key},
                            )
                            # Rename to YakOS schema if needed
                            if "player_name" not in live_pool.columns and "name" in live_pool.columns:
                                live_pool = live_pool.rename(columns={"name": "player_name"})
                            st.session_state["pool_df"] = live_pool
                            st.success(f"Loaded {len(live_pool)} players from API.")
                        except Exception as _e:
                            st.error(f"API fetch failed: {_e}")

        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            pool_df = rename_rg_columns_to_yakos(raw_df)
            st.session_state["pool_df"] = pool_df
        elif st.session_state.get("pool_df") is None:
            st.info(
                "Upload a RotoGrinders NBA CSV **or** click **Fetch Pool from API** to begin. "
                "RG projections are optional — they are used to calibrate when available."
            )

        pool_df = st.session_state.get("pool_df")

        # --- KPIs ---
        if pool_df is not None and not pool_df.empty:
            st.markdown("### 2. Pool KPIs")
            n_players = len(pool_df)
            avg_salary = pool_df["salary"].mean()
            median_proj = pool_df["proj"].median()
            top_proj = pool_df["proj"].max()
            max_own = pool_df["ownership"].max() if "ownership" in pool_df.columns else None
            avg_own = pool_df["ownership"].mean() if "ownership" in pool_df.columns else None

            kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
            kpi_c1.metric("Players", f"{n_players}")
            kpi_c2.metric("Avg Salary", f"${avg_salary:,.0f}")
            kpi_c3.metric("Median Proj", f"{median_proj:.1f}")
            kpi_c4.metric("Top Proj", f"{top_proj:.1f}")
            if max_own is not None:
                kpi_c5.metric("Max Ownership", f"{max_own:.1f}%")

            if avg_own is not None:
                value_leader = _slate_value_leader(pool_df)
                st.caption(
                    f"Avg ownership: {avg_own:.1f}% | Slate value leader: {value_leader}"
                )

            with st.expander("View cleaned pool", expanded=False):
                st.dataframe(pool_df, use_container_width=True, height=350)

            # --- Right Angle Ricky Edge Analysis ---
            st.markdown("### 3. 📐 Right Angle Ricky — Edge Analysis")

            col_edge_l, col_edge_r = st.columns(2)

            with col_edge_l:
                st.markdown("#### 🔥 Stack Alerts")
                stack_alerts = detect_stack_alerts(pool_df)
                if stack_alerts:
                    for alert in stack_alerts:
                        st.markdown(f"- {alert}")
                else:
                    st.info("No strong stack signals detected.")

                st.markdown("#### ⚡ Pace / Game Environment")
                pace_notes = detect_pace_environment(pool_df)
                if pace_notes:
                    for note in pace_notes:
                        st.markdown(f"- {note}")
                else:
                    st.info("Upload a pool with opponent data for game environment analysis.")

            with col_edge_r:
                st.markdown("#### 💎 High-Value Plays")
                value_plays = detect_high_value_plays(pool_df)
                if value_plays:
                    for play in value_plays:
                        st.markdown(f"- {play}")
                else:
                    st.info("No stand-out value plays identified.")

                st.markdown("#### 📋 Quick Notes")
                # Dynamic notes based on detected pool data
                quick_notes = []

                if stack_alerts:
                    quick_notes.append(
                        f"**Stack alerts**: {len(stack_alerts)} team stack(s) detected — "
                        "prioritise 2-man / 3-man stacks in GPPs for ceiling."
                    )
                else:
                    quick_notes.append(
                        "**Stack alerts**: No dominant team stacks detected — "
                        "spread exposure across multiple games."
                    )

                if pace_notes:
                    quick_notes.append(
                        f"**High-paced game**: {len(pace_notes)} high-total game(s) on slate — "
                        "stack teammates from high-total matchups."
                    )
                else:
                    quick_notes.append(
                        "**Pace environment**: Game-total data not available — "
                        "upload a pool with opponent data for environment analysis."
                    )

                if value_plays:
                    quick_notes.append(
                        f"**Value plays**: {len(value_plays)} stand-out value play(s) identified — "
                        "use as cheap differentiators to free up salary for studs."
                    )
                else:
                    quick_notes.append(
                        "**Value plays**: No stand-out values detected — "
                        "consider lowering min projection threshold."
                    )

                quick_notes.append(
                    "**Ownership leverage**: In GPPs, fade chalk (>30% own) and target "
                    "low-own upside plays to differentiate from the field."
                )

                for qn in quick_notes:
                    st.markdown(f"- {qn}")
                st.caption(
                    "Quick notes update automatically based on the loaded pool. "
                    "Sims and calibration learnings will sharpen the signals."
                )

            # --- Calibration Queue Lineups ---
            st.markdown("### 4. 📥 Lineups from Calibration Queue")

            promoted = st.session_state.get("promoted_lineups", [])
            if promoted:
                for i, entry in enumerate(promoted):
                    label = entry.get("label", f"Promoted Set {i + 1}")
                    lu_df = entry.get("lineups_df")
                    meta = entry.get("metadata", {})
                    with st.expander(f"**{label}** — {meta.get('contest_type', '')} | {len(lu_df['lineup_index'].unique()) if lu_df is not None else 0} lineups", expanded=(i == 0)):
                        if lu_df is not None and not lu_df.empty:
                            # Annotate with Ricky confidence scores
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
                    "No lineups have been promoted yet. "
                    "Run sims in the **🔬 Calibration Lab** and use "
                    "**Post to Ricky's Slate Room** to surface high-confidence lineups here."
                )

        else:
            st.info("Upload a projection CSV above to see edge analysis and lineups.")


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
                "Load a player pool in **🏀 Ricky's Slate Room** first — "
                "upload an RG CSV or use **Fetch Pool from API**."
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
            st.markdown("### 2. Build Controls")
            ctrl_c1, ctrl_c2, ctrl_c3 = st.columns(3)
            with ctrl_c1:
                num_lu_opt = st.slider("Lineups", 1, 300, num_lineups_user, key="opt_num_lu")
            with ctrl_c2:
                max_exp_opt = st.slider("Max exposure", 0.05, 1.0, max_exposure_user, step=0.05, key="opt_exp")
            with ctrl_c3:
                min_sal_opt = st.number_input("Min salary used", 0, 50000, min_salary_used_user, step=500, key="opt_sal")

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

            # Apply slate filter
            pool_for_opt = apply_slate_filters(pool_df_opt, slate_type_opt, showdown_game_opt)

            if st.button("🚀 Build Lineups", type="primary", key="opt_build_btn"):
                with st.spinner(f"Optimizing ({dk_contest_sel} / {archetype_sel})..."):
                    lu_df, exp_df = run_optimizer(
                        pool_for_opt,
                        num_lineups=num_lu_opt,
                        max_exposure=max_exp_opt,
                        min_salary_used=min_sal_opt,
                        proj_col=proj_style_opt,
                        archetype=archetype_sel,
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

                dl1, dl2 = st.columns(2)
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


# ============================================================
# Tab 3: 🔬 Calibration Lab
# ============================================================
with tab_lab:
    st.subheader("🔬 Calibration Lab")
    st.markdown(
        "Review prior-day lineups, run sims, tune configs, and promote "
        "high-confidence lineups to Ricky's Slate Room."
    )

    # Load historical data
    hist_df = load_historical_lineups()

    # ---- Section A: Calibration Queue ----
    st.markdown("### A. Calibration Queue — Prior-Day Lineups")

    if hist_df.empty:
        st.warning(
            "No historical data found. Add `data/historical_lineups.csv` to the repo "
            "with columns: slate_date, contest_name, lineup_id, pos, team, name, "
            "salary, own, actual."
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
            kq_cols = st.columns(4)
            n_lu = date_queue["lineup_id"].nunique()
            avg_actual = date_queue["actual"].mean() if "actual" in date_queue.columns else 0
            avg_own = date_queue["own"].mean() if "own" in date_queue.columns else 0
            n_reviewed = (date_queue["queue_status"] == "reviewed").sum() if "queue_status" in date_queue.columns else 0
            kq_cols[0].metric("Lineups in queue", n_lu)
            kq_cols[1].metric("Avg actual score", f"{avg_actual:.1f}")
            kq_cols[2].metric("Avg ownership", f"{avg_own:.1f}%")
            kq_cols[3].metric("Reviewed", n_reviewed)

            with st.expander("View queue lineups", expanded=True):
                st.dataframe(date_queue, use_container_width=True, height=300)

            # Individual / Bulk action
            st.markdown("#### Review & Action")
            all_lu_ids = sorted(date_queue["lineup_id"].unique().tolist())
            col_act_l, col_act_r = st.columns(2)
            with col_act_l:
                sel_ids = st.multiselect(
                    "Select lineup IDs to action",
                    all_lu_ids,
                    default=all_lu_ids,
                    key="queue_sel_ids",
                )
                action_choice = st.selectbox(
                    "Action",
                    ["reviewed", "apply_config", "dismissed"],
                    key="queue_action",
                )
            with col_act_r:
                st.markdown(" ")
                st.markdown(" ")
                if st.button("Apply action to selected", key="queue_apply_btn"):
                    updated_q = action_queue_items(
                        st.session_state["cal_queue_df"], sel_ids, action_choice
                    )
                    st.session_state["cal_queue_df"] = updated_q
                    st.success(f"Marked {len(sel_ids)} lineup(s) as '{action_choice}'.")

                if st.button("Bulk: mark all as reviewed", key="queue_bulk_btn"):
                    updated_q = action_queue_items(
                        st.session_state["cal_queue_df"], all_lu_ids, "reviewed"
                    )
                    st.session_state["cal_queue_df"] = updated_q
                    st.success("All lineups marked as reviewed.")

                if st.button("Suggest config changes from queue", key="queue_suggest_btn"):
                    cal_cfg = load_calibration_config()
                    suggested = suggest_config_from_queue(
                        st.session_state["cal_queue_df"], cal_cfg
                    )
                    st.success("Config suggestions derived from queue analysis.")
                    st.json(suggested)

    # ---- Section B: Ad Hoc Historical Lineup Builder ----
    st.markdown("---")
    st.markdown("### B. Ad Hoc Historical Lineup Builder")
    st.markdown(
        "Select a past slate date, apply YakOS projections to the RG pool, "
        "generate optimizer lineups, and compare against actuals."
    )

    if not hist_df.empty:
        available_dates_lab = sorted(hist_df["slate_date"].unique(), reverse=True)
        selected_date = st.selectbox("Slate date", available_dates_lab, key="lab_date_sel")
        slate_data = hist_df[hist_df["slate_date"] == selected_date].copy()

        # Historical slate KPIs
        st.markdown("#### Historical Slate Summary")
        kpi_lab = st.columns(4)
        n_lu_hist = slate_data["lineup_id"].nunique()
        avg_proj_hist = (
            slate_data.groupby("lineup_id")["proj"].sum().mean()
            if "proj" in slate_data.columns else 0
        )
        avg_actual_hist = (
            slate_data.groupby("lineup_id")["actual"].sum().mean()
            if "actual" in slate_data.columns else 0
        )
        proj_err_hist = avg_actual_hist - avg_proj_hist if avg_actual_hist else 0
        kpi_lab[0].metric("Lineups", n_lu_hist)
        kpi_lab[1].metric("Avg Projected", f"{avg_proj_hist:.1f}")
        kpi_lab[2].metric("Avg Actual", f"{avg_actual_hist:.1f}")
        kpi_lab[3].metric("Avg Error", f"{proj_err_hist:+.1f}")

        if "actual" in slate_data.columns:
            agg_cols_hist = {"actual": "mean", "salary": "first", "own": "mean"}
            if "proj" in slate_data.columns:
                agg_cols_hist["proj"] = "mean"
            player_agg = (
                slate_data.groupby("name").agg(agg_cols_hist).reset_index()
            )
            if "proj" in player_agg.columns:
                player_agg["error"] = player_agg["actual"] - player_agg["proj"]
                player_agg["abs_error"] = player_agg["error"].abs()
                player_agg = player_agg.sort_values("abs_error", ascending=False)
            with st.expander("Player accuracy (historical entries)", expanded=False):
                st.dataframe(player_agg, use_container_width=True, height=300)

        # RG pool vs actuals
        rg_file = RG_POOL_FILES.get(str(selected_date))
        if rg_file:
            st.markdown("##### RG Pool vs Actuals")
            rg_pool = load_rg_pool(rg_file)
            if (not rg_pool.empty) and ("actual" in slate_data.columns):
                merged_hist = rg_pool.merge(
                    slate_data[["name", "actual"]].rename(
                        columns={"name": "player_name"}
                    ).drop_duplicates(),
                    on="player_name",
                    how="left",
                )
                merged_hist["error"] = merged_hist["actual"] - merged_hist["proj"]
                disp_cols_hist = [
                    c for c in ["player_name", "pos", "team", "salary", "proj", "actual", "error", "ownership"]
                    if c in merged_hist.columns
                ]
                with st.expander("RG pool vs actuals table", expanded=False):
                    st.dataframe(
                        merged_hist[disp_cols_hist].sort_values("error", ascending=False),
                        use_container_width=True,
                        height=400,
                    )

        # Backtest optimizer
        st.markdown("#### Generate & Compare Backtest Lineups")
        rg_file_bt = RG_POOL_FILES.get(str(selected_date))
        if not rg_file_bt:
            st.info(
                f"No RG pool file configured for {selected_date}. "
                "Add an entry to `RG_POOL_FILES` to enable backtest generation."
            )
        else:
            rg_pool_bt = load_rg_pool(rg_file_bt)
            if rg_pool_bt.empty:
                st.warning("RG pool file loaded but contains no valid players.")
            else:
                bt_col_l, bt_col_r = st.columns(2)
                with bt_col_l:
                    bt_num_lu = st.slider("Backtest lineups", 1, 150, 20, key="bt_num_lu")
                    bt_min_sal = st.number_input("Min salary", 0, 50000, 46500, step=500, key="bt_min_sal")
                with bt_col_r:
                    bt_max_exp = st.slider("Max exposure", 0.05, 1.0, 0.35, step=0.05, key="bt_max_exp")
                    bt_dk_contest = st.selectbox(
                        "DraftKings Contest Type",
                        DK_CONTEST_TYPES,
                        key="bt_contest_sel",
                    )
                    bt_archetype = st.selectbox(
                        "DFS Archetype",
                        list(DFS_ARCHETYPES.keys()),
                        key="bt_archetype_sel",
                    )

                if st.button("▶ Run Backtest", type="primary", key="lab_backtest_btn"):
                    with st.spinner("Running backtest optimizer..."):
                        cal_config = load_calibration_config()
                        bt_internal_contest = DK_CONTEST_TYPE_MAP.get(bt_dk_contest, "GPP")
                        try:
                            # Apply archetype before backtest
                            bt_pool_arched = apply_archetype(rg_pool_bt, bt_archetype)
                            bt_lu_df, _ = run_backtest_lineups(
                                pool=bt_pool_arched,
                                num_lineups=bt_num_lu,
                                max_exposure=bt_max_exp,
                                min_salary_used=bt_min_sal,
                                contest_type=bt_internal_contest,
                                config=cal_config,
                            )
                            if "actual" in slate_data.columns:
                                actuals_lkp = (
                                    slate_data[["name", "actual"]]
                                    .rename(columns={"name": "player_name"})
                                    .drop_duplicates("player_name")
                                )
                                metrics = compute_calibration_metrics(bt_lu_df, actuals_lkp)
                            else:
                                metrics = None
                                st.warning(
                                    "No 'actual' column in historical data — "
                                    "lineups generated but comparison unavailable."
                                )
                            st.session_state["lab_lineups_df"] = bt_lu_df
                            st.session_state["lab_metrics"] = metrics
                            st.session_state["lab_contest_type"] = bt_dk_contest
                            st.success(
                                f"Generated {bt_lu_df['lineup_index'].nunique()} backtest lineups."
                            )
                        except Exception as e:
                            st.error(f"Backtest error: {e}")

                if st.session_state.get("lab_lineups_df") is not None:
                    bt_lu_df = st.session_state["lab_lineups_df"]
                    metrics = st.session_state["lab_metrics"]
                    used_dk_contest = st.session_state["lab_contest_type"]

                    st.markdown("#### Backtest Results")
                    if metrics is not None:
                        ll = metrics["lineup_level"]
                        kpi_bt = st.columns(4)
                        kpi_bt[0].metric("Lineups", len(ll["df"]))
                        kpi_bt[1].metric("Avg Projected", f"{ll['avg_proj']:.1f}")
                        kpi_bt[2].metric("Avg Actual", f"{ll['avg_actual']:.1f}")
                        kpi_bt[3].metric("Avg Error", f"{ll['avg_error']:+.1f}")

                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("MAE", f"{ll['mae']:.2f}")
                        col_m2.metric("RMSE", f"{ll['rmse']:.2f}")
                        col_m3.metric("R²", f"{ll['r_squared']:.3f}")

                        st.caption(
                            "⚠️ Actuals only cover players in historical contest entries. "
                            "Players not in those lineups are scored as 0."
                        )

                        with st.expander("Lineup-by-Lineup Results", expanded=True):
                            st.dataframe(ll["df"], use_container_width=True, height=300)

                        internal_used = DK_CONTEST_TYPE_MAP.get(used_dk_contest, "GPP")
                        insights = identify_calibration_gaps(metrics, internal_used)
                        if insights["findings"]:
                            st.markdown("##### Calibration Insights")
                            for finding in insights["findings"]:
                                st.markdown(f"- {finding}")

                        with st.expander("Player Accuracy", expanded=False):
                            st.dataframe(metrics["player_level"]["df"], use_container_width=True, height=400)

                        if "position_level" in metrics:
                            with st.expander("By Position", expanded=False):
                                st.dataframe(metrics["position_level"]["df"], use_container_width=True)

                        with st.expander("By Salary Bracket", expanded=False):
                            st.dataframe(metrics["salary_bracket"]["df"], use_container_width=True)

                    st.download_button(
                        "Download backtest lineups CSV",
                        data=to_csv_bytes(bt_lu_df),
                        file_name=f"yakos_backtest_{selected_date}.csv",
                        mime="text/csv",
                        key="bt_download",
                    )

    # ---- Section C: Archetype Config Knobs ----
    st.markdown("---")
    st.markdown("### C. 🎛️ Archetype Config Knobs")

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

    # ---- Section D: Sim Module ----
    st.markdown("---")
    st.markdown("### D. 🎲 Sim Module — Live Player Pool")
    st.markdown(
        "Run Monte Carlo sims on the live player pool. "
        "Apply news / injury updates, then promote high-confidence lineups to Ricky's Slate Room."
    )

    pool_for_sim = st.session_state.get("pool_df")
    if pool_for_sim is None:
        st.info(
            "Load a player pool in **🏀 Ricky's Slate Room** to enable sims. "
            "You can upload an RG CSV or fetch the pool from the Tank01 API."
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
                    value=pd.Timestamp.today().date(),
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
                    st.info(f"Applied {len(news_updates)} update(s) to player pool.")
                    changed = []
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
                st.dataframe(sim_res.sort_values("smash_prob", ascending=False), use_container_width=True, height=300)

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
                        updated_pool = active_sim_pool.copy()
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
                    st.success(
                        f"Promoted {len(high_conf_ids)} high-confidence lineup(s) "
                        "to 🏀 Ricky's Slate Room!"
                    )
                else:
                    st.warning("No lineups met the confidence threshold.")

    st.markdown("---")
    st.caption("YakOS Calibration Lab — data-driven lineup refinement.")

