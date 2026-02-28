"""YakOS DFS Optimizer - Ricky's Slate Room + Calibration Lab."""

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


# -----------------------------
# Core helpers
# -----------------------------


def rename_rg_columns_to_yakos(df: pd.DataFrame) -> pd.DataFrame:
    """Map RotoGrinders NBA CSV columns to YakOS schema."""
    col_map = {
        "Name": "name",
        "Position": "pos",
        "Team": "team",
        "Opponent": "opp",
        "Salary": "salary",
        "Projection": "proj",
        "Ceiling": "ceil",
        "Floor": "floor",
        "Ownership": "own",
    }
    out = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure required columns exist
    required = ["name", "pos", "team", "opp", "salary", "proj"]
    for col in required:
        if col not in out.columns:
            out[col] = 0

    # Standardize dtypes
    out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
    for c in ["proj", "ceil", "floor", "own"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop obviously bad rows
    out = out.dropna(subset=["name", "salary", "proj"])
    out = out[out["salary"] > 0]

    return out


def rename_rg_raw_to_yakos(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw RG export (PLAYER, SALARY, FPTS, POWN, etc.) to YakOS schema."""
    col_map = {
        "PLAYER": "name",
        "POS": "pos",
        "TEAM": "team",
        "OPP": "opp",
        "SALARY": "salary",
        "FPTS": "proj",
        "POWN": "own",
    }
    out = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    required = ["name", "pos", "team", "opp", "salary", "proj"]
    for col in required:
        if col not in out.columns:
            out[col] = 0

    out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
    for c in ["proj", "own"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Clean ownership: strip % if present, convert to float
    if "own" in out.columns:
        out["own"] = out["own"].astype(str).str.replace("%", "").astype(float)

    out = out.dropna(subset=["name", "salary", "proj"])
    out = out[out["salary"] > 0]

    # Handle multi-position: take first position for optimizer
    out["pos"] = out["pos"].astype(str).str.split("/").str[0]

    return out


def apply_slate_filters(
    pool: pd.DataFrame,
    slate_type: str,
    showdown_game: str | None,
) -> pd.DataFrame:
    if slate_type == "Classic":
        return pool

    if slate_type == "Showdown Captain" and showdown_game:
        home, away = showdown_game.split(" @ ")
        mask = ((pool["team"] == home) & (pool["opp"] == away)) | (
            (pool["team"] == away) & (pool["opp"] == home)
        )
        return pool[mask].copy()

    return pool


def get_showdown_games(pool: pd.DataFrame) -> list[str]:
    if pool.empty:
        return []
    games = set()
    for _, row in pool.iterrows():
        t, o = row.get("team"), row.get("opp")
        if pd.isna(t) or pd.isna(o):
            continue
        if f"{t} @ {o}" not in games and f"{o} @ {t}" not in games:
            games.add(f"{t} @ {o}")
    return sorted(games)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def ensure_session_state():
    if "lineups_df" not in st.session_state:
        st.session_state["lineups_df"] = None
    if "exposures_df" not in st.session_state:
        st.session_state["exposures_df"] = None
    if "current_lineup_index" not in st.session_state:
        st.session_state["current_lineup_index"] = 0
    if "pool_df" not in st.session_state:
        st.session_state["pool_df"] = None


def run_optimizer(
    pool: pd.DataFrame,
    num_lineups: int,
    max_exposure: float,
    min_salary_used: int,
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    cfg: Dict[str, Any] = {
        "SITE": "dk",
        "SPORT": "nba",
        "SLATE_TYPE": "classic",
        "NUM_LINEUPS": num_lineups,
        "MIN_SALARY_USED": min_salary_used,
        "MAX_EXPOSURE": max_exposure,
        "PROJ_COL": "proj",
    }

    try:
        lineups_df, exposures_df = build_multiple_lineups_with_exposure(pool, cfg)
    except Exception as e:
        st.error(f"Optimizer error: {e}")
        return None, None

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

tab_front, tab_lab = st.tabs(["Ricky's Slate Room", "Calibration + Sims (Lab)"])


# ============================================================
# Tab 1: Ricky's Slate Room (public)
# ============================================================
with tab_front:
    st.subheader("Ricky's Slate Room")

    # 1) Upload + pool
    st.markdown("### 1. Upload Today's Projection CSV")

    uploaded_file = st.file_uploader(
        "Upload RotoGrinders (or combined) NBA CSV",
        type=["csv"],
        help="Start with an RG export. Internal YakOS projections will sit on top of this.",
        key="front_rg_upload",
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        pool_df = rename_rg_columns_to_yakos(raw_df)
        st.session_state["pool_df"] = pool_df

        st.markdown("##### Pool KPIs")
        n_players = len(pool_df)
        avg_salary = pool_df["salary"].mean()
        median_proj = pool_df["proj"].median()
        max_own = pool_df["own"].max() if "own" in pool_df.columns else None

        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Player count", f"{n_players}")
        kpi_cols[1].metric("Avg salary", f"{avg_salary:,.0f}")
        kpi_cols[2].metric("Median proj", f"{median_proj:.2f}")
        if max_own is not None:
            kpi_cols[3].metric("Max ownership", f"{max_own:.1f}%")

        with st.expander("View cleaned pool data", expanded=False):
            st.dataframe(pool_df, use_container_width=True, height=400)
    else:
        st.info("Upload a RotoGrinders NBA CSV to begin.")
        st.session_state["pool_df"] = None
        pool_df = None

    # 2) Slate controls
    st.markdown("### 2. Slate & Contest Context")

    col_slate_left, col_slate_right = st.columns(2)
    with col_slate_left:
        slate_type = st.selectbox(
            "Slate Type",
            ["Classic", "Showdown Captain"],
            index=0,
        )

    showdown_game = None
    if st.session_state["pool_df"] is not None:
        games = get_showdown_games(st.session_state["pool_df"])
    else:
        games = []

    with col_slate_right:
        if slate_type == "Showdown Captain":
            if games:
                showdown_game = st.selectbox("Showdown Game", games)
            else:
                st.warning("No games detected in pool for Showdown. Upload a valid CSV.")
                showdown_game = None

    contest_type = st.selectbox(
        "Contest Type",
        ["GPP", "50/50", "Single Entry", "MME", "Captain"],
        index=0,
    )

    # 3) Ricky's analysis stub
    st.markdown("### 3. Ricky's Slate Notes")

    col_notes_left, col_notes_right = st.columns(2)

    with col_notes_left:
        st.markdown("#### Pace & Environment")
        st.write(
            "- Fast-paced spots and high total games will show up heavily in Ricky's lineups.\n"
            "- Slow, sloggy games will be used more for value or leverage than primary stacks.\n"
            "- Blowout risk reduces floor for thin rotations; Ricky will cap exposure there."
        )

        st.markdown("#### Stacks & Correlation")
        st.write(
            "- Same-team 2- and 3-man stacks are preferred in high total, close-spread games.\n"
            "- Bring-backs are encouraged when the opponent also has strong projection and minutes.\n"
            "- Showdown Captain lineups will emphasize strong captain candidates plus correlated flex pieces."
        )

    with col_notes_right:
        st.markdown("#### Contest-Type Lean")
        st.write(
            "- GPP: prioritize ceiling, leverage vs heavy chalk, and correlated stacks.\n"
            "- 50/50: tighter player pool, higher floor, less correlation risk.\n"
            "- Single Entry: in-between; strong lineups that aren't pure mega-chalk.\n"
            "- Captain: focus on high-usage, high-ceiling captains with complementary pieces."
        )
        st.info(
            "These notes will eventually be driven directly from YakOS sims and calibration for "
            "each slate and contest type."
        )

    # 4) Ricky's featured lineups
    st.markdown("### 4. Ricky's Featured Lineups (Preset)")

    pool_for_opt = None
    if st.session_state["pool_df"] is not None:
        pool_for_opt = apply_slate_filters(
            st.session_state["pool_df"], slate_type, showdown_game
        )

    if pool_for_opt is not None and not pool_for_opt.empty:
        preset_lineups_df, _ = run_optimizer(
            pool_for_opt,
            num_lineups=5,
            max_exposure=0.35,
            min_salary_used=46500,
        )

        if preset_lineups_df is not None and not preset_lineups_df.empty:
            st.success("Ricky generated 5 featured lineups for this context.")
            unique_lineups_preset = sorted(preset_lineups_df["lineup_id"].unique())
            num_preset = len(unique_lineups_preset)
            preset_idx = st.number_input(
                "Featured Lineup #",
                min_value=1,
                max_value=num_preset,
                value=1,
                step=1,
            )
            current_id = unique_lineups_preset[preset_idx - 1]
            rows = preset_lineups_df[preset_lineups_df["lineup_id"] == current_id].copy()
            total_salary = rows["salary"].sum()
            total_proj = rows["proj"].sum()
            st.markdown(
                f"**Featured Lineup {preset_idx}** -- Salary: {int(total_salary):,} | Proj: {total_proj:.2f}"
            )

            display_cols = ["name", "team", "pos", "salary", "proj"]
            lineup_display = rows[display_cols].copy()
            lineup_display["team"] = lineup_display["team"].fillna("")
            lineup_display["pos"] = lineup_display["pos"].fillna("")
            lineup_display["salary"] = lineup_display["salary"].astype(int)

            st.dataframe(lineup_display, use_container_width=True, height=260)
        else:
            st.info("No featured lineups available for this context yet.")
    else:
        st.info("Upload a pool and set a slate to see featured lineups.")

    # 5) User sandbox optimizer
    st.markdown("### 5. Build Your Own Lineups")

    st.write(
        "Use the same pool and context above to generate your own lineups in Ricky's style. "
        "These use the internal YakOS projection column."
    )

    col_knobs_left, col_knobs_right = st.columns(2)
    with col_knobs_left:
        num_lineups_user = st.slider(
            "Number of lineups",
            min_value=1,
            max_value=300,
            value=5,
            step=1,
        )
        min_salary_used_user = st.number_input(
            "Min salary used",
            min_value=0,
            max_value=50000,
            value=46500,
            step=500,
        )
    with col_knobs_right:
        max_

