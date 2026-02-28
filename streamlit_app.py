"""YakOS DFS Optimizer – Ricky's Slate Room + Calibration Lab."""

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
# Core helpers (from your current app)
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
    # Single projection knob: Ricky/YakOS internal projection
    cfg: Dict[str, Any] = {
        "SITE": "dk",
        "SPORT": "nba",
        "SLATE_TYPE": "classic",
        "NUM_LINEUPS": num_lineups,
        "MIN_SALARY_USED": min_salary_used,
        "MAX_EXPOSURE": max_exposure,
        "PROJ_COL": "proj",  # always use internal proj column for now
    }

    try:
        lineups_df, exposures_df = build_multiple_lineups_with_exposure(pool, cfg)
    except Exception as e:  # noqa: BLE001
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

        # Simple KPIs
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

    # Contest type selector
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

    # 4) Ricky's featured lineups (simple preset)
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
        max_exposure_user = st.slider(
            "Max exposure per player",
            min_value=0.05,
            max_value=1.0,
            value=0.35,
            step=0.05,
        )

    build_clicked = st.button("Build Lineups", type="primary")

    if build_clicked:
        if pool_for_opt is None or pool_for_opt.empty:
            st.warning("No valid player pool to optimize. Check your upload and slate settings.")
        else:
            lineups_df, exposures_df = run_optimizer(
                pool_for_opt,
                num_lineups=num_lineups_user,
                max_exposure=max_exposure_user,
                min_salary_used=min_salary_used_user,
            )
            st.session_state["lineups_df"] = lineups_df
            st.session_state["exposures_df"] = exposures_df
            st.session_state["current_lineup_index"] = 0

    lineups_df = st.session_state.get("lineups_df")
    exposures_df = st.session_state.get("exposures_df")

    if lineups_df is not None and exposures_df is not None and not lineups_df.empty:
        st.success(f"Built {len(lineups_df['lineup_id'].unique())} lineups.")

        st.markdown("#### Lineup Viewer")

        unique_lineups = sorted(lineups_df["lineup_id"].unique())
        num_unique = len(unique_lineups)

        col_view_left, col_view_right = st.columns([1, 3])
        with col_view_left:
            lineup_number = st.number_input(
                "Lineup #",
                min_value=1,
                max_value=num_unique,
                value=st.session_state.get("current_lineup_index", 0) + 1,
                step=1,
            )
            st.session_state["current_lineup_index"] = lineup_number - 1
            current_id = unique_lineups[st.session_state["current_lineup_index"]]

        lineup_rows = lineups_df[lineups_df["lineup_id"] == current_id].copy()
        total_salary = lineup_rows["salary"].sum()
        total_proj = lineup_rows["proj"].sum()

        with col_view_right:
            st.markdown(
                f"**Lineup {lineup_number}** -- Salary: {int(total_salary):,} | Proj: {total_proj:.2f}"
            )

        display_cols = ["name", "team", "pos", "salary", "proj"]
        lineup_display = lineup_rows[display_cols].copy()
        lineup_display["team"] = lineup_display["team"].fillna("")
        lineup_display["pos"] = lineup_display["pos"].fillna("")
        lineup_display["salary"] = lineup_display["salary"].astype(int)

        st.dataframe(lineup_display, use_container_width=True, height=280)

        st.markdown("#### Player Exposures")
        st.dataframe(exposures_df, use_container_width=True, height=320)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="Download Lineups CSV",
                data=to_csv_bytes(lineups_df),
                file_name="yakos_lineups.csv",
                mime="text/csv",
            )
        with col_dl2:
            st.download_button(
                label="Download Exposures CSV",
                data=to_csv_bytes(exposures_df),
                file_name="yakos_exposures.csv",
                mime="text/csv",
            )
    else:
        st.info("Build lineups to see the viewer and exposures.")


# ============================================================
# Tab 2: Calibration + Sims (Lab, admin only)
# ============================================================
with tab_lab:
    st.subheader("Calibration + Sims Lab")

    st.markdown(
        "This tab is your private lab to calibrate YakOS vs external projections, "
        "run sims by contest type, and tune Ricky's profiles."
    )

    # Simple admin gate (placeholder)
    admin_key = st.text_input("Admin key", type="password", help="Lab is restricted.")
    SECRET_KEY = "ricky-lab-123"  # TODO: move to env/secret later

    if admin_key != SECRET_KEY:
        st.warning("Restricted: enter the correct admin key to access the lab.")
        st.stop()

    st.success("Admin access granted. Welcome to the lab.")

    # --- Load historical data from repo ---
    hist_df = load_historical_lineups()

    if hist_df.empty:
        st.error("No historical lineups found in data/historical_lineups.csv.")
        st.stop()

    # --- Filters ---
    st.markdown("### 1. Filters")

    available_dates = sorted(hist_df["slate_date"].unique())
    available_contests = sorted(hist_df["contest_name"].unique())

    col_filt_left, col_filt_right = st.columns(2)
    with col_filt_left:
        lab_date = st.selectbox(
            "Slate date",
            options=available_dates,
            index=0,
            key="lab_slate_date",
        )
    with col_filt_right:
        # Filter contests to selected date
        contests_for_date = sorted(
            hist_df[hist_df["slate_date"] == lab_date]["contest_name"].unique()
        )
        lab_contest = st.selectbox(
            "Contest",
            options=["All"] + contests_for_date,
            index=0,
            key="lab_contest_name",
        )

    # Apply filters
    filtered_df = hist_df[hist_df["slate_date"] == lab_date].copy()
    if lab_contest != "All":
        filtered_df = filtered_df[filtered_df["contest_name"] == lab_contest].copy()

    st.markdown("---")

    # --- Panel A: Lineup Viewer + Actuals ---
    st.markdown("### 2. Historical Lineups & Actuals")

    lineup_ids = sorted(filtered_df["lineup_id"].unique())
    n_lineups = len(lineup_ids)

    if n_lineups == 0:
        st.warning("No lineups found for the selected filters.")
    else:
        # Lineup summary table
        lineup_summaries = []
        for lid in lineup_ids:
            lu = filtered_df[filtered_df["lineup_id"] == lid]
            lineup_summaries.append({
                "lineup_id": lid,
                "total_salary": int(lu["salary"].sum()),
                "total_actual_pts": lu["actual_points"].sum(),
                "avg_field_pct": lu["field_pct"].mean(),
                "players": len(lu),
            })

        summary_df = pd.DataFrame(lineup_summaries)
        st.dataframe(summary_df, use_container_width=True, height=200)

        # KPIs across all filtered lineups
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        col_kpi1.metric("Lineups", f"{n_lineups}")
        col_kpi2.metric("Best lineup pts", f"{summary_df['total_actual_pts'].max():.1f}")
        col_kpi3.metric("Avg lineup pts", f"{summary_df['total_actual_pts'].mean():.1f}")
        col_kpi4.metric("Avg field %", f"{summary_df['avg_field_pct'].mean():.1f}%")

        # Individual lineup viewer
        st.markdown("#### Lineup Detail")

        lab_lineup_idx = st.number_input(
            "Lineup #",
            min_value=1,
            max_value=n_lineups,
            value=1,
            step=1,
            key="lab_lineup_idx",
        )
        selected_lid = lineup_ids[lab_lineup_idx - 1]
        lu_detail = filtered_df[filtered_df["lineup_id"] == selected_lid].copy()

        total_sal = int(lu_detail["salary"].sum())
        total_pts = lu_detail["actual_points"].sum()
        avg_field = lu_detail["field_pct"].mean()

        st.markdown(
            f"**Lineup {lab_lineup_idx}** -- Salary: {total_sal:,} | "
            f"Actual pts: {total_pts:.1f} | Avg field: {avg_field:.1f}%"
        )

        detail_cols = ["pos", "player", "team", "salary", "field_pct", "actual_points"]
        # Add totals row
        totals_row = pd.DataFrame([{
            "pos": "TOTAL",
            "player": "",
            "team": "",
            "salary": lu_detail["salary"].sum(),
            "field_pct": lu_detail["field_pct"].mean(),
            "actual_points": lu_detail["actual_points"].sum(),
        }])
        lu_with_totals = pd.concat([lu_detail[detail_cols], totals_row], ignore_index=True)
        st.dataframe(
            lu_with_totals,
            use_container_width=True,
            height=320,
        )

    st.markdown("---")

    # --- Panel B: Player-level actuals across all filtered lineups ---
    st.markdown("### 3. Player Performance (Actuals)")

    if not filtered_df.empty:
        player_stats = (
            filtered_df.groupby(["player", "team", "salary"])
            .agg(
                appearances=("lineup_id", "nunique"),
                avg_actual_pts=("actual_points", "mean"),
                avg_field_pct=("field_pct", "mean"),
            )
            .reset_index()
            .sort_values("avg_actual_pts", ascending=False)
        )
        player_stats["pts_per_1k"] = (
            player_stats["avg_actual_pts"] / (player_stats["salary"] / 1000)
        ).round(2)

        st.dataframe(player_stats, use_container_width=True, height=350)

        # Value highlights
        st.markdown("#### Top Value Plays (pts per $1K salary)")
        top_value = player_stats.nlargest(5, "pts_per_1k")
        st.dataframe(top_value, use_container_width=True, height=200)

    st.markdown("---")

    # --- Panel C: Contest Sims (Placeholder) ---
    st.markdown("### 4. Contest Sims (Placeholder)")

    st.write(
        "This section will run game and contest simulations for the selected slate and contest type, "
        "and evaluate how different configs perform (ITM rate, ROI, etc.)."
    )

    if st.button("Run quick placeholder sim"):
        st.info(
            "Sim engine is not wired yet. Once past lineups are loaded, this will simulate outcomes "
            "and show distributions for different lineup archetypes."
        )

    st.markdown("---")

    # --- Panel D: Ricky Profile Editor ---
    st.markdown("### 5. Ricky Profile Editor (Placeholder)")

    st.write(
        "Adjust how Ricky balances projection, ceiling, ownership, and correlation for each contest type. "
        "Today these sliders are placeholders; later they will write to real config files."
    )

    lab_profile = st.selectbox(
        "Ricky Profile",
        ["Default", "Ricky_GPP", "Ricky_Cash", "Ricky_SE", "Ricky_Captain"],
        index=0,
        key="lab_profile",
    )

    col_prof_left, col_prof_right = st.columns(2)
    with col_prof_left:
        ceiling_weight = st.slider(
            "Ceiling weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )
        floor_weight = st.slider(
            "Floor weight",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )
    with col_prof_right:
        ownership_fade = st.slider(
            "Ownership fade strength",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        stack_priority = st.slider(
            "Stack priority",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )

    if st.button("Save Ricky Profile (placeholder)"):
        st.success(
            f"Saved profile '{lab_profile}' in memory "
            "(placeholder -- will later write to config and affect public lineups)."
        )

    st.caption("YakOS v0.9 | Ricky's Slate Room + Calibration Lab")
