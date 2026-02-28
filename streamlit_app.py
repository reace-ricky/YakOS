"""YakOS DFS Dashboard -- Streamlit v0 (read-only)."""
import os, sys, glob
import streamlit as st
import pandas as pd
import numpy as np

# --- path setup so yak_core is importable ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from yak_core.config import DEFAULT_CONFIG, YAKOS_ROOT
from yak_core.rg_loader import load_rg_projections, load_rg_contest, hit_rate
from yak_core.scoring import projection_pct, ownership_kpis, score_lineups, backtest_summary
from yak_core.projections import projection_quality_report

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="YakOS Dashboard",
    page_icon="\U0001F9AC",
    layout="wide",
)

st.title("\U0001F9AC YakOS DFS Dashboard")
st.caption("Phase 2 -- read-only dashboard | yak_core v0.2")

# ============================================================
# Sidebar: data source picker
# ============================================================
st.sidebar.header("Data Source")

DATA_DIR = os.path.join(YAKOS_ROOT, "data") if os.path.isdir(os.path.join(YAKOS_ROOT, "data")) else YAKOS_ROOT
DOWNLOADS = os.path.expanduser("~/Downloads")

# Find available RG projection files
rg_proj_files = sorted(glob.glob(os.path.join(DOWNLOADS, "projections_draftkings_*.csv")))
rg_contest_files = sorted(glob.glob(os.path.join(DOWNLOADS, "335*.csv")))

# Find available parquet pools
parquet_files = sorted(glob.glob(os.path.join(YAKOS_ROOT, "tank_opt_pool_*.parquet")))

data_mode = st.sidebar.radio("Mode", ["RG Projections", "Historical Parquet"], index=0)

# ============================================================
# Load data based on mode
# ============================================================

@st.cache_data
def load_rg_proj(path):
    return load_rg_projections(path)

@st.cache_data
def load_rg_contest_cached(path):
    return load_rg_contest(path)

@st.cache_data
def load_parquet(path):
    return pd.read_parquet(path)

pool_df = None
contest_df = None

if data_mode == "RG Projections":
    if rg_proj_files:
        chosen_proj = st.sidebar.selectbox(
            "Projection file",
            rg_proj_files,
            format_func=lambda x: os.path.basename(x),
        )
        pool_df = load_rg_proj(chosen_proj)
        st.sidebar.success(f"Loaded {len(pool_df)} players")
    else:
        st.sidebar.warning("No RG projection CSVs found in ~/Downloads")

    if rg_contest_files:
        chosen_contest = st.sidebar.selectbox(
            "Contest results file",
            ["(none)"] + rg_contest_files,
            format_func=lambda x: os.path.basename(x) if x != "(none)" else x,
        )
        if chosen_contest != "(none)":
            contest_df = load_rg_contest_cached(chosen_contest)
            st.sidebar.success(f"Contest: {len(contest_df)} players")
else:
    if parquet_files:
        chosen_pq = st.sidebar.selectbox(
            "Parquet pool",
            parquet_files,
            format_func=lambda x: os.path.basename(x),
        )
        pool_df = load_parquet(chosen_pq)
        st.sidebar.success(f"Loaded {len(pool_df)} players")
    else:
        st.sidebar.warning("No parquet files found in YakOS root")

# ============================================================
# TOP: KPI strip
# ============================================================
st.markdown("---")

if pool_df is not None:
    k1, k2, k3, k4, k5 = st.columns(5)
    proj_col = "proj" if "proj" in pool_df.columns else None
    salary_col = "salary" if "salary" in pool_df.columns else None

    k1.metric("Pool Size", f"{len(pool_df)} players")

    if proj_col:
        k2.metric("Avg Projection", f"{pool_df[proj_col].mean():.1f} FP")
        k3.metric("Max Projection", f"{pool_df[proj_col].max():.1f} FP")
    if salary_col:
        k4.metric("Avg Salary", f"${pool_df[salary_col].mean():,.0f}")
        k5.metric("Salary Range", f"${pool_df[salary_col].min():,.0f} - ${pool_df[salary_col].max():,.0f}")

    # Contest KPIs if loaded
    if contest_df is not None and "actual_fp" in contest_df.columns:
        st.markdown("#### Contest Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Actual FP", f"{contest_df['actual_fp'].mean():.1f}")
        c2.metric("Median Actual FP", f"{contest_df['actual_fp'].median():.1f}")
        if "ownership" in contest_df.columns:
            c3.metric("Avg Ownership", f"{contest_df['ownership'].mean():.1f}%")
            c4.metric("Max Ownership", f"{contest_df['ownership'].max():.1f}%")

        hr = hit_rate(contest_df, cash_line=30.0)
        if "pct_above_cash" in hr:
            st.info(f"Hit Rate (30+ FP cash line): **{hr['pct_above_cash']}%** of players cleared")
else:
    st.info("Select a data source in the sidebar to load player data.")

# ============================================================
# MIDDLE: Player Pool Table
# ============================================================
if pool_df is not None:
    st.markdown("---")
    st.subheader("Player Pool")

    # Position filter
    if "pos" in pool_df.columns:
        positions = ["All"] + sorted(pool_df["pos"].dropna().unique().tolist())
        sel_pos = st.selectbox("Filter by position", positions)
        display_df = pool_df if sel_pos == "All" else pool_df[pool_df["pos"] == sel_pos]
    else:
        display_df = pool_df

    # Sort options
    sort_col = st.selectbox(
        "Sort by",
        [c for c in display_df.columns if display_df[c].dtype in ["float64", "int64", "float32", "int32"]],
        index=0,
    )
    display_df = display_df.sort_values(sort_col, ascending=False)

    st.dataframe(
        display_df.head(50).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

# ============================================================
# BOTTOM: Calibration Chart (proj vs actual)
# ============================================================
if contest_df is not None and pool_df is not None:
    st.markdown("---")
    st.subheader("Calibration: Projected vs Actual")

    # Try to merge proj into contest data
    merged = None
    if "proj" in pool_df.columns and "actual_fp" in contest_df.columns:
        # Match by name
        if "name" in pool_df.columns and "name" in contest_df.columns:
            merged = contest_df.merge(
                pool_df[["name", "proj"]].drop_duplicates(subset="name"),
                on="name",
                how="inner",
            )

    if merged is not None and len(merged) > 0 and "proj" in merged.columns:
        # Scatter: proj vs actual
        chart_df = merged[["name", "proj", "actual_fp"]].dropna()
        if len(chart_df) > 0:
            col_chart, col_stats = st.columns([2, 1])

            with col_chart:
                st.scatter_chart(
                    chart_df.set_index("name")[["proj", "actual_fp"]],
                    height=400,
                )

            with col_stats:
                qr = projection_quality_report(chart_df)
                st.markdown("**Projection Quality**")
                for k, v in qr.items():
                    st.write(f"**{k}:** {v}")
        else:
            st.warning("No overlapping players between projection and contest data.")
    else:
        st.info("Load both a projection file and contest results to see calibration.")

elif contest_df is not None:
    st.markdown("---")
    st.subheader("Contest Actuals Distribution")
    if "actual_fp" in contest_df.columns:
        st.bar_chart(contest_df["actual_fp"].dropna().sort_values(ascending=False).reset_index(drop=True))

# ============================================================
# Exposure / Ownership distribution
# ============================================================
if pool_df is not None:
    own_col = None
    for c in ["ownership", "OWNERSHIP", "proj_own", "POWN"]:
        if c in pool_df.columns:
            own_col = c
            break
    if own_col is None and contest_df is not None:
        for c in ["ownership", "OWNERSHIP", "proj_own", "POWN"]:
            if c in contest_df.columns:
                own_col = c
                break

    src = contest_df if contest_df is not None and own_col and own_col in contest_df.columns else pool_df
    if own_col and own_col in src.columns:
        st.markdown("---")
        st.subheader("Ownership Distribution")
        own_data = src[["name", own_col]].dropna().sort_values(own_col, ascending=False).head(30)
        st.bar_chart(own_data.set_index("name")[own_col])

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("YakOS v0.2 | Phase 2 Streamlit Dashboard | Built by Right Angle Ricky")
