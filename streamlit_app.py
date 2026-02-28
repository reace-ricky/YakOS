"""YakOS DFS Optimizer – RotoGrinders CSV Frontend."""

import sys
from typing import Dict, Any, Tuple

import pandas as pd
import streamlit as st

# Make yak_core importable if this file lives at repo root
if "yak_core" not in sys.modules:
    try:
        import yak_core  # noqa: F401
    except ImportError:
        pass

from yak_core.lineups import build_multiple_lineups_with_exposure


# -----------------------------
# Helpers
# -----------------------------
RG_RENAME_MAP = {
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
}


def load_and_clean_rg_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.rename(columns=RG_RENAME_MAP)

    # Drop rows missing salary or projection
    drop_subset = [c for c in ["salary", "proj"] if c in df.columns]
    if drop_subset:
        df = df.dropna(subset=drop_subset)

    # Ensure salary numeric
    if "salary" in df.columns:
        df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
        df = df.dropna(subset=["salary"])
        df["salary"] = df["salary"].astype(int)

    return df


def build_cfg(
    num_lineups: int,
    max_exposure: float,
    min_salary_used: int,
    proj_col: str,
    sport: str,
    slate_type: str,
    game_key: str | None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    cfg["NUM_LINEUPS"] = int(num_lineups)
    cfg["MAX_EXPOSURE"] = float(max_exposure)
    cfg["MIN_SALARY_USED"] = int(min_salary_used)
    cfg["PROJ_COL"] = proj_col
    cfg["SPORT"] = sport
    cfg["SLATE_TYPE"] = slate_type
    if game_key is not None:
        cfg["GAME_KEY"] = game_key
    return cfg


def run_optimizer(pool_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lineups_df, exposures_df = build_multiple_lineups_with_exposure(pool_df, cfg)
    return lineups_df, exposures_df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_game_keys(df: pd.DataFrame) -> pd.Series:
    """
    Build strings like 'CLE @ DET' from team/opponent columns.
    """
    if "team" not in df.columns or "opponent" not in df.columns:
        return pd.Series(dtype=str)

    game_keys = df["team"].astype(str).str.upper() + " @ " + df["opponent"].astype(str).str.upper()
    return game_keys


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="YakOS DFS Optimizer",
    layout="wide",
)

st.title("YakOS DFS Optimizer – RotoGrinders CSV")

# Sport selector (PGA placeholder for now)
sport = st.sidebar.selectbox("Sport", ["NBA", "PGA"])
if sport == "PGA":
    st.sidebar.info("PGA support is a placeholder for now. Optimizer logic is tuned for NBA.")

# Tabs: Optimizer + Calibration
tab_optimizer, tab_calibration = st.tabs(["Optimizer", "Calibration"])

with tab_optimizer:
    st.subheader("Optimizer")

    st.markdown("#### 1. Upload RotoGrinders Projections CSV")
    uploaded = st.file_uploader("RotoGrinders CSV", type=["csv"])

    if uploaded is not None:
        try:
            pool_df = load_and_clean_rg_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            pool_df = None
    else:
        pool_df = None

    # ---- Slate + Game selection ----
    slate_type = st.sidebar.selectbox(
        "Slate Type",
        ["Classic", "Showdown Captain"],
        index=0,
    )

    if pool_df is not None:
        st.success(f"Loaded pool with {len(pool_df)} players.")
        with st.expander("Show player pool (first 100 rows)"):
            st.dataframe(pool_df.head(100), use_container_width=True)

        # Build game keys from team/opponent
        game_keys_series = build_game_keys(pool_df)
        unique_games = sorted(game_keys_series.dropna().unique().tolist())

        game_key: str | None = None
        if slate_type == "Showdown Captain":
            if unique_games:
                game_key = st.sidebar.selectbox("Game (for Showdown)", unique_games)
                # Filter pool to chosen game
                mask = game_keys_series == game_key
                filtered_df = pool_df[mask].copy()
                if filtered_df.empty:
                    st.warning(f"No players found for game {game_key}.")
                else:
                    pool_df = filtered_df
                    st.info(f"Filtered to game {game_key}: {len(pool_df)} players.")
            else:
                st.sidebar.warning("No team/opponent info to derive games. Showdown filter disabled.")
    else:
        game_key = None

        st.markdown("#### 2. Optimizer Settings")

    with st.sidebar:
        st.markdown("### Optimizer Settings")

        num_lineups = st.number_input(
            "NUM_LINEUPS",
            min_value=1,
            max_value=300,
            value=5,
            step=1,
        )

        max_exposure = st.slider(
            "Max exposure per player",
            min_value=0.05,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="Cap on how often any one player can appear across all lineups.",
        )

        min_salary_used = st.number_input(
            "MIN_SALARY_USED",
            min_value=0,
            max_value=60000,
            value=46500,
            step=500,
        )

        proj_cols = [c for c in ["proj", "floor", "ceil", "sim85"] if c in (pool_df.columns if pool_df is not None else [])]
        if not proj_cols:
            proj_cols = ["proj"]
        proj_col = st.selectbox(
            "Projection style",
            proj_cols,
            index=0,
            help="Which projection column to optimize: proj=median, floor=safer, ceil/sim85=more aggressive.",
        )



    if pool_df is None:
        st.info("Upload a RotoGrinders CSV to enable the optimizer.")
    else:
        build_btn = st.button("Build Lineups", type="primary")

        if build_btn:
            with st.spinner("Building lineups..."):
                cfg = build_cfg(
                    num_lineups=num_lineups,
                    max_exposure=max_exposure,
                    min_salary_used=min_salary_used,
                    proj_col=proj_col,
                    sport=sport,
                    slate_type=slate_type,
                    game_key=game_key,
                )

                try:
                    lineups_df, exposures_df = run_optimizer(pool_df, cfg)
                    st.session_state["lineups_df"] = lineups_df
                    st.session_state["exposures_df"] = exposures_df
                except Exception as e:
                    st.error(f"Optimizer error: {e}")
                    st.session_state["lineups_df"] = None
                    st.session_state["exposures_df"] = None

    # Outside the button block, for display:
    lineups_df = st.session_state.get("lineups_df")
    exposures_df = st.session_state.get("exposures_df")

        if lineups_df is not None and exposures_df is not None:
                num_unique = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else len(lineups_df)
                st.success(f"Built {num_unique} lineups.")

                # Lineup viewer (one lineup at a time)
                st.markdown("### Lineup viewer")

                if "lineup_index" in lineups_df.columns:
                    lineup_ids = sorted(lineups_df["lineup_index"].unique().tolist())
                    selected_lu = st.number_input(
                        "Lineup #",
                        min_value=min(lineup_ids),
                        max_value=max(lineup_ids),
                        value=min(lineup_ids),
                        step=1,
                    )

                    lu = lineups_df[lineups_df["lineup_index"] == selected_lu].copy()
                else:
                    selected_lu = 0
                    lu = lineups_df.copy()
                    lu["lineup_index"] = 0

                if "slot" in lu.columns:
                    lu = lu.sort_values("slot")

                display_cols = [c for c in ["slot", "team", "player_name", "salary", "proj"] if c in lu.columns]
                st.dataframe(lu[display_cols], use_container_width=True)

                total_salary = lu["salary"].sum() if "salary" in lu.columns else None
                total_proj = lu["proj"].sum() if "proj" in lu.columns else None
                if total_salary is not None and total_proj is not None:
                    st.markdown(f"**Total salary:** {total_salary} &nbsp;&nbsp; **Total proj:** {total_proj:.2f}")

                # Exposures summary
                st.markdown("### Player Exposures")
                st.dataframe(exposures_df, use_container_width=True, height=400)

                # Downloads
                st.markdown("### Download Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="Download Lineups CSV",
                        data=to_csv_bytes(lineups_df),
                        file_name="yakos_lineups.csv",
                        mime="text/csv",
                    )
                with col2:
                    st.download_button(
                        label="Download Exposures CSV",
                        data=to_csv_bytes(exposures_df),
                        file_name="yakos_exposures.csv",
                        mime="text/csv",
                    )
            else:
                st.warning("No lineups returned. Check logs/config for details.")

with tab_calibration:
    st.subheader("Calibration (Coming Soon)")
    st.info(
        "This tab will show calibration curves, proj vs actual, and tuning tools "
        "once we wire it to your historical data."
    )
