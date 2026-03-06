"""The Lab – YakOS unified workspace.

Responsibilities
----------------
- **Slate Loading** (formerly Slate Hub): Date + Sport picker, Contest Type,
  Fetch Available Slates, Load Player Pool, Game Filter, Publish Slate.
- **Simulations + Learnings**: Run sims, view player-level smash/bust/leverage,
  apply sim learnings — all in one section.
- **Edge Analysis**: Ownership edge, value plays, stacking labels.
- **Calibration**: Bucketed table, historical pipeline, rating weight tester,
  calibration profiles.
- **RCI**: Ricky Confidence Index gauges per contest type.
- **Contest-type Gauges**: Score-based gauges driven by sim smash probability.
- **Ricky Edge Check Gate**: Read-only gate status.

State read:  SlateState, RickyEdgeState, SimState
State written: SlateState, SimState
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import (  # noqa: E402
    get_slate_state, set_slate_state,
    get_edge_state, set_edge_state,
    get_sim_state, set_sim_state,
)
from yak_core.sims import (  # noqa: E402
    run_monte_carlo_for_lineups,
    build_sim_player_accuracy_table,
    compute_player_anomaly_table,
    run_sims_pipeline,
    run_calibration_pipeline,
    prepare_sims_table,
    ContestType,
    run_sims_for_contest_type,
    compute_sim_eligible,
    _INELIGIBLE_STATUSES,
)
from yak_core.edge import compute_edge_metrics  # noqa: E402
from yak_core.publishing import build_ricky_lineups, publish_edge_and_lineups  # noqa: E402
from yak_core.calibration import (  # noqa: E402
    load_calibration_config,
    save_calibration_config,
    compute_slate_kpis,
    DK_CONTEST_TYPES,
    DFS_ARCHETYPES,
)
from yak_core.config import (  # noqa: E402
    CONTEST_PRESETS,
    CONTEST_PRESET_LABELS,
    get_pool_size_range,
    merge_config,
    DK_POS_SLOTS,
    DK_LINEUP_SIZE,
    SALARY_CAP,
    DK_SHOWDOWN_SLOTS,
    DK_SHOWDOWN_LINEUP_SIZE,
    DK_CONTEST_MATCH_RULES,
    classify_draft_group,
    build_slate_options,
)
from yak_core.right_angle import (  # noqa: E402
    compute_stack_scores,
    compute_value_scores,
)
from yak_core.sim_rating import compare_rating_weights, get_weight_sets  # noqa: E402
from yak_core.context import get_lab_analysis  # noqa: E402
from yak_core.rci import (  # noqa: E402
    compute_rci,
    DEFAULT_WEIGHTS as RCI_DEFAULT_WEIGHTS,
)
from yak_core.dk_ingest import (  # noqa: E402
    fetch_dk_lobby_contests,
    fetch_dk_draftables,
    DK_GAME_TYPE_LABELS,
)
from yak_core.projections import (  # noqa: E402
    yakos_fp_projection,
    yakos_minutes_projection,
    apply_projections,
)
from yak_core.ownership import apply_ownership  # noqa: E402
from yak_core.rg_loader import load_rg_projections, merge_rg_with_pool  # noqa: E402
from yak_core.live import fetch_injury_updates, fetch_player_game_logs, fetch_betting_odds  # noqa: E402
from yak_core.salary_history import SalaryHistoryClient  # noqa: E402
from yak_core.dff_ingest import fetch_dff_pool  # noqa: E402


# ---------------------------------------------------------------------------
# Slate loading helpers (migrated from Slate Hub)
# ---------------------------------------------------------------------------

def _fetch_dk_draft_groups(sport: str = "NBA") -> list:
    """Fetch DraftGroup metadata from the LIVE DK lobby API."""
    import requests
    url = "https://www.draftkings.com/lobby/getcontests"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    resp = requests.get(url, params={"sport": sport.upper()}, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    draft_groups_raw = data.get("DraftGroups") or data.get("draftGroups") or []
    return draft_groups_raw


def _fetch_historical_draft_groups(date_str: str) -> list:
    """Fetch draft groups for a historical date via FantasyLabs → DK."""
    client = SalaryHistoryClient()
    fl_groups = client.get_draft_group_ids(date_str)
    if not fl_groups:
        return []
    dk_format = []
    for g in fl_groups:
        dk_format.append({
            "DraftGroupId": g.get("draft_group_id", 0),
            "GameCount": g.get("game_count", 0),
            "ContestStartTimeSuffix": g.get("suffix", g.get("display_name", "")),
            "GameTypeId": 81 if "showdown" in str(g.get("display_name", "")).lower() else 70,
            "GameStyle": "Showdown Captain Mode" if "showdown" in str(g.get("display_name", "")).lower() else "Classic",
            "StartDate": g.get("start_time", ""),
            "SortOrder": g.get("sort_order", 99),
        })
    return dk_format


_SHOWDOWN_GAME_TYPE_IDS = {
    gid for gid, label in DK_GAME_TYPE_LABELS.items() if "Showdown" in label
}
_PLAYER_IDENTITY_COLS = ["player_name", "team", "pos", "salary"]
_NUMERIC_AGG_COLS = ["proj", "floor", "ceil", "proj_minutes", "ownership"]


def _rules_from_preset(preset: dict) -> dict:
    is_showdown = preset.get("slate_type") == "Showdown Captain"
    return {
        "slots": DK_SHOWDOWN_SLOTS if is_showdown else DK_POS_SLOTS,
        "lineup_size": DK_SHOWDOWN_LINEUP_SIZE if is_showdown else DK_LINEUP_SIZE,
        "salary_cap": SALARY_CAP,
        "is_showdown": is_showdown,
    }


def _normalize_dk_pool(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    if "name" in pool.columns and "player_name" not in pool.columns:
        pool = pool.rename(columns={"name": "player_name"})
    if "positions" in pool.columns and "pos" not in pool.columns:
        pool = pool.rename(columns={"positions": "pos"})
    if "display_name" in pool.columns and "player_name" not in pool.columns:
        pool = pool.rename(columns={"display_name": "player_name"})
    return pool


def _enrich_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Add floor/ceil/proj_minutes/ownership from YakOS per-player models.

    Performance-optimised: vectorised operations instead of row-by-row
    iteration, and salary-rank ownership (instant) instead of field-sim
    (1000 PuLP solves).  Field sim ownership can be triggered separately
    from the Sims section if the user wants it.
    """
    has_floor = "floor" in pool.columns and pool["floor"].notna().any()
    has_ceil = "ceil" in pool.columns and pool["ceil"].notna().any()

    sal = pd.to_numeric(pool.get("salary", 0), errors="coerce").fillna(0).astype(float)

    # ── Vectorised FP projection (floor/ceil) ────────────────────────────
    # Try to load trained model ONCE (not per-row)
    _fp_model = None
    _fp_features = None
    try:
        import os, joblib
        from yak_core.config import YAKOS_ROOT
        _fp_path = os.path.join(YAKOS_ROOT, "models", "yakos_fp_model.pkl")
        if os.path.isfile(_fp_path):
            _fp_model = joblib.load(_fp_path)
            _fp_features = list(getattr(_fp_model, "feature_names_in_", []))
    except Exception:
        pass

    if _fp_model is not None and _fp_features:
        # Batch predict with trained model
        feat_df = pd.DataFrame({col: pd.to_numeric(pool.get(col, np.nan), errors="coerce") for col in _fp_features})
        pred = pd.Series(_fp_model.predict(feat_df), index=pool.index).clip(lower=0)

        # Blend with rolling signals (vectorised)
        _rolling_keys = [("rolling_fp_5", 0.30), ("rolling_fp_10", 0.20), ("rolling_fp_20", 0.10)]
        rolling_weighted = pd.Series(0.0, index=pool.index)
        rolling_w_sum = pd.Series(0.0, index=pool.index)
        for key, w in _rolling_keys:
            if key in pool.columns:
                vals = pd.to_numeric(pool[key], errors="coerce")
                mask = vals.notna()
                rolling_weighted = rolling_weighted + vals.fillna(0) * w * mask.astype(float)
                rolling_w_sum = rolling_w_sum + w * mask.astype(float)
        has_rolling = rolling_w_sum > 0
        rolling_signal = (rolling_weighted / rolling_w_sum.replace(0, 1))
        proj_fp = pred.copy()
        proj_fp[has_rolling] = pred[has_rolling] * 0.4 + rolling_signal[has_rolling] * 0.6
        proj_fp = proj_fp.clip(lower=0)
    else:
        # Formula fallback (vectorised)
        _FP_PER_K = 4.0
        sal_base = sal * _FP_PER_K / 1000.0
        _signal_cols = [("rolling_fp_5", 0.30), ("rolling_fp_10", 0.20),
                        ("rolling_fp_20", 0.10), ("tank01_proj", 0.20), ("rg_proj", 0.15)]
        sig_weighted = pd.Series(0.0, index=pool.index)
        sig_w_sum = pd.Series(0.0, index=pool.index)
        for key, w in _signal_cols:
            if key in pool.columns:
                vals = pd.to_numeric(pool[key], errors="coerce")
                mask = vals.notna()
                sig_weighted = sig_weighted + vals.fillna(0) * w * mask.astype(float)
                sig_w_sum = sig_w_sum + w * mask.astype(float)
        has_sig = sig_w_sum > 0
        signal_proj = sig_weighted / sig_w_sum.replace(0, 1)
        proj_fp = sal_base.copy()
        proj_fp[has_sig] = signal_proj[has_sig] * 0.70 + sal_base[has_sig] * 0.30
        proj_fp = proj_fp.clip(lower=0)

    if not has_floor:
        pool["floor"] = (proj_fp * 0.65).round(2)
    if not has_ceil:
        pool["ceil"] = (proj_fp * 1.45).round(2)

    # ── Vectorised minutes projection ────────────────────────────────────
    _min_model = None
    _min_features = None
    try:
        import os, joblib
        from yak_core.config import YAKOS_ROOT
        _min_path = os.path.join(YAKOS_ROOT, "models", "yakos_minutes_model.pkl")
        if os.path.isfile(_min_path):
            _min_model = joblib.load(_min_path)
            _min_features = list(getattr(_min_model, "feature_names_in_", []))
    except Exception:
        pass

    if _min_model is not None and _min_features:
        feat_df = pd.DataFrame({col: pd.to_numeric(pool.get(col, np.nan), errors="coerce") for col in _min_features})
        proj_min = pd.Series(_min_model.predict(feat_df), index=pool.index)
    else:
        # Formula: weighted rolling minutes or salary fallback
        _min_keys = [("rolling_min_5", 0.50), ("rolling_min_10", 0.30), ("rolling_min_20", 0.20)]
        min_weighted = pd.Series(0.0, index=pool.index)
        min_w_sum = pd.Series(0.0, index=pool.index)
        for key, w in _min_keys:
            if key in pool.columns:
                vals = pd.to_numeric(pool[key], errors="coerce")
                mask = vals.notna()
                min_weighted = min_weighted + vals.fillna(0) * w * mask.astype(float)
                min_w_sum = min_w_sum + w * mask.astype(float)
        has_min_sig = min_w_sum > 0
        sal_min_fallback = (sal / 300.0).clip(lower=10.0, upper=36.0)
        proj_min = sal_min_fallback.copy()
        proj_min[has_min_sig] = min_weighted[has_min_sig] / min_w_sum[has_min_sig].replace(0, 1)

        # Players with NO game-log signal AND bottom-tier salary are likely
        # deep bench / DNP-CD risks.  Cap their projected minutes at 4 so
        # compute_sim_eligible (min_proj_minutes=4) can filter them out.
        _MIN_SALARY_FOR_FALLBACK = 4000
        deep_bench_mask = (~has_min_sig) & (sal < _MIN_SALARY_FOR_FALLBACK)
        proj_min[deep_bench_mask] = proj_min[deep_bench_mask].clip(upper=3.5)

    # Contextual adjustments
    if "b2b" in pool.columns:
        b2b_mask = pool["b2b"].fillna(False).astype(bool)
        proj_min[b2b_mask] *= 0.93
    if "spread" in pool.columns:
        abs_spread = pd.to_numeric(pool["spread"], errors="coerce").fillna(0).abs()
        proj_min[abs_spread >= 15] *= 0.90
        proj_min[(abs_spread >= 10) & (abs_spread < 15)] *= 0.95

    pool["proj_minutes"] = proj_min.clip(lower=0).round(1)

    if "status" in pool.columns:
        inelig_mask = (
            pool["status"].fillna("").astype(str).str.strip().str.upper()
            .isin(_INELIGIBLE_STATUSES)
        )
        pool.loc[inelig_mask, "proj_minutes"] = 0.0

    # ── Ownership: use fast salary-rank (instant) instead of field sim ───
    # Field sim (1000 PuLP solves) can take 10+ minutes and is the primary
    # cause of the hang.  Salary-rank ownership is a good initial estimate;
    # users can run field sim separately from the Sims section.
    pool = apply_ownership(pool, use_field_sim=False)
    return compute_sim_eligible(pool)


def _filter_ineligible_players(pool: pd.DataFrame) -> pd.DataFrame:
    df = pool.copy()
    if "status" in df.columns:
        inelig_mask = (
            df["status"].fillna("").astype(str).str.strip().str.upper()
            .isin(_INELIGIBLE_STATUSES)
        )
        df = df[~inelig_mask]
    mins_col = "proj_minutes" if "proj_minutes" in df.columns else (
        "minutes" if "minutes" in df.columns else None
    )
    if mins_col is not None:
        mins = pd.to_numeric(df[mins_col], errors="coerce").fillna(0)
        df = df[mins > 0]
    return df.reset_index(drop=True)


def _extract_games(pool: pd.DataFrame) -> list[str]:
    opp_col = "opp" if "opp" in pool.columns else (
        "opponent" if "opponent" in pool.columns else None
    )
    if opp_col and "team" in pool.columns:
        teams = pool["team"].str.strip().str.upper().fillna("")
        opps = pool[opp_col].str.strip().str.upper().fillna("")
        pairs = {
            " vs ".join(sorted([t, o]))
            for t, o in zip(teams, opps)
            if t and o
        }
        return sorted(pairs)
    elif "team" in pool.columns:
        return sorted(pool["team"].dropna().str.strip().str.upper().unique().tolist())
    return []


def _filter_pool_by_games(pool: pd.DataFrame, selected_games: list[str], opp_col: str) -> pd.DataFrame:
    if not selected_games:
        return pool
    teams = pool["team"].str.strip().str.upper().fillna("")
    opps = pool[opp_col].str.strip().str.upper().fillna("")
    keys = pd.Series(
        [" vs ".join(sorted([t, o])) if t and o else t for t, o in zip(teams, opps)],
        index=pool.index,
    )
    return pool[keys.isin(selected_games)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sim helpers
# ---------------------------------------------------------------------------

_CONTEST_GAUGE_LABELS = ["Cash", "SE", "3-Max", "20-Max", "150-Max"]
_LAYER_ALL = ["Base", "Calibration", "Edge", "Sims"]


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


def _make_real_lineups_df(pool: pd.DataFrame, n_lineups: int = 5) -> pd.DataFrame:
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    rows = []
    available = pool.dropna(subset=["player_name"]).head(n_lineups * 8)
    for lu_idx in range(min(n_lineups, max(1, len(available) // 8))):
        chunk = available.iloc[lu_idx * 8: (lu_idx + 1) * 8]
        for i, (_, prow) in enumerate(chunk.iterrows()):
            rows.append({
                "lineup_index": lu_idx,
                "slot": slots[i % len(slots)],
                "player_name": prow.get("player_name", ""),
                "team": prow.get("team", ""),
                "pos": prow.get("pos", ""),
                "salary": prow.get("salary", 0),
                "proj": prow.get("proj", 0),
                "ownership": prow.get("ownership", 0),
            })
    return pd.DataFrame(rows)


def _build_player_level_sim_results(pool: pd.DataFrame, variance: float) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame()
    df = pool.copy()

    # Only include sim-eligible players (filters OUT/IR/0-min players)
    if "sim_eligible" in df.columns:
        df = df[df["sim_eligible"].astype(bool)].reset_index(drop=True)
    # For historical slates with actual minutes data, drop 0-minute players
    if "mp_actual" in df.columns:
        mp = pd.to_numeric(df["mp_actual"], errors="coerce").fillna(0)
        df = df[mp > 0].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    proj = pd.to_numeric(df.get("proj", 0), errors="coerce").fillna(0)
    ceil = pd.to_numeric(df.get("ceil", proj * 1.4), errors="coerce").fillna(proj * 1.4)
    floor = pd.to_numeric(df.get("floor", proj * 0.7), errors="coerce").fillna(proj * 0.7)
    own = pd.to_numeric(df.get("ownership", 5.0), errors="coerce").fillna(5.0)

    std = (ceil - floor) / 4 * variance
    std = std.clip(lower=0.5)
    smash_z = (ceil * 0.9 - proj) / std
    bust_z = (floor * 1.1 - proj) / std

    from scipy.stats import norm  # type: ignore
    smash_prob = 1 - norm.cdf(smash_z)
    bust_prob = norm.cdf(bust_z)

    own_frac = (own / 100.0).clip(lower=0.01)
    leverage = smash_prob / own_frac

    result = pd.DataFrame({
        "player_name": df.get("player_name", pd.Series(dtype=str)),
        "pos": df.get("pos", pd.Series(dtype=str)),
        "team": df.get("team", pd.Series(dtype=str)),
        "salary": df.get("salary", pd.Series(dtype=float)),
        "proj": proj,
        "floor": floor,
        "ceil": ceil,
        "ownership": own,
        "smash_prob": smash_prob.round(2),
        "bust_prob": bust_prob.round(2),
        "leverage": leverage.round(2),
    })
    return result.sort_values("leverage", ascending=False).reset_index(drop=True)


def _gauge_score(sim_results: Optional[pd.DataFrame], contest: str) -> float:
    if sim_results is None or sim_results.empty:
        return 0.0
    _TOP_N = 8
    if contest == "Cash":
        if "bust_prob" not in sim_results.columns:
            return 0.0
        bust = pd.to_numeric(sim_results["bust_prob"], errors="coerce").dropna()
        if bust.empty:
            return 0.0
        top_n_bust = bust.nsmallest(_TOP_N).mean()
        return float(np.clip(1.0 - top_n_bust, 0.0, 1.0))
    if "smash_prob" not in sim_results.columns:
        return 0.0
    smash = pd.to_numeric(sim_results["smash_prob"], errors="coerce").dropna()
    if smash.empty:
        return 0.0
    top_n_smash = smash.nlargest(_TOP_N).mean()
    weights = {"SE": 0.7, "3-Max": 0.85, "20-Max": 1.0, "150-Max": 1.2}
    w = weights.get(contest, 1.0)
    return float(np.clip(top_n_smash * w, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🧪 The Lab")
    st.caption("Load slate, run sims, analyze edge, calibrate projections.")

    slate = get_slate_state()
    edge = get_edge_state()
    sim = get_sim_state()

    # =====================================================================
    # SECTION 1: SLATE LOADING (formerly Slate Hub)
    # =====================================================================
    st.subheader("📥 Load Slate")

    # ── Row 1: Sport + Date ──────────────────────────────────────────────
    col_sport, col_date = st.columns(2)
    with col_sport:
        sport = st.selectbox("Sport", ["NBA", "PGA"], index=0 if slate.sport == "NBA" else 1)
    with col_date:
        from zoneinfo import ZoneInfo
        _today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        slate_date = st.date_input("Date", value=pd.to_datetime(_today))
        slate_date_str = str(slate_date)

    # Read Tank01 RapidAPI key from secrets
    rapidapi_key = st.secrets.get("TANK01_RAPIDAPI_KEY")
    if rapidapi_key:
        st.session_state["rapidapi_key"] = rapidapi_key

    # ── Row 2: Contest Type ──────────────────────────────────────────────
    contest_type_label = st.selectbox("Contest Type", CONTEST_PRESET_LABELS)
    preset = CONTEST_PRESETS[contest_type_label]
    st.caption(preset.get("description", ""))

    # Cache invalidation on date/sport/contest change
    _contest_safe = contest_type_label.lower().replace(" ", "_").replace("/", "-").replace("-", "_")
    _prev_date = st.session_state.get("_hub_prev_date")
    _prev_sport = st.session_state.get("_hub_prev_sport")
    _prev_contest = st.session_state.get("_hub_prev_contest")
    _date_changed = _prev_date is not None and _prev_date != slate_date_str
    _sport_changed = _prev_sport is not None and _prev_sport != sport
    _contest_changed = _prev_contest is not None and _prev_contest != contest_type_label
    if _date_changed or _sport_changed or _contest_changed:
        _stale_date = _prev_date if _prev_date is not None else slate_date_str
        _stale_sport = _prev_sport if _prev_sport is not None else sport
        _stale_contest = _prev_contest if _prev_contest is not None else contest_type_label
        _stale_contest_safe = (
            _stale_contest.lower().replace(" ", "_").replace("/", "-").replace("-", "_")
        )
        for key in [
            f"_hub_pool_{_stale_date}_{_stale_contest_safe}",
            f"_hub_rules_{_stale_date}_{_stale_contest_safe}",
            f"_hub_draft_group_id_{_stale_date}_{_stale_contest_safe}",
        ]:
            st.session_state.pop(key, None)
        old_slate_key = f"_hub_slates_{_stale_sport}_{_stale_date}"
        st.session_state.pop(old_slate_key, None)
    st.session_state["_hub_prev_date"] = slate_date_str
    st.session_state["_hub_prev_sport"] = sport
    st.session_state["_hub_prev_contest"] = contest_type_label

    proj_source = "model"

    # ── Slate Picker ─────────────────────────────────────────────────────
    from zoneinfo import ZoneInfo as _ZI2
    _today_date = pd.Timestamp.now(tz=_ZI2("America/New_York")).date()
    _is_historical = pd.to_datetime(slate_date_str).date() < _today_date
    _salary_client = SalaryHistoryClient()

    with st.expander(
        "📅 Historical slates" if _is_historical else "🟢 Live slates",
        expanded=False,
    ):
        if _is_historical:
            st.caption(f"Historical mode — slates fetched from FantasyLabs for {slate_date_str}")
        else:
            st.caption("Live mode — slates from DraftKings lobby (FantasyLabs fallback)")

        _slate_cache_key = f"_hub_slates_{sport}_{slate_date_str}"
        _cached_slates = st.session_state.get(_slate_cache_key)

        col_fetch_slate, col_clear_slate = st.columns([2, 1])
        with col_fetch_slate:
            if st.button("🔍 Fetch Available Slates", type="secondary"):
                with st.spinner("Fetching slates…"):
                    try:
                        if _is_historical:
                            raw_dgs = _fetch_historical_draft_groups(slate_date_str)
                            _source = "FantasyLabs"
                        else:
                            try:
                                raw_dgs = _fetch_dk_draft_groups(sport)
                                _source = "DraftKings"
                            except Exception:
                                raw_dgs = []
                            if not raw_dgs:
                                raw_dgs = _fetch_historical_draft_groups(slate_date_str)
                                _source = "FantasyLabs"
                        if not raw_dgs:
                            st.warning(f"No slates found on {_source} for {slate_date_str}. Try a different date.")
                        else:
                            slate_options = build_slate_options(raw_dgs)
                            st.session_state[_slate_cache_key] = slate_options
                            _cached_slates = slate_options
                            st.success(f"Found {len(slate_options)} slate(s) from {_source}.")
                    except Exception as exc:
                        st.error(f"Failed to fetch slates: {exc}")
        with col_clear_slate:
            if _cached_slates and st.button("Clear"):
                st.session_state.pop(_slate_cache_key, None)
                _cached_slates = None
                st.rerun()

        selected_dg_id: Optional[int] = None
        selected_slate_label: Optional[str] = None

        if _cached_slates:
            slate_labels = [s["label"] for s in _cached_slates]
            selected_idx = st.radio(
                "Choose a slate",
                range(len(slate_labels)),
                format_func=lambda i: slate_labels[i],
                key="_hub_slate_radio",
            )
            if selected_idx is not None:
                selected_slate = _cached_slates[selected_idx]
                selected_dg_id = selected_slate["draft_group_id"]
                selected_slate_label = selected_slate["label"]
                st.caption(
                    f"Draft Group **{selected_dg_id}** · "
                    f"{selected_slate['game_count']} game(s) · "
                    f"{selected_slate['game_style']}"
                )

        # Manual DG override
        manual_dg = st.number_input(
            "Manual Draft Group ID",
            min_value=0, step=1, value=0,
            help="Paste a DraftKings Draft Group ID if you know it. Overrides the slate picker.",
            key="_hub_manual_dg",
        )
        if manual_dg > 0:
            selected_dg_id = int(manual_dg)
            selected_slate_label = f"Manual (DG {manual_dg})"

    # Store selected_dg_id in session state so it persists after expander closes
    if selected_dg_id:
        st.session_state["_lab_selected_dg_id"] = selected_dg_id
    else:
        selected_dg_id = st.session_state.get("_lab_selected_dg_id")

    # ── Load Player Pool Button ──────────────────────────────────────────
    if st.button("📥 Load Player Pool", type="primary"):
        draft_group_id: Optional[int] = selected_dg_id

        with st.spinner("Loading player pool…"):
            try:
                _historical_salary_df: Optional[pd.DataFrame] = None
                _historical_dg_id: Optional[int] = None

                if _is_historical and not draft_group_id:
                    _cached = _salary_client.load_cached_salaries(slate_date_str)
                    if _cached is not None and not _cached.empty:
                        _historical_salary_df = _cached
                        st.info(f"Historical salaries loaded from cache for {slate_date_str}.")
                    else:
                        with st.spinner("Fetching historical salaries from DK…"):
                            _hist_df = _salary_client.get_historical_salaries(slate_date_str)
                        if not _hist_df.empty:
                            _historical_salary_df = _hist_df
                            _historical_dg_id = _hist_df.attrs.get("draft_group_id")
                            if _historical_dg_id:
                                st.info(f"Historical salaries loaded from DK (DraftGroup {_historical_dg_id})")
                        else:
                            st.caption("ℹ️ DK API unavailable — trying DailyFantasyFuel…")
                            _dff_pool = fetch_dff_pool(sport)
                            if not _dff_pool.empty:
                                _historical_salary_df = _dff_pool
                                st.info(f"✅ Player pool loaded from DailyFantasyFuel ({len(_dff_pool)} players)")
                elif _is_historical and draft_group_id:
                    with st.spinner(f"Fetching historical players for DraftGroup {draft_group_id}…"):
                        _hist_dg_df = _salary_client.get_draftables(draft_group_id)
                    if not _hist_dg_df.empty:
                        _historical_salary_df = _hist_dg_df
                        _historical_dg_id = draft_group_id
                        st.info(f"Historical salaries loaded from DK (DraftGroup {draft_group_id}, {slate_date_str})")
                    else:
                        st.caption("ℹ️ DK API unavailable — trying DailyFantasyFuel…")
                        _dff_pool = fetch_dff_pool(sport)
                        if not _dff_pool.empty:
                            _historical_salary_df = _dff_pool
                            st.info(f"✅ Player pool loaded from DailyFantasyFuel ({len(_dff_pool)} players)")
                        else:
                            st.warning(
                                f"No players found for DraftGroup {draft_group_id} on {slate_date_str}. "
                                "Both DK API and DailyFantasyFuel returned empty."
                            )

                if not draft_group_id and _historical_salary_df is None and not _is_historical:
                    pass
                elif _is_historical and not draft_group_id and _historical_salary_df is None:
                    st.warning(
                        'No slate selected. Use "Fetch Available Slates" to pick a slate, '
                        "or enter a Draft Group ID manually."
                    )
                elif _is_historical and draft_group_id and _historical_salary_df is None:
                    pass

                # ── Resolve the player pool ──────────────────────────────
                pool = None
                _pool_source = "DK"
                if _historical_salary_df is not None and not _historical_salary_df.empty:
                    pool = _historical_salary_df.copy()
                    if "position" in pool.columns and "pos" not in pool.columns:
                        pool = pool.rename(columns={"position": "pos"})
                    if "player_dk_id" in pool.columns and "dk_player_id" not in pool.columns:
                        pool = pool.rename(columns={"player_dk_id": "dk_player_id"})
                    if _historical_dg_id and not draft_group_id:
                        draft_group_id = _historical_dg_id
                    if "dff_proj" in pool.columns:
                        _pool_source = "DailyFantasyFuel"
                elif draft_group_id:
                    try:
                        pool = fetch_dk_draftables(draft_group_id)
                    except Exception:
                        pool = pd.DataFrame()
                    if pool.empty:
                        st.caption("ℹ️ DK API unavailable — loading from DailyFantasyFuel…")
                        pool = fetch_dff_pool(sport)
                        _pool_source = "DailyFantasyFuel"
                    if pool.empty:
                        st.error(
                            f"No players found for Draft Group ID {draft_group_id}. "
                            "Both DK API and DailyFantasyFuel returned empty."
                        )
                        st.stop()
                else:
                    st.caption("ℹ️ No draft group selected — loading from DailyFantasyFuel…")
                    pool = fetch_dff_pool(sport)
                    _pool_source = "DailyFantasyFuel"
                    if pool.empty:
                        st.error("No players found. DailyFantasyFuel returned empty.")
                        st.stop()

                if pool is not None and not pool.empty:
                    pool = _normalize_dk_pool(pool)
                    if "salary" not in pool.columns:
                        pool["salary"] = 0
                    pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce").fillna(0)

                    # Auto-save live salary data
                    if not _is_historical and not pool.empty:
                        _cache_col_map = {"pos": "position", "dk_player_id": "player_dk_id"}
                        _save_cols = [
                            c for c in ("player_name", "pos", "team", "salary", "dk_player_id")
                            if c in pool.columns
                        ]
                        if _save_cols:
                            _salary_client.save_salaries(
                                slate_date_str,
                                pool[_save_cols].rename(columns=_cache_col_map),
                            )

                    # Tank01 enrichment
                    _api_key = st.session_state.get("rapidapi_key", "")
                    if _api_key:
                        try:
                            _t01_id_map: dict = {}
                            for _id_col in ("player_id", "tank01_player_id", "t01_id"):
                                if _id_col in pool.columns:
                                    _t01_id_map = dict(
                                        zip(pool["player_name"].astype(str), pool[_id_col].astype(str))
                                    )
                                    break
                            _player_names = pool["player_name"].dropna().tolist()
                            with st.spinner("Fetching game log rolling stats from Tank01…"):
                                _game_log_df = fetch_player_game_logs(
                                    _player_names,
                                    _t01_id_map if _t01_id_map else None,
                                    _api_key,
                                )
                            if not _game_log_df.empty:
                                pool = pool.merge(_game_log_df, on="player_name", how="left")
                                st.caption(f"✅ Rolling stats merged for {_game_log_df['player_name'].nunique()} players.")
                            else:
                                st.caption("ℹ️ No game log rolling stats returned from Tank01.")

                            with st.spinner("Fetching Vegas odds from Tank01…"):
                                _odds_df = fetch_betting_odds(slate_date_str, _api_key)
                            if not _odds_df.empty:
                                _team_odds_rows = []
                                for _, _o in _odds_df.iterrows():
                                    _total = _o["vegas_total"]
                                    _spread = _o["spread"]
                                    if _o["home_team"]:
                                        _team_odds_rows.append({
                                            "team": _o["home_team"],
                                            "vegas_total": _total,
                                            "spread": _spread,
                                        })
                                    if _o["away_team"]:
                                        import math
                                        _away_spread = (
                                            -_spread
                                            if not (isinstance(_spread, float) and math.isnan(_spread))
                                            else float("nan")
                                        )
                                        _team_odds_rows.append({
                                            "team": _o["away_team"],
                                            "vegas_total": _total,
                                            "spread": _away_spread,
                                        })
                                if _team_odds_rows:
                                    _team_odds_df = pd.DataFrame(_team_odds_rows).drop_duplicates("team")
                                    for _vc in ("vegas_total", "spread"):
                                        if _vc in pool.columns:
                                            pool = pool.drop(columns=[_vc])
                                    pool = pool.merge(_team_odds_df, on="team", how="left")
                                    st.caption(f"✅ Vegas odds merged for {_team_odds_df['team'].nunique()} teams.")
                            else:
                                st.caption("ℹ️ No Vegas odds returned from Tank01.")
                        except Exception as t01_exc:
                            st.caption(f"ℹ️ Tank01 enrichment not available: {t01_exc}")

                    # Apply projection pipeline
                    parsed_rules = _rules_from_preset(preset)
                    cfg = merge_config({
                        "PROJ_SOURCE": proj_source,
                        "SLATE_DATE": slate_date_str,
                        "CONTEST_TYPE": preset["internal_contest"],
                    })
                    pool = apply_projections(pool, cfg)

                    # Apply calibration corrections from historical feedback
                    try:
                        from yak_core.calibration_feedback import get_correction_factors, apply_corrections
                        if "_cal_fb_store" not in st.session_state:
                            st.session_state["_cal_fb_store"] = {}
                        _cf = get_correction_factors(store=st.session_state["_cal_fb_store"])
                        if _cf.get("n_slates", 0) > 0:
                            pool = apply_corrections(pool, _cf, store=st.session_state["_cal_fb_store"])
                            st.caption(f"📐 Calibration corrections applied ({_cf['n_slates']} slate(s) of history)")
                    except Exception as _cf_exc:
                        pass  # silently skip if feedback module not ready

                    pool = _enrich_pool(pool)

                    # Dedup
                    _group_cols = [c for c in _PLAYER_IDENTITY_COLS if c in pool.columns]
                    _agg_cols = {c: "mean" for c in _NUMERIC_AGG_COLS if c in pool.columns}
                    if _group_cols and _agg_cols:
                        _extra_cols = [c for c in pool.columns if c not in _group_cols and c not in _agg_cols]
                        _extra_agg = {c: "first" for c in _extra_cols}
                        pool = pool.groupby(_group_cols, as_index=False).agg({**_agg_cols, **_extra_agg})
                    else:
                        if "dk_player_id" in pool.columns:
                            dedup_key = "dk_player_id"
                        else:
                            dedup_key = "player_name"
                        if "proj" in pool.columns:
                            pool = pool.sort_values("proj", ascending=False)
                        pool = pool.drop_duplicates(subset=[dedup_key], keep="first")
                    pool = pool.reset_index(drop=True)

                    # Filter ineligible
                    _before = len(pool)
                    pool = _filter_ineligible_players(pool)
                    _removed = _before - len(pool)
                    if _removed:
                        st.caption(f"ℹ️ {_removed} player(s) removed (OUT/DND/IR or 0 proj minutes).")

                    # For HISTORICAL slates, cross-reference box scores to drop
                    # players who got 0 actual minutes (DNP-CD, inactive, etc.)
                    if _is_historical and _api_key:
                        try:
                            from yak_core.live import fetch_actuals_from_api
                            _box_date = slate_date_str.replace("-", "")
                            _actuals = fetch_actuals_from_api(_box_date, {"RAPIDAPI_KEY": _api_key})
                            if not _actuals.empty and "actual_fp" in _actuals.columns:
                                # Players who appeared in box scores with > 0 FP
                                _played = set(_actuals[_actuals["actual_fp"] > 0]["player_name"].values)
                                if _played:
                                    # Merge actual_fp into pool before filtering
                                    _act_map = _actuals.set_index("player_name")["actual_fp"].to_dict()
                                    pool["actual_fp"] = pool["player_name"].map(_act_map)

                                    _before_box = len(pool)
                                    pool = pool[
                                        pool["player_name"].isin(_played)
                                    ].reset_index(drop=True)
                                    _dnp_removed = _before_box - len(pool)
                                    if _dnp_removed:
                                        st.caption(f"ℹ️ {_dnp_removed} DNP player(s) removed via box score cross-ref.")

                                    # Auto-record projection errors for calibration feedback
                                    if "proj" in pool.columns and "actual_fp" in pool.columns:
                                        try:
                                            from yak_core.calibration_feedback import record_slate_errors
                                            if "_cal_fb_store" not in st.session_state:
                                                st.session_state["_cal_fb_store"] = {}
                                            _fb_result = record_slate_errors(slate_date_str, pool, store=st.session_state["_cal_fb_store"])
                                            if "error" not in _fb_result:
                                                _fb_mae = _fb_result.get("overall", {}).get("mae", "?")
                                                _fb_corr = _fb_result.get("overall", {}).get("correlation", "?")
                                                st.caption(f"📐 Calibration feedback recorded (MAE: {_fb_mae}, Corr: {_fb_corr})")
                                        except Exception:
                                            pass
                        except Exception as _box_exc:
                            st.caption(f"ℹ️ Box score filter skipped: {_box_exc}")

                    # Store in session state
                    st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"] = pool
                    st.session_state[f"_hub_rules_{slate_date_str}_{_contest_safe}"] = parsed_rules
                    st.session_state[f"_hub_draft_group_id_{slate_date_str}_{_contest_safe}"] = draft_group_id
                    _salary_mode = "Historical" if _is_historical else "Live"
                    _source_label = f" (via {_pool_source})" if _pool_source != "DK" else ""
                    st.success(f"✅ Loaded {len(pool)} players — {_salary_mode}{_source_label}")

                    # Auto-publish the slate
                    _ts = datetime.now(timezone.utc).isoformat()
                    slate.sport = sport
                    slate.site = "DK"
                    slate.slate_date = slate_date_str
                    slate.proj_source = proj_source
                    slate.contest_name = contest_type_label
                    slate.draft_group_id = draft_group_id
                    if parsed_rules:
                        slate.apply_roster_rules(parsed_rules)
                    slate.contest_type = contest_type_label
                    slate.player_pool = pool
                    slate.published = True
                    slate.published_at = _ts
                    if "Base" not in slate.active_layers:
                        slate.active_layers = ["Base"]
                    set_slate_state(slate)

            except Exception as exc:
                st.error(f"Failed to load player pool: {exc}")

    # ── Pool Preview / Game Filter ───────────────────────────────────────
    hub_pool: Optional[pd.DataFrame] = st.session_state.get(f"_hub_pool_{slate_date_str}_{_contest_safe}")
    hub_rules: Optional[dict] = st.session_state.get(f"_hub_rules_{slate_date_str}_{_contest_safe}")

    if hub_pool is not None and not hub_pool.empty:
        all_games = _extract_games(hub_pool)
        is_showdown = (hub_rules or {}).get("is_showdown", False)

        if all_games:
            with st.expander("🎮 Game Filter", expanded=False):
                if is_showdown:
                    st.caption("Showdown: select exactly 1 game.")
                    sel_game = st.selectbox("Game", all_games, key="_hub_game_sd")
                    selected_games = [sel_game]
                else:
                    selected_games = st.multiselect(
                        "Filter to games (leave empty to keep all)",
                        all_games, default=[], key="_hub_games_multi",
                    )
                if selected_games:
                    opp_col = "opp" if "opp" in hub_pool.columns else (
                        "opponent" if "opponent" in hub_pool.columns else None
                    )
                    if opp_col:
                        hub_pool = _filter_pool_by_games(hub_pool, selected_games, opp_col)
                        # Update slate state with filtered pool
                        slate.player_pool = hub_pool
                        set_slate_state(slate)

        with st.expander(f"👥 Player Pool ({len(hub_pool)} players)", expanded=False):
            preview_cols = [c for c in [
                "player_name", "pos", "team", "opp", "opponent", "salary",
                "proj", "floor", "ceil", "proj_minutes", "ownership", "status", "sim_eligible",
            ] if c in hub_pool.columns]
            preview_df = hub_pool[preview_cols].sort_values("proj", ascending=False).copy()
            float_cols = [c for c in ["proj", "floor", "ceil", "proj_minutes", "ownership"]
                          if c in preview_df.columns]
            preview_df[float_cols] = preview_df[float_cols].round(1)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

            st.caption(f"{len(hub_pool)} players loaded.")

        # RG upload
        with st.expander("External Projections Upload", expanded=False):
            st.caption("Upload a RotoGrinders CSV to merge projections into the pool.")
            rg_file = st.file_uploader("RotoGrinders CSV", type="csv", key="_hub_rg_upload")
            if rg_file:
                try:
                    rg_df = load_rg_projections(rg_file)
                    st.session_state["_hub_rg_df"] = rg_df
                    st.success(f"RotoGrinders: {len(rg_df)} rows loaded.")
                    if st.button("Merge RG Projections into Pool"):
                        merged = merge_rg_with_pool(st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"], rg_df)
                        st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"] = merged
                        slate.player_pool = merged
                        set_slate_state(slate)
                        st.success(f"Merged RG data into pool ({len(merged)} rows).")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Failed to read RG CSV: {exc}")

        # Injury refresh
        with st.expander("🔃 Injury / News Refresh", expanded=False):
            if st.button("Refresh Injuries & News"):
                _key = st.session_state.get("rapidapi_key", "")
                if not _key:
                    st.warning("Tank01 RapidAPI key not configured — add TANK01_RAPIDAPI_KEY to .streamlit/secrets.toml.")
                else:
                    with st.spinner("Fetching latest injury updates…"):
                        try:
                            # date_key should be YYYYMMDD for Tank01
                            _injury_date = slate_date_str.replace("-", "")
                            updates = fetch_injury_updates(_injury_date, {"RAPIDAPI_KEY": _key})
                            if updates:
                                pool_names = set(hub_pool["player_name"].dropna().astype(str).values) if "player_name" in hub_pool.columns else set()
                                affected = [
                                    {"player": u.get("player_name", ""), "status": u.get("status", "")}
                                    for u in updates
                                    if u.get("player_name", "") in pool_names
                                ]
                                if affected:
                                    st.warning(f"⚠️ {len(affected)} player(s) in your pool have status changes:")
                                    st.dataframe(pd.DataFrame(affected), use_container_width=True, hide_index=True)
                                else:
                                    st.success(f"Fetched {len(updates)} league-wide updates — none affect your current pool.")
                            else:
                                st.info("No injury updates returned from Tank01.")
                        except Exception as exc:
                            _err_msg = str(exc)
                            if "429" in _err_msg or "rate" in _err_msg.lower():
                                st.warning("Tank01 API rate limit hit — try again in a minute.")
                            elif "401" in _err_msg or "403" in _err_msg:
                                st.error("Tank01 API key invalid or expired. Check TANK01_RAPIDAPI_KEY in secrets.")
                            else:
                                st.error(f"Injury refresh failed: {_err_msg}")

    st.divider()

    # =====================================================================
    # SECTION 2: SIMULATIONS + APPLY LEARNINGS (combined)
    # =====================================================================
    st.subheader("🎲 Simulations")

    # Use published pool
    pool: pd.DataFrame = slate.player_pool if slate.player_pool is not None else pd.DataFrame()

    # Auto-resolve draft group ID
    if slate.draft_group_id and slate.draft_group_id != sim.draft_group_id:
        sim.draft_group_id = int(slate.draft_group_id)
        set_sim_state(sim)

    # Sim controls
    col_mode, col_var = st.columns(2)
    with col_mode:
        sim_mode = st.radio("Mode", ["Live", "Historical"], horizontal=True, key="_lab_mode",
                            index=0 if sim.sim_mode == "Live" else 1)
        if sim_mode != sim.sim_mode:
            sim.sim_mode = sim_mode
            set_sim_state(sim)
    with col_var:
        variance = st.slider(
            "Sim Variance", min_value=0.5, max_value=2.0, step=0.1,
            value=float(sim.variance), key="_lab_variance",
        )
        if variance != sim.variance:
            sim.variance = variance
            set_sim_state(sim)

    with st.expander("Advanced sim settings", expanded=False):
        n_sims = st.number_input("Monte Carlo iterations", min_value=1000, max_value=50000,
                                 step=1000, value=int(sim.n_sims), key="_lab_nsims")
        if n_sims != sim.n_sims:
            sim.n_sims = int(n_sims)
            set_sim_state(sim)

    # Contest type for pipeline
    pipeline_contest_display = ["GPP Main", "GPP Early", "GPP Late", "Cash Main"]
    _CONTEST_NAME_TO_PIPELINE = {
        "GPP Main": "GPP_MAIN",
        "GPP Early": "GPP_EARLY",
        "GPP Late": "GPP_LATE",
        "Cash Main": "CASH",
        "Showdown": "GPP_EARLY",
    }
    _default_display_idx = pipeline_contest_display.index(slate.contest_name) if slate.contest_name in pipeline_contest_display else 1
    pipeline_contest_display_name = st.selectbox(
        "Contest Type",
        pipeline_contest_display,
        index=_default_display_idx,
        key="_lab_pipeline_contest",
    )
    pipeline_contest = _CONTEST_NAME_TO_PIPELINE.get(pipeline_contest_display_name, "GPP_20")

    if not pool.empty:
        if st.button("▶️ Run Sims Pipeline", type="primary", key="_lab_run_sims"):
            with st.spinner(f"Running {sim.n_sims:,} Monte Carlo iterations…"):
                try:
                    player_results = _build_player_level_sim_results(pool, sim.variance)
                    sim.player_results = player_results
                    # Build real optimized lineups instead of dummy placeholders
                    _PIPELINE_TO_OPTIMIZER = {"GPP_MAIN": "GPP_150", "GPP_EARLY": "GPP_20", "GPP_LATE": "GPP_20", "CASH": "CASH"}
                    optimizer_contest = _PIPELINE_TO_OPTIMIZER.get(pipeline_contest, "GPP_20")
                    real_lineups = build_ricky_lineups(edge_df=compute_edge_metrics(pool, calibration_state=slate.calibration_state, variance=sim.variance), contest_type=optimizer_contest, calibration_state=slate.calibration_state, salary_cap=SALARY_CAP)
                    if not real_lineups.empty:
                        pipeline_df = run_sims_pipeline(
                            pool=pool,
                            lineups_df=real_lineups,
                            contest_type=pipeline_contest,
                            n_sims=sim.n_sims,
                            variance=sim.variance,
                            slate_date=slate.slate_date,
                            draft_group_id=sim.draft_group_id,
                        )
                        sim.pipeline_output[pipeline_contest] = pipeline_df
                    set_sim_state(sim)

                    new_edge_df = compute_edge_metrics(
                        pool,
                        calibration_state=slate.calibration_state,
                        variance=sim.variance,
                    )
                    slate.edge_df = new_edge_df
                    if "Edge" not in slate.active_layers:
                        slate.active_layers.append("Edge")
                    set_slate_state(slate)
                    st.success(f"Sims complete — {len(player_results)} players.")
                except Exception as exc:
                    st.error(f"Sim failed: {exc}")
    else:
        st.info("Load a slate above first to run sims.")

    # Sim results display
    if sim.player_results is not None and not sim.player_results.empty:
        st.caption("Player-level smash / bust / leverage (sorted by leverage)")
        display_df = prepare_sims_table(sim.player_results)

        # Format all float columns to 2 decimal places for display
        _float_fmt = {c: "{:.2f}" for c in display_df.select_dtypes(include="number").columns
                      if c != "salary"}

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
            styled = display_df.style.apply(_style_row, axis=1).format(_float_fmt)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Pipeline output
    pipeline_output = sim.pipeline_output.get(pipeline_contest)
    if pipeline_output is not None and not pipeline_output.empty:
        with st.expander("📊 Pipeline Lineup Ratings", expanded=False):
            show_cols = [c for c in ["lineup_index", "projection", "total_pown", "top_x_rate",
                                     "itm_rate", "sim_roi", "leverage",
                                     "yakos_sim_rating", "rating_bucket"]
                         if c in pipeline_output.columns]
            st.dataframe(pipeline_output[show_cols], use_container_width=True, hide_index=True)

    # ── Score vs Actuals ────────────────────────────────────────────────
    if pipeline_output is not None and not pipeline_output.empty and "actual_fp" in pool.columns and pool["actual_fp"].notna().any():
        with st.expander("🎯 Score vs Actuals", expanded=True):
            st.caption("Lineup projections scored against real box-score results.")
            _lu_col = "lineup_index"
            _pn_col = "player_name"
            # Get the long-form lineup data from build_ricky_lineups
            # Reconstruct from the optimizer output stored during pipeline run
            try:
                _edge_for_score = compute_edge_metrics(pool, calibration_state=slate.calibration_state, variance=sim.variance)
                _PIPELINE_TO_OPTIMIZER_SC = {"GPP_MAIN": "GPP_150", "GPP_EARLY": "GPP_20", "GPP_LATE": "GPP_20", "CASH": "CASH"}
                _opt_contest = _PIPELINE_TO_OPTIMIZER_SC.get(pipeline_contest, "GPP_20")
                _lu_long = build_ricky_lineups(edge_df=_edge_for_score, contest_type=_opt_contest, calibration_state=slate.calibration_state, salary_cap=SALARY_CAP)

                if not _lu_long.empty and _pn_col in _lu_long.columns and _lu_col in _lu_long.columns:
                    # Map actuals onto lineup players
                    _act_map = pool.dropna(subset=["actual_fp"]).set_index("player_name")["actual_fp"].to_dict()
                    _lu_long["actual_fp"] = _lu_long[_pn_col].map(_act_map)

                    # Summarise per lineup
                    _lu_summary = _lu_long.groupby(_lu_col).agg(
                        proj_total=("proj", "sum"),
                        actual_total=("actual_fp", "sum"),
                        salary_total=("salary", "sum"),
                        players=(_pn_col, lambda x: ", ".join(x)),
                    ).reset_index()
                    _lu_summary["diff"] = (_lu_summary["actual_total"] - _lu_summary["proj_total"]).round(1)
                    _lu_summary["proj_total"] = _lu_summary["proj_total"].round(1)
                    _lu_summary["actual_total"] = _lu_summary["actual_total"].round(1)
                    _lu_summary = _lu_summary.sort_values("actual_total", ascending=False).reset_index(drop=True)

                    # Headline metrics
                    _best = _lu_summary.iloc[0]
                    _avg_proj = _lu_summary["proj_total"].mean()
                    _avg_actual = _lu_summary["actual_total"].mean()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Best Lineup Actual", f"{_best['actual_total']:.1f} FP")
                    c2.metric("Avg Projected", f"{_avg_proj:.1f}", delta=f"{_avg_actual - _avg_proj:+.1f} vs actual")
                    c3.metric("Lineups Built", len(_lu_summary))

                    # Summary table
                    _show = _lu_summary[[_lu_col, "proj_total", "actual_total", "diff", "salary_total"]].copy()
                    _show.columns = ["Lineup", "Projected", "Actual", "Diff", "Salary"]

                    def _color_diff(val):
                        if val > 0:
                            return "color: #4caf82"
                        elif val < 0:
                            return "color: #e05c5c"
                        return ""

                    st.dataframe(
                        _show.style.applymap(_color_diff, subset=["Diff"]),
                        use_container_width=True, hide_index=True,
                    )

                    # Best lineup detail
                    with st.expander(f"🏆 Best Lineup Detail (#{int(_best[_lu_col])})", expanded=False):
                        _best_players = _lu_long[_lu_long[_lu_col] == _best[_lu_col]][
                            [c for c in ["slot", _pn_col, "pos", "team", "salary", "proj", "actual_fp"] if c in _lu_long.columns]
                        ].copy()
                        if "actual_fp" in _best_players.columns and "proj" in _best_players.columns:
                            _best_players["diff"] = (_best_players["actual_fp"] - _best_players["proj"]).round(1)
                        st.dataframe(_best_players, use_container_width=True, hide_index=True)
            except Exception as _score_exc:
                st.warning(f"Score vs Actuals failed: {_score_exc}")

    # ── Apply Learnings (inline, not separate section) ───────────────────
    if sim.player_results is not None and not sim.player_results.empty:
        with st.expander("⚡ Apply Sim Learnings", expanded=False):
            st.caption(
                "Applies a non-destructive Sim Learnings layer (capped at ±15%). "
                "Does NOT overwrite base projections."
            )
            _BOOST_CAP = 0.15
            _BUST_REDUCTION = 0.08

            boost_threshold = st.slider(
                "Smash threshold for positive boost",
                min_value=0.10, max_value=0.50, step=0.01, value=0.20,
                key="_lab_boost_threshold",
            )
            bust_threshold = st.slider(
                "Bust threshold for reduction",
                min_value=0.20, max_value=0.60, step=0.01, value=0.30,
                key="_lab_bust_threshold",
            )

            col_apply, col_clear = st.columns(2)
            with col_apply:
                if st.button("⚡ Apply Learnings", key="_lab_apply_learnings"):
                    with st.spinner("Writing Sim Learnings layer…"):
                        pr = sim.player_results.copy()
                        applied = 0
                        for _, row in pr.iterrows():
                            pname = row.get("player_name", "")
                            smash = float(row.get("smash_prob", 0) or 0)
                            bust = float(row.get("bust_prob", 0) or 0)
                            if smash >= boost_threshold:
                                boost = min(_BOOST_CAP, smash * 0.5)
                                sim.apply_learning(pname, boost, f"smash_prob={smash:.2f}")
                                applied += 1
                            elif bust >= bust_threshold:
                                reduction = -min(_BUST_REDUCTION, bust * 0.25)
                                sim.apply_learning(pname, reduction, f"bust_prob={bust:.2f}")
                                applied += 1
                        if "Sims" not in slate.active_layers:
                            slate.active_layers.append("Sims")
                            set_slate_state(slate)
                        set_sim_state(sim)
                        st.success(f"Applied learnings for {applied} players. Layer 'Sims' activated.")
            with col_clear:
                if sim.sim_learnings and st.button("🗑️ Clear Learnings", key="_lab_clear_learnings"):
                    sim.clear_learnings()
                    if "Sims" in slate.active_layers:
                        slate.active_layers.remove("Sims")
                        set_slate_state(slate)
                    set_sim_state(sim)
                    st.info("Sim Learnings cleared.")

            if sim.sim_learnings:
                rows = [
                    {"Player": p, "Boost": f"{v['boost']:+.1%}", "Reason": v.get("reason", "")}
                    for p, v in sim.sim_learnings.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # =====================================================================
    # SECTION 3: EDGE ANALYSIS
    # =====================================================================
    st.subheader("📊 Edge Analysis")

    if not pool.empty and sim.player_results is not None and not sim.player_results.empty:
        pr = sim.player_results.copy()

        # ── Top leverage plays (smash prob vs ownership mismatch) ────────
        pos_edge = pr[pr["leverage"] > 1.2].nlargest(5, "leverage")
        neg_edge = pr[pr["leverage"] < 0.7].nsmallest(5, "leverage")

        ea_col1, ea_col2 = st.columns(2)
        with ea_col1:
            st.markdown("**Positive Leverage** — high smash, low owned")
            if not pos_edge.empty:
                _pe = pos_edge[["player_name", "salary", "ownership", "smash_prob", "leverage"]].copy()
                _pe["salary"] = _pe["salary"].astype(int)
                for _c in ["ownership", "smash_prob", "leverage"]:
                    if _c in _pe.columns:
                        _pe[_c] = _pe[_c].round(2)
                st.dataframe(_pe, use_container_width=True, hide_index=True)
            else:
                st.caption("No high-leverage plays found.")

        with ea_col2:
            st.markdown("**Negative Leverage** — bust risk, over-owned")
            if not neg_edge.empty:
                _ne = neg_edge[["player_name", "salary", "ownership", "bust_prob", "leverage"]].copy()
                _ne["salary"] = _ne["salary"].astype(int)
                for _c in ["ownership", "bust_prob", "leverage"]:
                    if _c in _ne.columns:
                        _ne[_c] = _ne[_c].round(2)
                st.dataframe(_ne, use_container_width=True, hide_index=True)
            else:
                st.caption("No over-owned bust risks found.")

        # ── Value plays ──────────────────────────────────────────────────
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

    # =====================================================================
    # SECTION 4: CALIBRATION
    # =====================================================================
    st.subheader("🔬 Calibration")

    _cal_slate_label = (
        f"{slate.sport or '—'} | {slate.site or '—'} | "
        f"{slate.slate_date or '—'} | DG {slate.draft_group_id or '—'}"
    )
    st.info(f"**Calibrating:** {_cal_slate_label}")

    with st.expander("Calibration Profiles", expanded=False):
        existing_profiles = list(sim.calibration_profiles.keys())
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            new_profile_name = st.text_input("New profile name", key="_lab_new_profile")
            if st.button("💾 Save Profile", key="_lab_save_profile"):
                if new_profile_name:
                    try:
                        current_cal = load_calibration_config()
                    except Exception:
                        current_cal = {}
                    sim.save_calibration_profile(new_profile_name, current_cal)
                    slate.calibration_state = dict(current_cal)
                    if "Calibration" not in slate.active_layers:
                        slate.active_layers.append("Calibration")
                    if pool is not None and not pool.empty:
                        slate.edge_df = compute_edge_metrics(
                            pool, calibration_state=slate.calibration_state, variance=sim.variance,
                        )
                    set_slate_state(slate)
                    set_sim_state(sim)
                    st.success(f"Profile '{new_profile_name}' saved.")

        with c_col2:
            if existing_profiles:
                active_profile = st.selectbox("Active profile", ["(none)"] + existing_profiles, key="_lab_active_profile")
                if active_profile != "(none)" and active_profile != sim.active_profile:
                    sim.active_profile = active_profile
                    profile_cal = sim.calibration_profiles.get(active_profile, {})
                    slate.calibration_state = dict(profile_cal)
                    if pool is not None and not pool.empty:
                        slate.edge_df = compute_edge_metrics(
                            pool, calibration_state=slate.calibration_state, variance=sim.variance,
                        )
                    set_slate_state(slate)
                    set_sim_state(sim)
                    st.success(f"Profile '{active_profile}' activated.")

                clone_src = st.selectbox("Clone profile", [""] + existing_profiles, key="_lab_clone_src")
                clone_dst = st.text_input("Clone to name", key="_lab_clone_dst")
                if st.button("📋 Clone", key="_lab_clone_btn") and clone_src and clone_dst:
                    ok = sim.clone_profile(clone_src, clone_dst)
                    set_sim_state(sim)
                    if ok:
                        st.success(f"Cloned '{clone_src}' → '{clone_dst}'.")
                    else:
                        st.error(f"Profile '{clone_src}' not found.")
            else:
                st.info("No profiles saved yet.")

    with st.expander("Bucketed Calibration Table", expanded=False):
        st.caption("Shows projection error by salary bucket and position. Requires at least 10 samples per bucket.")
        _MIN_SAMPLES = 10
        if not pool.empty and "proj" in pool.columns and "salary" in pool.columns:
            cal_pool = pool.copy()
            cal_pool["salary_bucket"] = pd.cut(
                pd.to_numeric(cal_pool["salary"], errors="coerce"),
                bins=[0, 4500, 5500, 6500, 7500, 8500, 99999],
                labels=["<4.5K", "4.5–5.5K", "5.5–6.5K", "6.5–7.5K", "7.5–8.5K", "8.5K+"],
            )
            bucket_counts = cal_pool.groupby("salary_bucket", observed=True).size().reset_index(name="n")
            valid_buckets = bucket_counts[bucket_counts["n"] >= _MIN_SAMPLES]["salary_bucket"].tolist()
            if valid_buckets:
                bucket_stats = (
                    cal_pool[cal_pool["salary_bucket"].isin(valid_buckets)]
                    .groupby("salary_bucket", observed=True)
                    .agg(n=("proj", "count"), avg_proj=("proj", "mean"), avg_salary=("salary", "mean"))
                    .reset_index()
                )
                st.dataframe(bucket_stats, use_container_width=True, hide_index=True)
            else:
                st.info(f"No salary buckets have ≥{_MIN_SAMPLES} samples.")
        else:
            st.info("Load a slate to see calibration buckets.")

    with st.expander("📈 Historical Calibration Pipeline (Bucket-level)", expanded=False):
        st.caption(
            "Accumulates historical sims output and computes realized ROI / top-finish rates "
            "per YakOS Sim Rating bucket (A/B/C/D)."
        )
        if st.button("🔄 Run Calibration Pipeline", key="_lab_run_cal_pipeline"):
            with st.spinner("Running calibration pipeline…"):
                try:
                    cal_summary = run_calibration_pipeline()
                    if not cal_summary.empty:
                        st.dataframe(cal_summary, use_container_width=True, hide_index=True)
                        _buckets_ready = cal_summary[cal_summary.get("meets_threshold", False)]["rating_bucket"].tolist() if "meets_threshold" in cal_summary.columns else []
                        if _buckets_ready:
                            st.success(f"Buckets with sufficient volume: {', '.join(_buckets_ready)}")
                        else:
                            st.info("Not enough volume yet to update weights.")
                    else:
                        st.info("No historical pipeline data found.")
                except Exception as exc:
                    st.error(f"Calibration pipeline failed: {exc}")

    with st.expander("🧪 Rating Weight Update Tester", expanded=False):
        st.caption("Compare two sets of rating weights on historical data before committing new weights.")
        default_weights = get_weight_sets().get("GPP_20", {})
        weight_keys = list(default_weights.keys())

        st.markdown("**Old weights (current):**")
        old_w_cols = st.columns(len(weight_keys))
        old_weights: dict = {}
        for i, k in enumerate(weight_keys):
            with old_w_cols[i]:
                old_weights[k] = st.number_input(
                    k, min_value=0.0, max_value=1.0, step=0.01,
                    value=float(default_weights.get(k, 0.0)), key=f"_lab_old_w_{k}",
                )

        st.markdown("**New weights (proposed):**")
        new_w_cols = st.columns(len(weight_keys))
        new_weights: dict = {}
        for i, k in enumerate(weight_keys):
            with new_w_cols[i]:
                new_weights[k] = st.number_input(
                    k, min_value=0.0, max_value=1.0, step=0.01,
                    value=float(default_weights.get(k, 0.0)), key=f"_lab_new_w_{k}",
                )

        if st.button("🔍 Compare Weights", key="_lab_compare_weights"):
            with st.spinner("Running before/after comparison…"):
                try:
                    comparison_df = compare_rating_weights(
                        old_params={"weights": old_weights},
                        new_params={"weights": new_weights},
                        contest_type=pipeline_contest,
                    )
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"Weight comparison failed: {exc}")

    with st.expander("📐 Calibration Feedback (Actuals Loop)", expanded=True):
        try:
            from yak_core.calibration_feedback import (
                record_slate_errors, get_calibration_summary,
                get_correction_factors, clear_calibration_history,
            )

            # Use session state as storage (survives on Streamlit Cloud)
            if "_cal_fb_store" not in st.session_state:
                st.session_state["_cal_fb_store"] = {}
            _fb_store = st.session_state["_cal_fb_store"]

            # ── Manual "Fetch Actuals & Record" button ──
            _fb_is_hist = False
            try:
                _fb_is_hist = pd.to_datetime(slate.slate_date).date() < _today_date if slate.slate_date else False
            except Exception:
                pass

            if _fb_is_hist and pool is not None and not pool.empty:
                _fb_has_actuals = "actual_fp" in pool.columns and pool["actual_fp"].notna().any()
                _fb_has_proj = "proj" in pool.columns and pool["proj"].notna().any()

                if _fb_has_actuals and _fb_has_proj:
                    # Actuals already in pool (from box score cross-ref or external projections)
                    if st.button("📐 Record Calibration from Current Pool", key="_lab_record_cal_pool"):
                        _fb_result = record_slate_errors(slate.slate_date, pool, store=_fb_store)
                        if "error" not in _fb_result:
                            _ov = _fb_result.get("overall", {})
                            st.success(f"Recorded: MAE {_ov.get('mae', '?')}, Corr {_ov.get('correlation', '?')}, {_ov.get('n_players', 0)} players")
                        else:
                            st.warning(f"Could not record: {_fb_result['error']}")
                else:
                    # Need to fetch actuals from Tank01
                    _fb_api_key = st.session_state.get("rapidapi_key", "")
                    if _fb_api_key:
                        if st.button("📐 Fetch Actuals & Record Calibration", key="_lab_fetch_record_cal"):
                            with st.spinner("Fetching box scores..."):
                                try:
                                    from yak_core.live import fetch_actuals_from_api
                                    _fb_date = slate.slate_date.replace("-", "")
                                    _fb_actuals = fetch_actuals_from_api(_fb_date, {"RAPIDAPI_KEY": _fb_api_key})
                                    if not _fb_actuals.empty and "actual_fp" in _fb_actuals.columns:
                                        _fb_act_map = _fb_actuals.set_index("player_name")["actual_fp"].to_dict()
                                        _fb_pool = pool.copy()
                                        _fb_pool["actual_fp"] = _fb_pool["player_name"].map(_fb_act_map)
                                        _fb_result = record_slate_errors(slate.slate_date, _fb_pool, store=_fb_store)
                                        if "error" not in _fb_result:
                                            _ov = _fb_result.get("overall", {})
                                            st.success(f"Recorded: MAE {_ov.get('mae', '?')}, Corr {_ov.get('correlation', '?')}, {_ov.get('n_players', 0)} players")
                                        else:
                                            st.warning(f"Could not record: {_fb_result['error']}")
                                    else:
                                        st.warning("No actuals returned from Tank01 for this date.")
                                except Exception as _fb_fetch_exc:
                                    st.error(f"Failed to fetch actuals: {_fb_fetch_exc}")
                    else:
                        st.caption(
                            "⚠️ Tank01 API key not set and pool has no actuals. "
                            "Either add TANK01_RAPIDAPI_KEY to secrets or upload an external projections file with actual results."
                        )
            elif not _fb_is_hist:
                st.caption("Switch to a historical date to record calibration data.")

            st.markdown("---")

            # ── Summary display ──
            _cal_fb = get_calibration_summary(store=_fb_store)
            _fb_status = _cal_fb.get("status", "no_data")

            if _fb_status == "no_data":
                st.info(
                    "No calibration data yet. Load a historical slate above, then click "
                    "**Record Calibration** to start building the correction model."
                )
            else:
                _n = _cal_fb.get("n_slates", 0)
                _status_emoji = "🟡" if _fb_status == "building" else "🟢"
                _status_label = f"{_status_emoji} {_fb_status.upper()} — {_n} slate(s) recorded"
                st.markdown(f"**{_status_label}**")

                if _fb_status == "building":
                    st.caption(f"Need {3 - _n} more slate(s) for full correction model. Keep running historical dates.")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Avg MAE", _cal_fb.get("avg_mae", "—"))
                with m2:
                    st.metric("Latest MAE", _cal_fb.get("latest_mae", "—"))
                with m3:
                    st.metric("Overall Bias", f"{_cal_fb.get('overall_bias', 0):+.2f}")

                _pos_corr = _cal_fb.get("position_corrections", [])
                if _pos_corr:
                    st.markdown("**Position corrections:**")
                    _pos_df = pd.DataFrame(_pos_corr)
                    _pos_df["correction"] = _pos_df["correction"].apply(lambda x: f"{x:+.2f}")
                    st.dataframe(_pos_df, use_container_width=True, hide_index=True)

                _tier_corr = _cal_fb.get("tier_corrections", [])
                if _tier_corr:
                    st.markdown("**Salary tier corrections:**")
                    _tier_df = pd.DataFrame(_tier_corr)
                    _tier_df["correction"] = _tier_df["correction"].apply(lambda x: f"{x:+.2f}")
                    st.dataframe(_tier_df, use_container_width=True, hide_index=True)

                _dates = _cal_fb.get("dates", [])
                if _dates:
                    st.caption(f"Slates: {', '.join(_dates)}")

                if st.button("🗑️ Reset Calibration History", key="_lab_reset_cal_fb"):
                    st.session_state["_cal_fb_store"] = {}
                    st.info("Calibration feedback cleared.")

        except Exception as _cal_fb_exc:
            st.caption(f"Calibration feedback unavailable: {_cal_fb_exc}")

    st.divider()

    # =====================================================================
    # SECTION 5: RCI — Ricky Confidence Index
    # =====================================================================
    st.subheader("🎯 Ricky Confidence Index (RCI)")
    st.caption(
        "Multi-signal gauge: Projection Confidence, Sim Alignment, "
        "Ownership Accuracy, and Historical ROI → per-contest 0–100 score."
    )

    rci_contests = list(edge.edge_analysis_by_contest.keys()) or list(CONTEST_PRESETS.keys())

    if not rci_contests:
        st.info("Run Edge Analysis on the Ricky Edge page to enable RCI gauges.")
    else:
        rci_cols = st.columns(min(len(rci_contests), 3))
        for ci, contest_label in enumerate(rci_contests):
            col_idx = ci % len(rci_cols)
            with rci_cols[col_idx]:
                edge_payload = edge.edge_analysis_by_contest.get(contest_label, {})
                custom_weights = sim.get_rci_weights(contest_label)
                try:
                    rci_result = compute_rci(
                        contest_label=contest_label,
                        edge_payload=edge_payload,
                        sim_results=sim.player_results,
                        weights=custom_weights,
                    )
                    sim.set_rci_result(contest_label, rci_result)
                    pct = int(rci_result.rci_score)
                    color = (
                        "green" if rci_result.rci_status == "green"
                        else "orange" if rci_result.rci_status == "yellow"
                        else "red"
                    )
                    st.markdown(f"**{contest_label}**")
                    st.progress(pct)
                    st.markdown(
                        f"<span style='color:{color}'>RCI: {pct}/100</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(rci_result.recommendation)
                except Exception as _rci_err:
                    st.warning(f"{contest_label}: RCI error — {_rci_err}")

        with st.expander("📊 RCI Signal Breakdown", expanded=False):
            for contest_label in rci_contests:
                stored = sim.get_rci_result(contest_label)
                if stored and stored.get("signals"):
                    st.markdown(f"**{contest_label}** — RCI {stored['rci_score']:.0f}/100")
                    signal_rows = [
                        {
                            "Signal": s["name"].replace("_", " ").title(),
                            "Score": f"{s['value']:.0f}/100",
                            "Weight": f"{s['weight']:.0%}",
                            "Status": s["status"].upper(),
                            "Description": s["description"],
                        }
                        for s in stored["signals"]
                    ]
                    st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)

        with st.expander("⚙️ Tune RCI Weights", expanded=False):
            st.caption("Override the default signal weights for each contest type.")
            tune_contest = st.selectbox("Contest type to tune", rci_contests, key="_lab_rci_tune_contest")
            if tune_contest:
                existing_w = sim.get_rci_weights(tune_contest) or dict(RCI_DEFAULT_WEIGHTS)
                w_cols = st.columns(len(RCI_DEFAULT_WEIGHTS))
                new_w: Dict[str, float] = {}
                for wi, (wk, wv) in enumerate(RCI_DEFAULT_WEIGHTS.items()):
                    with w_cols[wi]:
                        new_w[wk] = st.number_input(
                            wk.replace("_", " ").title(),
                            min_value=0.0, max_value=1.0, step=0.05,
                            value=float(existing_w.get(wk, wv)),
                            key=f"_lab_rci_w_{tune_contest}_{wk}",
                        )
                if st.button("💾 Save RCI Weights", key="_lab_rci_save_weights"):
                    sim.set_rci_weights(tune_contest, new_w)
                    set_sim_state(sim)
                    st.success(f"RCI weights saved for '{tune_contest}'.")
                if st.button("↩️ Reset to Defaults", key="_lab_rci_reset_weights"):
                    if tune_contest in sim.rci_weights:
                        del sim.rci_weights[tune_contest]
                    set_sim_state(sim)
                    st.success(f"RCI weights reset to defaults for '{tune_contest}'.")

    set_sim_state(sim)

    st.divider()

    # =====================================================================
    # SECTION 6: CONTEST-TYPE GAUGES
    # =====================================================================
    st.subheader("📈 Contest-type Gauges")
    st.caption("Score driven by sim smash probability and calibration outputs.")

    gauge_cols = st.columns(len(_CONTEST_GAUGE_LABELS))
    pr = sim.player_results

    for gcol, contest in zip(gauge_cols, _CONTEST_GAUGE_LABELS):
        with gcol:
            score = _gauge_score(pr, contest)
            pct = int(score * 100)
            color = "green" if pct >= 60 else "orange" if pct >= 35 else "red"
            st.markdown(f"**{contest}**")
            st.progress(pct)
            st.markdown(f"<span style='color:{color}'>{pct}%</span>", unsafe_allow_html=True)
            sim.contest_gauges[contest] = {"score": score, "label": contest}
    set_sim_state(sim)

    st.divider()

    # =====================================================================
    # SECTION 7: RICKY EDGE CHECK GATE
    # =====================================================================
    st.subheader("🔐 Ricky Edge Check Gate")

    # Historical slates auto-bypass the gate (no edge research needed for backtests)
    _lab_is_historical = False
    try:
        _lab_is_historical = pd.to_datetime(slate.slate_date).date() < _today_date if slate.slate_date else False
    except Exception:
        pass

    if _lab_is_historical:
        if not edge.ricky_edge_check:
            from datetime import datetime as _dt, timezone as _tz
            edge.approve_edge_check(f"{_dt.now(_tz.utc).isoformat()} (auto-historical)")
            set_edge_state(edge)
        st.success(f"✅ Auto-approved for historical slate. Build & Publish is unlocked.")
    elif edge.ricky_edge_check:
        st.success(f"✅ Ricky Edge Check approved at {edge.edge_check_ts}. Build & Publish is unlocked.")
        if st.button("🔓 Revoke Edge Check", key="_lab_revoke_edge"):
            edge.revoke_edge_check()
            set_edge_state(edge)
            st.rerun()
    else:
        st.warning(
            "⛔ **Ricky Edge Check not approved.** "
            "Approve here or go to **Ricky Edge** for full tagging."
        )
        if st.button("✅ Approve Edge Check", type="primary", key="_lab_approve_edge"):
            from datetime import datetime as _dt, timezone as _tz
            edge.approve_edge_check(_dt.now(_tz.utc).isoformat())
            set_edge_state(edge)
            st.rerun()


main()
