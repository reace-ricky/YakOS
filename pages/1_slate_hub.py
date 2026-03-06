"""Slate Hub – YakOS Sprint 1 page.

Responsibilities
----------------
- Date + Sport picker (date alone determines live vs historical — no Data Mode toggle).
- Contest Type picker from CONTEST_PRESETS.
- Fetch DK Lobby Contests → auto-resolves draft_group_id by matching contest type.
- Load Player Pool via DK draftables API (always API-first, no local files).
  Enriches with Tank01 game-log rolling stats and Vegas odds when
  TANK01_RAPIDAPI_KEY secret is present.  Does NOT merge Tank01 DFS data —
  DK is the sole source of truth for the player pool, salaries, and injury
  status.
- Game selector (multiselect) after pool loads to filter by matchup.
- Optional RG CSV overlay via merge_rg_with_pool().
- S1.7: Refresh action that re-pulls injuries via Tank01 API.
- Provide a "Publish Slate" action that writes the full configuration
  into SlateState.

Projection model is always YakOS Model (proj_source = "model").
Tank01 RapidAPI key is read from st.secrets["TANK01_RAPIDAPI_KEY"] — not prompted on this page.

All state is written exclusively to SlateState.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Ensure repo root is on sys.path when running directly
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import get_slate_state, set_slate_state  # noqa: E402
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
from yak_core.sims import compute_sim_eligible, _INELIGIBLE_STATUSES  # noqa: E402
from yak_core.live import fetch_injury_updates, fetch_player_game_logs, fetch_betting_odds  # noqa: E402
from yak_core.salary_history import SalaryHistoryClient  # noqa: E402


def _fetch_dk_draft_groups(sport: str = "NBA") -> list:
    """Fetch DraftGroup metadata from the DK lobby API.
    
    The lobby response contains both Contests and DraftGroups arrays.
    This function extracts the DraftGroups array which has slate-level
    metadata (GameCount, ContestStartTimeSuffix, GameStyle, etc.).
    """
    import requests
    url = "https://www.draftkings.com/lobby/getcontests"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    resp = requests.get(url, params={"sport": sport.upper()}, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    # Extract DraftGroups array
    draft_groups_raw = data.get("DraftGroups") or data.get("draftGroups") or []
    return draft_groups_raw


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# game_type_ids that represent Showdown contests
_SHOWDOWN_GAME_TYPE_IDS = {
    gid for gid, label in DK_GAME_TYPE_LABELS.items() if "Showdown" in label
}

# Columns that uniquely identify a player slot in the pool (used for group-dedup)
_PLAYER_IDENTITY_COLS = ["player_name", "team", "pos", "salary"]

# Numeric projection columns aggregated (averaged) across duplicate rows
_NUMERIC_AGG_COLS = ["proj", "floor", "ceil", "proj_minutes", "ownership"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rules_from_preset(preset: dict) -> dict:
    """Build a roster rules dict from a CONTEST_PRESETS entry."""
    is_showdown = preset.get("slate_type") == "Showdown Captain"
    return {
        "slots": DK_SHOWDOWN_SLOTS if is_showdown else DK_POS_SLOTS,
        "lineup_size": DK_SHOWDOWN_LINEUP_SIZE if is_showdown else DK_LINEUP_SIZE,
        "salary_cap": SALARY_CAP,
        "is_showdown": is_showdown,
    }


def _normalize_dk_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names from fetch_dk_draftables to YakOS conventions."""
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

    Preserves existing floor/ceil columns when already present.
    Ownership is computed pool-wide via salary_rank_ownership (from
    yak_core.ownership) which uses salary rank percentile and position
    scarcity — no external RG data required.

    Players with ineligible status (OUT, IR, DND, etc.) have their
    proj_minutes forced to 0 so the minutes-based sim_eligible filter
    correctly excludes them regardless of their salary.

    When rolling stat columns (rolling_fp_5/10/20, rolling_min_5/10/20)
    and Vegas context columns (vegas_total, spread) are present in the
    pool they are forwarded to yakos_fp_projection and
    yakos_minutes_projection so the YakOS model can use real observed data
    rather than salary-only estimates.
    """
    has_floor = "floor" in pool.columns and pool["floor"].notna().any()
    has_ceil = "ceil" in pool.columns and pool["ceil"].notna().any()

    # Columns to forward to the projection functions when present
    _OPTIONAL_FEAT_COLS = [
        "rolling_fp_5", "rolling_fp_10", "rolling_fp_20",
        "rolling_min_5", "rolling_min_10", "rolling_min_20",
        "vegas_total", "spread",
        "tank01_proj",
    ]

    floors, ceils, mins_proj = [], [], []
    for _, row in pool.iterrows():
        feats = {"salary": float(row.get("salary", 0) or 0)}
        # Forward any available real-data features to the projection functions
        for col in _OPTIONAL_FEAT_COLS:
            if col in pool.columns:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    feats[col] = float(val)

        fp_res = yakos_fp_projection(feats)
        min_res = yakos_minutes_projection(feats)
        floors.append(fp_res.get("floor", fp_res["proj"] * 0.7))
        ceils.append(fp_res.get("ceil", fp_res["proj"] * 1.4))
        mins_proj.append(min_res.get("proj_minutes", 0.0))

    if not has_floor:
        pool["floor"] = floors
    if not has_ceil:
        pool["ceil"] = ceils
    pool["proj_minutes"] = mins_proj

    # Force proj_minutes = 0 for players with ineligible status so the
    # minutes-based sim_eligible filter (min_proj_minutes=4.0) correctly
    # excludes them even when salary-based projection is non-zero.
    if "status" in pool.columns:
        inelig_mask = (
            pool["status"].fillna("").astype(str).str.strip().str.upper()
            .isin(_INELIGIBLE_STATUSES)
        )
        pool.loc[inelig_mask, "proj_minutes"] = 0.0

    # Pool-level ownership model (salary rank + position scarcity).
    # Replaces the broken per-row yakos_ownership_projection call which
    # required proj and rg_ownership features we don't have here.
    pool = apply_ownership(pool)

    return compute_sim_eligible(pool)


def _filter_ineligible_players(pool: pd.DataFrame) -> pd.DataFrame:
    """Remove players that should never appear in an optimisable pool.

    A player is removed when **either** condition is true:
      1. ``status`` is an ineligible designation (OUT, IR, DND, G-League,
         Suspended, etc.) – the player was not available for that slate.
      2. ``proj_minutes`` is zero (or missing) – no projected playing time,
         so the player contributes nothing to any lineup.

    This filter is applied before the pool is displayed AND before it is
    published to ``SlateState`` so that downstream pages (The Lab, Build &
    Publish) never see players who are ineligible for the slate date.
    """
    df = pool.copy()

    # Status-based removal
    if "status" in df.columns:
        inelig_mask = (
            df["status"].fillna("").astype(str).str.strip().str.upper()
            .isin(_INELIGIBLE_STATUSES)
        )
        df = df[~inelig_mask]

    # Minutes-based removal (zero projected minutes → no playing time)
    mins_col = "proj_minutes" if "proj_minutes" in df.columns else (
        "minutes" if "minutes" in df.columns else None
    )
    if mins_col is not None:
        mins = pd.to_numeric(df[mins_col], errors="coerce").fillna(0)
        df = df[mins > 0]

    return df.reset_index(drop=True)


def _render_status_bar(slate: "SlateState") -> None:
    """Render the top-of-page global status bar."""
    cols = st.columns([2, 2, 2, 2, 4])
    with cols[0]:
        st.metric("Sport", slate.sport or "—")
    with cols[1]:
        st.metric("Site", slate.site or "—")
    with cols[2]:
        st.metric("Date", slate.slate_date or "—")
    with cols[3]:
        st.metric("Contest", slate.contest_type or "—")
    with cols[4]:
        if slate.active_layers:
            chips = " ".join(f"`{l}`" for l in slate.active_layers)
            st.markdown(f"**Layers:** {chips}")
        if slate.published:
            st.success(f"✅ Slate published at {slate.published_at}")


def _extract_games(pool: pd.DataFrame) -> list[str]:
    """Extract unique game matchup strings from the pool."""
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
    """Return rows whose team+opponent matchup is in the selected games list."""
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
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🏀 Slate Hub")
    st.caption("Configure the slate and publish it.")

    slate = get_slate_state()
    _render_status_bar(slate)
    st.divider()

    # ── Row 1: Sport, Date ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        sport = st.selectbox("Sport", ["NBA", "PGA"], index=0 if slate.sport == "NBA" else 1)
    with col2:
        from zoneinfo import ZoneInfo
        _today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        slate_date = st.date_input("Date", value=pd.to_datetime(_today))
        slate_date_str = str(slate_date)

    # Read Tank01 RapidAPI key from secrets (not prompted on this page)
    rapidapi_key = st.secrets.get("TANK01_RAPIDAPI_KEY")
    if rapidapi_key:
        st.session_state["rapidapi_key"] = rapidapi_key

    # ── Row 2: Contest Type ────────────────────────────────────────────────
    contest_type_label = st.selectbox("Contest Type", CONTEST_PRESET_LABELS)
    preset = CONTEST_PRESETS[contest_type_label]
    st.caption(preset.get("description", ""))

    # Clear cached pool when date, sport, or contest type changes to prevent stale data.
    # Cache key includes both date and contest type so switching contests forces a fresh load.
    _contest_safe = contest_type_label.lower().replace(" ", "_").replace("/", "-").replace("-", "_")
    _prev_date = st.session_state.get("_hub_prev_date")
    _prev_sport = st.session_state.get("_hub_prev_sport")
    _prev_contest = st.session_state.get("_hub_prev_contest")
    _date_changed = _prev_date is not None and _prev_date != slate_date_str
    _sport_changed = _prev_sport is not None and _prev_sport != sport
    _contest_changed = _prev_contest is not None and _prev_contest != contest_type_label
    if _date_changed or _sport_changed or _contest_changed:
        # Clear previous pool data for the old date + contest combination.
        # Use explicit `if … else` guards so we never accidentally key on the
        # *current* value when a previous value was not recorded yet.
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
        # Clear slate cache keyed to the previous sport
        old_slate_key = f"_hub_slates_{_stale_sport}_{_stale_date}"
        st.session_state.pop(old_slate_key, None)
    st.session_state["_hub_prev_date"] = slate_date_str
    st.session_state["_hub_prev_sport"] = sport
    st.session_state["_hub_prev_contest"] = contest_type_label

    # Projection model is always YakOS Model
    proj_source = "model"

    # ── Slate Picker ──────────────────────────────────────────────────────
    st.subheader("Select Slate")
    
    _slate_cache_key = f"_hub_slates_{sport}_{slate_date_str}"
    _cached_slates = st.session_state.get(_slate_cache_key)
    
    col_fetch_slate, col_clear_slate = st.columns([2, 1])
    with col_fetch_slate:
        if st.button("🔍 Fetch Available Slates", type="secondary"):
            with st.spinner("Fetching slates from DraftKings..."):
                try:
                    raw_dgs = _fetch_dk_draft_groups(sport)
                    if not raw_dgs:
                        st.warning("No slates found on DraftKings for this sport. Try a different date.")
                    else:
                        slate_options = build_slate_options(raw_dgs)
                        st.session_state[_slate_cache_key] = slate_options
                        _cached_slates = slate_options
                        st.success(f"Found {len(slate_options)} slate(s).")
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
        # Build radio options: "Main Slate (6 games)" etc.
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
    
    # Manual DG ID override (for historical dates or when lobby is unavailable)
    with st.expander("Manual Draft Group ID (advanced)", expanded=_cached_slates is None):
        manual_dg = st.number_input(
            "Draft Group ID",
            min_value=0,
            step=1,
            value=0,
            help="Paste a DraftKings Draft Group ID if you know it. Overrides the slate picker.",
            key="_hub_manual_dg",
        )
        if manual_dg > 0:
            selected_dg_id = int(manual_dg)
            selected_slate_label = f"Manual (DG {manual_dg})"

    # ── Row 5: Load Player Pool ────────────────────────────────────────────
    _today_date = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()
    _is_historical = pd.to_datetime(slate_date_str).date() < _today_date
    _salary_client = SalaryHistoryClient()

    if st.button("📥 Load Player Pool", type="primary"):
        # Resolve draft_group_id from the slate picker or manual override.
        draft_group_id: Optional[int] = selected_dg_id

        with st.spinner("Loading player pool…"):
            try:
                # ── Historical salary path ────────────────────────────────
                # For historical dates use the SalaryHistoryClient pipeline
                # (FantasyLabs + DK draftables) instead of the live DK lobby.
                _historical_salary_df: Optional[pd.DataFrame] = None
                _historical_dg_id: Optional[int] = None

                if _is_historical and not draft_group_id:
                    # Check cache first
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
                                st.info(
                                    f"Historical salaries loaded from DK "
                                    f"(DraftGroup {_historical_dg_id})"
                                )

                if not draft_group_id and _historical_salary_df is None:
                    st.warning(
                        "No slate selected. Use \"Fetch Available Slates\" to pick a slate, "
                        "or enter a Draft Group ID manually in the advanced section above."
                    )
                else:
                    # Step 1: Fetch DK draftables (salaries, positions, teams)
                    # Use historical salary cache when available; fall back to live DK API.
                    if _historical_salary_df is not None and not _historical_salary_df.empty:
                        pool = _historical_salary_df.copy()
                        # Rename SalaryHistoryClient columns to YakOS conventions
                        if "position" in pool.columns and "pos" not in pool.columns:
                            pool = pool.rename(columns={"position": "pos"})
                        if "player_dk_id" in pool.columns and "dk_player_id" not in pool.columns:
                            pool = pool.rename(columns={"player_dk_id": "dk_player_id"})
                        if _historical_dg_id and not draft_group_id:
                            draft_group_id = _historical_dg_id
                    else:
                        pool = fetch_dk_draftables(draft_group_id)
                        if pool.empty:
                            st.error(f"No players found for Draft Group ID {draft_group_id}.")
                            st.stop()

                    # Step 2: Normalize column names
                    pool = _normalize_dk_pool(pool)

                    # Normalize salary
                    if "salary" not in pool.columns:
                        pool["salary"] = 0
                    pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce").fillna(0)

                    # Auto-save live salary data to cache for future historical lookups
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

                    # ----------------------------------------------------------------
                    # Step 3: Enrich pool with Tank01 game-log stats and Vegas odds.
                    #
                    # Architecture: DK is the sole source of truth for the player
                    # pool, salaries, positions, and injury status.  Tank01 is used
                    # ONLY for game-log rolling averages (fp/min) and Vegas totals.
                    # We do NOT call fetch_live_opt_pool / getNBADFS here — that
                    # endpoint overwrites injury status and reduces the pool.
                    # ----------------------------------------------------------------
                    _api_key = st.session_state.get("rapidapi_key", "")
                    if _api_key:
                        try:
                            # -- 3a: Build Tank01 playerID map from pool if available --
                            # The DK draftables payload sometimes includes a Tank01-
                            # compatible player_id; use it when present.
                            _t01_id_map: dict = {}
                            for _id_col in ("player_id", "tank01_player_id", "t01_id"):
                                if _id_col in pool.columns:
                                    _t01_id_map = dict(
                                        zip(
                                            pool["player_name"].astype(str),
                                            pool[_id_col].astype(str),
                                        )
                                    )
                                    break

                            # -- 3b: Fetch per-player game log rolling stats --
                            _player_names = pool["player_name"].dropna().tolist()
                            with st.spinner("Fetching game log rolling stats from Tank01…"):
                                _game_log_df = fetch_player_game_logs(
                                    _player_names,
                                    _t01_id_map if _t01_id_map else None,
                                    _api_key,
                                )

                            if not _game_log_df.empty:
                                # Left-join rolling stats into pool — does NOT touch status
                                pool = pool.merge(
                                    _game_log_df,
                                    on="player_name",
                                    how="left",
                                )
                                st.caption(
                                    f"✅ Rolling stats merged for "
                                    f"{_game_log_df['player_name'].nunique()} players."
                                )
                            else:
                                st.caption("ℹ️ No game log rolling stats returned from Tank01.")

                            # -- 3c: Fetch Vegas betting odds for this date --
                            with st.spinner("Fetching Vegas odds from Tank01…"):
                                _odds_df = fetch_betting_odds(slate_date_str, _api_key)

                            if not _odds_df.empty:
                                # Merge Vegas total + spread into pool by matching
                                # each player's team against home/away team columns.
                                # Strategy: build a team → (vegas_total, spread) lookup
                                # where away teams get the same total but inverted spread.
                                _team_odds_rows = []
                                for _, _o in _odds_df.iterrows():
                                    _total = _o["vegas_total"]
                                    _spread = _o["spread"]
                                    # Home team: spread as-is
                                    if _o["home_team"]:
                                        _team_odds_rows.append({
                                            "team": _o["home_team"],
                                            "vegas_total": _total,
                                            "spread": _spread,
                                        })
                                    # Away team: same total, inverted spread
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
                                    # Temporarily strip existing vegas cols to avoid _x/_y suffixes
                                    for _vc in ("vegas_total", "spread"):
                                        if _vc in pool.columns:
                                            pool = pool.drop(columns=[_vc])
                                    pool = pool.merge(
                                        _team_odds_df,
                                        on="team",
                                        how="left",
                                    )
                                    st.caption(
                                        f"✅ Vegas odds merged for "
                                        f"{_team_odds_df['team'].nunique()} teams."
                                    )
                            else:
                                st.caption("ℹ️ No Vegas odds returned from Tank01.")

                        except Exception as t01_exc:
                            st.caption(f"ℹ️ Tank01 enrichment not available: {t01_exc}")

                    # Step 4: Apply projection pipeline
                    parsed_rules = _rules_from_preset(preset)
                    cfg = merge_config({
                        "PROJ_SOURCE": proj_source,
                        "SLATE_DATE": slate_date_str,
                        "CONTEST_TYPE": preset["internal_contest"],
                    })
                    pool = apply_projections(pool, cfg)

                    # Step 5: Add floor/ceil/minutes/ownership per player
                    pool = _enrich_pool(pool)

                    # Step 6: Group-then-deduplicate – aggregate duplicate rows per player
                    # across sources (e.g. multiple projection providers for the same player)
                    # before falling back to a single-key drop_duplicates for any residual dupes.
                    _group_cols = [c for c in _PLAYER_IDENTITY_COLS if c in pool.columns]
                    _agg_cols = {
                        c: "mean"
                        for c in _NUMERIC_AGG_COLS
                        if c in pool.columns
                    }
                    if _group_cols and _agg_cols:
                        # Preserve non-aggregated columns by taking the first value per group
                        _extra_cols = [c for c in pool.columns if c not in _group_cols and c not in _agg_cols]
                        _extra_agg = {c: "first" for c in _extra_cols}
                        pool = pool.groupby(_group_cols, as_index=False).agg({**_agg_cols, **_extra_agg})
                    else:
                        # Fallback: single-key dedup when group cols or agg cols are absent
                        if "dk_player_id" in pool.columns:
                            dedup_key = "dk_player_id"
                        else:
                            dedup_key = "player_name"
                            st.caption("ℹ️ dk_player_id not found; deduplicating by player_name.")
                        if "proj" in pool.columns:
                            pool = pool.sort_values("proj", ascending=False)
                        pool = pool.drop_duplicates(subset=[dedup_key], keep="first")
                    pool = pool.reset_index(drop=True)

                    # Step 7: Remove players who are ineligible for this slate
                    # (OUT/DND/IR status or zero projected minutes).
                    _before_filter = pool.copy()  # keep a snapshot for the diagnostic expander
                    _before = len(pool)
                    pool = _filter_ineligible_players(pool)
                    _removed = _before - len(pool)
                    if _removed:
                        st.caption(f"ℹ️ {_removed} player(s) removed (OUT/DND/IR or 0 proj minutes).")

                    # ── Diagnostic expander (post Step 7) ────────────────────────
                    with st.expander(
                        f"🔍 Pool Diagnostics — {_removed} player(s) dropped",
                        expanded=(_removed > 0),
                    ):
                        # ── Dropped players breakdown ─────────────────────────
                        _dropped_rows = _before_filter[
                            ~_before_filter.index.isin(
                                _before_filter.merge(
                                    pool[["player_name"]],
                                    on="player_name",
                                    how="inner",
                                ).index
                            )
                        ].copy() if _removed > 0 else pd.DataFrame()

                        if not _dropped_rows.empty:
                            # Label the reason for each dropped player
                            def _drop_reason(r):
                                status_val = str(r.get("status", "")).strip().upper()
                                if status_val in _INELIGIBLE_STATUSES:
                                    return f"Status: {r.get('status', '')}"
                                mins_val = r.get("proj_minutes", None)
                                try:
                                    if float(mins_val) <= 0:
                                        return "Zero proj_minutes"
                                except (TypeError, ValueError):
                                    return "Zero proj_minutes"
                                return "Unknown"

                            _dropped_rows["drop_reason"] = _dropped_rows.apply(_drop_reason, axis=1)

                            st.markdown("**Dropped players:**")
                            _diag_show_cols = [
                                c for c in [
                                    "player_name", "status", "proj_minutes", "salary", "drop_reason"
                                ]
                                if c in _dropped_rows.columns
                            ]
                            st.dataframe(
                                _dropped_rows[_diag_show_cols]
                                .sort_values("drop_reason")
                                .reset_index(drop=True),
                                use_container_width=True,
                                hide_index=True,
                            )

                            # Breakdown counts by reason
                            _reason_counts = _dropped_rows["drop_reason"].value_counts()
                            for _reason, _cnt in _reason_counts.items():
                                st.caption(f"  • {_reason}: {_cnt} player(s)")
                        else:
                            st.success("No players were dropped at this step.")

                        st.divider()

                        # ── Projection coverage stats ─────────────────────────
                        st.markdown("**Projection coverage (active pool):**")
                        _total_active = len(pool)

                        # Rolling FP coverage
                        _fp_col = "rolling_fp_5"
                        if _fp_col in pool.columns:
                            _has_rolling = pool[_fp_col].notna() & (pool[_fp_col] != 0)
                            _n_rolling = int(_has_rolling.sum())
                            _n_salary_only = _total_active - _n_rolling
                            st.caption(
                                f"Players with real rolling FP data: **{_n_rolling}** / {_total_active}  "
                                f"(salary-implied only: {_n_salary_only})"
                            )
                        else:
                            st.caption(
                                f"Rolling FP columns not present — all {_total_active} players "
                                "use salary-implied projections only."
                            )

                        # Rolling minutes coverage
                        _min_col = "rolling_min_5"
                        if _min_col in pool.columns:
                            _has_rolling_min = pool[_min_col].notna() & (pool[_min_col] != 0)
                            st.caption(
                                f"Players with real rolling minutes data: "
                                f"**{int(_has_rolling_min.sum())}** / {_total_active}"
                            )

                        # Vegas coverage
                        if "vegas_total" in pool.columns:
                            _has_vegas = pool["vegas_total"].notna()
                            st.caption(
                                f"Players with Vegas total: "
                                f"**{int(_has_vegas.sum())}** / {_total_active}"
                            )

                    st.session_state[f"_hub_pool_{slate_date_str}_{_contest_safe}"] = pool
                    st.session_state[f"_hub_rules_{slate_date_str}_{_contest_safe}"] = parsed_rules
                    st.session_state[f"_hub_draft_group_id_{slate_date_str}_{_contest_safe}"] = draft_group_id
                    _salary_mode = "Historical" if _is_historical else "Live"
                    st.info(f"✅ {_salary_mode} salaries loaded for {slate_date_str}")
                    st.success(f"Loaded {len(pool)} players. Roster: {parsed_rules['slots']}")
                    # Sanity check: warn if draft group changed from what was previously published
                    if draft_group_id and slate.draft_group_id and draft_group_id != slate.draft_group_id:
                        st.warning(
                            f"⚠️ Draft Group changed from {slate.draft_group_id} to {draft_group_id}. "
                            "Verify contest settings and re-publish the slate before building lineups."
                        )
            except Exception as exc:
                st.error(f"Failed to load player pool: {exc}")

    # ── Pool Preview ──────────────────────────────────────────────────────
    hub_pool: Optional[pd.DataFrame] = st.session_state.get(f"_hub_pool_{slate_date_str}_{_contest_safe}")
    hub_rules: Optional[dict] = st.session_state.get(f"_hub_rules_{slate_date_str}_{_contest_safe}")

    if hub_pool is not None:
        # ── Game Selector ─────────────────────────────────────────────────
        all_games = _extract_games(hub_pool)
        is_showdown = (hub_rules or {}).get("is_showdown", False)

        if all_games:
            st.subheader("Game Filter")
            if is_showdown:
                st.caption("Showdown: select exactly 1 game.")
                sel_game = st.selectbox("Game", all_games, key="_hub_game_sd")
                selected_games = [sel_game]
            else:
                selected_games = st.multiselect(
                    "Filter to games (leave empty to keep all)",
                    all_games,
                    default=[],
                    key="_hub_games_multi",
                )

            if selected_games:
                # Determine opponent column
                opp_col = "opp" if "opp" in hub_pool.columns else (
                    "opponent" if "opponent" in hub_pool.columns else None
                )
                if opp_col:
                    filtered_pool = _filter_pool_by_games(hub_pool, selected_games, opp_col)
                    if not filtered_pool.empty:
                        hub_pool = filtered_pool
                        st.caption(f"Showing {len(hub_pool)} players in selected games.")

        st.subheader("Player Pool Preview")
        preview_cols = [c for c in [
            "player_name", "pos", "team", "opp", "opponent", "salary",
            "proj", "floor", "ceil", "proj_minutes", "ownership", "status", "sim_eligible",
            "actual_fp",
        ] if c in hub_pool.columns]
        preview_df = hub_pool[preview_cols].sort_values("proj", ascending=False).copy()
        float_cols = [c for c in ["proj", "floor", "ceil", "proj_minutes", "ownership", "actual_fp"]
                      if c in preview_df.columns]
        preview_df[float_cols] = preview_df[float_cols].round(1)
        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
        )

        # ── Pool Size Gauge ───────────────────────────────────────────────
        pool_count = len(hub_pool)
        pmin, pmax = get_pool_size_range(contest_type_label)
        if pmin <= pool_count <= pmax:
            st.success(
                f"✅ {pool_count} players — in range for {contest_type_label}"
                f" (target {pmin}–{pmax})"
            )
        elif pool_count < pmin:
            st.warning(
                f"⚠️ {pool_count} players — below target for {contest_type_label}"
                f" (need {pmin}–{pmax}). Edge may be too concentrated."
            )
        else:
            st.warning(
                f"⚠️ {pool_count} players — above target for {contest_type_label}"
                f" (target {pmin}–{pmax}). Edge may be diluted."
            )

        if hub_rules:
            with st.expander("Roster Rules", expanded=False):
                st.json(hub_rules)

        # ── External Projections Upload ───────────────────────────────────
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
                        st.success(f"Merged RG data into pool ({len(merged)} rows).")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Failed to read RG CSV: {exc}")

        # ── S1.7 — Late Swap / Refresh ────────────────────────────────────
        st.subheader("Injury / News Refresh (Late Swap)")

        if st.button("🔃 Refresh Injuries & News"):
            _key = st.session_state.get("rapidapi_key", "")
            if not _key:
                st.warning("Tank01 RapidAPI key not configured. Set `TANK01_RAPIDAPI_KEY` in `.streamlit/secrets.toml` or Streamlit Cloud secrets to enable injury refresh.")
            else:
                with st.spinner("Fetching latest injury updates…"):
                    try:
                        updates = fetch_injury_updates(
                            slate_date_str,
                            {"RAPIDAPI_KEY": _key},
                        )
                        if updates:
                            st.session_state["_hub_injury_updates"] = updates
                            st.success(f"Fetched {len(updates)} injury updates.")

                            # Flag affected players in current pool
                            _hub_pool_val = st.session_state.get(f"_hub_pool_{slate_date_str}_{_contest_safe}")
                            pool_copy = _hub_pool_val.copy() if _hub_pool_val is not None else pd.DataFrame()
                            affected = []
                            for update in updates:
                                pname = update.get("player_name", "")
                                status = update.get("status", "")
                                if pname in pool_copy.get("player_name", pd.Series(dtype=str)).values:
                                    affected.append({"player": pname, "status": status})
                            if affected:
                                st.warning(f"⚠️ {len(affected)} players in your pool have status updates:")
                                st.dataframe(pd.DataFrame(affected), use_container_width=True, hide_index=True)
                        else:
                            st.info("No injury updates found.")
                    except Exception as exc:
                        st.error(f"Refresh failed: {exc}")

        # ── Publish Slate ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Publish Slate")
        st.caption("Publishing writes the full slate configuration into SlateState for use by all other pages.")

        if st.button("✅ Publish Slate", type="primary"):
            from datetime import datetime, timezone
            _ts = datetime.now(timezone.utc).isoformat()

            slate.sport = sport
            slate.site = "DK"
            slate.slate_date = slate_date_str
            slate.proj_source = proj_source
            slate.contest_name = contest_type_label
            slate.draft_group_id = st.session_state.get(f"_hub_draft_group_id_{slate_date_str}_{_contest_safe}")

            if hub_rules:
                slate.apply_roster_rules(hub_rules)

            # Store the full contest type label (e.g. "GPP - 150 Max") so
            # downstream pages can read it from SlateState.contest_type.
            # apply_roster_rules sets contest_type to "Classic"/"Showdown Captain",
            # so we overwrite it with the user-selected preset label here.
            slate.contest_type = contest_type_label

            # Publish the filtered pool (game filter applied) rather than the
            # raw session-state pool so Lab and other pages see the same set.
            slate.player_pool = hub_pool
            slate.published = True
            slate.published_at = _ts

            if "Base" not in slate.active_layers:
                slate.active_layers = ["Base"]

            set_slate_state(slate)
            st.success(f"✅ Slate published! {len(hub_pool)} players, cap ${slate.salary_cap:,}, slots: {slate.roster_slots}")
            st.balloons()
    else:
        st.info("Click **Load Player Pool** to load the player pool.")


main()
