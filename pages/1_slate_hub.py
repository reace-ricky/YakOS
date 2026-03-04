"""Slate Hub – YakOS Sprint 1 page.

Responsibilities
----------------
- Date + Sport picker (date alone determines live vs historical — no Data Mode toggle).
- Contest Type picker from CONTEST_PRESETS.
- Fetch DK Lobby Contests → auto-resolves draft_group_id by matching contest type.
- Load Player Pool via DK draftables API (always API-first, no local files).
  Optionally enriches with Tank01 stats when TANK01_RAPIDAPI_KEY secret is present.
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
    merge_config,
    DK_POS_SLOTS,
    DK_LINEUP_SIZE,
    SALARY_CAP,
    DK_SHOWDOWN_SLOTS,
    DK_SHOWDOWN_LINEUP_SIZE,
)
from yak_core.sims import compute_sim_eligible  # noqa: E402
from yak_core.live import fetch_injury_updates  # noqa: E402

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
    """
    has_floor = "floor" in pool.columns and pool["floor"].notna().any()
    has_ceil = "ceil" in pool.columns and pool["ceil"].notna().any()

    floors, ceils, mins_proj = [], [], []
    for _, row in pool.iterrows():
        feats = {"salary": float(row.get("salary", 0) or 0)}
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

    # Pool-level ownership model (salary rank + position scarcity).
    # Replaces the broken per-row yakos_ownership_projection call which
    # required proj and rg_ownership features we don't have here.
    pool = apply_ownership(pool)

    return compute_sim_eligible(pool)


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


def _match_contests_to_preset(lobby_df: pd.DataFrame, preset: dict) -> pd.DataFrame:
    """Filter lobby contests to those whose game_type_id matches the preset's slate type."""
    if lobby_df.empty:
        return lobby_df
    is_showdown = preset.get("slate_type") == "Showdown Captain"
    if is_showdown:
        mask = lobby_df["game_type_id"].isin(_SHOWDOWN_GAME_TYPE_IDS)
    else:
        mask = ~lobby_df["game_type_id"].isin(_SHOWDOWN_GAME_TYPE_IDS)
    return lobby_df[mask].drop_duplicates(subset=["draft_group_id"]).reset_index(drop=True)


def _auto_pick_best_contest(lobby_df: pd.DataFrame, preset: dict) -> Optional[int]:
    """Pick the best-matching DraftKings draft_group_id for the given preset.

    Filters lobby contests to match the preset's slate type (Showdown vs. Classic),
    then returns the draft_group_id of the contest with the highest prize pool.
    Returns None if no matching contest is found.
    """
    matched = _match_contests_to_preset(lobby_df, preset)
    if matched.empty:
        return None
    best_row = matched.sort_values("prize_pool", ascending=False).iloc[0]
    return int(best_row["draft_group_id"])


def _filter_lobby_by_date(lobby_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """Filter lobby contests to those whose start_time falls on target_date (YYYY-MM-DD).

    The DK lobby returns all upcoming contests (potentially days away).
    This ensures we only auto-pick from contests actually scheduled for
    the user's selected slate date.
    """
    if lobby_df.empty or "start_time" not in lobby_df.columns:
        return lobby_df
    try:
        st_parsed = pd.to_datetime(lobby_df["start_time"], errors="coerce", utc=True)
        # Convert UTC to US/Eastern for date comparison (NBA games are evening ET)
        from zoneinfo import ZoneInfo
        st_eastern = st_parsed.dt.tz_convert(ZoneInfo("America/New_York"))
        mask = st_eastern.dt.strftime("%Y-%m-%d") == target_date
        filtered = lobby_df[mask].reset_index(drop=True)
        return filtered
    except Exception:
        return lobby_df

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
        slate_date = st.date_input("Date", value=pd.to_datetime(slate.slate_date or _today))
        slate_date_str = str(slate_date)

    # Read Tank01 RapidAPI key from secrets (not prompted on this page)
    rapidapi_key = st.secrets.get("TANK01_RAPIDAPI_KEY")
    if rapidapi_key:
        st.session_state["rapidapi_key"] = rapidapi_key
    else:
        # Show a dismissible info banner only when the key is absent.
        # Once dismissed it stays hidden for the remainder of the session.
        if not st.session_state.get("_tank01_banner_dismissed", False):
            banner_col, dismiss_col = st.columns([6, 1])
            with banner_col:
                st.info(
                    "ℹ️ Tank01 stats enrichment and injury refresh are disabled. "
                    "Set `TANK01_RAPIDAPI_KEY` in `.streamlit/secrets.toml` or "
                    "Streamlit Cloud secrets to enable."
                )
            with dismiss_col:
                if st.button("Dismiss", key="_dismiss_tank01_banner", help="Hide this message for the session"):
                    st.session_state["_tank01_banner_dismissed"] = True
                    st.rerun()

    # ── Row 2: Contest Type ────────────────────────────────────────────────
    contest_type_label = st.selectbox("Contest Type", CONTEST_PRESET_LABELS)
    preset = CONTEST_PRESETS[contest_type_label]
    st.caption(preset.get("description", ""))

    # Projection model is always YakOS Model
    proj_source = "model"

    # ── DK lobby cache (used both by debug UI and auto-pick) ──────────────
    lobby_key = f"_hub_lobby_{sport}_{slate_date_str}"
    lobby_df: Optional[pd.DataFrame] = st.session_state.get(lobby_key)

    # ── Debug Mode: DK Contest Selection (admin only) ─────────────────────
    # Show the full DK contest selection UI only when debug_mode is enabled.
    # In normal operation the contest is resolved automatically.
    _debug_mode = st.session_state.get("debug_mode", False)
    _debug_draft_group_id: Optional[int] = None

    if _debug_mode:
        st.subheader("DK Contest Selection (Debug)")
        _is_live = slate_date_str == _today
        if _is_live:
            st.caption("📡 Live slate — fetching from DK lobby.")
        else:
            st.caption(f"📂 Historical slate — date: {slate_date_str}. Enter Draft Group ID directly.")

        col_fetch, col_clear = st.columns([2, 1])
        with col_fetch:
            if st.button("🔍 Fetch Contests from DK", help="Pulls current DK lobby contests for this sport."):
                with st.spinner("Fetching DK lobby…"):
                    try:
                        fetched = fetch_dk_lobby_contests(sport)
                        st.session_state[lobby_key] = fetched
                        lobby_df = fetched
                        if fetched.empty:
                            st.warning("No contests found in DK lobby for this sport/date.")
                        else:
                            st.success(f"Found {len(fetched)} contests.")
                    except Exception as exc:
                        st.error(f"Failed to fetch DK lobby: {exc}")
        with col_clear:
            if lobby_df is not None and st.button("Clear"):
                st.session_state.pop(lobby_key, None)
                lobby_df = None
                st.rerun()

        if lobby_df is not None and not lobby_df.empty:
            matched = _match_contests_to_preset(lobby_df, preset)
            if not matched.empty:
                contest_options = {
                    f"{row['name']} (DG {row['draft_group_id']})": int(row["draft_group_id"])
                    for _, row in matched.iterrows()
                }
                selected_label = st.selectbox("Select Contest", list(contest_options.keys()))
                _debug_draft_group_id = contest_options[selected_label]
                st.caption(f"Draft Group ID: **{_debug_draft_group_id}**")
            else:
                st.info("No contests matched the selected Contest Type. Enter Draft Group ID manually.")

        with st.expander("Manual Draft Group ID override", expanded=_debug_draft_group_id is None):
            dg_val = st.number_input(
                "Draft Group ID",
                min_value=0,
                step=1,
                value=int(_debug_draft_group_id or slate.draft_group_id or 0),
                help="DraftKings draft group ID (visible in DK contest URLs). Overrides lobby selection.",
                key="_hub_dg_manual",
            )
            if dg_val > 0:
                _debug_draft_group_id = int(dg_val)

    # ── Row 5: Load Player Pool ────────────────────────────────────────────
    if st.button("📥 Load Player Pool", type="primary"):
        # Resolve draft_group_id:
        # • Debug mode  → use the manually selected / entered value.
        # • Normal mode → auto-fetch the DK lobby and pick the highest prize-pool
        #                 contest that matches the selected Contest Type.
        draft_group_id: Optional[int] = _debug_draft_group_id if _debug_mode else None

        with st.spinner("Loading player pool…"):
            try:
                # Auto-pick contest when no override is set
                if not draft_group_id:
                    _lobby = lobby_df
                    if _lobby is None:
                        _lobby = fetch_dk_lobby_contests(sport)
                        st.session_state[lobby_key] = _lobby
                                                _lobby = _filter_lobby_by_date(_lobby, slate_date_str)
                    draft_group_id = _auto_pick_best_contest(_lobby, preset)
                    if draft_group_id:
                        st.caption(f"ℹ️ Auto-selected Draft Group ID: **{draft_group_id}**")

                if not draft_group_id:
                    st.warning(
                        "Could not determine a Draft Group ID for the selected Contest Type. "
                        "No matching contests were found in the DK lobby. "
                        "Try a different sport, date, or contest type, or check your network connection."
                    )
                else:
                    # Step 1: Fetch DK draftables (salaries, positions, teams)
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

                    # Step 3: Optionally enrich with Tank01 stats (game logs, rolling avgs, Vegas)
                    _api_key = st.session_state.get("rapidapi_key", "")
                    if _api_key:
                        try:
                            from yak_core.live import fetch_live_opt_pool
                            tank01_pool = fetch_live_opt_pool(
                                slate_date_str,
                                {"RAPIDAPI_KEY": _api_key},
                            )
                            if not tank01_pool.empty:
                                # Rename 'proj' → 'tank01_proj' before merge to preserve the
                                # DK salary-based proj that will be overwritten by apply_projections.
                                if "proj" in tank01_pool.columns and "tank01_proj" not in tank01_pool.columns:
                                    tank01_pool = tank01_pool.rename(columns={"proj": "tank01_proj"})
                                # Select only useful columns for the merge
                                merge_cols = ["player_name"]
                                for col in ("opp", "opponent", "tank01_proj", "own_proj", "actual_fp"):
                                    if col in tank01_pool.columns:
                                        merge_cols.append(col)
                                pool = pool.merge(
                                    tank01_pool[merge_cols],
                                    on="player_name",
                                    how="left",
                                    suffixes=("", "_tank01"),
                                )
                                st.caption(f"✅ Tank01 stats merged for {len(tank01_pool)} players.")
                        except Exception as t01_exc:
                            st.caption(f"ℹ️ Tank01 stats not available: {t01_exc}")

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

                    st.session_state["_hub_pool"] = pool
                    st.session_state["_hub_rules"] = parsed_rules
                    st.session_state["_hub_draft_group_id"] = draft_group_id
                    st.success(f"Loaded {len(pool)} players. Roster: {parsed_rules['slots']}")
            except Exception as exc:
                st.error(f"Failed to load player pool: {exc}")

    # ── Pool Preview ──────────────────────────────────────────────────────
    hub_pool: Optional[pd.DataFrame] = st.session_state.get("_hub_pool")
    hub_rules: Optional[dict] = st.session_state.get("_hub_rules")

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
                        merged = merge_rg_with_pool(st.session_state["_hub_pool"], rg_df)
                        st.session_state["_hub_pool"] = merged
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
                            pool_copy = st.session_state["_hub_pool"].copy()
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
            slate.slate_date = slate_date_str
            slate.proj_source = proj_source
            slate.contest_name = contest_type_label
            slate.draft_group_id = st.session_state.get("_hub_draft_group_id")

            if hub_rules:
                slate.apply_roster_rules(hub_rules)

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
