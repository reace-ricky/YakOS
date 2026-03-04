"""Slate Hub – YakOS Sprint 1 page.

Responsibilities
----------------
- Load DK contests by sport / date and group them like the DK lobby
  (Main, Late Night, Showdown by game, Turbo).
- On slate selection, pull draftables + game-type rules and
  auto-configure roster template, salary cap, scoring, and captain
  multipliers into SlateState.
- Merge projections (floor / median / ceiling) and ownership from the
  chosen sources, surfacing projected minutes as a visible column.
- Provide a "Publish Slate" action that writes the full configuration
  into SlateState.
- S1.7: Refresh action that re-pulls news / injuries, updates projections,
  and flags affected lineups per contest type.

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
    fetch_game_type_rules,
    parse_roster_rules,
    build_contest_scoped_pool,
    is_dk_integration_enabled,
    DK_GAME_TYPE_LABELS,
)
from yak_core.projections import (  # noqa: E402
    salary_implied_proj,
    yakos_fp_projection,
    yakos_minutes_projection,
    yakos_ownership_projection,
)
from yak_core.sims import compute_sim_eligible  # noqa: E402
from yak_core.live import fetch_injury_updates  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOBBY_GROUP_ORDER = ["Main", "Late Night", "Turbo", "Showdown"]


def _group_contests(contests_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group contests by lobby category (Main / Late Night / Showdown / Turbo)."""
    groups: dict[str, list] = {g: [] for g in _LOBBY_GROUP_ORDER}
    other: list = []
    for _, row in contests_df.iterrows():
        name = str(row.get("name", "")).lower()
        game_type = int(row.get("gameTypeId", 1))
        if game_type in (96, 114) or "showdown" in name:
            groups["Showdown"].append(row)
        elif "turbo" in name or "turbo" in str(row.get("contestType", "")).lower():
            groups["Turbo"].append(row)
        elif "late" in name or "late night" in name:
            groups["Late Night"].append(row)
        else:
            groups["Main"].append(row)
    result = {}
    for label in _LOBBY_GROUP_ORDER:
        rows = groups[label]
        if rows:
            result[label] = pd.DataFrame(rows)
    if other:
        result["Other"] = pd.DataFrame(other)
    return result


def _build_default_pool(draftables_df: pd.DataFrame) -> pd.DataFrame:
    """Add salary-implied projections to a raw draftables DataFrame."""
    pool = draftables_df.copy()
    if "salary" not in pool.columns:
        pool["salary"] = 0
    pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce").fillna(0)

    # Salary-implied projections
    pool["proj"] = salary_implied_proj(pool["salary"])

    # YakOS projections per player
    floors, ceils, mins_proj, own_proj = [], [], [], []
    for _, row in pool.iterrows():
        feats = {"salary": float(row.get("salary", 0) or 0)}
        fp_res = yakos_fp_projection(feats)
        min_res = yakos_minutes_projection(feats)
        own_res = yakos_ownership_projection(feats)
        floors.append(fp_res.get("floor", fp_res["proj"] * 0.7))
        ceils.append(fp_res.get("ceil", fp_res["proj"] * 1.4))
        mins_proj.append(min_res.get("proj_minutes", 0.0))
        own_proj.append(own_res.get("proj_own", 0.0))

    pool["floor"] = floors
    pool["ceil"] = ceils
    pool["proj_minutes"] = mins_proj
    pool["ownership"] = own_proj

    pool = compute_sim_eligible(pool)
    return pool


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


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🏀 Slate Hub")
    st.caption("Select a DK contest, configure the slate, and publish it.")

    slate = get_slate_state()
    _render_status_bar(slate)
    st.divider()

    # ── Inputs ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        sport = st.selectbox("Sport", ["NBA", "PGA"], index=0 if slate.sport == "NBA" else 1)
    with col2:
        from zoneinfo import ZoneInfo
        _today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        slate_date = st.date_input("Date", value=pd.to_datetime(slate.slate_date or _today))
        slate_date_str = str(slate_date)
    with col3:
        proj_source = st.selectbox(
            "Projection Source",
            ["salary_implied", "yakos", "RotoGrinders", "FantasyPros"],
            index=["salary_implied", "yakos", "RotoGrinders", "FantasyPros"].index(
                slate.proj_source if slate.proj_source in ["salary_implied", "yakos", "RotoGrinders", "FantasyPros"]
                else "salary_implied"
            ),
        )

    # ── Fetch DK Lobby ────────────────────────────────────────────────────
    if st.button("🔄 Load DK Lobby", type="primary"):
        with st.spinner("Fetching DK lobby contests…"):
            try:
                contests_df = fetch_dk_lobby_contests(sport)
                st.session_state["_hub_contests_df"] = contests_df
                st.session_state["_hub_sport"] = sport
                st.session_state["_hub_date"] = slate_date_str
                st.success(f"Loaded {len(contests_df)} contests.")
            except Exception as exc:
                st.error(f"Failed to load DK lobby: {exc}")

    # ── Contest Picker ────────────────────────────────────────────────────
    contests_df: Optional[pd.DataFrame] = st.session_state.get("_hub_contests_df")

    if contests_df is not None and not contests_df.empty:
        st.subheader("Select Contest")
        groups = _group_contests(contests_df)

        if not groups:
            st.info("No contests found. Try a different date or sport.")
            return

        tab_labels = list(groups.keys())
        tabs = st.tabs(tab_labels)

        selected_contest_row = None
        for tab, label in zip(tabs, tab_labels):
            with tab:
                grp_df = groups[label]
                if grp_df.empty:
                    st.info("No contests in this category.")
                    continue

                # Show selectable table
                display_cols = [c for c in ["name", "entries", "totalPayoutAmount", "gameTypeId", "draftGroupId"] if c in grp_df.columns]
                if not display_cols:
                    display_cols = list(grp_df.columns[:5])
                st.dataframe(grp_df[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

                contest_names = grp_df["name"].tolist() if "name" in grp_df.columns else grp_df.index.tolist()
                chosen = st.selectbox(f"Pick contest ({label})", contest_names, key=f"_hub_pick_{label}")
                if chosen:
                    mask = grp_df["name"] == chosen if "name" in grp_df.columns else grp_df.index == chosen
                    if mask.any():
                        selected_contest_row = grp_df[mask].iloc[0]
                        st.session_state["_hub_selected_row"] = selected_contest_row
                        st.session_state["_hub_selected_label"] = label

        # Pull draftables when a contest is selected
        sel_row = st.session_state.get("_hub_selected_row")
        if sel_row is not None:
            draft_group_id = int(sel_row.get("draftGroupId", 0) or 0)
            game_type_id = int(sel_row.get("gameTypeId", 1) or 1)
            contest_id = int(sel_row.get("id", 0) or sel_row.get("contestId", 0) or 0)

            if st.button("📥 Load Draftables + Rules", type="secondary"):
                with st.spinner("Fetching draftables and roster rules…"):
                    try:
                        draftables_df = fetch_dk_draftables(draft_group_id)
                        rules_json = fetch_game_type_rules(game_type_id)
                        parsed_rules = parse_roster_rules(rules_json)
                        pool = _build_default_pool(draftables_df)

                        st.session_state["_hub_pool"] = pool
                        st.session_state["_hub_rules"] = parsed_rules
                        st.session_state["_hub_draft_group_id"] = draft_group_id
                        st.session_state["_hub_game_type_id"] = game_type_id
                        st.session_state["_hub_contest_id"] = contest_id
                        st.session_state["_hub_contest_name"] = str(sel_row.get("name", ""))
                        st.success(f"Loaded {len(pool)} draftables. Roster: {parsed_rules['slots']}")
                    except Exception as exc:
                        st.error(f"Failed to load draftables: {exc}")

    # ── Pool Preview ──────────────────────────────────────────────────────
    hub_pool: Optional[pd.DataFrame] = st.session_state.get("_hub_pool")
    hub_rules: Optional[dict] = st.session_state.get("_hub_rules")

    if hub_pool is not None:
        st.subheader("Player Pool Preview")
        preview_cols = [c for c in [
            "player_name", "pos", "team", "opponent", "salary",
            "proj", "floor", "ceil", "proj_minutes", "ownership", "status", "sim_eligible",
        ] if c in hub_pool.columns]
        st.dataframe(
            hub_pool[preview_cols].sort_values("proj", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        if hub_rules:
            with st.expander("Roster Rules", expanded=False):
                st.json(hub_rules)

        # ── Projection source merge ───────────────────────────────────────
        with st.expander("External Projections Upload", expanded=False):
            st.caption("Upload RotoGrinders or FantasyPros CSV to merge projections.")
            rg_file = st.file_uploader("RotoGrinders CSV", type="csv", key="_hub_rg_upload")
            if rg_file:
                try:
                    rg_df = pd.read_csv(rg_file)
                    st.session_state["_hub_rg_df"] = rg_df
                    st.success(f"RotoGrinders: {len(rg_df)} rows")
                except Exception as exc:
                    st.error(f"Failed to read RG CSV: {exc}")

            fp_file = st.file_uploader("FantasyPros CSV", type="csv", key="_hub_fp_upload")
            if fp_file:
                try:
                    fp_df = pd.read_csv(fp_file)
                    st.session_state["_hub_fp_df"] = fp_df
                    st.success(f"FantasyPros: {len(fp_df)} rows")
                except Exception as exc:
                    st.error(f"Failed to read FP CSV: {exc}")

        # ── S1.7 — Late Swap / Refresh ────────────────────────────────────
        st.subheader("Injury / News Refresh (Late Swap)")
        rapidapi_key = st.text_input(
            "Tank01 RapidAPI Key",
            value=st.session_state.get("rapidapi_key", ""),
            type="password",
            key="_hub_rapidapi_key",
        )
        if rapidapi_key:
            st.session_state["rapidapi_key"] = rapidapi_key

        if st.button("🔃 Refresh Injuries & News"):
            _key = st.session_state.get("rapidapi_key", "")
            if not _key:
                st.warning("Enter your Tank01 RapidAPI key to refresh injuries.")
            else:
                with st.spinner("Fetching latest injury updates…"):
                    try:
                        updates = fetch_injury_updates(_key)
                        if updates:
                            st.session_state["_hub_injury_updates"] = updates
                            st.success(f"Fetched {len(updates)} injury updates.")

                            # Flag affected players in current pool
                            pool_copy = hub_pool.copy()
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
            slate.draft_group_id = st.session_state.get("_hub_draft_group_id")
            slate.game_type_id = st.session_state.get("_hub_game_type_id")
            slate.contest_id = st.session_state.get("_hub_contest_id")
            slate.contest_name = st.session_state.get("_hub_contest_name", "")

            if hub_rules:
                slate.apply_roster_rules(hub_rules)

            slate.player_pool = hub_pool
            slate.published = True
            slate.published_at = _ts

            if "Base" not in slate.active_layers:
                slate.active_layers = ["Base"]

            set_slate_state(slate)
            st.success(f"✅ Slate published! {len(hub_pool)} players, cap ${slate.salary_cap:,}, slots: {slate.roster_slots}")
            st.balloons()
    else:
        if st.session_state.get("_hub_contests_df") is not None:
            st.info("Select a contest and click **Load Draftables + Rules** to proceed.")
        else:
            st.info("Click **Load DK Lobby** to browse available contests.")

    # Demo mode banner when DK integration is disabled
    if not is_dk_integration_enabled():
        st.warning(
            "⚠️ **DK integration is disabled** (set `DK_INTEGRATION_ENABLED=true` to enable). "
            "Showing mock data."
        )


main()
