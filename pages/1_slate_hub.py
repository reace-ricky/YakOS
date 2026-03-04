"""Slate Hub – YakOS Sprint 1 page.

Responsibilities
----------------
- Data Mode toggle: Live (DK draftables API) or Historical (local parquet/CSV).
- Contest Type picker from CONTEST_PRESETS (replaces DK lobby contest picker).
- Projection Source picker: salary_implied / regression / model / blend / parquet.
- Load Player Pool via DK draftables (Live) or local data file (Historical).
- Run full apply_projections() pipeline on the loaded pool.
- Optional RG CSV merge after pool load.
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
from yak_core.dk_ingest import fetch_dk_draftables  # noqa: E402
from yak_core.projections import (  # noqa: E402
    yakos_fp_projection,
    yakos_minutes_projection,
    yakos_ownership_projection,
    apply_projections,
    load_historical_slate,
)
from yak_core.rg_loader import load_rg_projections, merge_rg_with_pool  # noqa: E402
from yak_core.config import (  # noqa: E402
    CONTEST_PRESETS,
    CONTEST_PRESET_LABELS,
    merge_config,
    YAKOS_ROOT,
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

_PROJ_SOURCES = ["salary_implied", "regression", "model", "blend", "parquet"]

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


def _enrich_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Add floor/ceil/proj_minutes/ownership from YakOS per-player models.

    Preserves existing floor/ceil columns when already present.
    """
    has_floor = "floor" in pool.columns and pool["floor"].notna().any()
    has_ceil = "ceil" in pool.columns and pool["ceil"].notna().any()

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

    if not has_floor:
        pool["floor"] = floors
    if not has_ceil:
        pool["ceil"] = ceils
    pool["proj_minutes"] = mins_proj
    pool["ownership"] = own_proj

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


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🏀 Slate Hub")
    st.caption("Configure the slate and publish it.")

    slate = get_slate_state()
    _render_status_bar(slate)
    st.divider()

    # ── Row 1: Sport, Date, Data Mode ─────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        sport = st.selectbox("Sport", ["NBA", "PGA"], index=0 if slate.sport == "NBA" else 1)
    with col2:
        from zoneinfo import ZoneInfo
        _today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        slate_date = st.date_input("Date", value=pd.to_datetime(slate.slate_date or _today))
        slate_date_str = str(slate_date)
    with col3:
        data_mode = st.selectbox("Data Mode", ["Live", "Historical"], index=0)

    # ── Row 2: Contest Type ────────────────────────────────────────────────
    contest_type_label = st.selectbox("Contest Type", CONTEST_PRESET_LABELS)
    preset = CONTEST_PRESETS[contest_type_label]
    st.caption(preset.get("description", ""))

    # ── Row 3: Projection Source ───────────────────────────────────────────
    _proj_idx = _PROJ_SOURCES.index(slate.proj_source) if slate.proj_source in _PROJ_SOURCES else 0
    proj_source = st.selectbox("Projection Source", _PROJ_SOURCES, index=_proj_idx)

    # ── Row 4: Live → Draft Group ID / Historical → auto-detect ───────────
    draft_group_id: Optional[int] = None
    if data_mode == "Live":
        dg_val = st.number_input(
            "Draft Group ID",
            min_value=0,
            step=1,
            value=int(slate.draft_group_id or 0),
            help="DraftKings draft group ID (visible in DK contest URLs).",
        )
        draft_group_id = int(dg_val) if dg_val > 0 else None
    else:
        date_compact = slate_date_str.replace("-", "")
        st.info(
            f"Historical mode: will look for `tank_opt_pool_{date_compact}.parquet` "
            f"or `*DK{date_compact}*.csv` in the `data/` folder."
        )

    # ── Row 5: Load Player Pool ────────────────────────────────────────────
    if st.button("📥 Load Player Pool", type="primary"):
        with st.spinner("Loading player pool…"):
            try:
                if data_mode == "Historical":
                    pool = load_historical_slate(slate_date_str, YAKOS_ROOT)
                    if pool.empty:
                        st.error(
                            f"No historical data found for {slate_date_str}. "
                            f"Make sure a `tank_opt_pool_{date_compact}.parquet` or "
                            f"`*DK{date_compact}*.csv` file exists in the `data/` folder."
                        )
                        return
                    parsed_rules = _rules_from_preset(preset)
                else:
                    if not draft_group_id:
                        st.warning("Enter a Draft Group ID to load a live slate.")
                        return
                    pool = fetch_dk_draftables(draft_group_id)
                    parsed_rules = _rules_from_preset(preset)

                # Normalize salary column
                if "salary" not in pool.columns:
                    pool["salary"] = 0
                pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce").fillna(0)

                # Apply full projection pipeline
                cfg = merge_config({
                    "PROJ_SOURCE": proj_source,
                    "SLATE_DATE": slate_date_str,
                    "CONTEST_TYPE": preset["internal_contest"],
                })
                pool = apply_projections(pool, cfg)

                # Add floor/ceil/minutes/ownership per player
                pool = _enrich_pool(pool)

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
        st.subheader("Player Pool Preview")
        preview_cols = [c for c in [
            "player_name", "pos", "team", "opp", "salary",
            "proj", "floor", "ceil", "proj_minutes", "ownership", "status", "sim_eligible",
            "actual_fp",
        ] if c in hub_pool.columns]
        st.dataframe(
            hub_pool[preview_cols].sort_values("proj", ascending=False),
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
                        merged = merge_rg_with_pool(hub_pool, rg_df)
                        st.session_state["_hub_pool"] = merged
                        st.success(f"Merged RG data into pool ({len(merged)} rows).")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Failed to read RG CSV: {exc}")

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
            slate.contest_name = contest_type_label
            slate.draft_group_id = st.session_state.get("_hub_draft_group_id")

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
        st.info("Click **Load Player Pool** to load the player pool.")


main()
