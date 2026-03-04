"""Friends / Edge Share – YakOS Sprint 1 page.

Responsibilities (S1.6)
-----------------------
- Read-only view of published lineups by contest type.
- Show Ricky's edge analysis: slate notes, edge labels, core/value/fade
  reasoning.
- Simple lineup builder for friends constrained to Ricky's tagged pool.
- Show last updated timestamps and late swap status.

State read: LineupSetState (published_sets), RickyEdgeState, SlateState
State written: None (read-only for published data; friend builder writes to
               a local session key only)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import (  # noqa: E402
    get_slate_state,
    get_edge_state,
    get_lineup_state,
    get_sim_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TAG_COLORS = {
    "core": "🟢",
    "secondary": "🔵",
    "value": "🟡",
    "punt": "⚪",
    "fade": "🔴",
}


def _render_status_bar(slate: "SlateState") -> None:
    cols = st.columns([2, 2, 2, 2, 4])
    with cols[0]:
        st.metric("Sport", slate.sport or "—")
    with cols[1]:
        st.metric("Date", slate.slate_date or "—")
    with cols[2]:
        st.metric("Site", slate.site or "—")
    with cols[3]:
        st.metric("Contest", slate.contest_type or "—")
    with cols[4]:
        if slate.active_layers:
            chips = " ".join(f"`{l}`" for l in slate.active_layers)
            st.markdown(f"**Layers:** {chips}")


def _render_edge_panel(edge: "RickyEdgeState") -> None:
    """Read-only display of Ricky's edge analysis."""
    st.subheader("🎯 Ricky's Edge Analysis")

    if edge.slate_notes:
        st.markdown(f"**Slate Notes:** {edge.slate_notes}")

    if edge.edge_labels:
        st.markdown("**Edge Labels:**")
        for lbl in edge.edge_labels:
            st.markdown(f"- {lbl}")

    if edge.player_tags:
        st.markdown("**Player Tags:**")
        for tag_type in ["core", "secondary", "value", "punt", "fade"]:
            tagged = edge.get_tagged(tag_type)
            if tagged:
                icon = _TAG_COLORS.get(tag_type, "")
                st.markdown(f"{icon} **{tag_type.title()}:** {', '.join(tagged)}")

    if edge.stacks:
        st.markdown("**Stacks:**")
        for s in edge.stacks:
            players_str = ", ".join(s.get("players", []))
            rationale = s.get("rationale", "")
            st.markdown(f"- **{s.get('team', '')}**: {players_str}" + (f" — {rationale}" if rationale else ""))


def _render_lineup_card(lu_df: pd.DataFrame, lineup_index: int) -> None:
    """Render a single lineup card."""
    if "lineup_index" not in lu_df.columns:
        return
    lu = lu_df[lu_df["lineup_index"] == lineup_index]
    if lu.empty:
        return

    show_cols = [c for c in ["slot", "player_name", "pos", "team", "salary", "proj", "ownership"] if c in lu.columns]
    st.dataframe(lu[show_cols], use_container_width=True, hide_index=True)

    # Footer totals
    if "salary" in lu.columns:
        total_salary = pd.to_numeric(lu["salary"], errors="coerce").sum()
    else:
        total_salary = 0
    if "proj" in lu.columns:
        total_proj = pd.to_numeric(lu["proj"], errors="coerce").sum()
    else:
        total_proj = 0
    st.caption(f"💰 Salary: ${total_salary:,.0f} | 📊 Proj: {total_proj:.1f}")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("👥 Friends / Edge Share")
    st.caption("View Ricky's published lineups and build your own from his tagged pool.")

    slate = get_slate_state()
    edge = get_edge_state()
    lu_state = get_lineup_state()

    _render_status_bar(slate)
    st.divider()

    # ── Published lineup status ───────────────────────────────────────────
    published_labels = lu_state.get_published_labels()

    if not published_labels:
        st.info(
            "No lineups published yet. "
            "Ricky needs to complete the Edge Check and publish lineups in **Build & Publish**."
        )
    else:
        # Show late swap status
        injury_updates = st.session_state.get("_hub_injury_updates", [])
        if injury_updates:
            st.warning(f"⚠️ **Late Swap Active** — {len(injury_updates)} injury updates since last publish.")
        else:
            st.success("✅ No late swap flags.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 1: Ricky's Edge Analysis (read-only)
    # ─────────────────────────────────────────────────────────────────────
    _render_edge_panel(edge)
    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 2: Published Lineups by Contest Type (read-only)
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("📋 Published Lineups")

    if not published_labels:
        st.info("No published lineups available.")
    else:
        tab_labels = published_labels
        tabs = st.tabs(tab_labels)

        for tab, label in zip(tabs, tab_labels):
            with tab:
                pub = lu_state.published_sets.get(label, {})
                pub_df: pd.DataFrame = pub.get("lineups_df", pd.DataFrame())
                pub_ts = pub.get("published_at", "")
                pub_config = pub.get("config", {})

                # Timestamp and metadata
                if pub_ts:
                    st.caption(f"📅 Published: {pub_ts}")
                if pub_config:
                    st.caption(
                        f"Build mode: {pub_config.get('build_mode', '?')} | "
                        f"Archetype: {pub_config.get('archetype', '?')} | "
                        f"# Lineups: {pub_config.get('num_lineups', '?')}"
                    )

                if pub_df.empty:
                    st.info("No lineup data.")
                    continue

                n_lineups = len(pub_df["lineup_index"].unique()) if "lineup_index" in pub_df.columns else 0
                st.markdown(f"**{n_lineups} lineup(s)**")

                # Navigate through lineups
                lu_idx_key = f"_fes_lu_nav_{label}"
                if lu_idx_key not in st.session_state:
                    st.session_state[lu_idx_key] = 0

                nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
                with nav_col1:
                    if st.button("◀", key=f"_fes_prev_{label}") and st.session_state[lu_idx_key] > 0:
                        st.session_state[lu_idx_key] -= 1
                with nav_col2:
                    current_idx = st.session_state[lu_idx_key]
                    unique_idxs = sorted(pub_df["lineup_index"].unique().tolist()) if "lineup_index" in pub_df.columns else [0]
                    st.markdown(f"<center>Lineup {current_idx + 1} of {n_lineups}</center>", unsafe_allow_html=True)
                with nav_col3:
                    if st.button("▶", key=f"_fes_next_{label}") and st.session_state[lu_idx_key] < n_lineups - 1:
                        st.session_state[lu_idx_key] += 1

                cur_lu_idx = st.session_state[lu_idx_key]
                if unique_idxs:
                    actual_idx = unique_idxs[min(cur_lu_idx, len(unique_idxs) - 1)]
                    _render_lineup_card(pub_df, actual_idx)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 3: Friend Lineup Builder
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🛠️ Friend Lineup Builder")
    st.caption(
        "Build your own lineup constrained to Ricky's tagged pool. "
        "Only core / secondary / value players are available."
    )

    pool: pd.DataFrame = slate.player_pool if slate.player_pool is not None else pd.DataFrame()

    if pool.empty:
        st.info("No player pool available. Ricky must publish a slate first.")
        return

    # Filter to Ricky's tagged pool (exclude fades/punts)
    allowed_tags = {"core", "secondary", "value"}
    tagged_players = {p for p, v in edge.player_tags.items() if v.get("tag") in allowed_tags}
    fade_players = edge.get_tagged("fade")

    if tagged_players:
        filtered_pool = pool[
            pool["player_name"].isin(tagged_players) &
            ~pool["player_name"].isin(fade_players)
        ].copy()
        st.caption(f"Showing {len(filtered_pool)} tagged players (core/secondary/value)")
    else:
        # No tags yet — show full pool but exclude fades
        filtered_pool = pool[~pool["player_name"].isin(fade_players)].copy()
        st.caption(f"No tags set yet. Showing all {len(filtered_pool)} non-faded players.")

    if filtered_pool.empty:
        st.info("No players in tagged pool. Ricky needs to tag players in **Ricky Edge**.")
        return

    # Show available pool
    pool_display_cols = [c for c in ["player_name", "pos", "team", "salary", "proj", "ceil", "ownership"] if c in filtered_pool.columns]
    st.dataframe(filtered_pool[pool_display_cols].sort_values("proj", ascending=False),
                 use_container_width=True, hide_index=True)

    # Friend's lineup builder — manual player selection
    st.markdown("**Pick your lineup:**")
    slots = slate.roster_slots or ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    salary_cap = slate.salary_cap or 50000

    player_choices = [""] + sorted(filtered_pool["player_name"].dropna().tolist())
    friend_lineup: list[dict] = []
    total_salary = 0

    with st.form("_fes_builder_form"):
        for slot in slots:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                pick = st.selectbox(slot, player_choices, key=f"_fes_{slot}_pick")
            with col_b:
                if pick and "salary" in filtered_pool.columns:
                    sal = filtered_pool[filtered_pool["player_name"] == pick]["salary"].values
                    sal_val = int(sal[0]) if len(sal) > 0 else 0
                    st.caption(f"${sal_val:,}")
                    total_salary += sal_val
                else:
                    st.caption("")

            if pick:
                friend_lineup.append({"slot": slot, "player_name": pick})

        submitted = st.form_submit_button("📊 Evaluate Lineup")

    if submitted and friend_lineup:
        total_salary = 0
        total_proj = 0.0
        for entry in friend_lineup:
            pname = entry["player_name"]
            row = filtered_pool[filtered_pool["player_name"] == pname]
            if not row.empty:
                total_salary += int(row.iloc[0].get("salary", 0) or 0)
                total_proj += float(row.iloc[0].get("proj", 0) or 0)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            salary_color = "green" if total_salary <= salary_cap else "red"
            st.metric("Total Salary", f"${total_salary:,}")
        with col_b:
            st.metric("Cap Remaining", f"${salary_cap - total_salary:,}")
        with col_c:
            st.metric("Projected Points", f"{total_proj:.1f}")

        if total_salary > salary_cap:
            st.error(f"❌ Over salary cap by ${total_salary - salary_cap:,}. Adjust your picks.")
        elif len(friend_lineup) == len(slots):
            st.success("✅ Valid lineup! You're within the salary cap.")
        else:
            st.warning(f"Fill all {len(slots)} slots to complete your lineup.")


main()
