"""Edge Share – YakOS page.

Responsibilities
----------------
- Read-only view of Ricky's Edge Analysis and Optimizer results.
- Cross-contest Ricky Confidence strip (one pill per contest with data).
- Per-contest blocks: Edge Analysis (left) + Optimizer/lineups (right).

State read: LineupSetState (published_sets), RickyEdgeState, SlateState
State written: None (fully read-only)
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
from yak_core.components import render_lineup_cards_paged  # noqa: E402
from yak_core.edge_metrics import (  # noqa: E402
    compute_ricky_confidence_for_contest,
    get_confidence_color,
)
from yak_core.config import CONTEST_PRESETS  # noqa: E402

# Fixed display order — canonical labels from CONTEST_PRESETS
CONTEST_ORDER = [
    "GPP Main",
    "Cash Main",
    "Showdown",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
            chips = " ".join(f"`{layer}`" for layer in slate.active_layers)
            st.markdown(f"**Layers:** {chips}")


def _render_confidence_strip(edge: "RickyEdgeState") -> None:
    """Render the cross-contest Ricky Confidence strip."""
    available_contests = [c for c in CONTEST_ORDER if c in edge.edge_analysis_by_contest]

    if not available_contests:
        st.info("No edge analysis published yet. Run Edge Analysis on Ricky Edge to populate this view.")
        return

    st.subheader("🎯 Ricky Confidence by Contest")
    cols = st.columns(len(available_contests))
    for col, contest_label in zip(cols, available_contests, strict=True):
        payload = edge.edge_analysis_by_contest[contest_label]
        score = compute_ricky_confidence_for_contest(payload)
        color = get_confidence_color(score)
        with col:
            if color == "green":
                st.success(f"**{contest_label}**\n\nConfidence: {score:.0f}/100")
            elif color == "yellow":
                st.warning(f"**{contest_label}**\n\nConfidence: {score:.0f}/100")
            else:
                st.error(f"**{contest_label}**\n\nConfidence: {score:.0f}/100")


def _render_edge_analysis_col(edge: "RickyEdgeState", contest_label: str) -> None:
    """Render the left (Edge Analysis) column for a single contest block."""
    st.markdown("#### 🎯 Ricky's Edge Analysis")

    payload = edge.edge_analysis_by_contest.get(contest_label)
    if payload is None:
        st.info("No edge analysis for this contest type yet.")
        return

    summary = payload.get("edge_summary", "")
    if summary:
        st.markdown(summary)

    core_value = payload.get("core_value_players", [])
    leverage = payload.get("leverage_players", [])
    fades = payload.get("fade_players", [])

    _PLAYER_COLS = ["Player", "Team", "Salary", "Proj", "Ceil", "Own", "Confidence", "Tag", "Suggestion"]

    def _to_df(players: list) -> pd.DataFrame:
        if not players:
            return pd.DataFrame(columns=_PLAYER_COLS)
        rows = []
        for p in players:
            if not isinstance(p, dict):
                continue
            rows.append({
                "Player": p.get("player_name", p.get("Player", "")),
                "Team": p.get("team", p.get("Team", "")),
                "Salary": p.get("salary", p.get("Salary", "")),
                "Proj": p.get("proj", p.get("Proj", "")),
                "Ceil": p.get("ceil", p.get("Ceil", "")),
                "Own": p.get("own", p.get("Own", p.get("own_pct", ""))),
                "Confidence": p.get("confidence", p.get("Confidence", "")),
                "Tag": p.get("tag", p.get("Tag", "")),
                "Suggestion": p.get("suggestion", p.get("Suggestion", "")),
            })
        return pd.DataFrame(rows, columns=_PLAYER_COLS)

    tab1, tab2, tab3 = st.tabs(["Core & Value", "Leverage", "Fades / Risky Chalk"])
    with tab1:
        df = _to_df(core_value)
        if df.empty:
            st.caption("No core/value players tagged.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    with tab2:
        df = _to_df(leverage)
        if df.empty:
            st.caption("No leverage players tagged.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    with tab3:
        df = _to_df(fades)
        if df.empty:
            st.caption("No fade players tagged.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    warnings = payload.get("contest_fit_warnings", [])
    for w in warnings:
        st.caption(f"⚠️ {w}")


def _render_optimizer_col(
    lu_state: "LineupSetState",
    sim_state: "SimState",
    slate: "SlateState",
    contest_label: str,
) -> None:
    """Render the right (Optimizer) column for a single contest block."""
    st.markdown("#### ⚡ Ricky's Optimizer")

    pub = lu_state.published_sets.get(contest_label)
    if pub is None:
        st.info("No lineups published for this contest type.")
        return

    pub_ts = pub.get("published_at", "")
    if pub_ts:
        st.caption(f"Published: {pub_ts}")

    config = pub.get("config", {})
    if config:
        st.caption(
            f"Mode: {config.get('build_mode', '?')} | "
            f"Lineups: {config.get('num_lineups', '?')}"
        )

    pub_df: pd.DataFrame = pub.get("lineups_df", pd.DataFrame())
    if pub_df.empty:
        st.info("No lineup data.")
        return

    boom_bust_df = pub.get("boom_bust_df")
    exposure_df = pub.get("exposure_df")

    # ── Boom/bust summary strip ────────────────────────────────────────────
    if boom_bust_df is not None and not boom_bust_df.empty:
        n_lineups = len(boom_bust_df)
        n_ab = len(boom_bust_df[boom_bust_df["lineup_grade"].isin(["A", "B"])]) if "lineup_grade" in boom_bust_df.columns else 0
        avg_boom = boom_bust_df["boom_score"].mean() if "boom_score" in boom_bust_df.columns else 0.0
        avg_bust = boom_bust_df["bust_risk"].mean() if "bust_risk" in boom_bust_df.columns else 0.0

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Lineups", n_lineups)
        with m2:
            st.metric("A/B Grades", f"{n_ab}/{n_lineups}")
        with m3:
            st.metric("Avg Boom", f"{avg_boom:.0f}")

        preset = CONTEST_PRESETS.get(contest_label, {})
        if preset.get("tagging_mode") == "ceiling":
            st.caption(f"GPP lineup set — {n_ab} high-ceiling lineups, avg bust risk {avg_bust:.0f}/100")
        else:
            st.caption(f"Cash lineup set — {n_ab} safe-floor lineups, avg bust risk {avg_bust:.0f}/100")

    # Fall back to "GPP_20" — the internal sim key for GPP - 20 Max runs, which is
    # the most common contest type and a reasonable proxy when the exact label is absent.
    pipeline_df = (
        sim_state.pipeline_output.get(contest_label)
        or sim_state.pipeline_output.get("GPP_20")
    )

    render_lineup_cards_paged(
        lineups_df=pub_df,
        sim_results_df=pipeline_df,
        salary_cap=slate.salary_cap,
        nav_key=f"es_{contest_label}",
        boom_bust_df=boom_bust_df,
    )

    # ── Exposure summary expander ──────────────────────────────────────────
    if exposure_df is not None and not exposure_df.empty:
        with st.expander("📊 Exposure Summary", expanded=False):
            display_cols = [c for c in [
                "player", "team", "salary", "your_exposure_pct",
                "field_own_pct", "delta", "leverage_ratio",
            ] if c in exposure_df.columns]
            st.dataframe(
                exposure_df[display_cols].head(25),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("📊 Edge Share")
    st.caption("Ricky's Edge Analysis and Optimizer results — all contest types.")

    slate = get_slate_state()
    edge = get_edge_state()
    lu_state = get_lineup_state()
    sim_state = get_sim_state()

    _render_status_bar(slate)
    st.divider()

    # Determine which contests have data (edge analysis OR published lineups)
    contests_with_data = [
        c for c in CONTEST_ORDER
        if c in edge.edge_analysis_by_contest or c in lu_state.published_sets
    ]

    # Empty state
    if not edge.edge_analysis_by_contest and not lu_state.published_sets:
        st.info(
            "No Edge Share data yet. Run Edge Analysis on Ricky Edge and publish "
            "lineups from Build & Publish."
        )
        return

    # ── Cross-contest Ricky Confidence strip ──────────────────────────────
    _render_confidence_strip(edge)
    st.divider()

    # ── Per-contest blocks ─────────────────────────────────────────────────
    for contest_label in contests_with_data:
        preset = CONTEST_PRESETS.get(contest_label, {})
        desc = preset.get("description", "")
        pool_min = preset.get("pool_size_min", "?")
        pool_max = preset.get("pool_size_max", "?")
        own_min = preset.get("target_avg_ownership_min")
        own_max = preset.get("target_avg_ownership_max")

        st.markdown(f"### {contest_label}")
        caption_parts = []
        if desc:
            caption_parts.append(desc)
        caption_parts.append(f"Target pool: {pool_min}–{pool_max}")
        if own_min is not None:
            caption_parts.append(f"Avg own: {own_min}–{own_max}%")
        st.caption(" | ".join(caption_parts))

        left_col, right_col = st.columns(2)
        with left_col:
            _render_edge_analysis_col(edge, contest_label)
        with right_col:
            _render_optimizer_col(lu_state, sim_state, slate, contest_label)

        st.divider()

    # ── Friend lineup builder ─────────────────────────────────────────────
    st.warning(
        "⚠️ **Friend Lineup Builder not yet wired.** "
        "The simple lineup builder constrained to Ricky's tagged pool (S1.6) "
        "is pending implementation. Use Build & Publish to build your lineups."
    )


main()
