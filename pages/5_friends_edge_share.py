"""Edge Share – YakOS page.

Responsibilities
----------------
- Two-tab layout: Ricky's Edge Analysis (writeup + top lineups) and Optimizer
  (full paginated lineup browser per contest type).
- Clean, minimal UI — no clutter.

State read: LineupSetState (published_sets), RickyEdgeState, SlateState, SimState
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
from yak_core.components import render_premium_lineup_card, render_premium_cards_paged  # noqa: E402
from yak_core.edge_metrics import (  # noqa: E402
    compute_ricky_confidence_for_contest,
    get_confidence_color,
)
from yak_core.config import CONTEST_PRESETS, UI_CONTEST_LABELS, UI_CONTEST_MAP  # noqa: E402

# Internal preset labels in display order
_CONTEST_ORDER = [UI_CONTEST_MAP[k] for k in UI_CONTEST_LABELS]
# Reverse map: preset label → short UI label  ("GPP Main" → "GPP")
_LABEL_SHORT = {v: k for k, v in UI_CONTEST_MAP.items()}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_strip(slate: "SlateState") -> None:
    """Compact one-line status bar."""
    parts = []
    if slate.sport:
        parts.append(f"**{slate.sport}**")
    if slate.slate_date:
        parts.append(slate.slate_date)
    if slate.site:
        parts.append(slate.site)
    if parts:
        st.caption(" · ".join(parts))


def _confidence_pills(edge: "RickyEdgeState") -> None:
    """Render compact confidence pills across contests that have data."""
    available = [c for c in _CONTEST_ORDER if c in edge.edge_analysis_by_contest]
    if not available:
        return

    cols = st.columns(len(available))
    for col, contest_label in zip(cols, available, strict=True):
        payload = edge.edge_analysis_by_contest[contest_label]
        score = compute_ricky_confidence_for_contest(payload)
        color = get_confidence_color(score)
        short = _LABEL_SHORT.get(contest_label, contest_label)
        with col:
            if color == "green":
                st.success(f"**{short}** — {score:.0f}/100")
            elif color == "yellow":
                st.warning(f"**{short}** — {score:.0f}/100")
            else:
                st.error(f"**{short}** — {score:.0f}/100")


def _render_edge_writeup(edge: "RickyEdgeState", contest_label: str) -> None:
    """Render edge analysis writeup for one contest type."""
    payload = edge.edge_analysis_by_contest.get(contest_label)
    if payload is None:
        st.caption("No edge analysis for this contest type yet.")
        return

    summary = payload.get("edge_summary", "")
    if summary:
        st.markdown(summary)

    core_value = payload.get("core_value_players", [])
    leverage = payload.get("leverage_players", [])
    fades = payload.get("fade_players", [])

    _COLS = ["Player", "Team", "Salary", "Proj", "Own", "Confidence", "Tag"]

    def _to_df(players: list) -> pd.DataFrame:
        if not players:
            return pd.DataFrame(columns=_COLS)
        rows = []
        for p in players:
            if not isinstance(p, dict):
                continue
            rows.append({
                "Player": p.get("player_name", p.get("Player", "")),
                "Team": p.get("team", p.get("Team", "")),
                "Salary": p.get("salary", p.get("Salary", "")),
                "Proj": p.get("proj", p.get("Proj", "")),
                "Own": p.get("own", p.get("Own", p.get("own_pct", ""))),
                "Confidence": p.get("confidence", p.get("Confidence", "")),
                "Tag": p.get("tag", p.get("Tag", "")),
            })
        return pd.DataFrame(rows, columns=_COLS)

    # Show all three player groups inline — compact
    for group_label, group_data, empty_msg in [
        ("Core & Value", core_value, "No core/value plays tagged."),
        ("Leverage", leverage, "No leverage plays tagged."),
        ("Fades", fades, "No fades tagged."),
    ]:
        df = _to_df(group_data)
        if df.empty:
            continue
        st.markdown(f"**{group_label}**")
        st.dataframe(df, use_container_width=True, hide_index=True, height=min(38 * len(df) + 38, 300))

    warnings = payload.get("contest_fit_warnings", [])
    for w in warnings:
        st.caption(f"⚠️ {w}")


def _get_best_lineup(
    lu_state: "LineupSetState",
    sim_state: "SimState",
    contest_label: str,
) -> tuple:
    """Return (lineup_rows_df, sim_metrics_dict, boom_bust_dict) for the #1 lineup.

    Returns (None, None, None) if no data.
    """
    pub = lu_state.published_sets.get(contest_label)
    if pub is None:
        return None, None, None

    pub_df: pd.DataFrame = pub.get("lineups_df", pd.DataFrame())
    if pub_df.empty:
        return None, None, None

    boom_bust_df = pub.get("boom_bust_df")
    pipeline_df = (
        sim_state.pipeline_output.get(contest_label)
        or sim_state.pipeline_output.get("GPP_20")
    )

    # Determine best lineup — prefer highest boom_score, else first
    best_idx = 0
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        if "boom_score" in boom_bust_df.columns:
            best_idx = int(boom_bust_df.sort_values("boom_score", ascending=False).iloc[0]["lineup_index"])
        else:
            best_idx = int(boom_bust_df.iloc[0]["lineup_index"])
    elif "lineup_index" in pub_df.columns:
        best_idx = int(pub_df["lineup_index"].min())

    lu_rows = pub_df[pub_df["lineup_index"] == best_idx] if "lineup_index" in pub_df.columns else pub_df

    # Sim metrics
    sim_metrics = {}
    if pipeline_df is not None and not pipeline_df.empty and "lineup_index" in pipeline_df.columns:
        match = pipeline_df[pipeline_df["lineup_index"] == best_idx]
        if not match.empty:
            sim_metrics = match.iloc[0].to_dict()

    # Boom/bust row
    bb_row = None
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        bb_match = boom_bust_df[boom_bust_df["lineup_index"] == best_idx]
        if not bb_match.empty:
            bb_row = bb_match.iloc[0].to_dict()

    return lu_rows, sim_metrics or None, bb_row


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("📊 Edge Share")
    st.caption("Ricky's Edge Analysis and Optimizer — read-only view.")

    slate = get_slate_state()
    edge = get_edge_state()
    lu_state = get_lineup_state()
    sim_state = get_sim_state()

    _status_strip(slate)

    # ── Empty state check ──────────────────────────────────────────────
    has_edge = bool(edge.edge_analysis_by_contest)
    has_lineups = bool(lu_state.published_sets)

    if not has_edge and not has_lineups:
        st.info(
            "No Edge Share data yet. Run Edge Analysis on Ricky's Edge and "
            "publish lineups from Build & Publish."
        )
        return

    # ── Confidence strip ───────────────────────────────────────────────
    _confidence_pills(edge)
    st.divider()

    # ── Two main tabs ──────────────────────────────────────────────────
    tab_edge, tab_optimizer = st.tabs(["Ricky's Edge Analysis", "Optimizer"])

    # ==================================================================
    # TAB 1 — Ricky's Edge Analysis
    # ==================================================================
    with tab_edge:
        # Show each contest type that has edge analysis OR published lineups
        contests_shown = [
            c for c in _CONTEST_ORDER
            if c in edge.edge_analysis_by_contest or c in lu_state.published_sets
        ]

        if not contests_shown:
            st.info("No edge analysis available. Run analysis on Ricky's Edge page first.")
        else:
            for contest_label in contests_shown:
                short = _LABEL_SHORT.get(contest_label, contest_label)
                st.markdown(f"### {short}")

                # Edge writeup
                if contest_label in edge.edge_analysis_by_contest:
                    _render_edge_writeup(edge, contest_label)

                # Top lineup card for this contest
                lu_rows, sim_metrics, bb_row = _get_best_lineup(lu_state, sim_state, contest_label)
                if lu_rows is not None and not lu_rows.empty:
                    st.markdown(f"**Top {short} Lineup**")
                    render_premium_lineup_card(
                        lineup_rows=lu_rows,
                        sim_metrics=sim_metrics,
                        lineup_label=f"#{1} {short}",
                        salary_cap=slate.salary_cap,
                        boom_bust_row=bb_row,
                        compact=True,
                    )

                st.divider()

    # ==================================================================
    # TAB 2 — Optimizer (paginated lineup browser)
    # ==================================================================
    with tab_optimizer:
        pub_contests = [
            c for c in _CONTEST_ORDER
            if c in lu_state.published_sets
        ]

        if not pub_contests:
            st.info("No lineups published yet. Build and publish from the Build & Publish page.")
        else:
            for contest_label in pub_contests:
                short = _LABEL_SHORT.get(contest_label, contest_label)
                pub = lu_state.published_sets[contest_label]

                st.markdown(f"### {short}")

                # Metadata strip
                pub_ts = pub.get("published_at", "")
                config = pub.get("config", {})
                meta_parts = []
                if pub_ts:
                    meta_parts.append(f"Published: {pub_ts}")
                if config:
                    meta_parts.append(f"Mode: {config.get('build_mode', '?')}")
                    meta_parts.append(f"Lineups: {config.get('num_lineups', '?')}")
                if meta_parts:
                    st.caption(" · ".join(meta_parts))

                pub_df: pd.DataFrame = pub.get("lineups_df", pd.DataFrame())
                if pub_df.empty:
                    st.caption("No lineup data.")
                    continue

                boom_bust_df = pub.get("boom_bust_df")
                pipeline_df = (
                    sim_state.pipeline_output.get(contest_label)
                    or sim_state.pipeline_output.get("GPP_20")
                )

                # Paginated premium cards
                render_premium_cards_paged(
                    lineups_df=pub_df,
                    sim_results_df=pipeline_df,
                    salary_cap=slate.salary_cap,
                    nav_key=f"es_{contest_label}",
                    boom_bust_df=boom_bust_df,
                )

                # Exposure summary
                exposure_df = pub.get("exposure_df")
                if exposure_df is not None and not exposure_df.empty:
                    with st.expander("Exposure Summary", expanded=False):
                        display_cols = [c for c in [
                            "player", "team", "salary", "your_exposure_pct",
                            "field_own_pct", "delta", "leverage_ratio",
                        ] if c in exposure_df.columns]
                        st.dataframe(
                            exposure_df[display_cols].head(25),
                            use_container_width=True,
                            hide_index=True,
                        )

                st.divider()


main()
