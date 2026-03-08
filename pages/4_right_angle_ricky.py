"""Right Angle Ricky -- the public-facing edge analysis and lineup showcase.

This is what friends see. Clean, confident, no clutter.

Layout:
  - Slate header (sport, date, site)
  - Confidence pills per contest type
  - Tab 1: Ricky's Analysis (edge writeup + top lineup per contest)
  - Tab 2: All Lineups (paginated browser per contest type)

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
from yak_core.ricky_signals import compute_ricky_signals, generate_slate_overview, SIGNAL_BADGES  # noqa: E402
from yak_core.context import get_lab_analysis  # noqa: E402
from yak_core.display_format import normalise_ownership, standard_player_format  # noqa: E402

# Internal preset labels in display order
_CONTEST_ORDER = [UI_CONTEST_MAP[k] for k in UI_CONTEST_LABELS]
# Reverse map: preset label -> short UI label  ("GPP Main" -> "GPP")
_LABEL_SHORT = {v: k for k, v in UI_CONTEST_MAP.items()}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_strip(slate) -> None:
    """Compact one-line slate header."""
    parts = []
    if slate.sport:
        parts.append(f"**{slate.sport}**")
    if slate.slate_date:
        parts.append(slate.slate_date)
    if slate.site:
        parts.append(slate.site)
    if parts:
        st.caption(" \u00b7 ".join(parts))


# Muted status colors for dark backgrounds
_PILL_COLORS = {
    "green": {"bg": "#1a3a1a", "text": "#6abf69", "border": "#3a6b3a"},
    "yellow": {"bg": "#3a3418", "text": "#d4a046", "border": "#6b5a2a"},
    "red": {"bg": "#3a1a1a", "text": "#c27a7a", "border": "#6b3a3a"},
}


def _confidence_pills(edge) -> None:
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
                st.success(f"**{short}** \u2014 {score:.0f}/100")
            elif color == "yellow":
                st.warning(f"**{short}** \u2014 {score:.0f}/100")
            else:
                # Use muted red instead of st.error() bright red
                _bg = _PILL_COLORS["red"]["bg"]
                _tx = _PILL_COLORS["red"]["text"]
                _bd = _PILL_COLORS["red"]["border"]
                st.markdown(
                    f"<div style='padding:0.75rem 1rem;border-radius:0.5rem;"
                    f"background:{_bg};border:1px solid {_bd};"
                    f"color:{_tx};font-size:0.9rem;'>"
                    f"<strong>{short}</strong> \u2014 {score:.0f}/100</div>",
                    unsafe_allow_html=True,
                )


def _render_edge_writeup(edge, contest_label: str) -> None:
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

    _COLS = ["Player", "Team", "Salary", "Proj", "Own", "Confidence", "Tag", "Catalyst"]

    def _to_df(players: list) -> pd.DataFrame:
        if not players:
            return pd.DataFrame(columns=_COLS)
        rows = []
        for p in players:
            if not isinstance(p, dict):
                continue
            cat = p.get("pop_catalyst_tag", p.get("Catalyst", ""))
            rows.append({
                "Player": p.get("player_name", p.get("Player", "")),
                "Team": p.get("team", p.get("Team", "")),
                "Salary": p.get("salary", p.get("Salary", "")),
                "Proj": p.get("proj", p.get("Proj", "")),
                "Own": p.get("own", p.get("Own", p.get("own_pct", ""))),
                "Confidence": p.get("confidence", p.get("Confidence", "")),
                "Tag": p.get("tag", p.get("Tag", "")),
                "Catalyst": f"\U0001f680 {cat}" if cat else "",
            })
        df = pd.DataFrame(rows, columns=_COLS)
        # Drop Catalyst column if no data
        if df["Catalyst"].str.strip().eq("").all():
            df = df.drop(columns=["Catalyst"])
        return df

    _writeup_fmt = {"Salary": "${:,.0f}", "Proj": "{:.1f}", "Own": "{:.1f}%"}
    for group_label, group_data in [
        ("Core & Value", core_value),
        ("Leverage", leverage),
        ("Fades", fades),
    ]:
        df = _to_df(group_data)
        if df.empty:
            continue
        # Normalise ownership values
        if "Own" in df.columns:
            df["Own"] = normalise_ownership(pd.to_numeric(df["Own"], errors="coerce").fillna(0))
        st.markdown(f"**{group_label}**")
        st.dataframe(
            df.style.format({k: v for k, v in _writeup_fmt.items() if k in df.columns}, na_rep=""),
            use_container_width=True, hide_index=True, height=min(38 * len(df) + 38, 300),
        )

    warnings = payload.get("contest_fit_warnings", [])
    for w in warnings:
        st.caption(w)


def _get_best_lineup(lu_state, sim_state, contest_label: str) -> tuple:
    """Return (lineup_rows_df, sim_metrics_dict, boom_bust_dict) for the #1 lineup."""
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

    # Determine best lineup -- prefer highest boom_score, else first
    best_idx = 0
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        if "boom_score" in boom_bust_df.columns:
            best_idx = int(boom_bust_df.sort_values("boom_score", ascending=False).iloc[0]["lineup_index"])
        else:
            best_idx = int(boom_bust_df.iloc[0]["lineup_index"])
    elif "lineup_index" in pub_df.columns:
        best_idx = int(pub_df["lineup_index"].min())

    lu_rows = pub_df[pub_df["lineup_index"] == best_idx] if "lineup_index" in pub_df.columns else pub_df

    sim_metrics = {}
    if pipeline_df is not None and not pipeline_df.empty and "lineup_index" in pipeline_df.columns:
        match = pipeline_df[pipeline_df["lineup_index"] == best_idx]
        if not match.empty:
            sim_metrics = match.iloc[0].to_dict()

    bb_row = None
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        bb_match = boom_bust_df[boom_bust_df["lineup_index"] == best_idx]
        if not bb_match.empty:
            bb_row = bb_match.iloc[0].to_dict()

    return lu_rows, sim_metrics or None, bb_row


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ricky flavor lines (rotated per session)
# ---------------------------------------------------------------------------
import hashlib as _hl

_RICKY_LINES = [
    "Hoodie on. Cold brew in hand. Let's find some edges.",
    "Running quiet from the coffee shop. The angles don't lie.",
    "Process over picks. Edges over hype.",
    "Perpendicular to nonsense since day one.",
    "Low-key analytics. High-key results.",
    "The spreadsheet doesn't care about your feelings.",
    "Calm process, sharp lines. That's the Ricky way.",
    "Half-awake, fully locked in.",
]

def _ricky_quote() -> str:
    """Pick a deterministic-but-rotating Ricky line based on session id."""
    seed = st.session_state.get("_ricky_seed", "default")
    idx = int(_hl.md5(str(seed).encode()).hexdigest(), 16) % len(_RICKY_LINES)
    return _RICKY_LINES[idx]


def main() -> None:
    st.title("📐 Right Angle Ricky")
    st.caption(_ricky_quote())

    slate = get_slate_state()
    edge = get_edge_state()
    lu_state = get_lineup_state()
    sim_state = get_sim_state()

    _status_strip(slate)

    # -- Empty state check --
    has_edge = bool(edge.edge_analysis_by_contest)
    has_lineups = bool(lu_state.published_sets)
    has_pool = slate.player_pool is not None and not slate.player_pool.empty

    if not has_edge and not has_lineups and not has_pool:
        st.info(
            "Ricky's got nothing to show yet. Load a slate in **The Lab**, "
            "approve in **Ricky's Edge Analysis**, and publish from **Build & Publish**."
        )
        return

    # -- Confidence strip --
    _confidence_pills(edge)
    if has_edge:
        st.divider()

    # -- Two main tabs --
    tab_analysis, tab_lineups = st.tabs(["Ricky's Analysis", "All Lineups"])

    # ==================================================================
    # TAB 1 -- Ricky's Analysis
    # ==================================================================
    with tab_analysis:
        # If we have a pool but no published edge analysis, generate the
        # signal-driven overview so the page isn't empty
        if not has_edge and has_pool:
            pool = slate.player_pool.copy()
            # Merge sim data if available
            _analysis = get_lab_analysis()
            if not _analysis["pool"].empty:
                pool = _analysis["pool"]

            signals_df = compute_ricky_signals(pool)
            contest_type = slate.contest_name or "GPP"
            overview = generate_slate_overview(pool, signals_df, contest_type=contest_type)

            # Slate overview bullets
            for bullet in overview["bullets"]:
                st.markdown(f"- {bullet}")

            if overview["recommendation"]:
                st.info(overview["recommendation"])

            st.divider()

            # Top edges table
            top_edges = signals_df[signals_df["edge_composite"] > 0].head(10)
            if not top_edges.empty:
                st.markdown("**Top Edges**")
                display_df = pd.DataFrame()
                display_df["Player"] = top_edges["player_name"].values
                if "pos" in top_edges.columns:
                    display_df["Pos"] = top_edges["pos"].values
                if "team" in top_edges.columns:
                    display_df["Team"] = top_edges["team"].values
                display_df["Salary"] = top_edges["salary"].values if "salary" in top_edges.columns else 0
                display_df["Proj"] = top_edges["proj"].values if "proj" in top_edges.columns else 0
                own_col = "ownership" if "ownership" in top_edges.columns else "own_pct"
                if own_col in top_edges.columns:
                    display_df["Own%"] = normalise_ownership(pd.Series(top_edges[own_col].values)).values
                display_df["Edge"] = (top_edges["edge_composite"].values * 100).round(0).astype(int)
                display_df["Signals"] = top_edges["signal_badges"].values

                # Pop catalyst tag if available
                if "pop_catalyst_tag" in top_edges.columns:
                    pop_tags = top_edges["pop_catalyst_tag"].values
                    has_any = any(bool(t) for t in pop_tags)
                    if has_any:
                        display_df["Catalyst"] = [
                            f"\U0001f680 {t}" if t else "" for t in pop_tags
                        ]

                _fmt = standard_player_format(display_df)
                st.dataframe(
                    display_df.style.format(_fmt, na_rep=""),
                    use_container_width=True,
                    hide_index=True,
                )

        # Published edge analysis by contest
        contests_shown = [
            c for c in _CONTEST_ORDER
            if c in edge.edge_analysis_by_contest or c in lu_state.published_sets
        ]

        if contests_shown:
            for contest_label in contests_shown:
                short = _LABEL_SHORT.get(contest_label, contest_label)
                st.markdown(f"### {short}")

                if contest_label in edge.edge_analysis_by_contest:
                    _render_edge_writeup(edge, contest_label)

                # Top lineup card for this contest
                lu_rows, sim_metrics, bb_row = _get_best_lineup(lu_state, sim_state, contest_label)
                if lu_rows is not None and not lu_rows.empty:
                    st.markdown(f"**Top {short} Lineup**")
                    render_premium_lineup_card(
                        lineup_rows=lu_rows,
                        sim_metrics=sim_metrics,
                        lineup_label=f"#1 {short}",
                        salary_cap=slate.salary_cap,
                        boom_bust_row=bb_row,
                        compact=True,
                    )

                st.divider()

        elif not has_pool:
            st.info("No analysis available yet.")

    # ==================================================================
    # TAB 2 -- All Lineups (paginated browser)
    # ==================================================================
    with tab_lineups:
        pub_contests = [
            c for c in _CONTEST_ORDER
            if c in lu_state.published_sets
        ]

        if not pub_contests:
            st.info("No lineups published yet. Build and publish from **Build & Publish**.")
        else:
            for contest_label in pub_contests:
                short = _LABEL_SHORT.get(contest_label, contest_label)
                pub = lu_state.published_sets[contest_label]

                st.markdown(f"### {short}")

                pub_ts = pub.get("published_at", "")
                config = pub.get("config", {})
                meta_parts = []
                if pub_ts:
                    meta_parts.append(f"Published: {pub_ts}")
                if config:
                    meta_parts.append(f"Mode: {config.get('build_mode', '?')}")
                    meta_parts.append(f"Lineups: {config.get('num_lineups', '?')}")
                if meta_parts:
                    st.caption(" \u00b7 ".join(meta_parts))

                pub_df: pd.DataFrame = pub.get("lineups_df", pd.DataFrame())
                if pub_df.empty:
                    st.caption("No lineup data.")
                    continue

                boom_bust_df = pub.get("boom_bust_df")
                pipeline_df = (
                    sim_state.pipeline_output.get(contest_label)
                    or sim_state.pipeline_output.get("GPP_20")
                )

                render_premium_cards_paged(
                    lineups_df=pub_df,
                    sim_results_df=pipeline_df,
                    salary_cap=slate.salary_cap,
                    nav_key=f"rar_{contest_label}",
                    boom_bust_df=boom_bust_df,
                )

                exposure_df = pub.get("exposure_df")
                if exposure_df is not None and not exposure_df.empty:
                    with st.expander("Exposure Summary", expanded=False):
                        display_cols = [c for c in [
                            "player", "team", "salary", "your_exposure_pct",
                            "field_own_pct", "delta", "leverage_ratio",
                        ] if c in exposure_df.columns]
                        _expo_display = exposure_df[display_cols].head(25)
                        _expo_fmt = {}
                        for c in ("salary",):
                            if c in _expo_display.columns:
                                _expo_fmt[c] = "${:,.0f}"
                        for c in ("your_exposure_pct", "field_own_pct"):
                            if c in _expo_display.columns:
                                _expo_fmt[c] = "{:.1f}%"
                        for c in ("delta", "leverage_ratio"):
                            if c in _expo_display.columns:
                                _expo_fmt[c] = "{:.2f}"
                        st.dataframe(
                            _expo_display.style.format(_expo_fmt, na_rep=""),
                            use_container_width=True,
                            hide_index=True,
                        )

                st.divider()


main()
