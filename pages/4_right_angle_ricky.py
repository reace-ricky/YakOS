"""Right Angle Ricky -- the public-facing page for friends.

Two tabs per RICKY_BLUEPRINT.md:

  Tab 1 – Ricky's Edge Analysis
    Published top lineups (GPP, Cash, Showdown) + quick analysis
    (core plays, value plays, leverage plays, fades).

  Tab 2 – Optimizer
    Friends can build their own lineups using Ricky's same player
    pool and projections.  Simplified controls — no admin knobs.

State read:  SlateState, RickyEdgeState, LineupSetState, SimState
State written: None (fully read-only — friend lineups stored in session_state only)
"""

from __future__ import annotations

import hashlib as _hl
import io
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
from yak_core.components import render_premium_lineup_card, render_lineup_cards_paged  # noqa: E402
from yak_core.edge_metrics import (  # noqa: E402
    compute_ricky_confidence_for_contest,
    get_confidence_color,
)
from yak_core.config import CONTEST_PRESETS, UI_CONTEST_LABELS, UI_CONTEST_MAP  # noqa: E402
from yak_core.ricky_signals import compute_ricky_signals, generate_slate_overview  # noqa: E402
from yak_core.context import get_lab_analysis  # noqa: E402
from yak_core.display_format import normalise_ownership, standard_player_format  # noqa: E402
from yak_core.lineups import (  # noqa: E402
    build_multiple_lineups_with_exposure,
    build_showdown_lineups,
    to_dk_upload_format,
    to_dk_showdown_upload_format,
)
from yak_core.calibration import apply_archetype, DFS_ARCHETYPES  # noqa: E402

# ---------------------------------------------------------------------------
# Contest display helpers
# ---------------------------------------------------------------------------

# Internal preset labels in display order
_CONTEST_ORDER = [UI_CONTEST_MAP[k] for k in UI_CONTEST_LABELS]
# Reverse map: preset label -> short UI label
_LABEL_SHORT = {v: k for k, v in UI_CONTEST_MAP.items()}

# Build mode auto-selection per contest type
_CONTEST_TO_BUILD_MODE = {
    "GPP Main": "ceiling",
    "GPP Early": "ceiling",
    "GPP Late": "ceiling",
    "Cash Main": "floor",
    "Showdown": "ceiling",
}
_BUILD_MODE_PROJ_COL = {
    "floor": "floor",
    "median": "proj",
    "ceiling": "proj",
}

# ---------------------------------------------------------------------------
# Ricky flavor lines
# ---------------------------------------------------------------------------

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
    seed = st.session_state.get("_ricky_seed", "default")
    idx = int(_hl.md5(str(seed).encode()).hexdigest(), 16) % len(_RICKY_LINES)
    return _RICKY_LINES[idx]


# ---------------------------------------------------------------------------
# Shared UI helpers
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
                st.markdown(
                    f"<div style='padding:0.75rem 1rem;border-radius:0.5rem;"
                    f"background:#3a1a1a;border:1px solid #6b3a3a;"
                    f"color:#c27a7a;font-size:0.9rem;'>"
                    f"<strong>{short}</strong> \u2014 {score:.0f}/100</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Tab 1 helpers — Ricky's Edge Analysis
# ---------------------------------------------------------------------------

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

    # Pick the best lineup — prefer highest boom_score
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
# Tab 1 — Ricky's Edge Analysis
# ---------------------------------------------------------------------------

def _render_tab_analysis(slate, edge, lu_state, sim_state) -> None:
    """Tab 1: Published top lineups + edge analysis per contest type."""

    has_edge = bool(edge.edge_analysis_by_contest)
    has_lineups = bool(lu_state.published_sets)
    has_pool = slate.player_pool is not None and not slate.player_pool.empty

    # If no published data, show signal-driven overview from pool
    if not has_edge and not has_lineups and has_pool:
        pool = slate.player_pool.copy()
        _analysis = get_lab_analysis()
        if not _analysis["pool"].empty:
            pool = _analysis["pool"]

        signals_df = compute_ricky_signals(pool)
        contest_type = slate.contest_name or "GPP"
        overview = generate_slate_overview(pool, signals_df, contest_type=contest_type)

        for bullet in overview["bullets"]:
            st.markdown(f"- {bullet}")
        if overview["recommendation"]:
            st.info(overview["recommendation"])

        st.divider()

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

            if "pop_catalyst_tag" in top_edges.columns:
                pop_tags = top_edges["pop_catalyst_tag"].values
                if any(bool(t) for t in pop_tags):
                    display_df["Catalyst"] = [f"\U0001f680 {t}" if t else "" for t in pop_tags]

            _fmt = standard_player_format(display_df)
            st.dataframe(
                display_df.style.format(_fmt, na_rep=""),
                use_container_width=True, hide_index=True,
            )
        return

    if not has_edge and not has_lineups:
        st.info("No analysis available yet. Check back after Ricky publishes.")
        return

    # Published edge analysis + top lineup per contest type
    contests_shown = [
        c for c in _CONTEST_ORDER
        if c in edge.edge_analysis_by_contest or c in lu_state.published_sets
    ]

    for contest_label in contests_shown:
        short = _LABEL_SHORT.get(contest_label, contest_label)
        st.markdown(f"### {short}")

        if contest_label in edge.edge_analysis_by_contest:
            _render_edge_writeup(edge, contest_label)

        # Top lineup card for this contest
        lu_rows, sim_metrics, bb_row = _get_best_lineup(lu_state, sim_state, contest_label)
        if lu_rows is not None and not lu_rows.empty:
            st.markdown(f"**Ricky's Top {short} Lineup**")
            render_premium_lineup_card(
                lineup_rows=lu_rows,
                sim_metrics=sim_metrics,
                lineup_label=f"#1 {short}",
                salary_cap=slate.salary_cap,
                boom_bust_row=bb_row,
                compact=True,
            )

        st.divider()


# ---------------------------------------------------------------------------
# Tab 2 — Friends Optimizer
# ---------------------------------------------------------------------------

def _render_tab_optimizer(slate) -> None:
    """Tab 2: Simplified optimizer for friends to build their own lineups."""

    if not slate.is_ready() or slate.player_pool is None or slate.player_pool.empty:
        st.info(
            "No slate loaded yet. Ricky needs to load a slate in The Lab "
            "before you can build lineups."
        )
        return

    pool = slate.player_pool.copy()

    st.markdown(
        "Build your own lineups using Ricky's player pool and projections. "
        "Pick a contest type, set the number of lineups, and hit **Build**."
    )

    # ── Controls ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        ui_contest = st.selectbox(
            "Contest Type", UI_CONTEST_LABELS, index=0, key="_rar_opt_contest"
        )
        contest_label = UI_CONTEST_MAP[ui_contest]
        preset = CONTEST_PRESETS.get(contest_label, {})

    with col2:
        num_lineups = st.number_input(
            "# Lineups",
            min_value=1,
            max_value=50,
            value=min(int(preset.get("default_lineups", 3)), 50),
            key="_rar_opt_num",
        )

    # Auto-configure from preset
    build_mode = _CONTEST_TO_BUILD_MODE.get(contest_label, "ceiling")
    archetype = preset.get("archetype", "Balanced")
    max_exposure = float(preset.get("default_max_exposure", 0.5))
    min_salary = int(preset.get("min_salary", 46000))
    proj_col = _BUILD_MODE_PROJ_COL.get(build_mode, "proj")
    if proj_col not in pool.columns:
        proj_col = "proj"

    is_showdown = contest_label == "Showdown" or slate.is_showdown

    st.caption(
        f"**{len(pool)} players** \u00b7 "
        f"Mode: {build_mode} \u00b7 "
        f"Archetype: {archetype} \u00b7 "
        f"Cap: ${slate.salary_cap:,}"
    )

    # ── Build ─────────────────────────────────────────────────────────
    if st.button("Build Lineups", type="primary", key="_rar_opt_build"):
        # Ensure player_id exists
        _pool = pool.copy()
        if "player_id" not in _pool.columns:
            if "player_name" in _pool.columns:
                _pool["player_id"] = _pool["player_name"]
            elif "dk_player_id" in _pool.columns:
                _pool["player_id"] = _pool["dk_player_id"]

        _contest_type_map = {
            "GPP Main": "gpp", "GPP Early": "gpp", "GPP Late": "gpp",
            "Cash Main": "cash", "Showdown": "showdown",
        }
        cfg = {
            "NUM_LINEUPS": int(num_lineups),
            "SALARY_CAP": slate.salary_cap,
            "MAX_EXPOSURE": max_exposure,
            "MIN_SALARY_USED": min_salary,
            "LOCK": [],
            "EXCLUDE": [],
            "PROJ_COL": proj_col,
            "CONTEST_TYPE": _contest_type_map.get(contest_label, "gpp"),
        }

        try:
            with st.spinner(f"Building {num_lineups} {ui_contest} lineup(s)..."):
                if is_showdown:
                    lineups_df, expo_df = build_showdown_lineups(_pool, cfg)
                else:
                    opt_pool = apply_archetype(_pool.copy(), archetype)
                    lineups_df, expo_df = build_multiple_lineups_with_exposure(opt_pool, cfg)

            if lineups_df is not None and not lineups_df.empty:
                st.session_state["_rar_friend_lineups"] = lineups_df
                st.session_state["_rar_friend_expo"] = expo_df
                st.session_state["_rar_friend_contest"] = contest_label
                st.session_state["_rar_friend_is_showdown"] = is_showdown
                n_built = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else 1
                st.success(f"Built {n_built} lineup(s).")
            else:
                st.error("Optimizer returned no lineups. Try different settings.")
        except Exception as exc:
            st.error(f"Optimizer error: {exc}")

    # ── Display built lineups ─────────────────────────────────────────
    friend_lineups = st.session_state.get("_rar_friend_lineups")
    if friend_lineups is not None and not friend_lineups.empty:
        st.divider()
        st.markdown("### Your Lineups")

        render_lineup_cards_paged(
            lineups_df=friend_lineups,
            sim_results_df=None,
            salary_cap=slate.salary_cap,
            nav_key="rar_friend",
        )

        # ── DK CSV Export ─────────────────────────────────────────────
        st.divider()
        _is_sd = st.session_state.get("_rar_friend_is_showdown", False)
        try:
            if _is_sd:
                dk_csv = to_dk_showdown_upload_format(friend_lineups)
            else:
                dk_csv = to_dk_upload_format(friend_lineups)

            if dk_csv is not None and not dk_csv.empty:
                buf = io.StringIO()
                dk_csv.to_csv(buf, index=False)
                st.download_button(
                    label="Download DK CSV",
                    data=buf.getvalue(),
                    file_name="ricky_lineups.csv",
                    mime="text/csv",
                    key="_rar_dk_export",
                )
        except Exception as exc:
            st.caption(f"CSV export unavailable: {exc}")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("\U0001f4d0 Right Angle Ricky")
    st.caption(_ricky_quote())

    slate = get_slate_state()
    edge = get_edge_state()
    lu_state = get_lineup_state()
    sim_state = get_sim_state()

    _status_strip(slate)

    # Empty state
    has_edge = bool(edge.edge_analysis_by_contest)
    has_lineups = bool(lu_state.published_sets)
    has_pool = slate.player_pool is not None and not slate.player_pool.empty

    if not has_edge and not has_lineups and not has_pool:
        st.info(
            "Ricky's got nothing to show yet. Load a slate in **The Lab**, "
            "approve in **Ricky's Edge Analysis**, and publish from **Build & Publish**."
        )
        return

    # Confidence strip
    _confidence_pills(edge)
    if has_edge:
        st.divider()

    # Two tabs
    tab_analysis, tab_optimizer = st.tabs(
        ["\U0001f3af Ricky's Edge Analysis", "\U0001f527 Optimizer"]
    )

    with tab_analysis:
        _render_tab_analysis(slate, edge, lu_state, sim_state)

    with tab_optimizer:
        _render_tab_optimizer(slate)


main()
