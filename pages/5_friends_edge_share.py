"""Friends / Edge Share – YakOS read-only lineup showcase.

This page is what friends see: Ricky's published lineups and edge analysis
for each contest type.  It is fully read-only – no state is written here.

Layout
------
- Slate header (sport, date, site, active layers)
- Confidence pills per contest type
- For each contest with data:
    - Edge summary + core/value/leverage player list
    - Published lineup card browser (paged)
    - Boom/bust summary strip (if available)
    - Exposure breakdown (if available)

State read:  SlateState, RickyEdgeState, LineupSetState
State written: None
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
)
from yak_core.edge_metrics import (  # noqa: E402
    compute_ricky_confidence_for_contest,
    get_confidence_color,
)
from yak_core.config import CONTEST_PRESETS  # noqa: E402
from yak_core.display_format import normalise_ownership  # noqa: E402

# ---------------------------------------------------------------------------
# Contest display order — Cash first (floor/certainty), then GPP variants
# ---------------------------------------------------------------------------

CONTEST_ORDER: list[str] = [
    "Cash Main",
    "GPP Main",
    "GPP Early",
    "GPP Late",
    "Showdown",
]

# Short labels for display pills
_SHORT_LABEL: dict[str, str] = {
    "Cash Main": "Cash",
    "GPP Main": "GPP",
    "GPP Early": "GPP-E",
    "GPP Late": "GPP-L",
    "Showdown": "SD",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_strip(slate) -> None:
    """Compact one-line slate header."""
    parts = []
    if slate.sport:
        parts.append(f"**{slate.sport}**")
    if getattr(slate, "slate_date", None):
        parts.append(slate.slate_date)
    if slate.site:
        parts.append(slate.site)
    if parts:
        st.caption(" \u00b7 ".join(parts))


def _render_confidence_pills(edge) -> None:
    """Render compact confidence pills for contests with edge data."""
    available = [c for c in CONTEST_ORDER if c in edge.edge_analysis_by_contest]
    if not available:
        return

    cols = st.columns(len(available))
    for col, label in zip(cols, available):
        payload = edge.edge_analysis_by_contest[label]
        score = compute_ricky_confidence_for_contest(payload)
        color = get_confidence_color(score)
        short = _SHORT_LABEL.get(label, label)
        with col:
            if color == "green":
                st.success(f"**{short}** — {score:.0f}/100")
            elif color == "yellow":
                st.warning(f"**{short}** — {score:.0f}/100")
            else:
                st.error(f"**{short}** — {score:.0f}/100")


def _render_edge_summary(payload: dict, contest_label: str) -> None:
    """Render the edge summary text and player tables for one contest."""
    summary = payload.get("edge_summary", "")
    if summary:
        st.markdown(f"_{summary}_")

    core_players = payload.get("core_value_players", [])
    leverage_players = payload.get("leverage_players", [])
    fade_players = payload.get("fade_players", [])
    warnings = payload.get("contest_fit_warnings", [])

    if core_players:
        st.markdown("**Core / Value Plays**")
        df = pd.DataFrame(core_players)
        if "own" in df.columns:
            df["own"] = normalise_ownership(pd.to_numeric(df["own"], errors="coerce").fillna(0))
        st.dataframe(df, use_container_width=True, hide_index=True)

    if leverage_players:
        st.markdown("**Leverage Plays**")
        df = pd.DataFrame(leverage_players)
        if "own" in df.columns:
            df["own"] = normalise_ownership(pd.to_numeric(df["own"], errors="coerce").fillna(0))
        st.dataframe(df, use_container_width=True, hide_index=True)

    if fade_players:
        with st.expander("🚫 Fades", expanded=False):
            df = pd.DataFrame(fade_players)
            st.dataframe(df, use_container_width=True, hide_index=True)

    if warnings:
        for w in warnings:
            st.warning(w)


def _render_boom_bust_summary(boom_bust_df: pd.DataFrame, contest_label: str) -> None:
    """Render a compact boom/bust summary strip for published lineups."""
    if boom_bust_df is None or boom_bust_df.empty:
        return

    preset = CONTEST_PRESETS.get(contest_label, {})
    tagging_mode = preset.get("tagging_mode", "ceiling")

    n_ab = len(boom_bust_df[boom_bust_df["lineup_grade"].isin(["A", "B"])])
    avg_boom = boom_bust_df["boom_score"].mean()
    avg_bust = boom_bust_df["bust_risk"].mean()

    if tagging_mode == "ceiling":
        caption = (
            f"GPP lineup set — {n_ab} high-ceiling lineups, "
            f"avg bust risk {avg_bust:.0f}/100"
        )
    else:
        caption = (
            f"Cash lineup set — {n_ab} safe-floor lineups, "
            f"avg bust risk {avg_bust:.0f}/100"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Boom Score", f"{avg_boom:.0f}/100")
    with col2:
        st.metric("Avg Bust Risk", f"{avg_bust:.0f}/100")
    with col3:
        st.metric("A/B Lineups", n_ab)

    st.caption(caption)


def _render_exposure_table(exposure_df: pd.DataFrame) -> None:
    """Render exposure vs field ownership breakdown."""
    if exposure_df is None or exposure_df.empty:
        return

    wanted = [
        "player", "team", "salary",
        "your_exposure_pct", "field_own_pct", "delta", "leverage_ratio",
    ]
    display_cols = [c for c in wanted if c in exposure_df.columns]
    if display_cols:
        st.dataframe(exposure_df[display_cols], use_container_width=True, hide_index=True)


def _render_optimizer_col(
    contest_label: str,
    pub: dict,
    edge_payload: dict | None = None,
) -> None:
    """Render the right-hand optimizer column for one contest type.

    Parameters
    ----------
    contest_label : str
        The contest preset label (e.g. "GPP Main").
    pub : dict
        The published_sets entry for this contest.  Keys: lineups_df,
        published_at, config, boom_bust_df (optional), exposure_df (optional).
    edge_payload : dict, optional
        Edge analysis payload from RickyEdgeState for this contest.
    """
    if not pub:
        return

    lineups_df = pub.get("lineups_df")
    boom_bust_df = pub.get("boom_bust_df")
    exposure_df = pub.get("exposure_df")
    published_at = pub.get("published_at", "")

    if published_at:
        st.caption(f"Published: {published_at}")

    if lineups_df is not None and not lineups_df.empty:
        n_lineups = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else 1
        st.markdown(f"**{n_lineups} lineup{'s' if n_lineups != 1 else ''}**")
        st.dataframe(lineups_df, use_container_width=True, hide_index=True)

    if boom_bust_df is not None and not boom_bust_df.empty:
        with st.expander("📊 Boom/Bust Summary", expanded=False):
            _render_boom_bust_summary(boom_bust_df, contest_label)

    if exposure_df is not None and not exposure_df.empty:
        with st.expander("📈 Exposure vs Field", expanded=False):
            _render_exposure_table(exposure_df)


# ---------------------------------------------------------------------------
# Main page entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Friends / Edge Share page."""
    st.set_page_config(page_title="Edge Share", page_icon="🤝", layout="wide")
    st.title("🤝 Edge Share")
    st.caption("Read-only view of Ricky's published lineups and edge analysis.")

    slate = get_slate_state()
    edge = get_edge_state()
    lu = get_lineup_state()

    _status_strip(slate)

    # Determine which contests have data
    contests_with_data = [
        c for c in CONTEST_ORDER
        if c in edge.edge_analysis_by_contest or c in lu.published_sets
    ]

    if not contests_with_data:
        st.info("No published lineups or edge analysis yet. Check back after Ricky publishes.")
        return

    # Confidence pills
    _render_confidence_pills(edge)
    st.divider()

    # Render one section per contest
    for contest_label in contests_with_data:
        short = _SHORT_LABEL.get(contest_label, contest_label)
        st.subheader(f"🏀 {short} — {contest_label}")

        has_edge = contest_label in edge.edge_analysis_by_contest
        has_lineups = contest_label in lu.published_sets

        edge_col, lineup_col = st.columns([1, 1])

        with edge_col:
            st.markdown("**Ricky's Analysis**")
            if has_edge:
                _render_edge_summary(
                    edge.edge_analysis_by_contest[contest_label],
                    contest_label,
                )
            else:
                st.caption("No edge analysis published for this contest.")

        with lineup_col:
            st.markdown("**Lineups**")
            if has_lineups:
                _render_optimizer_col(
                    contest_label,
                    lu.published_sets[contest_label],
                    edge_payload=edge.edge_analysis_by_contest.get(contest_label),
                )
            else:
                st.caption("No lineups published for this contest.")

        st.divider()


if __name__ == "__main__":
    main()
