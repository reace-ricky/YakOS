"""yak_core.components – Reusable Streamlit UI components for YakOS.

Provides SimLabs-style lineup card rendering and other shared UI helpers.
Import these functions in any page instead of re-implementing inline HTML/CSS.

Usage
-----
    from yak_core.components import render_lineup_card

    render_lineup_card(
        lineup_rows=lu_df[lu_df["lineup_index"] == i],
        sim_metrics=pipeline_df[pipeline_df["lineup_index"] == i].iloc[0].to_dict(),
        lineup_label="Lineup 3 of 20",
        salary_cap=50000,
    )
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

try:
    import streamlit as st
except ImportError:  # allow import outside Streamlit (tests, etc.)
    st = None  # type: ignore[assignment]

from yak_core.lineup_scoring import GRADE_COLORS as _GRADE_COLORS, GRADE_EMOJI as _GRADE_EMOJI  # noqa: E402


# ---------------------------------------------------------------------------
# Colour palette (dark-mode)
# ---------------------------------------------------------------------------

_BAR_GREEN = "#28a745"
_BAR_BLUE = "#1f77b4"
_BAR_ORANGE = "#fd7e14"
_BAR_RED = "#dc3545"
_BAR_GOLD = "#ffc107"
_CARD_BG = "#1e2130"
_HEADER_BG = "#262c40"
_TEXT_PRIMARY = "#f0f2f6"
_TEXT_SECONDARY = "#9da5b4"

# Bucket colours for the overall rating badge
_BUCKET_COLORS = {
    "A": _BAR_GREEN,
    "B": _BAR_BLUE,
    "C": _BAR_ORANGE,
    "D": _BAR_RED,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct_bar(label: str, value: float, lo: float, hi: float, color: str = _BAR_BLUE) -> str:
    """Return an HTML string for a labelled horizontal glow bar.

    Parameters
    ----------
    label : str
        Human-readable label displayed to the left of the bar.
    value : float
        Raw metric value (will be scaled to [0, 100] within [lo, hi]).
    lo, hi : float
        Clipping range used to normalise *value*.
    color : str
        CSS hex color for the bar fill.

    Returns
    -------
    str
        HTML snippet ready to be passed to ``st.markdown(..., unsafe_allow_html=True)``.
    """
    pct = int(max(0, min(100, (value - lo) / max(hi - lo, 1e-9) * 100)))
    return (
        f"<div style='margin-bottom:4px;'>"
        f"<span style='color:{_TEXT_SECONDARY};font-size:0.75rem;'>{label}</span>"
        f"<div style='background:#2d3350;border-radius:4px;height:10px;margin-top:2px;'>"
        f"<div style='width:{pct}%;background:{color};"
        f"border-radius:4px;height:10px;"
        f"box-shadow:0 0 6px {color}80;'></div>"
        f"</div>"
        f"<span style='color:{_TEXT_PRIMARY};font-size:0.7rem;'>{value:.3g}</span>"
        f"</div>"
    )


def _rating_badge(rating: float, bucket: str) -> str:
    """Return an HTML badge for the overall YakOS Sim Rating."""
    color = _BUCKET_COLORS.get(bucket, _BAR_BLUE)
    return (
        f"<div style='display:inline-block;padding:4px 12px;"
        f"background:{color};border-radius:8px;"
        f"color:#fff;font-weight:700;font-size:1.1rem;"
        f"box-shadow:0 0 10px {color}80;'>"
        f"⭐ {rating:.0f} <span style='font-size:0.85rem;'>[{bucket}]</span>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_lineup_card(
    lineup_rows: pd.DataFrame,
    sim_metrics: Optional[Dict[str, Any]] = None,
    lineup_label: str = "",
    salary_cap: int = 50000,
    show_rating: bool = True,
    boom_bust_row: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a SimLabs-style lineup card in the current Streamlit context.

    The card has three sections:

    **Header** – lineup label, remaining salary, total projection, total pOwn.

    **Body** – player list with position, name, team, matchup, salary,
    individual projection.

    **Footer** – horizontal "glow" bars for projection, pOwn, top-X% finish
    rate, ITM rate, sim ROI, leverage, and overall YakOS Sim Rating badge.

    Parameters
    ----------
    lineup_rows : pd.DataFrame
        Rows for a single lineup in long format (one player per row).
        Expected columns (all optional): ``slot``, ``player_name``, ``pos``,
        ``team``, ``matchup``, ``salary``, ``proj``, ``ownership``.
    sim_metrics : dict, optional
        Pipeline-level metrics for this lineup.  Expected keys:
        ``projection``, ``total_pown``, ``top_x_rate``, ``itm_rate``,
        ``sim_roi``, ``leverage``, ``yakos_sim_rating``, ``rating_bucket``.
        When *None* or missing keys, bars are omitted / show neutral values.
    lineup_label : str
        Header label e.g. "Lineup 3 of 20".
    salary_cap : int
        DK salary cap used to compute remaining salary.
    show_rating : bool
        If False, the footer rating panel is omitted.
    boom_bust_row : dict, optional
        A single row from the boom/bust rankings DataFrame (as a dict).
        When provided, a grade badge is shown in the header.
        Expected keys: ``lineup_grade``, ``boom_score``, ``bust_risk``.
    """
    if st is None:
        raise RuntimeError("Streamlit is not available.")

    lu = lineup_rows.copy() if lineup_rows is not None else pd.DataFrame()
    metrics = sim_metrics or {}

    # ── Derived header values ─────────────────────────────────────────────
    total_salary = int(pd.to_numeric(lu.get("salary", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    remaining_salary = salary_cap - total_salary

    # Use stored pipeline projection when available, else sum player projs
    projection = float(metrics.get("projection") or
                       pd.to_numeric(lu.get("proj", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    total_pown_frac = float(metrics.get("total_pown") or
                            pd.to_numeric(lu.get("ownership", pd.Series(dtype=float)),
                                          errors="coerce").fillna(0).sum() / 100.0)
    total_pown_pct = total_pown_frac * 100.0

    yakos_rating = float(metrics.get("yakos_sim_rating") or 0.0)
    bucket = str(metrics.get("rating_bucket") or "-")

    # ── Boom/bust grade badge ─────────────────────────────────────────────
    grade_badge_html = ""
    if boom_bust_row:
        grade = str(boom_bust_row.get("lineup_grade", ""))
        boom  = boom_bust_row.get("boom_score", None)
        bust  = boom_bust_row.get("bust_risk", None)
        if grade:
            color = _GRADE_COLORS.get(grade, "#9da5b4")
            emoji = _GRADE_EMOJI.get(grade, "")
            tooltip = ""
            if boom is not None and bust is not None:
                tooltip = f" | Boom: {boom:.0f} | Bust Risk: {bust:.0f}"
            grade_badge_html = (
                f"<span style='margin-left:8px;padding:2px 8px;"
                f"border-radius:12px;background:{color};color:#fff;"
                f"font-size:0.8rem;font-weight:700;'>"
                f"Grade: {grade} {emoji}</span>"
                f"<span style='font-size:0.75rem;color:#9da5b4;margin-left:6px;'>{tooltip}</span>"
            )

    # ── Header ───────────────────────────────────────────────────────────
    header_html = (
        f"<div style='background:{_HEADER_BG};border-radius:8px 8px 0 0;"
        f"padding:10px 14px;margin-bottom:0;'>"
        f"<span style='font-weight:700;font-size:1rem;color:{_TEXT_PRIMARY};'>"
        f"{'📋 ' + lineup_label if lineup_label else '📋 Lineup'}</span>"
        f"{grade_badge_html}"
        f"<span style='float:right;font-size:0.85rem;color:{_TEXT_SECONDARY};'>"
        f"Remaining: <b style='color:{'#28a745' if remaining_salary >= 0 else '#dc3545'};'>"
        f"${remaining_salary:+,}</b>"
        f"</span>"
        f"</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # Sub-header metrics
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.metric("Proj", f"{projection:.1f}")
    with h_col2:
        st.metric("pOwn", f"{total_pown_pct:.1f}%")
    with h_col3:
        st.metric("Salary", f"${total_salary:,}")

    # ── Body ─────────────────────────────────────────────────────────────
    if not lu.empty:
        show_cols = [c for c in ["slot", "player_name", "pos", "team", "matchup", "salary", "proj", "ownership"]
                     if c in lu.columns]
        # Rename for display
        rename_map = {"player_name": "Player", "pos": "Pos", "team": "Team",
                      "matchup": "Match", "salary": "Salary", "proj": "Proj",
                      "ownership": "Own%", "slot": "Slot"}
        display_df = lu[show_cols].rename(columns=rename_map)

        # Format numeric columns
        for col in ["Salary"]:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce").apply(
                    lambda v: f"${v:,.0f}" if pd.notna(v) else ""
                )
        for col in ["Proj"]:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce").apply(
                    lambda v: f"{v:.1f}" if pd.notna(v) else ""
                )
        for col in ["Own%"]:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce").apply(
                    lambda v: f"{v:.1f}%" if pd.notna(v) else ""
                )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Footer: sim rating bars ───────────────────────────────────────────
    if show_rating and metrics:
        st.markdown(
            f"<div style='background:{_CARD_BG};border-radius:0 0 8px 8px;"
            f"padding:10px 14px;margin-top:0;border-top:1px solid #333;'>",
            unsafe_allow_html=True,
        )

        bar_col, badge_col = st.columns([3, 1])

        with bar_col:
            bars_html = ""
            bars_html += _pct_bar("Projection",    projection,                         220, 350, _BAR_BLUE)
            bars_html += _pct_bar("pOwn (field)",  total_pown_pct,                      20,  70, _BAR_ORANGE)
            bars_html += _pct_bar("Top-X% rate",   float(metrics.get("top_x_rate") or 0), 0,  0.5, _BAR_GREEN)
            bars_html += _pct_bar("ITM rate",       float(metrics.get("itm_rate") or 0),   0.1, 0.7, _BAR_BLUE)
            bars_html += _pct_bar("Sim EV / ROI",  float(metrics.get("sim_roi") or 0),  -0.5, 1.0, _BAR_GOLD)
            bars_html += _pct_bar("Leverage",       float(metrics.get("leverage") or 1),   0.5, 2.5, _BAR_GREEN)
            st.markdown(bars_html, unsafe_allow_html=True)

        with badge_col:
            st.markdown(
                "<div style='text-align:center;padding-top:12px;'>"
                "<div style='font-size:0.75rem;color:#9da5b4;margin-bottom:4px;'>YakOS Rating</div>"
                + _rating_badge(yakos_rating, bucket) +
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


def render_lineup_cards_paged(
    lineups_df: pd.DataFrame,
    sim_results_df: Optional[pd.DataFrame] = None,
    contest_type: str = "GPP_20",
    salary_cap: int = 50000,
    nav_key: str = "lineup_nav",
    boom_bust_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render a paginated set of lineup cards with navigation controls.

    Shows one lineup at a time with Previous / Next buttons.

    Parameters
    ----------
    lineups_df : pd.DataFrame
        Full lineup DataFrame in long format with ``lineup_index`` column.
    sim_results_df : pd.DataFrame, optional
        Pipeline output from ``run_sims_pipeline``.  When provided, metrics
        are pulled from this table and displayed in the card footer.
    contest_type : str
        Passed to the rating system if metrics are not pre-computed.
    salary_cap : int
        DK salary cap.
    nav_key : str
        Base key for Streamlit session-state navigation.  Use unique values
        per call site to avoid key collisions.
    boom_bust_df : pd.DataFrame, optional
        Rankings DataFrame from ``compute_lineup_boom_bust``.  When provided,
        a grade badge is shown in each lineup card header.
    """
    if st is None:
        raise RuntimeError("Streamlit is not available.")

    if lineups_df is None or lineups_df.empty:
        st.info("No lineups to display.")
        return

    unique_idxs = sorted(lineups_df["lineup_index"].unique().tolist()) if "lineup_index" in lineups_df.columns else [0]
    n_lineups = len(unique_idxs)

    nav_state_key = f"_{nav_key}_idx"
    if nav_state_key not in st.session_state:
        st.session_state[nav_state_key] = 0

    # Navigation controls
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    with nav_col1:
        if st.button("◀ Prev", key=f"_{nav_key}_prev", disabled=st.session_state[nav_state_key] == 0):
            st.session_state[nav_state_key] = max(0, st.session_state[nav_state_key] - 1)
    with nav_col2:
        cur = st.session_state[nav_state_key]
        st.markdown(
            f"<div style='text-align:center;padding-top:8px;font-size:0.9rem;color:{_TEXT_SECONDARY};'>"
            f"Lineup {cur + 1} of {n_lineups}</div>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button("Next ▶", key=f"_{nav_key}_next", disabled=st.session_state[nav_state_key] >= n_lineups - 1):
            st.session_state[nav_state_key] = min(n_lineups - 1, st.session_state[nav_state_key] + 1)

    cur = st.session_state[nav_state_key]
    actual_idx = unique_idxs[min(cur, len(unique_idxs) - 1)]
    lu_rows = lineups_df[lineups_df["lineup_index"] == actual_idx]

    # Resolve sim metrics from pipeline output
    sim_metrics: Dict[str, Any] = {}
    if sim_results_df is not None and not sim_results_df.empty and "lineup_index" in sim_results_df.columns:
        match = sim_results_df[sim_results_df["lineup_index"] == actual_idx]
        if not match.empty:
            sim_metrics = match.iloc[0].to_dict()

    # Resolve boom/bust row for this lineup
    bb_row: Optional[Dict[str, Any]] = None
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        bb_match = boom_bust_df[boom_bust_df["lineup_index"] == actual_idx]
        if not bb_match.empty:
            bb_row = bb_match.iloc[0].to_dict()

    render_lineup_card(
        lineup_rows=lu_rows,
        sim_metrics=sim_metrics if sim_metrics else None,
        lineup_label=f"Lineup {cur + 1} of {n_lineups}",
        salary_cap=salary_cap,
        show_rating=bool(sim_metrics),
        boom_bust_row=bb_row,
    )
