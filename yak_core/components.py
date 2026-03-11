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
from yak_core.display_format import normalise_ownership  # noqa: E402


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
_ROW_DARK = "#1a1e2e"
_ROW_LIGHT = "#222840"
_TEXT_PRIMARY = "#f0f2f6"
_TEXT_SECONDARY = "#9da5b4"
_ACCENT_GREEN = "#00d26a"

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


def _pct_bar_slim(label: str, value: float, lo: float, hi: float, color: str = _ACCENT_GREEN) -> str:
    """Compact glow bar for premium card footer — label + bar + value on one line."""
    pct = int(max(0, min(100, (value - lo) / max(hi - lo, 1e-9) * 100)))
    return (
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:3px;'>"
        f"<span style='color:{_TEXT_SECONDARY};font-size:0.7rem;min-width:60px;text-align:right;'>{label}</span>"
        f"<div style='flex:1;background:#2d3350;border-radius:3px;height:8px;'>"
        f"<div style='width:{pct}%;background:{color};border-radius:3px;height:8px;"
        f"box-shadow:0 0 4px {color}80;'></div>"
        f"</div>"
        f"<span style='color:{_TEXT_PRIMARY};font-size:0.7rem;min-width:40px;'>{value:.2f}</span>"
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


def _rating_badge_compact(rating: float, bucket: str) -> str:
    """Smaller rating badge for premium card."""
    color = _BUCKET_COLORS.get(bucket, _BAR_BLUE)
    return (
        f"<div style='text-align:center;'>"
        f"<div style='font-size:0.65rem;color:{_TEXT_SECONDARY};margin-bottom:2px;'>YakOS Rating</div>"
        f"<div style='display:inline-block;padding:6px 14px;"
        f"background:{color};border-radius:10px;"
        f"color:#fff;font-weight:700;font-size:1.2rem;"
        f"box-shadow:0 0 12px {color}80;'>"
        f"⭐ {rating:.0f}"
        f"</div>"
        f"<div style='font-size:0.7rem;color:{_TEXT_SECONDARY};margin-top:2px;'>Grade {bucket}</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Premium lineup card (SimLabs-style HTML — self-contained single block)
# ---------------------------------------------------------------------------

def render_premium_lineup_card(
    lineup_rows: pd.DataFrame,
    sim_metrics: Optional[Dict[str, Any]] = None,
    lineup_label: str = "",
    salary_cap: int = 50000,
    boom_bust_row: Optional[Dict[str, Any]] = None,
    compact: bool = False,
) -> None:
    """Render a SimLabs-inspired premium lineup card as a single HTML block.

    Dark banded rows showing: Slot | Player Name (Pos) | @Team Time | pOwn% | $Salary | Proj
    Footer with glow rating bars and YakOS rating badge.

    Parameters
    ----------
    lineup_rows : pd.DataFrame
        Rows for a single lineup (one player per row).
    sim_metrics : dict, optional
        Pipeline-level metrics for this lineup.
    lineup_label : str
        Header label e.g. "Lineup 1 of 20".
    salary_cap : int
        DK salary cap used to compute remaining salary.
    boom_bust_row : dict, optional
        A row from boom/bust rankings — adds grade badge.
    compact : bool
        If True, renders a smaller card (for Edge Analysis top picks).
    """
    if st is None:
        raise RuntimeError("Streamlit is not available.")

    lu = lineup_rows.copy() if lineup_rows is not None else pd.DataFrame()
    metrics = sim_metrics or {}

    # ── Derived header values ──────────────────────────────────────────
    total_salary = int(
        pd.to_numeric(lu.get("salary", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    )
    remaining_salary = salary_cap - total_salary

    projection = float(
        metrics.get("projection")
        or pd.to_numeric(lu.get("proj", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    )

    _own_raw = pd.to_numeric(lu.get("ownership", pd.Series(dtype=float)), errors="coerce").fillna(0)
    _own_normed = normalise_ownership(_own_raw)  # guaranteed 0-100
    total_pown_pct = float(metrics.get("total_pown", 0) or 0)
    if total_pown_pct:
        # Pipeline now stores ownership on 0-100 scale (enforced by ownership_scale.py).
        # If value looks fractional (< 1.5), scale up for backward compat with old data.
        if total_pown_pct < 1.5:
            total_pown_pct = total_pown_pct * 100.0
    else:
        total_pown_pct = float(_own_normed.sum())

    yakos_rating = float(metrics.get("yakos_sim_rating") or 0.0)
    bucket = str(metrics.get("rating_bucket") or "-")

    # Grade badge
    grade_html = ""
    if boom_bust_row:
        grade = str(boom_bust_row.get("lineup_grade", ""))
        boom = boom_bust_row.get("boom_score")
        bust = boom_bust_row.get("bust_risk")
        if grade:
            gcolor = _GRADE_COLORS.get(grade, "#9da5b4")
            grade_html = (
                f"<span style='margin-left:8px;padding:2px 8px;"
                f"border-radius:12px;background:{gcolor};color:#fff;"
                f"font-size:0.75rem;font-weight:700;'>"
                f"Grade {grade}</span>"
            )
            if boom is not None:
                grade_html += (
                    f"<span style='font-size:0.7rem;color:{_TEXT_SECONDARY};margin-left:6px;'>"
                    f"Boom {boom:.0f} | Bust {bust:.0f}</span>"
                )

    # Remaining salary colour
    rem_color = _ACCENT_GREEN if remaining_salary >= 0 else _BAR_RED

    # ── Build the card as a single HTML block ──────────────────────────
    font_size = "0.8rem" if compact else "0.85rem"
    row_pad = "8px 12px" if compact else "10px 14px"

    html_parts = []

    # HEADER
    html_parts.append(
        f"<div style='background:{_HEADER_BG};border-radius:10px 10px 0 0;"
        f"padding:12px 16px;border-bottom:1px solid #333;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<div>"
        f"<span style='font-weight:700;font-size:1rem;color:{_TEXT_PRIMARY};'>"
        f"{lineup_label or 'Lineup'}</span>"
        f"{grade_html}"
        f"</div>"
        f"<span style='font-size:0.85rem;color:{_TEXT_SECONDARY};'>"
        f"Remaining: <b style='color:{rem_color};'>${remaining_salary:+,}</b></span>"
        f"</div>"
        f"<div style='display:flex;gap:24px;margin-top:6px;'>"
        f"<span style='font-size:0.8rem;color:{_TEXT_SECONDARY};'>"
        f"Proj <b style='color:{_TEXT_PRIMARY};'>{projection:.1f}</b></span>"
        f"<span style='font-size:0.8rem;color:{_TEXT_SECONDARY};'>"
        f"pOwn <b style='color:{_TEXT_PRIMARY};'>{total_pown_pct:.1f}</b></span>"
        f"<span style='font-size:0.8rem;color:{_TEXT_SECONDARY};'>"
        f"Salary <b style='color:{_TEXT_PRIMARY};'>${total_salary:,}</b></span>"
        f"</div>"
        f"</div>"
    )

    # PLAYER ROWS
    if not lu.empty:
        for i, (_, row) in enumerate(lu.iterrows()):
            bg = _ROW_DARK if i % 2 == 0 else _ROW_LIGHT
            slot = str(row.get("slot", row.get("pos", "")))
            name = str(row.get("player_name", ""))
            pos = str(row.get("pos", ""))
            team = str(row.get("team", ""))
            matchup = str(row.get("matchup", ""))
            sal = pd.to_numeric(row.get("salary", 0), errors="coerce")
            sal = int(sal) if pd.notna(sal) else 0
            proj = pd.to_numeric(row.get("proj", 0), errors="coerce")
            proj = float(proj) if pd.notna(proj) else 0.0
            own = pd.to_numeric(row.get("ownership", 0), errors="coerce")
            own = float(own) if pd.notna(own) else 0.0
            # Normalise: if all player ownerships are < 1, it's fractional
            if own > 0 and own < 1.0:
                own = own * 100.0

            # Format salary as $X.XK
            sal_str = f"${sal / 1000:.1f}K" if sal >= 1000 else f"${sal:,}"

            # Matchup line
            match_line = f"@{matchup}" if matchup and matchup != "nan" else team

            html_parts.append(
                f"<div style='background:{bg};padding:{row_pad};"
                f"display:flex;align-items:center;gap:0;"
                f"border-bottom:1px solid #2a2e42;'>"
                # Slot badge
                f"<div style='min-width:42px;font-weight:700;font-size:{font_size};"
                f"color:{_TEXT_PRIMARY};'>{slot}</div>"
                # Player name + pos + matchup
                f"<div style='flex:1;'>"
                f"<div style='font-weight:600;font-size:{font_size};color:{_TEXT_PRIMARY};'>"
                f"{name} <span style='color:{_TEXT_SECONDARY};font-weight:400;'>({pos})</span></div>"
                f"<div style='font-size:0.7rem;color:{_TEXT_SECONDARY};'>{match_line}</div>"
                f"</div>"
                # pOwn
                f"<div style='min-width:55px;text-align:right;'>"
                f"<div style='font-weight:600;font-size:{font_size};color:{_TEXT_PRIMARY};'>"
                f"{own:.1f}%</div>"
                f"<div style='font-size:0.65rem;color:{_TEXT_SECONDARY};'>pOwn</div>"
                f"</div>"
                # Salary
                f"<div style='min-width:55px;text-align:right;margin-left:12px;'>"
                f"<div style='font-weight:700;font-size:{font_size};color:{_TEXT_PRIMARY};'>"
                f"{sal_str}</div>"
                f"</div>"
                # Projection
                f"<div style='min-width:55px;text-align:right;margin-left:12px;'>"
                f"<div style='font-size:0.7rem;color:{_TEXT_SECONDARY};'>Proj</div>"
                f"<div style='font-weight:600;font-size:{font_size};color:{_TEXT_PRIMARY};'>"
                f"{proj:.1f}</div>"
                f"</div>"
                f"</div>"
            )

    # FOOTER — rating bars
    show_footer = bool(metrics) and yakos_rating > 0
    if show_footer:
        bars_html = ""
        bars_html += _pct_bar_slim("Projection", projection, 220, 350, _BAR_BLUE)
        bars_html += _pct_bar_slim("pOwn", total_pown_pct, 20, 70, _BAR_ORANGE)
        bars_html += _pct_bar_slim("Top-X%", float(metrics.get("top_x_rate") or 0), 0, 0.5, _ACCENT_GREEN)
        bars_html += _pct_bar_slim("ITM", float(metrics.get("itm_rate") or 0), 0.1, 0.7, _BAR_BLUE)
        bars_html += _pct_bar_slim("Sim ROI", float(metrics.get("sim_roi") or 0), -0.5, 1.0, _BAR_GOLD)
        bars_html += _pct_bar_slim("Leverage", float(metrics.get("leverage") or 1), 0.5, 2.5, _ACCENT_GREEN)

        rating_html = _rating_badge_compact(yakos_rating, bucket)

        html_parts.append(
            f"<div style='background:{_CARD_BG};padding:12px 16px;"
            f"border-top:1px solid #333;border-radius:0 0 10px 10px;'>"
            f"<div style='font-size:0.75rem;font-weight:700;color:{_TEXT_PRIMARY};"
            f"margin-bottom:6px;'>YakOS Sim Rating</div>"
            f"<div style='display:flex;gap:16px;'>"
            f"<div style='flex:1;'>{bars_html}</div>"
            f"<div style='min-width:90px;display:flex;align-items:center;'>{rating_html}</div>"
            f"</div>"
            f"</div>"
        )
    else:
        # Close card with rounded bottom
        html_parts.append(
            f"<div style='background:{_CARD_BG};padding:4px;border-radius:0 0 10px 10px;'></div>"
        )

    # Render the whole card
    full_html = "\n".join(html_parts)
    st.markdown(
        f"<div style='border-radius:10px;overflow:hidden;border:1px solid #333;"
        f"margin-bottom:16px;'>{full_html}</div>",
        unsafe_allow_html=True,
    )


def render_premium_cards_paged(
    lineups_df: pd.DataFrame,
    sim_results_df: Optional[pd.DataFrame] = None,
    contest_type: str = "GPP_20",
    salary_cap: int = 50000,
    nav_key: str = "lineup_nav",
    boom_bust_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render paginated premium lineup cards with Prev/Next navigation."""
    if st is None:
        raise RuntimeError("Streamlit is not available.")

    if lineups_df is None or lineups_df.empty:
        st.info("No lineups to display.")
        return

    unique_idxs = (
        sorted(lineups_df["lineup_index"].unique().tolist())
        if "lineup_index" in lineups_df.columns
        else [0]
    )
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
    lu_rows = lineups_df[lineups_df["lineup_index"] == actual_idx] if "lineup_index" in lineups_df.columns else lineups_df

    # Resolve sim metrics
    sim_metrics: Dict[str, Any] = {}
    if sim_results_df is not None and not sim_results_df.empty and "lineup_index" in sim_results_df.columns:
        match = sim_results_df[sim_results_df["lineup_index"] == actual_idx]
        if not match.empty:
            sim_metrics = match.iloc[0].to_dict()

    # Resolve boom/bust row
    bb_row: Optional[Dict[str, Any]] = None
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        bb_match = boom_bust_df[boom_bust_df["lineup_index"] == actual_idx]
        if not bb_match.empty:
            bb_row = bb_match.iloc[0].to_dict()

    render_premium_lineup_card(
        lineup_rows=lu_rows,
        sim_metrics=sim_metrics if sim_metrics else None,
        lineup_label=f"Lineup {cur + 1} of {n_lineups}",
        salary_cap=salary_cap,
        boom_bust_row=bb_row,
    )


# ---------------------------------------------------------------------------
# Legacy public API (still used by other pages)
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

    _own_raw_lg = pd.to_numeric(lu.get("ownership", pd.Series(dtype=float)), errors="coerce").fillna(0)
    _own_normed_lg = normalise_ownership(_own_raw_lg)  # guaranteed 0-100
    total_pown_pct = float(metrics.get("total_pown", 0) or 0)
    if total_pown_pct:
        # Pipeline now stores ownership on 0-100 scale (enforced by ownership_scale.py).
        # If value looks fractional (< 1.5), scale up for backward compat with old data.
        if total_pown_pct < 1.5:
            total_pown_pct = total_pown_pct * 100.0
    else:
        total_pown_pct = float(_own_normed_lg.sum())

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
                _own_series = normalise_ownership(pd.to_numeric(display_df[col], errors="coerce").fillna(0))
                display_df[col] = _own_series.apply(
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


def render_lineup_cards_scrollable(
    lineups_df: pd.DataFrame,
    sim_results_df: Optional[pd.DataFrame] = None,
    contest_type: str = "GPP_20",
    salary_cap: int = 50000,
    nav_key: str = "lineup_nav",
    boom_bust_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render all lineup cards in a scrollable vertical list sorted best-to-worst.

    Shows all lineups at once sorted by grade/boom score so the user can
    quickly identify the best lineups without clicking through one at a time.

    Parameters
    ----------
    lineups_df : pd.DataFrame
        Full lineup DataFrame in long format with ``lineup_index`` column.
    sim_results_df : pd.DataFrame, optional
        Pipeline output from ``run_sims_pipeline``.  When provided, metrics
        are pulled from this table and displayed in each card footer.
    contest_type : str
        Passed to the rating system if metrics are not pre-computed.
    salary_cap : int
        DK salary cap.
    nav_key : str
        Base key (unused for navigation but kept for API compatibility).
    boom_bust_df : pd.DataFrame, optional
        Rankings DataFrame from ``compute_lineup_boom_bust``.  When provided,
        lineups are sorted by rank/score and grade badges are shown in headers.
        Expected columns: ``lineup_index``, ``boom_bust_rank``, ``boom_score``,
        ``lineup_grade``.
    """
    if st is None:
        raise RuntimeError("Streamlit is not available.")

    if lineups_df is None or lineups_df.empty:
        st.info("No lineups to display.")
        return

    unique_idxs = (
        sorted(lineups_df["lineup_index"].unique().tolist())
        if "lineup_index" in lineups_df.columns
        else [0]
    )
    n_lineups = len(unique_idxs)

    # ── Sort lineup indices by grade ──────────────────────────────────────
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        if "boom_bust_rank" in boom_bust_df.columns:
            rank_map = dict(zip(boom_bust_df["lineup_index"], boom_bust_df["boom_bust_rank"]))
            unique_idxs = sorted(unique_idxs, key=lambda i: rank_map.get(i, 9999))
        elif "boom_score" in boom_bust_df.columns:
            score_map = dict(zip(boom_bust_df["lineup_index"], boom_bust_df["boom_score"]))
            unique_idxs = sorted(unique_idxs, key=lambda i: score_map.get(i, 0), reverse=True)

    # ── Summary header ────────────────────────────────────────────────────
    # Pre-build lookup dicts for O(1) access inside loops
    bb_row_map: Dict[Any, Dict[str, Any]] = {}
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        for _, row in boom_bust_df.iterrows():
            bb_row_map[row["lineup_index"]] = row.to_dict()

    sim_metrics_map: Dict[Any, Dict[str, Any]] = {}
    if sim_results_df is not None and not sim_results_df.empty and "lineup_index" in sim_results_df.columns:
        for _, row in sim_results_df.iterrows():
            sim_metrics_map[row["lineup_index"]] = row.to_dict()

    grade_dist_html = ""
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_grade" in boom_bust_df.columns:
        grade_map = dict(zip(boom_bust_df["lineup_index"], boom_bust_df["lineup_grade"]))
        grade_counts: Dict[str, int] = {}
        for idx in unique_idxs:
            g = str(grade_map.get(idx, "?"))
            grade_counts[g] = grade_counts.get(g, 0) + 1
        parts = []
        for grade in ["A", "B", "C", "D", "F"]:
            if grade in grade_counts:
                color = _GRADE_COLORS.get(grade, "#9da5b4")
                parts.append(
                    f"<span style='color:{color};font-weight:700;'>"
                    f"{grade_counts[grade]}{grade}</span>"
                )
        grade_dist_html = " · ".join(parts)

    summary_html = (
        f"<div style='padding:8px 0 10px 0;font-size:0.9rem;color:{_TEXT_SECONDARY};'>"
        f"Showing <b style='color:{_TEXT_PRIMARY};'>{n_lineups}</b> lineup"
        f"{'s' if n_lineups != 1 else ''} (sorted best → worst)"
        + (f"&nbsp;&nbsp;{grade_dist_html}" if grade_dist_html else "")
        + "</div>"
    )
    st.markdown(summary_html, unsafe_allow_html=True)

    # ── Scrollable container (open) ───────────────────────────────────────
    st.markdown(
        "<div style='max-height:800px;overflow-y:auto;padding-right:6px;'>",
        unsafe_allow_html=True,
    )

    for rank_pos, actual_idx in enumerate(unique_idxs, start=1):
        lu_rows = lineups_df[lineups_df["lineup_index"] == actual_idx]

        # Resolve sim metrics and boom/bust row via pre-built lookups
        sim_metrics: Dict[str, Any] = sim_metrics_map.get(actual_idx, {})
        bb_row: Optional[Dict[str, Any]] = bb_row_map.get(actual_idx)

        # Build label with grade when available
        if bb_row:
            grade = bb_row.get("lineup_grade", "")
            grade_str = f" — Grade {grade}" if grade else ""
            lineup_label = f"#{rank_pos}{grade_str} (Lineup {actual_idx})"
        else:
            lineup_label = f"Lineup {rank_pos} of {n_lineups}"

        render_lineup_card(
            lineup_rows=lu_rows,
            sim_metrics=sim_metrics if sim_metrics else None,
            lineup_label=lineup_label,
            salary_cap=salary_cap,
            show_rating=bool(sim_metrics),
            boom_bust_row=bb_row,
        )

        # Spacer between cards
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ── Scrollable container (close) ──────────────────────────────────────
    st.markdown("</div>", unsafe_allow_html=True)
