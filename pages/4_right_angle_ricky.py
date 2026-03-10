"""Right Angle Ricky -- the public-facing page for friends.

Two tabs per RICKY_BLUEPRINT.md:

  Tab 1 – Ricky's Edge Analysis
    4 analysis boxes: Slate Analysis, Game Environment, Ricky's Plays, Best Stacks.
    Below: 3 published lineup columns (GPP, Cash, Showdown) side by side.

  Tab 2 – Optimizer
    Friends build lineups from Ricky's player pool.  Auto-optimizer
    (pick contest, hit Build) plus manual lineup builder.

State read:  SlateState, RickyEdgeState, LineupSetState, SimState
State written: None (fully read-only — friend lineups in session_state only)
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
from yak_core.display_format import normalise_ownership, standard_player_format  # noqa: E402
from yak_core.lineups import (  # noqa: E402
    build_multiple_lineups_with_exposure,
    build_showdown_lineups,
    to_dk_upload_format,
    to_dk_showdown_upload_format,
)
from yak_core.calibration import apply_archetype, DFS_ARCHETYPES  # noqa: E402
from yak_core.right_angle import compute_stack_scores, compute_game_environment_cards  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTEST_ORDER = [UI_CONTEST_MAP[k] for k in UI_CONTEST_LABELS]
_LABEL_SHORT = {v: k for k, v in UI_CONTEST_MAP.items()}

_CONTEST_TO_BUILD_MODE = {
    "GPP Main": "ceiling", "GPP Early": "ceiling", "GPP Late": "ceiling",
    "Cash Main": "floor", "Showdown": "ceiling",
}
_BUILD_MODE_PROJ_COL = {"floor": "floor", "median": "proj", "ceiling": "proj"}

# DK Classic roster slots
_ROSTER_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

# ---------------------------------------------------------------------------
# Ricky flavor
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
# Shared helpers
# ---------------------------------------------------------------------------

def _status_strip(slate) -> None:
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
    available = [c for c in _CONTEST_ORDER if c in edge.edge_analysis_by_contest]
    if not available:
        return
    cols = st.columns(len(available))
    for col, label in zip(cols, available, strict=True):
        payload = edge.edge_analysis_by_contest[label]
        score = compute_ricky_confidence_for_contest(payload)
        color = get_confidence_color(score)
        short = _LABEL_SHORT.get(label, label)
        with col:
            if color == "green":
                st.success(f"**{short}** \u2014 {score:.0f}/100")
            elif color == "yellow":
                st.warning(f"**{short}** \u2014 {score:.0f}/100")
            else:
                st.markdown(
                    f"<div style='padding:0.6rem 1rem;border-radius:0.5rem;"
                    f"background:#3a1a1a;border:1px solid #6b3a3a;"
                    f"color:#c27a7a;font-size:0.9rem;'>"
                    f"<strong>{short}</strong> \u2014 {score:.0f}/100</div>",
                    unsafe_allow_html=True,
                )


def _own_pct(pool: pd.DataFrame) -> pd.Series:
    """Return ownership as 0-100 scale using canonical normaliser."""
    raw = pd.to_numeric(
        pool.get("ownership", pool.get("own_proj", pd.Series(dtype=float))),
        errors="coerce",
    ).fillna(0)
    return normalise_ownership(raw)


# ---------------------------------------------------------------------------
# Styled box helper
# ---------------------------------------------------------------------------

_BOX_CSS = (
    "padding:1rem 1.25rem;border-radius:0.75rem;"
    "border-left:4px solid {accent};"
    "background:rgba(255,255,255,0.03);"
    "margin-bottom:0.75rem;"
)


def _box_open(title: str, icon: str, accent: str = "#3b82f6") -> None:
    st.markdown(
        f"<div style='{_BOX_CSS.format(accent=accent)}'>"
        f"<div style='font-size:1.1rem;font-weight:700;margin-bottom:0.6rem;'>"
        f"{icon} {title}</div>",
        unsafe_allow_html=True,
    )


def _box_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _player_row(name: str, team: str, badge_text: str, badge_bg: str, stats: str) -> None:
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:0.35rem 0;border-bottom:1px solid rgba(255,255,255,0.05);'>"
        f"<div><strong>{name}</strong> ({team})"
        f"<span style='background:{badge_bg};color:#fff;padding:2px 8px;"
        f"border-radius:4px;font-size:0.72rem;font-weight:700;margin-left:6px;'>"
        f"{badge_text}</span></div>"
        f"<div style='color:#aaa;font-size:0.85rem;'>{stats}</div></div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# BOX 1 — Slate Analysis
# ═══════════════════════════════════════════════════════════════════════════

def _render_slate_analysis(pool: pd.DataFrame) -> None:
    proj = pd.to_numeric(pool.get("proj", 0), errors="coerce").fillna(0)
    own = _own_pct(pool)

    top_proj = round(float(proj.max()), 1)
    avg_proj = round(float(proj.mean()), 1)
    high_owned = int((own >= 20).sum())
    n_players = len(pool)

    if high_owned >= 5:
        stype, scolor = "CHALKY", "#ef4444"
    elif high_owned >= 3:
        stype, scolor = "BALANCED", "#eab308"
    else:
        stype, scolor = "VOLATILE", "#22c55e"

    _box_open("SLATE ANALYSIS", "\U0001f3b0", accent="#8b5cf6")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label, color in [
        (c1, stype, "Slate Type", scolor),
        (c2, str(top_proj), "Top Projection", "#22d3ee"),
        (c3, str(avg_proj), "Avg Projection", "#22d3ee"),
        (c4, str(high_owned), "High Owned (>20%)", "#f97316"),
        (c5, str(n_players), "Player Pool", "#a78bfa"),
    ]:
        with col:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-size:1.3rem;font-weight:800;color:{color};'>{val}</div>"
                f"<div style='font-size:0.7rem;color:#888;'>{label}</div></div>",
                unsafe_allow_html=True,
            )
    _box_close()


# ═══════════════════════════════════════════════════════════════════════════
# BOX 2 — Game Environment
# ═══════════════════════════════════════════════════════════════════════════

def _render_game_environment(pool: pd.DataFrame) -> None:
    _pool = pool.copy()
    if "opp" in _pool.columns and "opponent" not in _pool.columns:
        _pool["opponent"] = _pool["opp"]

    cards = compute_game_environment_cards(_pool)
    if not cards:
        return

    _box_open("GAME ENVIRONMENT", "\U0001f3c0", accent="#22c55e")
    for row_start in range(0, len(cards), 3):
        row_cards = cards[row_start:row_start + 3]
        cols = st.columns(len(row_cards))
        for col, gc in zip(cols, row_cards):
            with col:
                ou = gc.get("combined_ou", 0)
                pace = gc.get("pace_rating", "N/A")
                flags = gc.get("flags", [])
                if pace == "Fast":
                    pace_icon, pace_color = "\U0001f525", "#ef4444"
                elif pace == "Slow":
                    pace_icon, pace_color = "\u2744\ufe0f", "#3b82f6"
                else:
                    pace_icon, pace_color = "\u26a1", "#eab308"
                flag_str = " ".join(flags) if flags else ""
                st.markdown(
                    f"<div style='padding:0.5rem 0.75rem;border-radius:0.5rem;"
                    f"background:rgba(255,255,255,0.05);margin-bottom:0.3rem;'>"
                    f"<div style='font-weight:700;font-size:0.9rem;'>"
                    f"{gc['home']} vs {gc['away']}</div>"
                    f"<div style='font-size:0.8rem;color:#aaa;'>"
                    f"O/U: {ou} | {pace_icon} "
                    f"<span style='color:{pace_color};font-weight:600;'>"
                    f"{pace.upper()}</span> {flag_str}</div></div>",
                    unsafe_allow_html=True,
                )
    _box_close()


# ═══════════════════════════════════════════════════════════════════════════
# BOX 3 — Ricky's Plays (Core + Leverage + Value in one box)
# ═══════════════════════════════════════════════════════════════════════════

def _section_header(text: str, color: str) -> None:
    st.markdown(
        f"<div style='font-size:0.85rem;font-weight:700;color:{color};"
        f"margin-top:0.6rem;margin-bottom:0.3rem;'>{text}</div>",
        unsafe_allow_html=True,
    )


def _render_plays(pool: pd.DataFrame) -> None:
    proj = pd.to_numeric(pool.get("proj", 0), errors="coerce").fillna(0)
    sal = pd.to_numeric(pool.get("salary", 0), errors="coerce").fillna(0)
    own = _own_pct(pool)
    ceil_vals = pd.to_numeric(pool.get("ceil", proj * 1.25), errors="coerce").fillna(proj * 1.25)

    df = pool.copy()
    df["_proj"], df["_sal"], df["_own"], df["_ceil"] = proj, sal, own, ceil_vals
    df["_value"] = (proj / (sal / 1000.0)).round(2).replace([float("inf")], 0)
    df["_leverage"] = (ceil_vals / (own.clip(lower=0.5))).round(2)

    _box_open("RICKY'S PLAYS", "\U0001f3af", accent="#f97316")

    # Core Plays (Chalk)
    core = df[(df["_proj"] >= df["_proj"].quantile(0.85)) & (df["_own"] >= 15)].nlargest(3, "_proj")
    _section_header("\U0001f3af CORE PLAYS (Chalk)", "#ef4444")
    if not core.empty:
        for _, r in core.iterrows():
            _player_row(
                r.get("player_name", "?"), r.get("team", ""),
                "CHALK", "#dc2626",
                f"{r['_proj']:.1f} pts | {r['_own']:.1f}% owned | ${r['_sal']:,.0f}",
            )
    else:
        st.caption("No clear chalk plays.")

    # Leverage Plays (GPP Gold)
    leverage = df[
        (df["_own"] < 15) & (df["_ceil"] >= df["_ceil"].quantile(0.6)) & (df["_sal"] >= 3500)
    ].nlargest(3, "_leverage")
    _section_header("\u26a1 LEVERAGE PLAYS (GPP Gold)", "#eab308")
    if not leverage.empty:
        for _, r in leverage.iterrows():
            _player_row(
                r.get("player_name", "?"), r.get("team", ""),
                "UNDEROWNED", "#ca8a04",
                f"{r['_proj']:.1f} pts | {r['_own']:.1f}% owned | {r['_value']:.2f} val",
            )
    else:
        st.caption("No clear leverage plays.")

    # Value Plays (Salary Savers)
    value = df[
        (df["_sal"] <= 5500) & (df["_value"] >= df["_value"].quantile(0.75))
    ].nlargest(3, "_value")
    _section_header("\U0001f4b0 VALUE PLAYS (Salary Savers)", "#22c55e")
    if not value.empty:
        for _, r in value.iterrows():
            _player_row(
                r.get("player_name", "?"), r.get("team", ""),
                "VALUE", "#16a34a",
                f"${r['_sal']:,.0f} | {r['_value']:.2f} pts/$1K | Unlocks salary for studs",
            )
    else:
        st.caption("No clear value plays.")

    _box_close()


# ═══════════════════════════════════════════════════════════════════════════
# BOX 4 — Best Stacks
# ═══════════════════════════════════════════════════════════════════════════

def _render_stacks(pool: pd.DataFrame) -> None:
    stacks_df = compute_stack_scores(pool, top_n=5)
    if stacks_df.empty:
        return

    _box_open("BEST STACKS", "\U0001f9e9", accent="#3b82f6")
    for _, r in stacks_df.head(3).iterrows():
        team = r.get("team", "?")
        players = r.get("key_players", "")
        top_proj = r.get("top_proj", 0)
        lev = r.get("leverage_tag", "")
        lev_html = ""
        if lev and "low" in str(lev).lower():
            lev_html = (
                " <span style='background:#16a34a;color:#fff;padding:1px 6px;"
                "border-radius:3px;font-size:0.7rem;'>LOW OWNED</span>"
            )
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.05);'>"
            f"<div><strong>{team}:</strong> {players}{lev_html}</div>"
            f"<div style='color:#aaa;font-size:0.85rem;'>"
            f"Combined {top_proj:.1f} pts \u2014 GPP correlation target"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    _box_close()


# ═══════════════════════════════════════════════════════════════════════════
# PUBLISHED LINEUPS — 3 columns (GPP / Cash / Showdown)
# ═══════════════════════════════════════════════════════════════════════════

def _render_published_lineups(lu_state, sim_state, slate) -> None:
    """Render published lineups in 3 side-by-side columns by contest type."""
    pub_contests = [c for c in _CONTEST_ORDER if c in lu_state.published_sets]
    if not pub_contests:
        return

    st.markdown("### Ricky's Lineups")

    # Up to 3 columns side by side
    cols = st.columns(min(len(pub_contests), 3))
    for i, contest_label in enumerate(pub_contests[:3]):
        short = _LABEL_SHORT.get(contest_label, contest_label)
        pub = lu_state.published_sets[contest_label]
        pub_df = pub.get("lineups_df", pd.DataFrame())

        with cols[i % 3]:
            st.markdown(
                f"<div style='font-size:1rem;font-weight:700;margin-bottom:0.5rem;'>"
                f"{short}</div>",
                unsafe_allow_html=True,
            )

            if pub_df.empty:
                st.caption("No lineups published.")
                continue

            # Show published lineup(s) as a clean table
            # Get the first (best) lineup
            if "lineup_index" in pub_df.columns:
                best_idx = int(pub_df["lineup_index"].min())
                lu_rows = pub_df[pub_df["lineup_index"] == best_idx].copy()
            else:
                lu_rows = pub_df.copy()

            # Build display table
            display_cols = []
            if "pos" in lu_rows.columns:
                display_cols.append("pos")
            if "player_name" in lu_rows.columns:
                display_cols.append("player_name")
            if "salary" in lu_rows.columns:
                display_cols.append("salary")
            if "proj" in lu_rows.columns:
                display_cols.append("proj")

            own_col = None
            for oc in ("ownership", "own_proj", "own_pct"):
                if oc in lu_rows.columns:
                    own_col = oc
                    break

            if display_cols:
                show_df = lu_rows[display_cols].copy()
                show_df.columns = [c.replace("player_name", "Player").replace("pos", "Pos")
                                   .replace("salary", "Salary").replace("proj", "Proj")
                                   for c in show_df.columns]
                if own_col:
                    show_df["Own"] = normalise_ownership(
                        pd.to_numeric(lu_rows[own_col], errors="coerce").fillna(0)
                    )

                _fmt = {}
                if "Salary" in show_df.columns:
                    _fmt["Salary"] = "${:,.0f}"
                if "Proj" in show_df.columns:
                    _fmt["Proj"] = "{:.1f}"
                if "Own" in show_df.columns:
                    _fmt["Own"] = "{:.1f}%"

                st.dataframe(
                    show_df.style.format(_fmt, na_rep=""),
                    use_container_width=True,
                    hide_index=True,
                    height=min(38 * len(show_df) + 38, 380),
                )

                # Summary line
                total_sal = pd.to_numeric(lu_rows.get("salary", 0), errors="coerce").sum()
                total_proj = pd.to_numeric(lu_rows.get("proj", 0), errors="coerce").sum()
                st.caption(f"${total_sal:,.0f} salary \u00b7 {total_proj:.1f} projected")

            pub_ts = pub.get("published_at", "")
            if pub_ts:
                st.caption(f"Published: {pub_ts[:16]}")

    # If more than 3 contest types published, overflow to next row
    if len(pub_contests) > 3:
        cols2 = st.columns(len(pub_contests) - 3)
        for i, contest_label in enumerate(pub_contests[3:]):
            short = _LABEL_SHORT.get(contest_label, contest_label)
            with cols2[i]:
                st.markdown(f"**{short}**")
                st.caption("(See above pattern)")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Ricky's Edge Analysis
# ═══════════════════════════════════════════════════════════════════════════

def _render_tab_analysis(slate, edge, lu_state, sim_state) -> None:
    has_pool = slate.player_pool is not None and not slate.player_pool.empty
    has_lineups = bool(lu_state.published_sets)

    if not has_pool and not has_lineups:
        st.info("No analysis available yet. Check back after Ricky publishes.")
        return

    # 4 boxes
    if has_pool:
        pool = slate.player_pool.copy()
        _render_slate_analysis(pool)
        _render_game_environment(pool)
        _render_plays(pool)
        _render_stacks(pool)

    # Published lineups below (3 columns)
    if has_lineups:
        st.divider()
        _render_published_lineups(lu_state, sim_state, slate)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Optimizer
# ═══════════════════════════════════════════════════════════════════════════

def _pos_eligible(pos_str: str, slot: str) -> bool:
    """Check if a player's position string is eligible for a roster slot."""
    if not pos_str or not slot:
        return False
    positions = [p.strip().upper() for p in str(pos_str).replace("/", ",").split(",")]
    slot = slot.upper()
    if slot == "UTIL":
        return True
    if slot == "G":
        return any(p in ("PG", "SG", "G") for p in positions)
    if slot == "F":
        return any(p in ("SF", "PF", "F") for p in positions)
    return slot in positions


def _render_tab_optimizer(slate) -> None:
    if not slate.is_ready() or slate.player_pool is None or slate.player_pool.empty:
        st.info("No slate loaded yet. Ricky needs to load a slate first.")
        return

    pool = slate.player_pool.copy()
    is_showdown = getattr(slate, "is_showdown", False)

    # ── Auto Optimizer ────────────────────────────────────────────────
    st.markdown("### Auto-Build Lineups")
    st.caption("Pick contest type, hit Build. Uses Ricky's pool and projections.")

    col1, col2 = st.columns(2)
    with col1:
        ui_contest = st.selectbox("Contest Type", UI_CONTEST_LABELS, index=0, key="_rar_opt_contest")
        contest_label = UI_CONTEST_MAP[ui_contest]
        preset = CONTEST_PRESETS.get(contest_label, {})
    with col2:
        num_lineups = st.number_input(
            "# Lineups", min_value=1, max_value=50,
            value=min(int(preset.get("default_lineups", 3)), 50),
            key="_rar_opt_num",
        )

    build_mode = _CONTEST_TO_BUILD_MODE.get(contest_label, "ceiling")
    archetype = preset.get("archetype", "Balanced")
    max_exposure = float(preset.get("default_max_exposure", 0.5))
    min_salary = int(preset.get("min_salary", 46000))
    proj_col = _BUILD_MODE_PROJ_COL.get(build_mode, "proj")
    if proj_col not in pool.columns:
        proj_col = "proj"

    st.caption(
        f"**{len(pool)} players** \u00b7 Mode: {build_mode} \u00b7 "
        f"Archetype: {archetype} \u00b7 Cap: ${slate.salary_cap:,}"
    )

    if st.button("Build Lineups", type="primary", key="_rar_opt_build"):
        _pool = pool.copy()
        if "player_id" not in _pool.columns:
            _pool["player_id"] = _pool.get("player_name", _pool.get("dk_player_id", ""))

        cfg = {
            "NUM_LINEUPS": int(num_lineups),
            "SALARY_CAP": slate.salary_cap,
            "MAX_EXPOSURE": max_exposure,
            "MIN_SALARY_USED": min_salary,
            "LOCK": [], "EXCLUDE": [],
            "PROJ_COL": proj_col,
            "CONTEST_TYPE": {"GPP Main": "gpp", "Cash Main": "cash", "Showdown": "showdown"}.get(contest_label, "gpp"),
        }
        try:
            with st.spinner(f"Building {num_lineups} {ui_contest} lineup(s)..."):
                if is_showdown or contest_label == "Showdown":
                    lineups_df, expo_df = build_showdown_lineups(_pool, cfg)
                else:
                    opt_pool = apply_archetype(_pool.copy(), archetype)
                    lineups_df, expo_df = build_multiple_lineups_with_exposure(opt_pool, cfg)

            if lineups_df is not None and not lineups_df.empty:
                st.session_state["_rar_friend_lineups"] = lineups_df
                st.session_state["_rar_friend_is_showdown"] = is_showdown or contest_label == "Showdown"
                n = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else 1
                st.success(f"Built {n} lineup(s).")
            else:
                st.error("Optimizer returned no lineups.")
        except Exception as exc:
            st.error(f"Optimizer error: {exc}")

    # Display auto-built lineups
    friend_lineups = st.session_state.get("_rar_friend_lineups")
    if friend_lineups is not None and not friend_lineups.empty:
        st.divider()
        render_lineup_cards_paged(
            lineups_df=friend_lineups,
            sim_results_df=None,
            salary_cap=slate.salary_cap,
            nav_key="rar_friend",
        )

        # DK CSV export
        _is_sd = st.session_state.get("_rar_friend_is_showdown", False)
        try:
            dk_csv = to_dk_showdown_upload_format(friend_lineups) if _is_sd else to_dk_upload_format(friend_lineups)
            if dk_csv is not None and not dk_csv.empty:
                buf = io.StringIO()
                dk_csv.to_csv(buf, index=False)
                st.download_button("Download DK CSV", buf.getvalue(), "ricky_lineups.csv", "text/csv", key="_rar_dk")
        except Exception as exc:
            st.caption(f"CSV export unavailable: {exc}")

    # ── Manual Lineup Builder ─────────────────────────────────────────
    st.divider()
    st.markdown("### Manual Lineup Builder")
    st.caption("Click a player to add them to your lineup. Salary and projections update live.")

    # Initialize manual lineup state
    if "_rar_manual_lineup" not in st.session_state:
        st.session_state["_rar_manual_lineup"] = {slot: None for slot in _ROSTER_SLOTS}

    manual_lu = st.session_state["_rar_manual_lineup"]

    # Lineup card (right side concept, but stacked for Streamlit)
    filled = {s: v for s, v in manual_lu.items() if v is not None}
    used_salary = sum(v.get("salary", 0) for v in filled.values())
    used_proj = sum(v.get("proj", 0) for v in filled.values())
    remaining = slate.salary_cap - used_salary

    m1, m2, m3 = st.columns(3)
    m1.metric("Remaining Salary", f"${remaining:,}")
    m2.metric("Total Pts", f"{used_proj:.1f}")
    m3.metric("Players", f"{len(filled)}/{len(_ROSTER_SLOTS)}")

    # Lineup slots
    for slot in _ROSTER_SLOTS:
        player = manual_lu.get(slot)
        if player:
            col_slot, col_name, col_sal, col_pts, col_rm = st.columns([1, 4, 2, 2, 1])
            col_slot.markdown(f"**{slot}**")
            col_name.markdown(player.get("player_name", "?"))
            col_sal.markdown(f"${player.get('salary', 0):,}")
            col_pts.markdown(f"{player.get('proj', 0):.1f}")
            if col_rm.button("\u2716", key=f"_rar_rm_{slot}"):
                st.session_state["_rar_manual_lineup"][slot] = None
                st.rerun()
        else:
            col_slot, col_empty = st.columns([1, 9])
            col_slot.markdown(f"**{slot}**")
            col_empty.markdown(
                f"<span style='color:#555;font-style:italic;'>empty</span>",
                unsafe_allow_html=True,
            )

    if st.button("Clear Lineup", key="_rar_clear_manual"):
        st.session_state["_rar_manual_lineup"] = {slot: None for slot in _ROSTER_SLOTS}
        st.rerun()

    # Player pool browser — filter by position
    st.divider()
    pos_filter = st.selectbox(
        "Filter by position", ["All"] + _ROSTER_SLOTS[:5], key="_rar_pos_filter"
    )

    used_names = {v.get("player_name") for v in filled.values() if v}
    browse_df = pool[~pool["player_name"].isin(used_names)].copy() if "player_name" in pool.columns else pool.copy()

    if pos_filter != "All" and "pos" in browse_df.columns:
        browse_df = browse_df[browse_df["pos"].str.contains(pos_filter, case=False, na=False)]

    # Sort by projection desc
    if "proj" in browse_df.columns:
        browse_df = browse_df.sort_values("proj", ascending=False)

    # Show top 20 players with "+" buttons
    for _, row in browse_df.head(20).iterrows():
        name = row.get("player_name", "?")
        pos = row.get("pos", "?")
        sal = row.get("salary", 0)
        proj_val = row.get("proj", 0)
        own_val = _own_pct(pd.DataFrame([row])).iloc[0] if not pd.isna(row.get("ownership", row.get("own_proj", None))) else 0

        # Find first empty eligible slot
        target_slot = None
        for slot in _ROSTER_SLOTS:
            if manual_lu.get(slot) is None and _pos_eligible(str(pos), slot):
                target_slot = slot
                break

        cols = st.columns([1, 4, 2, 2, 2, 1])
        cols[0].markdown(f"**{pos}**")
        cols[1].markdown(name)
        cols[2].markdown(f"${sal:,}")
        cols[3].markdown(f"{proj_val:.1f}")
        cols[4].markdown(f"{own_val:.1f}%")

        if target_slot:
            if cols[5].button("+", key=f"_rar_add_{name}"):
                st.session_state["_rar_manual_lineup"][target_slot] = row.to_dict()
                st.rerun()
        else:
            cols[5].markdown("<span style='color:#555;'>\u2014</span>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("\U0001f4d0 Right Angle Ricky")
    st.caption(_ricky_quote())

    slate = get_slate_state()
    edge = get_edge_state()
    lu_state = get_lineup_state()
    sim_state = get_sim_state()

    _status_strip(slate)

    has_edge = bool(edge.edge_analysis_by_contest)
    has_lineups = bool(lu_state.published_sets)
    has_pool = slate.player_pool is not None and not slate.player_pool.empty

    if not has_edge and not has_lineups and not has_pool:
        st.info(
            "Ricky's got nothing to show yet. Load a slate in **The Lab**, "
            "approve in **Ricky's Edge Analysis**, and publish from **Build & Publish**."
        )
        return

    _confidence_pills(edge)
    if has_edge:
        st.divider()

    tab_analysis, tab_optimizer = st.tabs(
        ["\U0001f3af Ricky's Edge Analysis", "\U0001f527 Optimizer"]
    )

    with tab_analysis:
        _render_tab_analysis(slate, edge, lu_state, sim_state)

    with tab_optimizer:
        _render_tab_optimizer(slate)


main()
