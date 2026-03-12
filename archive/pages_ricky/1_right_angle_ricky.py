"""Right Angle Ricky — standalone page for friends.

Reads ALL data from disk (``data/published/`` for NBA,
``data/publish_pga/`` for PGA) via the existing persistence helpers.
Each browser session is fully independent — no shared state with the
main YakOS app.

Two tabs:
  Tab 1 — Ricky's Edge Analysis  (4-box dashboard + published lineups)
  Tab 2 — Optimizer              (data_editor pool table + build)

NEW: star/select players from Edge Analysis → auto-Lock in Optimizer;
     Fade players → auto-Exclude in Optimizer.
"""

from __future__ import annotations

import hashlib as _hl
import io
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import SlateState, RickyEdgeState, LineupSetState, SimState  # noqa: E402
from yak_core.components import render_premium_lineup_card, render_lineup_cards_paged  # noqa: E402
from yak_core.edge_metrics import (  # noqa: E402
    compute_ricky_confidence_for_contest,
    get_confidence_color,
)
from yak_core.config import (  # noqa: E402
    CONTEST_PRESETS, UI_CONTEST_LABELS, UI_CONTEST_MAP,
    PGA_UI_CONTEST_LABELS, PGA_UI_CONTEST_MAP,
    DK_POS_SLOTS, DK_LINEUP_SIZE,
)
from yak_core.ricky_signals import compute_ricky_signals, generate_slate_overview  # noqa: E402
from yak_core.right_angle import (  # noqa: E402
    compute_pga_breakout_signals,
    generate_pga_slate_overview,
    apply_edge_adjustments,
    compute_breakout_candidates,
)
from yak_core.display_format import normalise_ownership, standard_player_format  # noqa: E402
from yak_core.lineups import (  # noqa: E402
    build_multiple_lineups_with_exposure,
    build_showdown_lineups,
    to_dk_upload_format,
    to_dk_pga_upload_format,
    to_dk_showdown_upload_format,
)
from yak_core.calibration import apply_archetype, DFS_ARCHETYPES  # noqa: E402
from yak_core.edge import compute_edge_metrics  # noqa: E402
from yak_core.lineup_store import (  # noqa: E402
    load_all_published,
    load_slate,
    load_edge,
    clear_published,
)
from yak_core.pga_state import (  # noqa: E402
    _pga_load_all_published,
    _pga_load_slate,
    _pga_load_edge,
    _pga_clear_published,
)

# ── GitHub Sync Check (diagnostic) ───────────────────────────────────

import json as _json
import os as _os
import subprocess as _sp

from yak_core.config import YAKOS_ROOT as _YAKOS_ROOT

_PUBLISHED_DIR = _os.path.join(_YAKOS_ROOT, "data", "published")
_SYNC_TEST_PATH = _os.path.join(_PUBLISHED_DIR, "sync_test.json")


@st.cache_data(ttl=60)
def _pull_latest_from_github() -> str:
    """Try git pull to get latest commits. Returns status message."""
    try:
        result = _sp.run(
            ["git", "pull", "origin", "main"],
            capture_output=True, text=True, timeout=15,
            cwd=_YAKOS_ROOT,
        )
        if result.returncode == 0:
            return f"git pull OK: {result.stdout.strip()}"
        return f"git pull failed (rc={result.returncode}): {result.stderr.strip()}"
    except Exception as e:
        return f"git pull error: {e}"


def _fetch_sync_test_via_api() -> dict | None:
    """Fallback: fetch sync_test.json via GitHub API if git pull didn't work."""
    try:
        token = None
        try:
            token = st.secrets.get("GITHUB_TOKEN")
        except Exception:
            token = _os.environ.get("GITHUB_TOKEN") or _os.environ.get("GH_TOKEN")
        if not token:
            return None
        import requests
        resp = requests.get(
            "https://api.github.com/repos/reace-ricky/YakOS/contents/data/published/sync_test.json",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"},
            params={"ref": "main"},
            timeout=10,
        )
        if resp.status_code == 200:
            import base64
            content = base64.b64decode(resp.json()["content"]).decode()
            return _json.loads(content)
    except Exception:
        pass
    return None


with st.expander("🔧 GitHub Sync Status (diagnostic)", expanded=False):
    _pull_msg = _pull_latest_from_github()
    st.text(_pull_msg)

    # Show sync_test.json if it exists on disk
    if _os.path.isfile(_SYNC_TEST_PATH):
        with open(_SYNC_TEST_PATH) as _f:
            _sync_data = _json.load(_f)
        st.success(f"Sync test file found on disk — timestamp: {_sync_data.get('timestamp', '?')}")
        st.json(_sync_data)
    else:
        st.warning("No sync test file found on disk.")
        # Try GitHub API fallback
        _api_data = _fetch_sync_test_via_api()
        if _api_data:
            st.info(f"Found via GitHub API — timestamp: {_api_data.get('timestamp', '?')}")
            st.json(_api_data)
        else:
            st.caption("Could not fetch via GitHub API either (no token or file doesn't exist yet).")

    # List files in data/published/
    st.markdown("**Files in `data/published/`:**")
    if _os.path.isdir(_PUBLISHED_DIR):
        _pub_files = []
        for _root, _dirs, _files in _os.walk(_PUBLISHED_DIR):
            for _file in _files:
                _pub_files.append(_os.path.relpath(_os.path.join(_root, _file), _PUBLISHED_DIR))
        if _pub_files:
            for _pf in sorted(_pub_files):
                st.text(f"  {_pf}")
        else:
            st.caption("Directory exists but is empty.")
    else:
        st.caption("Directory does not exist yet.")

# ── Session-state key prefix (unique to standalone app) ──────────────
_K = "_ricky_sa_"

# ---------------------------------------------------------------------------
# Contest display helpers (same as main app)
# ---------------------------------------------------------------------------

_CONTEST_ORDER = [UI_CONTEST_MAP[k] for k in UI_CONTEST_LABELS]
_LABEL_SHORT = {v: k for k, v in UI_CONTEST_MAP.items()}
_LABEL_SHORT.update({v: k for k, v in PGA_UI_CONTEST_MAP.items()})

_CONTEST_TO_BUILD_MODE = {
    "GPP Main": "ceiling", "GPP Early": "ceiling", "GPP Late": "ceiling",
    "Cash Main": "floor", "Showdown": "ceiling",
    "PGA GPP": "ceiling", "PGA Cash": "floor", "PGA Showdown": "ceiling",
}
_BUILD_MODE_PROJ_COL = {"floor": "floor", "median": "proj", "ceiling": "proj"}

_NBA_POS_FILTERS = ["All", "PG", "SG", "SF", "PF", "C"]

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
    seed = st.session_state.get(f"{_K}seed", "default")
    idx = int(_hl.md5(str(seed).encode()).hexdigest(), 16) % len(_RICKY_LINES)
    return _RICKY_LINES[idx]


# ---------------------------------------------------------------------------
# Disk-based data loading (cached 60s)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def _load_nba_slate_data() -> Dict[str, Any]:
    """Load NBA slate state from data/published/."""
    slate = SlateState()
    ok = load_slate(slate)
    if not ok:
        return {"ok": False, "slate_dict": {}, "pool": pd.DataFrame()}
    return {
        "ok": True,
        "slate_dict": {
            "sport": slate.sport, "site": slate.site,
            "slate_date": slate.slate_date,
            "contest_name": slate.contest_name,
            "contest_type": slate.contest_type,
            "is_showdown": slate.is_showdown,
            "roster_slots": slate.roster_slots,
            "lineup_size": slate.lineup_size,
            "salary_cap": slate.salary_cap,
            "captain_multiplier": slate.captain_multiplier,
            "proj_source": slate.proj_source,
            "published": slate.published,
            "published_at": slate.published_at,
            "active_layers": slate.active_layers,
            "selected_games": slate.selected_games,
            "calibration_state": slate.calibration_state,
        },
        "pool": slate.player_pool if slate.player_pool is not None else pd.DataFrame(),
        "edge_df": slate.edge_df if slate.edge_df is not None else pd.DataFrame(),
    }


@st.cache_data(ttl=60)
def _load_nba_edge_data() -> Dict[str, Any]:
    """Load NBA edge state from data/published/."""
    edge = RickyEdgeState()
    ok = load_edge(edge)
    if not ok:
        return {"ok": False}
    return {
        "ok": True,
        "player_tags": edge.player_tags,
        "game_tags": edge.game_tags,
        "stacks": edge.stacks,
        "edge_labels": edge.edge_labels,
        "slate_notes": edge.slate_notes,
        "ricky_edge_check": edge.ricky_edge_check,
        "edge_check_ts": edge.edge_check_ts,
        "approved_not_with_pairs": edge.approved_not_with_pairs,
        "auto_tags": edge.auto_tags,
        "auto_tag_reasons": edge.auto_tag_reasons,
        "confidence_scores": edge.confidence_scores,
        "player_tags_manual": edge.player_tags_manual,
        "edge_analysis_by_contest": edge.edge_analysis_by_contest,
    }


@st.cache_data(ttl=60)
def _load_nba_lineups() -> Dict[str, Dict[str, Any]]:
    return load_all_published()


@st.cache_data(ttl=60)
def _load_pga_slate_data() -> Dict[str, Any]:
    """Load PGA slate state from data/publish_pga/."""
    slate = SlateState(sport="PGA")
    ok = _pga_load_slate(slate)
    if not ok:
        return {"ok": False, "slate_dict": {}, "pool": pd.DataFrame()}
    return {
        "ok": True,
        "slate_dict": {
            "sport": slate.sport, "site": slate.site,
            "slate_date": slate.slate_date,
            "contest_name": slate.contest_name,
            "contest_type": slate.contest_type,
            "is_showdown": slate.is_showdown,
            "roster_slots": slate.roster_slots,
            "lineup_size": slate.lineup_size,
            "salary_cap": slate.salary_cap,
            "captain_multiplier": slate.captain_multiplier,
            "proj_source": slate.proj_source,
            "published": slate.published,
            "published_at": slate.published_at,
            "active_layers": slate.active_layers,
            "selected_games": slate.selected_games,
            "calibration_state": slate.calibration_state,
        },
        "pool": slate.player_pool if slate.player_pool is not None else pd.DataFrame(),
        "edge_df": slate.edge_df if slate.edge_df is not None else pd.DataFrame(),
    }


@st.cache_data(ttl=60)
def _load_pga_edge_data() -> Dict[str, Any]:
    """Load PGA edge state from data/publish_pga/."""
    edge = RickyEdgeState()
    ok = _pga_load_edge(edge)
    if not ok:
        return {"ok": False}
    return {
        "ok": True,
        "player_tags": edge.player_tags,
        "game_tags": edge.game_tags,
        "stacks": edge.stacks,
        "edge_labels": edge.edge_labels,
        "slate_notes": edge.slate_notes,
        "ricky_edge_check": edge.ricky_edge_check,
        "edge_check_ts": edge.edge_check_ts,
        "approved_not_with_pairs": edge.approved_not_with_pairs,
        "auto_tags": edge.auto_tags,
        "auto_tag_reasons": edge.auto_tag_reasons,
        "confidence_scores": edge.confidence_scores,
        "player_tags_manual": edge.player_tags_manual,
        "edge_analysis_by_contest": edge.edge_analysis_by_contest,
    }


@st.cache_data(ttl=60)
def _load_pga_lineups() -> Dict[str, Dict[str, Any]]:
    return _pga_load_all_published()


def _hydrate_slate(data: Dict[str, Any]) -> SlateState:
    """Reconstruct a SlateState from cached dict data."""
    slate = SlateState()
    d = data.get("slate_dict", {})
    for k, v in d.items():
        if hasattr(slate, k):
            setattr(slate, k, v)
    pool = data.get("pool", pd.DataFrame())
    if not pool.empty:
        slate.player_pool = pool
    edge_df = data.get("edge_df", pd.DataFrame())
    if not edge_df.empty:
        slate.edge_df = edge_df
    return slate


def _hydrate_edge(data: Dict[str, Any]) -> RickyEdgeState:
    """Reconstruct a RickyEdgeState from cached dict data."""
    edge = RickyEdgeState()
    for k in (
        "player_tags", "game_tags", "stacks", "edge_labels", "slate_notes",
        "ricky_edge_check", "edge_check_ts", "approved_not_with_pairs",
        "auto_tags", "auto_tag_reasons", "confidence_scores",
        "player_tags_manual", "edge_analysis_by_contest",
    ):
        if k in data:
            setattr(edge, k, data[k])
    return edge


def _hydrate_lineup_state(published: Dict[str, Dict[str, Any]]) -> LineupSetState:
    """Reconstruct a LineupSetState from cached published data."""
    ls = LineupSetState()
    ls.published_sets = published
    return ls


# ---------------------------------------------------------------------------
# Admin pin gate
# ---------------------------------------------------------------------------

_ADMIN_PIN = st.secrets.get("ADMIN_PIN", "2018")


def _render_admin_clear(key_suffix: str, sport: str) -> None:
    """Pin-protected clear button for standalone app."""
    with st.expander("\u2699\ufe0f Admin", expanded=False):
        pin = st.text_input(
            "Enter PIN to clear", type="password",
            key=f"{_K}pin_{key_suffix}", max_chars=4,
        )
        if pin and pin == _ADMIN_PIN:
            if st.button(
                "\U0001f5d1\ufe0f Clear All Published Data",
                key=f"{_K}clear_pub_{key_suffix}",
                type="secondary",
            ):
                if sport == "PGA":
                    _pga_clear_published()
                else:
                    clear_published()
                st.cache_data.clear()
                st.rerun()
        elif pin:
            st.error("Wrong PIN.")


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------

def _status_strip(slate: SlateState) -> None:
    parts = []
    if slate.sport:
        parts.append(f"**{slate.sport}**")
    if slate.slate_date:
        parts.append(slate.slate_date)
    if slate.site:
        parts.append(slate.site)
    if parts:
        st.caption(" \u00b7 ".join(parts))


def _confidence_pills(edge: RickyEdgeState) -> None:
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
# Tab 1 helpers — best lineup from published
# ---------------------------------------------------------------------------

def _get_best_lineup(lu_state: LineupSetState, contest_label: str) -> tuple:
    pub = lu_state.published_sets.get(contest_label)
    if pub is None:
        return None, None, None
    pub_df = pub.get("lineups_df", pd.DataFrame())
    boom_bust_df = pub.get("boom_bust_df")
    if pub_df is None or pub_df.empty:
        return None, None, None

    best_idx = 0
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        if "boom_score" in boom_bust_df.columns:
            best_idx = int(boom_bust_df.sort_values("boom_score", ascending=False).iloc[0]["lineup_index"])
        else:
            best_idx = int(boom_bust_df.iloc[0]["lineup_index"])
    elif "lineup_index" in pub_df.columns:
        best_idx = int(pub_df["lineup_index"].min())

    lu_rows = pub_df[pub_df["lineup_index"] == best_idx] if "lineup_index" in pub_df.columns else pub_df

    bb_row = None
    if boom_bust_df is not None and not boom_bust_df.empty and "lineup_index" in boom_bust_df.columns:
        bb_match = boom_bust_df[boom_bust_df["lineup_index"] == best_idx]
        if not bb_match.empty:
            bb_row = bb_match.iloc[0].to_dict()

    return lu_rows, None, bb_row


# ---------------------------------------------------------------------------
# HTML card renderer
# ---------------------------------------------------------------------------

_CARD_COLORS = {
    "core": "#f7931e",
    "leverage": "#a855f7",
    "value": "#4ade80",
    "fade": "#ef4444",
}


def _render_play_card(
    title: str,
    players: pd.DataFrame,
    color: str,
    badge_label: str,
    stat_format: str = "proj_salary",
) -> None:
    _badge_text = {"#4ade80": "#000", "#f7931e": "#000"}
    badge_fg = _badge_text.get(color, "#fff")

    rows_html = ""
    if players.empty:
        rows_html = (
            "<div style='padding:12px 0;font-size:12px;color:rgba(255,255,255,0.4);'>"
            "No players match this category for the current slate.</div>"
        )
    else:
        for _, r in players.iterrows():
            name = r.get("player_name", "")
            sal = float(r.get("salary", 0) or 0)
            proj = float(r.get("proj", 0) or 0)
            val = proj / (sal / 1000) if sal > 0 else 0

            if stat_format == "value":
                stat_str = f"${sal:,.0f} | {val:.2f} pts/$1K"
            elif stat_format == "proj_val":
                stat_str = f"{proj:.1f} pts | {val:.2f} val"
            else:
                stat_str = f"{proj:.1f} pts | ${sal:,.0f}"

            rows_html += (
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;padding:6px 0;"
                f"border-bottom:1px solid rgba(255,255,255,0.08);'>"
                f"<div>"
                f"<span style='font-weight:600;font-size:13px;'>{name}</span>"
                f"<span style='display:inline-block;padding:2px 6px;border-radius:4px;"
                f"font-size:9px;font-weight:bold;margin-left:6px;"
                f"background:{color};color:{badge_fg};'>{badge_label}</span>"
                f"</div>"
                f"<div style='text-align:right;font-size:12px;color:rgba(255,255,255,0.85);'>"
                f"{stat_str}</div>"
                f"</div>"
            )

    html = (
        f"<div style='background:rgba(255,255,255,0.06);border-radius:10px;"
        f"padding:12px;border-left:3px solid {color};margin-bottom:4px;'>"
        f"<h3 style='font-size:14px;margin-bottom:10px;color:{color};'>"
        f"{title}</h3>"
        f"{rows_html}"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Classify players into 4 buckets (NBA)
# ---------------------------------------------------------------------------

def _safe_col(frame: pd.DataFrame, name: str, default: float = 0) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index)


def _classify_nba(signals_df: pd.DataFrame):
    """Return (core, leverage, value, fades) DataFrames for NBA."""
    sdf = signals_df.copy()
    _sal = _safe_col(sdf, "salary")
    _proj = _safe_col(sdf, "proj")
    _own_col = "ownership" if "ownership" in sdf.columns else "own_pct"
    _own = normalise_ownership(_safe_col(sdf, _own_col))
    _edge = _safe_col(sdf, "edge_composite")
    _val = np.where(_sal > 0, _proj / (_sal / 1000), 0)
    sdf["_sal"] = _sal
    sdf["_proj"] = _proj
    sdf["_own"] = _own
    sdf["_edge"] = _edge
    sdf["_val"] = _val

    core = sdf[sdf["_sal"] >= 7000].nlargest(5, "_proj")
    _used = set(core["player_name"].tolist())

    _lev_pool = sdf[(sdf["_own"] < 15) & (~sdf["player_name"].isin(_used))]
    leverage = _lev_pool.nlargest(5, "_edge")
    _used.update(leverage["player_name"].tolist())

    _val_pool = sdf[(sdf["_sal"] < 6500) & (sdf["_sal"] > 0) & (~sdf["player_name"].isin(_used))]
    value = _val_pool.nlargest(5, "_val")
    _used.update(value["player_name"].tolist())

    _fade_pool = sdf[~sdf["player_name"].isin(_used)].copy()
    _fade_high_own = _fade_pool[_fade_pool["_own"] >= 10]
    if len(_fade_high_own) >= 3:
        fades = _fade_high_own.nsmallest(5, "_edge")
    else:
        _fade_sal = _fade_pool[_fade_pool["_sal"] >= 5000]
        fades = _fade_sal.nsmallest(5, "_edge") if not _fade_sal.empty else _fade_pool.nsmallest(5, "_edge")

    return core, leverage, value, fades


def _classify_pga(signals_df: pd.DataFrame):
    """Return (core, leverage, value, fades) DataFrames for PGA."""
    sdf = signals_df.copy()
    _sal = _safe_col(sdf, "salary")
    _proj = _safe_col(sdf, "proj")
    _own_col = "ownership" if "ownership" in sdf.columns else "own_pct"
    _own = normalise_ownership(_safe_col(sdf, _own_col))
    _edge_col = "pga_edge_composite" if "pga_edge_composite" in sdf.columns else "edge_composite"
    _edge = _safe_col(sdf, _edge_col)
    _val = np.where(_sal > 0, _proj / (_sal / 1000), 0)
    sdf["_sal"] = _sal
    sdf["_proj"] = _proj
    sdf["_own"] = _own
    sdf["_edge"] = _edge
    sdf["_val"] = _val

    core = sdf[sdf["_sal"] >= 8000].nlargest(5, "_proj")
    _used = set(core["player_name"].tolist())

    _lev_pool = sdf[(sdf["_own"] < 15) & (~sdf["player_name"].isin(_used))]
    leverage = _lev_pool.nlargest(5, "_edge")
    _used.update(leverage["player_name"].tolist())

    _val_pool = sdf[(sdf["_sal"] < 7500) & (sdf["_sal"] > 0) & (~sdf["player_name"].isin(_used))]
    value = _val_pool.nlargest(5, "_val")
    _used.update(value["player_name"].tolist())

    _fade_pool = sdf[~sdf["player_name"].isin(_used)].copy()
    _fade_high_own = _fade_pool[_fade_pool["_own"] >= 10]
    if len(_fade_high_own) >= 3:
        fades = _fade_high_own.nsmallest(5, "_edge")
    else:
        _fade_sal = _fade_pool[_fade_pool["_sal"] >= 7000]
        fades = _fade_sal.nsmallest(5, "_edge") if not _fade_sal.empty else _fade_pool.nsmallest(5, "_edge")

    return core, leverage, value, fades


# ---------------------------------------------------------------------------
# PGA info cards (course, weather, waves, history)
# ---------------------------------------------------------------------------

def _render_pga_info_cards(pool: pd.DataFrame) -> None:
    attrs = pool.attrs if hasattr(pool, "attrs") else {}
    cards: list[tuple[str, str, str]] = []

    # 1. Course
    course_name = attrs.get("course_name", "")
    course_city = attrs.get("course_city", "")
    fit_cols = {
        "driving_dist_adj": "Distance", "driving_acc_adj": "Accuracy",
        "approach_fit": "Approach", "short_game_fit": "Short Game",
    }
    demands = []
    for col, label in fit_cols.items():
        if col in pool.columns:
            v = pool[col].dropna().var()
            if v and v > 0:
                demands.append((label, v))
    demands.sort(key=lambda x: x[1], reverse=True)
    if course_name:
        loc = f" — {course_city}" if course_city else ""
        body = f"<b>{course_name}</b>{loc}"
        if demands:
            tags = " &middot; ".join(d[0] for d in demands[:3])
            body += f"<br><span style='color:rgba(255,255,255,0.5);font-size:11px;'>Key demands: {tags}</span>"
        cards.append(("⛳ Course", body, "#3b82f6"))

    # 2. Weather
    weather = attrs.get("weather", {})
    cur = weather.get("current", {})
    daily = weather.get("daily", [])
    if cur:
        temp = cur.get("temp_f", "")
        wind = cur.get("wind_mph", "")
        wdir = cur.get("wind_dir", "")
        cond = cur.get("conditions", "")
        body = f"{temp}\u00b0F &middot; {wind} mph {wdir}"
        if cond:
            body += f" &middot; {cond}"
        if daily:
            rows = []
            for d in daily[:4]:
                dt = d.get("date", "")[-5:]
                hi = d.get("high_f", "")
                dw = d.get("wind_mph", "")
                rain = d.get("precip_chance", 0)
                rain_flag = " \U0001f327\ufe0f" if int(rain or 0) >= 40 else ""
                rows.append(f"{dt}: {hi}\u00b0 / {dw}mph{rain_flag}")
            body += "<br><span style='color:rgba(255,255,255,0.45);font-size:11px;'>" + " &nbsp;|&nbsp; ".join(rows) + "</span>"
        si = weather.get("scoring_impact", "")
        if si:
            body += f"<br><span style='font-size:11px;color:#f59e0b;'>{si}</span>"
        cards.append(("\u2600\ufe0f Weather", body, "#f59e0b"))

    # 3. Wave split
    if "early_late_wave" in pool.columns:
        wave_str = pool["early_late_wave"].astype(str).str.strip().str.lower()
        n_early = wave_str.str.contains("early", na=False).sum()
        n_late = wave_str.str.contains("late", na=False).sum()
        if n_early > 0 or n_late > 0:
            body = f"Early: {n_early} &middot; Late: {n_late}"
            body += "<br><span style='color:rgba(255,255,255,0.45);font-size:11px;'>AM wave typically 0.15-0.30 strokes easier</span>"
            early_df = pool[wave_str.str.contains("early", na=False)]
            late_df = pool[wave_str.str.contains("late", na=False)]
            parts = []
            if not early_df.empty and "proj" in early_df.columns:
                names = ", ".join(early_df.nlargest(3, "proj")["player_name"].tolist())
                parts.append(f"<span style='color:#06b6d4;'>AM:</span> {names}")
            if not late_df.empty and "proj" in late_df.columns:
                names = ", ".join(late_df.nlargest(3, "proj")["player_name"].tolist())
                parts.append(f"<span style='color:#06b6d4;'>PM:</span> {names}")
            if parts:
                body += "<br><span style='font-size:11px;'>" + " &nbsp;|&nbsp; ".join(parts) + "</span>"
            cards.append(("\U0001f30a Waves", body, "#06b6d4"))

    # 4. Course history
    if "course_history" in pool.columns:
        ch = pool[["player_name", "course_history"]].dropna(subset=["course_history"])
        ch = ch[ch["course_history"].abs() >= 0.2]
        if not ch.empty:
            top = ch.nlargest(4, "course_history")
            items = []
            for _, r in top.iterrows():
                adj = float(r["course_history"])
                sign = "+" if adj >= 0 else ""
                items.append(f"{r['player_name']} <span style='color:#8b5cf6;'>{sign}{adj:.1f}</span>")
            cards.append(("\U0001f4ca History", " &middot; ".join(items), "#8b5cf6"))

    if not cards:
        return

    cols = st.columns(len(cards))
    for col, (title, body, color) in zip(cols, cards):
        with col:
            html = (
                f"<div style='background:rgba(255,255,255,0.05);border-radius:8px;"
                f"padding:10px 12px;border-left:3px solid {color};min-height:80px;'>"
                f"<div style='font-size:11px;font-weight:700;color:{color};"
                f"margin-bottom:6px;letter-spacing:0.5px;'>{title}</div>"
                f"<div style='font-size:12px;color:rgba(255,255,255,0.85);line-height:1.5;'>"
                f"{body}</div></div>"
            )
            st.markdown(html, unsafe_allow_html=True)
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Player star/select widget (NEW FEATURE)
# ---------------------------------------------------------------------------

def _render_player_star_select(
    core: pd.DataFrame,
    leverage: pd.DataFrame,
    value: pd.DataFrame,
    fades: pd.DataFrame,
) -> None:
    """Multiselect widget below the 4-box grid for starring players.

    Updates ``_ricky_sa_starred_players`` and ``_ricky_sa_fade_players``
    in session state.
    """
    # Collect candidate players (core + leverage + value — NOT fades)
    candidate_names = []
    for df in [core, leverage, value]:
        if not df.empty and "player_name" in df.columns:
            candidate_names.extend(df["player_name"].tolist())
    candidate_names = list(dict.fromkeys(candidate_names))  # preserve order, dedupe

    # Collect fade players
    fade_names = []
    if not fades.empty and "player_name" in fades.columns:
        fade_names = fades["player_name"].tolist()

    # Always store fade_players
    st.session_state[f"{_K}fade_players"] = set(fade_names)

    if not candidate_names:
        return

    prev_starred = st.session_state.get(f"{_K}starred_players", set())
    # Filter previous starred to only include currently valid candidates
    valid_prev = [p for p in prev_starred if p in candidate_names]

    starred = st.multiselect(
        "\u2b50 Select players for Optimizer",
        options=candidate_names,
        default=valid_prev,
        key=f"{_K}star_multiselect",
    )
    st.session_state[f"{_K}starred_players"] = set(starred)

    if starred:
        st.caption(
            f"\u2b50 {len(starred)} player{'s' if len(starred) != 1 else ''} "
            f"selected for optimizer: {', '.join(starred)}"
        )


# ===========================================================================
# TAB 1 — EDGE ANALYSIS (NBA)
# ===========================================================================

def _render_tab_analysis_nba(
    slate: SlateState,
    edge: RickyEdgeState,
    lu_state: LineupSetState,
) -> None:
    has_pool = slate.player_pool is not None and not slate.player_pool.empty
    if not has_pool:
        st.info("No player pool available. Publish a slate from the main app first.")
        return

    pool = slate.player_pool.copy()

    # ── Display-time manual exclude filter ────────────────────────────
    _nba_excludes = st.session_state.get("_nba_manual_excludes", [])
    if _nba_excludes and "player_name" in pool.columns:
        _lower_ex = [n.lower() for n in _nba_excludes]
        pool = pool[~pool["player_name"].str.lower().isin(_lower_ex)].reset_index(drop=True)

    signals_df = compute_ricky_signals(pool)
    contest_type = slate.contest_name or "GPP"
    overview = generate_slate_overview(pool, signals_df, contest_type=contest_type)

    for bullet in overview["bullets"]:
        st.markdown(f"- {bullet}")
    if overview["recommendation"]:
        st.info(overview["recommendation"])

    st.divider()

    core, leverage, value, fades = _classify_nba(signals_df)

    # 4-box layout
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        _render_play_card("CORE PLAYS (Chalk)", core, _CARD_COLORS["core"], "CHALK", stat_format="proj_salary")
    with row1_c2:
        _render_play_card("LEVERAGE PLAYS (GPP Gold)", leverage, _CARD_COLORS["leverage"], "UNDEROWNED", stat_format="proj_val")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        _render_play_card("VALUE PLAYS (Salary Savers)", value, _CARD_COLORS["value"], "VALUE", stat_format="value")
    with row2_c2:
        _render_play_card("FADE CANDIDATES", fades, _CARD_COLORS["fade"], "FADE", stat_format="proj_salary")

    # NEW: Star/select players for optimizer
    _render_player_star_select(core, leverage, value, fades)

    # Published lineups
    _lineup_data = []
    for contest_label in _CONTEST_ORDER:
        lu_rows, sim_metrics, bb_row = _get_best_lineup(lu_state, contest_label)
        if lu_rows is not None and not lu_rows.empty:
            short = _LABEL_SHORT.get(contest_label, contest_label)
            _lineup_data.append((contest_label, short, lu_rows, sim_metrics, bb_row))

    if _lineup_data:
        st.divider()
        for i in range(0, len(_lineup_data), 2):
            chunk = _lineup_data[i:i + 2]
            cols = st.columns(len(chunk))
            for col, (contest_label, short, lu_rows, sim_metrics, bb_row) in zip(cols, chunk):
                with col:
                    render_premium_lineup_card(
                        lineup_rows=lu_rows,
                        sim_metrics=sim_metrics,
                        lineup_label=f"Top {short}",
                        salary_cap=slate.salary_cap,
                        boom_bust_row=bb_row,
                        compact=True,
                    )

        _render_admin_clear("nba", "NBA")


# ===========================================================================
# TAB 1 — EDGE ANALYSIS (PGA)
# ===========================================================================

def _render_tab_analysis_pga(
    slate: SlateState,
    lu_state: LineupSetState,
) -> None:
    has_pool = slate.player_pool is not None and not slate.player_pool.empty
    if not has_pool:
        st.info("No PGA slate loaded yet. Publish a PGA pool from the main app first.")
        return

    pool = slate.player_pool.copy()

    # ── Display-time manual exclude filter ────────────────────────────
    _pga_excludes = st.session_state.get("_pga_manual_excludes", [])
    if _pga_excludes and "player_name" in pool.columns:
        _lower_ex = [n.lower() for n in _pga_excludes]
        pool = pool[~pool["player_name"].str.lower().isin(_lower_ex)].reset_index(drop=True)

    _render_pga_info_cards(pool)

    signals_df = compute_pga_breakout_signals(pool)
    event_name = pool.attrs.get("event_name", "") if hasattr(pool, "attrs") else ""
    overview = generate_pga_slate_overview(pool, signals_df, event_name=event_name)

    _bullet_colors = [
        ("\U0001f4aa", "#f7931e"),
        ("\U0001f3af", "#4ade80"),
        ("\U0001f4b0", "#3b82f6"),
        ("\U0001f50d", "#a855f7"),
    ]
    for i, bullet in enumerate(overview["bullets"]):
        color = _bullet_colors[i][1] if i < len(_bullet_colors) else "rgba(255,255,255,0.3)"
        _html_bullet = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', bullet)
        st.markdown(
            f"<div style='padding:6px 10px;margin-bottom:4px;border-radius:6px;"
            f"border-left:3px solid {color};background:rgba(255,255,255,0.04);"
            f"font-size:13px;'>{_html_bullet}</div>",
            unsafe_allow_html=True,
        )
    if overview["recommendation"]:
        rec = overview["recommendation"]
        if "thin" in rec.lower() or "balanced" in rec.lower():
            _rec_bg = "rgba(239,68,68,0.12)"
            _rec_border = "#ef4444"
        elif "strong" in rec.lower() or "5+" in rec:
            _rec_bg = "rgba(74,222,128,0.12)"
            _rec_border = "#4ade80"
        else:
            _rec_bg = "rgba(59,130,246,0.12)"
            _rec_border = "#3b82f6"
        st.markdown(
            f"<div style='padding:10px 14px;margin-top:6px;border-radius:8px;"
            f"background:{_rec_bg};border:1px solid {_rec_border};"
            f"font-size:13px;font-style:italic;color:rgba(255,255,255,0.9);'>"
            f"{rec}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    core, leverage, value, fades = _classify_pga(signals_df)

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        _render_play_card("CORE PLAYS (Chalk)", core, _CARD_COLORS["core"], "CHALK", stat_format="proj_salary")
    with row1_c2:
        _render_play_card("LEVERAGE PLAYS (GPP Gold)", leverage, _CARD_COLORS["leverage"], "UNDEROWNED", stat_format="proj_val")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        _render_play_card("VALUE PLAYS (Salary Savers)", value, _CARD_COLORS["value"], "VALUE", stat_format="value")
    with row2_c2:
        _render_play_card("FADE CANDIDATES", fades, _CARD_COLORS["fade"], "FADE", stat_format="proj_salary")

    # NEW: Star/select players for optimizer
    _render_player_star_select(core, leverage, value, fades)

    # PGA lineups
    _pga_lineup_data = []
    for _pga_cl in ["PGA GPP", "PGA Cash", "PGA Showdown"]:
        lu_rows, sim_metrics, bb_row = _get_best_lineup(lu_state, _pga_cl)
        if lu_rows is not None and not lu_rows.empty:
            _short = _LABEL_SHORT.get(_pga_cl, _pga_cl)
            _pga_lineup_data.append((_pga_cl, _short, lu_rows, sim_metrics, bb_row))

    if _pga_lineup_data:
        st.divider()
        for i in range(0, len(_pga_lineup_data), 2):
            chunk = _pga_lineup_data[i:i + 2]
            cols = st.columns(len(chunk))
            for col, (_pga_cl, _short, lu_rows, sim_metrics, bb_row) in zip(cols, chunk):
                with col:
                    render_premium_lineup_card(
                        lineup_rows=lu_rows,
                        sim_metrics=sim_metrics,
                        lineup_label=f"Top {_short}",
                        salary_cap=slate.salary_cap,
                        boom_bust_row=bb_row,
                        compact=True,
                    )

        _render_admin_clear("pga", "PGA")


# ===========================================================================
# TAB 2 — OPTIMIZER
# ===========================================================================

def _extract_games(pool: pd.DataFrame) -> list[str]:
    opp_col = "opp" if "opp" in pool.columns else (
        "opponent" if "opponent" in pool.columns else None
    )
    if opp_col and "team" in pool.columns:
        teams = pool["team"].str.strip().str.upper().fillna("")
        opps = pool[opp_col].str.strip().str.upper().fillna("")
        pairs = {
            " vs ".join(sorted([t, o]))
            for t, o in zip(teams, opps)
            if t and o
        }
        return sorted(pairs)
    return []


def _filter_pool_by_games(pool: pd.DataFrame, selected_games: list[str]) -> pd.DataFrame:
    if not selected_games:
        return pool
    opp_col = "opp" if "opp" in pool.columns else (
        "opponent" if "opponent" in pool.columns else None
    )
    if not opp_col:
        return pool
    teams = pool["team"].str.strip().str.upper().fillna("")
    opps = pool[opp_col].str.strip().str.upper().fillna("")
    keys = pd.Series(
        [" vs ".join(sorted([t, o])) if t and o else t for t, o in zip(teams, opps)],
        index=pool.index,
    )
    return pool[keys.isin(selected_games)].reset_index(drop=True)


def _build_pool_display(
    pool: pd.DataFrame,
    edge_df: Optional[pd.DataFrame],
    sport: str,
    contest_label: str,
    pos_filter: str = "All",
) -> pd.DataFrame:
    df = pool.copy()

    if edge_df is not None and not edge_df.empty:
        _edge_cols = ["player_name"]
        for c in ["edge_score", "edge_label", "smash_prob", "bust_prob",
                   "leverage", "own_pct"]:
            if c in edge_df.columns and c not in df.columns:
                _edge_cols.append(c)
        if len(_edge_cols) > 1:
            _sub = edge_df[_edge_cols].drop_duplicates(subset=["player_name"])
            df = df.merge(_sub, on="player_name", how="left")

    _sal = _safe_col(df, "salary")
    _proj = _safe_col(df, "proj")
    df["value"] = np.where(_sal > 0, _proj / (_sal / 1000.0), 0.0)

    _own_col_opt = "ownership" if "ownership" in df.columns else "own_pct"
    _own = _safe_col(df, _own_col_opt)
    df["own_display"] = _own

    # Add Lock/Exclude from starred/fade players
    _starred = st.session_state.get(f"{_K}starred_players", set())
    _faded = st.session_state.get(f"{_K}fade_players", set())
    _prev_lock = st.session_state.get(f"{_K}locked_players", _starred)
    _prev_excl = st.session_state.get(f"{_K}excluded_players", _faded)
    df["Lock"] = df["player_name"].isin(_prev_lock)
    df["Exclude"] = df["player_name"].isin(_prev_excl)

    if sport == "NBA" and pos_filter != "All" and "pos" in df.columns:
        df = df[df["pos"].str.contains(pos_filter, case=False, na=False)].copy()

    is_cash = "cash" in contest_label.lower()
    is_pga = sport == "PGA"

    if is_pga:
        cols = ["Lock", "Exclude", "player_name", "salary", "proj", "own_display",
                "edge_score", "value"]
        rename = {
            "player_name": "Player", "salary": "Salary",
            "proj": "Proj", "own_display": "Own%",
            "edge_score": "Edge", "value": "Value",
        }
        for c, label in [("sg_total", "SG Total"), ("course_fit", "Course Fit")]:
            if c in df.columns:
                cols.insert(-2, c)
                rename[c] = label
    elif is_cash:
        cols = ["Lock", "Exclude", "player_name", "team", "opp", "pos",
                "salary", "proj", "floor", "own_display", "value"]
        rename = {
            "player_name": "Player", "team": "Team", "opp": "Opp",
            "pos": "Pos", "salary": "Salary", "proj": "Proj",
            "floor": "Floor", "own_display": "Own%", "value": "Value",
        }
    else:
        cols = ["Lock", "Exclude", "player_name", "team", "opp", "pos",
                "salary", "proj", "own_display", "edge_score", "value",
                "edge_label"]
        rename = {
            "player_name": "Player", "team": "Team", "opp": "Opp",
            "pos": "Pos", "salary": "Salary", "proj": "Proj",
            "own_display": "Own%", "edge_score": "Edge",
            "value": "Value", "edge_label": "Label",
        }

    cols = [c for c in cols if c in df.columns]
    display = df[cols].copy()
    display = display.rename(columns={k: v for k, v in rename.items() if k in display.columns})

    if "Edge" in display.columns and not is_cash:
        display = display.sort_values("Edge", ascending=False, na_position="last")
    elif "Proj" in display.columns:
        display = display.sort_values("Proj", ascending=False, na_position="last")

    return display.reset_index(drop=True)


def _render_tab_optimizer(slate: SlateState) -> None:
    has_pool = slate.player_pool is not None and not slate.player_pool.empty
    if not has_pool:
        st.info(
            "No slate loaded yet. Publish a slate from the main app "
            "before you can build lineups."
        )
        return

    pool = slate.player_pool.copy()
    sport = slate.sport or "NBA"

    # 1. Contest type selector
    if sport == "PGA":
        _labels = PGA_UI_CONTEST_LABELS
        _label_map = PGA_UI_CONTEST_MAP
    else:
        _labels = UI_CONTEST_LABELS
        _label_map = UI_CONTEST_MAP

    _ui_contest = st.radio(
        "Contest Type", _labels, horizontal=True, key=f"{_K}opt_contest",
    )
    contest_label = _label_map[_ui_contest]
    preset = CONTEST_PRESETS.get(contest_label, {})

    # 2. Build settings
    col_lu, col_exp, col_sal = st.columns(3)
    with col_lu:
        num_lineups = st.number_input(
            "# Lineups", min_value=1, max_value=150,
            value=int(preset.get("default_lineups", preset.get("num_lineups", 20))),
            key=f"{_K}opt_num_lineups",
        )
    with col_exp:
        max_exp = st.slider(
            "Max Exposure", min_value=0.10, max_value=1.0, step=0.05,
            value=float(preset.get("default_max_exposure", preset.get("max_exposure", 0.50))),
            key=f"{_K}opt_max_exp",
        )
    with col_sal:
        min_salary = st.number_input(
            "Min Salary Used", min_value=40000, max_value=50000, step=500,
            value=int(preset.get("min_salary", preset.get("min_salary_used", 46000))),
            key=f"{_K}opt_min_salary",
        )

    _n_games = len(_extract_games(pool))
    _n_players = len(pool)
    _game_str = f"{_n_games} games" if _n_games else ""
    _cap_str = f"${slate.salary_cap:,} cap"
    _date_str = slate.slate_date or ""
    _parts = [p for p in [_date_str, _game_str, f"{_n_players} players", _cap_str] if p]
    st.caption(f"\U0001f4cb {' \u00b7 '.join(_parts)}")

    # Game filter (NBA only)
    all_games = _extract_games(pool)
    build_games: list[str] = []
    if all_games and sport == "NBA":
        _lab_games = slate.selected_games if hasattr(slate, "selected_games") else []
        _default_all = not _lab_games
        with st.expander(f"Games ({len(all_games)})", expanded=False):
            for _g in all_games:
                _default_on = _g in _lab_games if _lab_games else _default_all
                if st.checkbox(_g, value=_default_on, key=f"{_K}gf_{_g}"):
                    build_games.append(_g)
        if build_games and len(build_games) < len(all_games):
            pool = _filter_pool_by_games(pool, build_games)

    # Compute edge metrics
    _edge_df = None
    try:
        _edge_df = compute_edge_metrics(
            pool, calibration_state=slate.calibration_state, sport=sport,
        )
    except Exception:
        pass

    _breakout_df = None
    try:
        _breakout_df = compute_breakout_candidates(pool, top_n=15)
    except Exception:
        pass

    try:
        pool, _edge_overrides = apply_edge_adjustments(pool, edge_df=_edge_df, breakout_df=_breakout_df)
    except Exception:
        _edge_overrides = {}

    # 3. Player pool table
    st.divider()

    pos_filter = "All"
    if sport == "NBA":
        pos_filter = st.radio(
            "Position", _NBA_POS_FILTERS, horizontal=True,
            key=f"{_K}opt_pos_filter", label_visibility="collapsed",
        )

    display_df = _build_pool_display(pool, _edge_df, sport, contest_label, pos_filter)

    col_config: Dict[str, Any] = {
        "Lock": st.column_config.CheckboxColumn("\U0001f512", width="small", default=False),
        "Exclude": st.column_config.CheckboxColumn("\u2715", width="small", default=False),
    }
    if "Salary" in display_df.columns:
        col_config["Salary"] = st.column_config.NumberColumn("Salary", format="$%d")
    if "Proj" in display_df.columns:
        col_config["Proj"] = st.column_config.NumberColumn("Proj", format="%.1f")
    if "Floor" in display_df.columns:
        col_config["Floor"] = st.column_config.NumberColumn("Floor", format="%.1f")
    if "Own%" in display_df.columns:
        col_config["Own%"] = st.column_config.NumberColumn("Own%", format="%.1f%%")
    if "Edge" in display_df.columns:
        col_config["Edge"] = st.column_config.NumberColumn("Edge", format="%.2f")
    if "Value" in display_df.columns:
        col_config["Value"] = st.column_config.NumberColumn("Value", format="%.1fx")
    if "SG Total" in display_df.columns:
        col_config["SG Total"] = st.column_config.NumberColumn("SG Total", format="%.2f")
    if "Course Fit" in display_df.columns:
        col_config["Course Fit"] = st.column_config.NumberColumn("Fit", format="%.2f")
    if "Label" in display_df.columns:
        col_config["Label"] = st.column_config.TextColumn("Label", width="medium")
    if "Player" in display_df.columns:
        col_config["Player"] = st.column_config.TextColumn("Player", width="medium")

    edited_df = st.data_editor(
        display_df,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        key=f"{_K}pool_editor",
        height=min(600, 40 + len(display_df) * 35),
    )

    # Extract lock/exclude from editor
    _locked = set()
    _excluded = set()
    _name_col = "Player" if "Player" in edited_df.columns else "player_name"
    if _name_col in edited_df.columns:
        if "Lock" in edited_df.columns:
            _locked = set(edited_df.loc[edited_df["Lock"] == True, _name_col].tolist())
        if "Exclude" in edited_df.columns:
            _excluded = set(edited_df.loc[edited_df["Exclude"] == True, _name_col].tolist())
    st.session_state[f"{_K}locked_players"] = _locked
    st.session_state[f"{_K}excluded_players"] = _excluded

    _lock_excl_parts = []
    if _locked:
        _lock_excl_parts.append(f"\U0001f512 {len(_locked)} locked")
    if _excluded:
        _lock_excl_parts.append(f"\u2715 {len(_excluded)} excluded")
    _auto_excl = _edge_overrides.get("auto_exclude", []) if isinstance(_edge_overrides, dict) else []
    if _auto_excl:
        _lock_excl_parts.append(f"\u26a0 {len(_auto_excl)} auto-excluded (bust risk)")
    if _lock_excl_parts:
        st.caption(" \u00b7 ".join(_lock_excl_parts))

    # 4. Build button
    st.divider()

    archetype = preset.get("archetype", "Balanced")
    _merged_exclude = list(_excluded | set(_auto_excl))
    is_showdown = contest_label == "Showdown" or slate.is_showdown

    _contest_type_map = {
        "GPP Main": "gpp", "GPP Early": "gpp", "GPP Late": "gpp",
        "Cash Main": "cash", "Showdown": "showdown",
        "PGA GPP": "gpp", "PGA Cash": "cash", "PGA Showdown": "gpp",
    }

    build_mode = _CONTEST_TO_BUILD_MODE.get(contest_label, "ceiling")
    proj_col = _BUILD_MODE_PROJ_COL.get(build_mode, "proj")
    if "cash" in contest_label.lower() and "floor" in pool.columns:
        proj_col = "floor"

    if st.button("\u26a1 Build Lineups", type="primary", key=f"{_K}opt_build", use_container_width=True):
        _pool = pool.copy()
        if "player_id" not in _pool.columns:
            if "player_name" in _pool.columns:
                _pool["player_id"] = _pool["player_name"]
            elif "dk_player_id" in _pool.columns:
                _pool["player_id"] = _pool["dk_player_id"]

        cfg = {
            "NUM_LINEUPS": int(num_lineups),
            "SALARY_CAP": slate.salary_cap,
            "MAX_EXPOSURE": float(max_exp),
            "MIN_SALARY_USED": int(min_salary),
            "LOCK": list(_locked),
            "EXCLUDE": _merged_exclude,
            "PROJ_COL": proj_col,
            "CONTEST_TYPE": _contest_type_map.get(contest_label, "gpp"),
            "POS_SLOTS": preset.get("pos_slots", DK_POS_SLOTS),
            "LINEUP_SIZE": preset.get("lineup_size", DK_LINEUP_SIZE),
        }

        if contest_label.startswith("PGA"):
            _pga_preset = CONTEST_PRESETS.get(contest_label, {})
            cfg["POS_SLOTS"] = _pga_preset.get("pos_slots", ["G"] * 6)
            cfg["LINEUP_SIZE"] = _pga_preset.get("lineup_size", 6)
            cfg["POS_CAPS"] = _pga_preset.get("pos_caps", {})
            cfg["GPP_MAX_PUNT_PLAYERS"] = _pga_preset.get("max_punt_players", 1)
            cfg["GPP_MIN_MID_PLAYERS"] = _pga_preset.get("min_mid_salary_players", 2)
            cfg["GPP_OWN_CAP"] = _pga_preset.get("own_cap", 5.0)
            cfg["GPP_MIN_LOW_OWN_PLAYERS"] = _pga_preset.get("min_low_own_players", 1)
            cfg["GPP_LOW_OWN_THRESHOLD"] = _pga_preset.get("low_own_threshold", 0.40)
            cfg["GPP_FORCE_GAME_STACK"] = _pga_preset.get("force_game_stack", False)

        if isinstance(_edge_overrides, dict):
            if _edge_overrides.get("max_exposure_players"):
                cfg["PLAYER_MAX_EXPOSURE"] = _edge_overrides["max_exposure_players"]
            if _edge_overrides.get("tier_player_names"):
                cfg["TIER_CONSTRAINTS"] = {
                    "tier_player_names": _edge_overrides["tier_player_names"],
                    "tier_min_players": _edge_overrides.get("tier_min_players", {}),
                    "tier_max_players": _edge_overrides.get("tier_max_players", {}),
                }

        try:
            with st.spinner(f"Building {num_lineups} {_ui_contest} lineups..."):
                if is_showdown:
                    lineups_df, expo_df = build_showdown_lineups(_pool, cfg)
                else:
                    opt_pool = apply_archetype(_pool.copy(), archetype)
                    lineups_df, expo_df = build_multiple_lineups_with_exposure(opt_pool, cfg)

            if lineups_df is not None and not lineups_df.empty:
                st.session_state[f"{_K}friend_lineups"] = lineups_df
                st.session_state[f"{_K}friend_expo"] = expo_df
                st.session_state[f"{_K}friend_contest"] = contest_label
                st.session_state[f"{_K}friend_is_showdown"] = is_showdown
                n_built = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else 1
                st.success(f"Built {n_built} lineups for **{_ui_contest}**.")
            else:
                st.error("Optimizer returned no lineups. Try different settings.")
        except Exception as exc:
            st.error(f"Optimizer error: {exc}")

    # 5. Lineup results
    friend_lineups = st.session_state.get(f"{_K}friend_lineups")
    if friend_lineups is None or friend_lineups.empty:
        return

    st.divider()

    n_lu = len(friend_lineups["lineup_index"].unique()) if "lineup_index" in friend_lineups.columns else 0

    if "proj" in friend_lineups.columns or "total_proj" in friend_lineups.columns:
        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        _sc1.metric("Lineups", n_lu)

        _total_sal_col = "total_salary" if "total_salary" in friend_lineups.columns else None
        if _total_sal_col:
            _lu_sals = friend_lineups.groupby("lineup_index")[_total_sal_col].first()
            _sc2.metric("Avg Salary", f"${_lu_sals.mean():,.0f}")
        elif "salary" in friend_lineups.columns:
            _lu_sals = friend_lineups.groupby("lineup_index")["salary"].sum()
            _sc2.metric("Avg Salary", f"${_lu_sals.mean():,.0f}")

        _total_proj_col = "total_proj" if "total_proj" in friend_lineups.columns else None
        if _total_proj_col:
            _lu_projs = friend_lineups.groupby("lineup_index")[_total_proj_col].first()
            _sc3.metric("Avg Proj", f"{_lu_projs.mean():.1f}")
            _sc4.metric("Top Lineup", f"{_lu_projs.max():.1f}")
        elif "proj" in friend_lineups.columns:
            _lu_projs = friend_lineups.groupby("lineup_index")["proj"].sum()
            _sc3.metric("Avg Proj", f"{_lu_projs.mean():.1f}")
            _sc4.metric("Top Lineup", f"{_lu_projs.max():.1f}")

    # Exposure table
    expo_df = st.session_state.get(f"{_K}friend_expo")
    if expo_df is not None and not expo_df.empty:
        with st.expander("Player Exposures", expanded=False):
            _expo_fmt = standard_player_format(expo_df)
            st.dataframe(
                expo_df.style.format(_expo_fmt, na_rep=""),
                use_container_width=True, hide_index=True,
            )

    # Lineup cards (paged)
    render_lineup_cards_paged(
        lineups_df=friend_lineups,
        sim_results_df=None,
        salary_cap=slate.salary_cap,
        nav_key=f"{_K}friend",
    )

    # DK CSV Export
    st.divider()
    _is_sd = st.session_state.get(f"{_K}friend_is_showdown", False)
    _friend_contest = st.session_state.get(f"{_K}friend_contest", "")
    try:
        if _is_sd:
            dk_csv = to_dk_showdown_upload_format(friend_lineups)
        elif str(_friend_contest).startswith("PGA"):
            dk_csv = to_dk_pga_upload_format(friend_lineups)
        else:
            dk_csv = to_dk_upload_format(friend_lineups)

        if dk_csv is not None and not dk_csv.empty:
            buf = io.StringIO()
            dk_csv.to_csv(buf, index=False)
            st.download_button(
                label="\U0001f4e5 Download DK CSV",
                data=buf.getvalue(),
                file_name="ricky_lineups.csv",
                mime="text/csv",
                key=f"{_K}dk_export",
            )
    except Exception as exc:
        st.caption(f"CSV export unavailable: {exc}")


# ===========================================================================
# MAIN PAGE
# ===========================================================================

def main() -> None:
    st.title("\U0001f4d0 Right Angle Ricky")
    st.caption(_ricky_quote())

    # Sport toggle — session-state-first pattern (no dynamic index override)
    if f"{_K}sport_toggle" not in st.session_state:
        st.session_state[f"{_K}sport_toggle"] = "NBA"
    sport = st.radio(
        "Sport",
        ["NBA", "PGA"],
        horizontal=True,
        key=f"{_K}sport_toggle",
    )
    _is_pga = sport == "PGA"

    # Load data from disk based on sport
    if _is_pga:
        slate_data = _load_pga_slate_data()
        edge_data = _load_pga_edge_data()
        published = _load_pga_lineups()
    else:
        slate_data = _load_nba_slate_data()
        edge_data = _load_nba_edge_data()
        published = _load_nba_lineups()

    if not slate_data.get("ok"):
        st.info(
            f"No {sport} data published yet. Publish a slate from the "
            f"main YakOS app and it will appear here automatically."
        )
        return

    slate = _hydrate_slate(slate_data)
    edge = _hydrate_edge(edge_data) if edge_data.get("ok") else RickyEdgeState()
    lu_state = _hydrate_lineup_state(published)

    _status_strip(slate)

    has_edge = bool(edge.edge_analysis_by_contest)
    has_lineups = bool(published)
    has_pool = slate.player_pool is not None and not slate.player_pool.empty

    if not has_edge and not has_lineups and not has_pool:
        if _is_pga:
            st.info("No PGA data published yet. Publish from the main PGA app first.")
        else:
            st.info(
                "Ricky's got nothing to show yet. Publish a slate from the "
                "main YakOS app."
            )
        return

    # Confidence strip (NBA only)
    if not _is_pga:
        _confidence_pills(edge)
        if has_edge:
            st.divider()

    # Two tabs
    tab_analysis, tab_optimizer = st.tabs(
        ["\U0001f3af Ricky's Edge Analysis", "\U0001f527 Optimizer"]
    )

    with tab_analysis:
        if _is_pga:
            _render_tab_analysis_pga(slate, lu_state)
        else:
            _render_tab_analysis_nba(slate, edge, lu_state)

    with tab_optimizer:
        _render_tab_optimizer(slate)


main()
