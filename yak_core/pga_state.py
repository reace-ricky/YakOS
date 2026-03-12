"""PGA-specific session state accessors.

Mirrors yak_core.state accessors but uses PGA-prefixed session state keys
and PGA-specific persistence paths so PGA and NBA apps are fully isolated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from yak_core.state import (
    SlateState,
    RickyEdgeState,
    LineupSetState,
    SimState,
)

# ---------------------------------------------------------------------------
# Session state keys (PGA-prefixed — never collide with NBA keys)
# ---------------------------------------------------------------------------
_KEY_SLATE = "_pga_slate_state"
_KEY_EDGE = "_pga_edge_state"
_KEY_LINEUP = "_pga_lineup_state"
_KEY_SIM = "_pga_sim_state"

# ---------------------------------------------------------------------------
# PGA publish directory (separate from NBA's data/published/)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
PGA_PUBLISH_DIR = _REPO_ROOT / "data" / "publish_pga"


def _ss():
    import streamlit as st
    return st.session_state


# ---------------------------------------------------------------------------
# PGA persistence helpers (mirror lineup_store.py but use PGA_PUBLISH_DIR)
# ---------------------------------------------------------------------------

def _pga_save_published_set(contest_label: str, published_data: Dict[str, Any]) -> None:
    import json
    PGA_PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    slug = contest_label.lower().replace(" ", "_")

    lineups_df = published_data.get("lineups_df")
    if lineups_df is not None and not lineups_df.empty:
        lineups_df.to_parquet(PGA_PUBLISH_DIR / f"{slug}_lineups.parquet", index=False)

    bb_df = published_data.get("boom_bust_df")
    if bb_df is not None and isinstance(bb_df, pd.DataFrame) and not bb_df.empty:
        bb_df.to_parquet(PGA_PUBLISH_DIR / f"{slug}_boom_bust.parquet", index=False)

    expo_df = published_data.get("exposure_df")
    if expo_df is not None and isinstance(expo_df, pd.DataFrame) and not expo_df.empty:
        expo_df.to_parquet(PGA_PUBLISH_DIR / f"{slug}_exposure.parquet", index=False)

    meta = {
        "contest_label": contest_label,
        "published_at": published_data.get("published_at", ""),
        "config": published_data.get("config", {}),
    }
    (PGA_PUBLISH_DIR / f"{slug}_meta.json").write_text(
        __import__("json").dumps(meta, indent=2, default=str), encoding="utf-8",
    )


def _pga_load_all_published() -> Dict[str, Dict[str, Any]]:
    import json
    result: Dict[str, Dict[str, Any]] = {}
    if not PGA_PUBLISH_DIR.exists():
        return result

    for meta_path in sorted(PGA_PUBLISH_DIR.glob("*_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        contest_label = meta.get("contest_label", "")
        if not contest_label:
            continue
        slug = contest_label.lower().replace(" ", "_")

        lu_path = PGA_PUBLISH_DIR / f"{slug}_lineups.parquet"
        if not lu_path.exists():
            continue
        try:
            lineups_df = pd.read_parquet(lu_path)
        except Exception:
            continue

        bb_df = None
        bb_path = PGA_PUBLISH_DIR / f"{slug}_boom_bust.parquet"
        if bb_path.exists():
            try:
                bb_df = pd.read_parquet(bb_path)
            except Exception:
                pass

        expo_df = None
        expo_path = PGA_PUBLISH_DIR / f"{slug}_exposure.parquet"
        if expo_path.exists():
            try:
                expo_df = pd.read_parquet(expo_path)
            except Exception:
                pass

        result[contest_label] = {
            "lineups_df": lineups_df,
            "config": meta.get("config", {}),
            "published_at": meta.get("published_at", ""),
            "boom_bust_df": bb_df,
            "exposure_df": expo_df,
        }
    return result


def _pga_save_slate(slate: SlateState) -> None:
    import json
    PGA_PUBLISH_DIR.mkdir(parents=True, exist_ok=True)

    if slate.player_pool is not None and not slate.player_pool.empty:
        slate.player_pool.to_parquet(PGA_PUBLISH_DIR / "slate_pool.parquet", index=False)

    if slate.edge_df is not None and not slate.edge_df.empty:
        slate.edge_df.to_parquet(PGA_PUBLISH_DIR / "slate_edge_df.parquet", index=False)

    meta = {
        "sport": slate.sport,
        "site": slate.site,
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
    }
    (PGA_PUBLISH_DIR / "slate_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8",
    )


def _pga_load_slate(slate: SlateState) -> bool:
    import json
    meta_path = PGA_PUBLISH_DIR / "slate_meta.json"
    pool_path = PGA_PUBLISH_DIR / "slate_pool.parquet"
    if not meta_path.exists() or not pool_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    try:
        pool = pd.read_parquet(pool_path)
    except Exception:
        return False

    for k in (
        "sport", "site", "slate_date", "contest_name", "contest_type",
        "is_showdown", "roster_slots", "lineup_size", "salary_cap",
        "captain_multiplier", "proj_source", "published", "published_at",
        "active_layers", "selected_games", "calibration_state",
    ):
        setattr(slate, k, meta.get(k, getattr(slate, k)))

    slate.player_pool = pool

    edge_path = PGA_PUBLISH_DIR / "slate_edge_df.parquet"
    if edge_path.exists():
        try:
            slate.edge_df = pd.read_parquet(edge_path)
        except Exception:
            pass
    return True


def _pga_save_edge(edge: RickyEdgeState) -> None:
    import json
    PGA_PUBLISH_DIR.mkdir(parents=True, exist_ok=True)

    edge_by_contest = {}
    for label, payload in edge.edge_analysis_by_contest.items():
        serialised = {}
        for k, v in payload.items():
            if isinstance(v, pd.DataFrame):
                serialised[k] = {"__df__": True, "data": v.to_dict(orient="list")}
            else:
                serialised[k] = v
        edge_by_contest[label] = serialised

    data = {
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
        "edge_analysis_by_contest": edge_by_contest,
    }
    (PGA_PUBLISH_DIR / "edge_state.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8",
    )


def _pga_load_edge(edge: RickyEdgeState) -> bool:
    import json
    path = PGA_PUBLISH_DIR / "edge_state.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    for k in (
        "player_tags", "game_tags", "stacks", "edge_labels", "slate_notes",
        "ricky_edge_check", "edge_check_ts", "approved_not_with_pairs",
        "auto_tags", "auto_tag_reasons", "confidence_scores", "player_tags_manual",
    ):
        setattr(edge, k, data.get(k, getattr(edge, k)))

    raw_ebc = data.get("edge_analysis_by_contest", {})
    for label, payload in raw_ebc.items():
        restored = {}
        for k, v in payload.items():
            if isinstance(v, dict) and v.get("__df__"):
                restored[k] = pd.DataFrame(v["data"])
            else:
                restored[k] = v
        edge.edge_analysis_by_contest[label] = restored
    return True


def _pga_clear_published(contest_label=None) -> None:
    if not PGA_PUBLISH_DIR.exists():
        return
    if contest_label is not None:
        slug = contest_label.lower().replace(" ", "_")
        for f in PGA_PUBLISH_DIR.glob(f"{slug}_*"):
            f.unlink(missing_ok=True)
    else:
        for f in PGA_PUBLISH_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Session state accessors (PGA-specific)
# ---------------------------------------------------------------------------

def get_slate_state() -> SlateState:
    ss = _ss()
    if _KEY_SLATE not in ss:
        ss[_KEY_SLATE] = SlateState(sport="PGA")
    state: SlateState = ss[_KEY_SLATE]
    if state.player_pool is None or (hasattr(state.player_pool, "empty") and state.player_pool.empty):
        try:
            _pga_load_slate(state)
        except Exception:
            pass
    return state


def set_slate_state(state: SlateState) -> None:
    _ss()[_KEY_SLATE] = state
    if state.published:
        try:
            _pga_save_slate(state)
        except Exception:
            pass


def get_edge_state() -> RickyEdgeState:
    ss = _ss()
    if _KEY_EDGE not in ss:
        ss[_KEY_EDGE] = RickyEdgeState()
    state: RickyEdgeState = ss[_KEY_EDGE]
    if not state.edge_analysis_by_contest and not state.ricky_edge_check:
        try:
            _pga_load_edge(state)
        except Exception:
            pass
    return state


def set_edge_state(state: RickyEdgeState) -> None:
    _ss()[_KEY_EDGE] = state
    if state.ricky_edge_check:
        try:
            _pga_save_edge(state)
        except Exception:
            pass


def get_lineup_state() -> LineupSetState:
    ss = _ss()
    if _KEY_LINEUP not in ss:
        ss[_KEY_LINEUP] = LineupSetState()
    state: LineupSetState = ss[_KEY_LINEUP]
    if not state.published_sets:
        try:
            persisted = _pga_load_all_published()
            if persisted:
                state.published_sets = persisted
        except Exception:
            pass
    return state


def set_lineup_state(state: LineupSetState) -> None:
    _ss()[_KEY_LINEUP] = state


def get_sim_state() -> SimState:
    ss = _ss()
    if _KEY_SIM not in ss:
        ss[_KEY_SIM] = SimState()
    return ss[_KEY_SIM]


def set_sim_state(state: SimState) -> None:
    _ss()[_KEY_SIM] = state


def pga_publish(lu_state: LineupSetState, contest_label: str, ts: str) -> None:
    """Publish lineups using PGA persistence path."""
    df = lu_state.lineups.get(contest_label)
    if df is not None:
        lu_state.published_sets[contest_label] = {
            "lineups_df": df.copy(),
            "config": lu_state.build_configs.get(contest_label, {}),
            "published_at": ts,
            "boom_bust_df": (
                lu_state.boom_bust_rankings[contest_label].copy()
                if contest_label in lu_state.boom_bust_rankings
                and lu_state.boom_bust_rankings[contest_label] is not None
                else None
            ),
            "exposure_df": (
                lu_state.exposures[contest_label].copy()
                if contest_label in lu_state.exposures
                and lu_state.exposures[contest_label] is not None
                else None
            ),
        }
        lu_state.snapshot_times[contest_label] = ts
        try:
            _pga_save_published_set(contest_label, lu_state.published_sets[contest_label])
        except Exception:
            pass
