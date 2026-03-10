"""File-based persistence for published state.

Persists lineup sets, slate state (player pool + config), and edge
state to ``data/published/``.  This allows Right Angle Ricky to work
for anyone opening the app — no need to go through The Lab first.
Data survives page refreshes and new sessions until explicitly cleared.

Re-publishing from The Lab overwrites the persisted data (late swaps,
new projections, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISH_DIR = _REPO_ROOT / "data" / "published"


def _safe_label(contest_label: str) -> str:
    """Convert a contest label to a filesystem-safe slug."""
    return contest_label.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_published_set(contest_label: str, published_data: Dict[str, Any]) -> None:
    """Persist a single published lineup set to disk.

    Parameters
    ----------
    contest_label : str
        e.g. ``"GPP Main"``, ``"PGA GPP"``
    published_data : dict
        The dict stored in ``LineupSetState.published_sets[label]``.
        Expected keys: ``lineups_df``, ``config``, ``published_at``,
        ``boom_bust_df`` (optional), ``exposure_df`` (optional).
    """
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    slug = _safe_label(contest_label)

    # Lineups DataFrame (required)
    lineups_df = published_data.get("lineups_df")
    if lineups_df is not None and not lineups_df.empty:
        lineups_df.to_parquet(PUBLISH_DIR / f"{slug}_lineups.parquet", index=False)

    # Boom/bust DataFrame (optional)
    bb_df = published_data.get("boom_bust_df")
    if bb_df is not None and isinstance(bb_df, pd.DataFrame) and not bb_df.empty:
        bb_df.to_parquet(PUBLISH_DIR / f"{slug}_boom_bust.parquet", index=False)

    # Exposure DataFrame (optional)
    expo_df = published_data.get("exposure_df")
    if expo_df is not None and isinstance(expo_df, pd.DataFrame) and not expo_df.empty:
        expo_df.to_parquet(PUBLISH_DIR / f"{slug}_exposure.parquet", index=False)

    # Metadata JSON (config, timestamps, original label)
    meta = {
        "contest_label": contest_label,
        "published_at": published_data.get("published_at", ""),
        "config": published_data.get("config", {}),
    }
    (PUBLISH_DIR / f"{slug}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_all_published() -> Dict[str, Dict[str, Any]]:
    """Load every published set from disk.

    Returns a dict matching the shape of
    ``LineupSetState.published_sets``:
    ``{contest_label: {"lineups_df": ..., "config": ..., ...}}``.
    """
    result: Dict[str, Dict[str, Any]] = {}
    if not PUBLISH_DIR.exists():
        return result

    for meta_path in sorted(PUBLISH_DIR.glob("*_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        contest_label = meta.get("contest_label", "")
        if not contest_label:
            continue

        slug = _safe_label(contest_label)

        # Lineups
        lu_path = PUBLISH_DIR / f"{slug}_lineups.parquet"
        if not lu_path.exists():
            continue
        try:
            lineups_df = pd.read_parquet(lu_path)
        except Exception:
            continue

        # Boom/bust (optional)
        bb_df = None
        bb_path = PUBLISH_DIR / f"{slug}_boom_bust.parquet"
        if bb_path.exists():
            try:
                bb_df = pd.read_parquet(bb_path)
            except Exception:
                pass

        # Exposure (optional)
        expo_df = None
        expo_path = PUBLISH_DIR / f"{slug}_exposure.parquet"
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


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def clear_published(contest_label: Optional[str] = None) -> None:
    """Delete persisted published data.

    Parameters
    ----------
    contest_label : str or None
        If provided, only delete files for that contest.
        If ``None``, delete *all* published files (lineups, slate, edge).
    """
    if not PUBLISH_DIR.exists():
        return

    if contest_label is not None:
        slug = _safe_label(contest_label)
        for f in PUBLISH_DIR.glob(f"{slug}_*"):
            f.unlink(missing_ok=True)
    else:
        for f in PUBLISH_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)


# ===========================================================================
# Slate State persistence
# ===========================================================================

def save_slate(slate) -> None:
    """Persist the SlateState to disk.

    Saves:
    - player_pool as parquet
    - edge_df as parquet (if present)
    - scalar config fields as JSON
    """
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)

    # Player pool
    if slate.player_pool is not None and not slate.player_pool.empty:
        slate.player_pool.to_parquet(
            PUBLISH_DIR / "slate_pool.parquet", index=False,
        )

    # Edge DataFrame
    if slate.edge_df is not None and not slate.edge_df.empty:
        slate.edge_df.to_parquet(
            PUBLISH_DIR / "slate_edge_df.parquet", index=False,
        )

    # Scalar / JSON-serialisable config
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
    (PUBLISH_DIR / "slate_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8",
    )


def load_slate(slate) -> bool:
    """Restore SlateState from disk.  Returns True if data was restored.

    Parameters
    ----------
    slate : SlateState
        The state object to populate in-place.
    """
    meta_path = PUBLISH_DIR / "slate_meta.json"
    pool_path = PUBLISH_DIR / "slate_pool.parquet"
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

    # Restore scalar fields
    slate.sport = meta.get("sport", slate.sport)
    slate.site = meta.get("site", slate.site)
    slate.slate_date = meta.get("slate_date", slate.slate_date)
    slate.contest_name = meta.get("contest_name", slate.contest_name)
    slate.contest_type = meta.get("contest_type", slate.contest_type)
    slate.is_showdown = meta.get("is_showdown", slate.is_showdown)
    slate.roster_slots = meta.get("roster_slots", slate.roster_slots)
    slate.lineup_size = meta.get("lineup_size", slate.lineup_size)
    slate.salary_cap = meta.get("salary_cap", slate.salary_cap)
    slate.captain_multiplier = meta.get("captain_multiplier", slate.captain_multiplier)
    slate.proj_source = meta.get("proj_source", slate.proj_source)
    slate.published = meta.get("published", slate.published)
    slate.published_at = meta.get("published_at", slate.published_at)
    slate.active_layers = meta.get("active_layers", slate.active_layers)
    slate.selected_games = meta.get("selected_games", slate.selected_games)
    slate.calibration_state = meta.get("calibration_state", slate.calibration_state)

    # Restore DataFrames
    slate.player_pool = pool

    edge_path = PUBLISH_DIR / "slate_edge_df.parquet"
    if edge_path.exists():
        try:
            slate.edge_df = pd.read_parquet(edge_path)
        except Exception:
            pass

    return True


# ===========================================================================
# Edge State persistence
# ===========================================================================

def save_edge(edge) -> None:
    """Persist RickyEdgeState to disk as JSON."""
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)

    # edge_analysis_by_contest contains nested dicts with possible
    # DataFrames — serialise DataFrames as dicts
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
    (PUBLISH_DIR / "edge_state.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8",
    )


def load_edge(edge) -> bool:
    """Restore RickyEdgeState from disk.  Returns True if data was restored."""
    path = PUBLISH_DIR / "edge_state.json"
    if not path.exists():
        return False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    edge.player_tags = data.get("player_tags", edge.player_tags)
    edge.game_tags = data.get("game_tags", edge.game_tags)
    edge.stacks = data.get("stacks", edge.stacks)
    edge.edge_labels = data.get("edge_labels", edge.edge_labels)
    edge.slate_notes = data.get("slate_notes", edge.slate_notes)
    edge.ricky_edge_check = data.get("ricky_edge_check", edge.ricky_edge_check)
    edge.edge_check_ts = data.get("edge_check_ts", edge.edge_check_ts)
    edge.approved_not_with_pairs = data.get("approved_not_with_pairs", edge.approved_not_with_pairs)
    edge.auto_tags = data.get("auto_tags", edge.auto_tags)
    edge.auto_tag_reasons = data.get("auto_tag_reasons", edge.auto_tag_reasons)
    edge.confidence_scores = data.get("confidence_scores", edge.confidence_scores)
    edge.player_tags_manual = data.get("player_tags_manual", edge.player_tags_manual)

    # Restore edge_analysis_by_contest, re-hydrating DataFrames
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
