"""Shared data loading functions for the YakOS Streamlit app.

All public-tab data is read from data/published/{sport}/ and cached until the
published files change.  The cache key includes a ``published_version`` derived
from the modification times of all relevant artifacts, so the cache busts
automatically whenever a new publish lands — no manual Refresh required.

Lab tab reads bypass the cache to always get fresh data after writes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "published"

# Files tracked for cache-busting (relative to the sport directory).
_VERSIONED_FILES = [
    "slate_meta.json",
    "slate_pool.parquet",
    "edge_analysis.json",
    "edge_state.json",
]


def get_published_version(sport: str) -> str:
    """Return a version string for the published data of *sport*.

    The version is a colon-separated list of ``<filename>:<mtime_ns>`` pairs
    for every tracked artifact in ``data/published/{sport}/``.  It changes
    whenever any of those files is updated, so callers can pass it as a cache
    key argument to :func:`load_published_data` to get automatic cache
    invalidation without waiting for the TTL to expire.

    Returns an empty string when the directory does not exist (treated as a
    single, stable "empty" version by the cache).
    """
    base = DATA_DIR / sport.lower()
    if not base.is_dir():
        return ""

    parts: list[str] = []
    # Fixed tracked files
    for fname in _VERSIONED_FILES:
        p = base / fname
        try:
            parts.append(f"{fname}:{p.stat().st_mtime_ns}")
        except FileNotFoundError:
            parts.append(f"{fname}:missing")

    # Dynamic lineup files (names vary by contest slug)
    for lf in sorted(base.glob("*_lineups.parquet")):
        try:
            parts.append(f"{lf.name}:{lf.stat().st_mtime_ns}")
        except FileNotFoundError:
            pass

    return ":".join(parts)


@st.cache_data(ttl=3600)
def load_published_data(sport: str, published_version: str = "") -> Tuple[
    Dict[str, Any],        # slate_meta
    pd.DataFrame,          # slate_pool
    Dict[str, Any],        # edge_analysis
    Dict[str, Any],        # edge_state
    Dict[str, pd.DataFrame],  # lineups by contest slug
]:
    """Load all published data for a sport.

    Results are cached until *published_version* changes.  Pass the return
    value of :func:`get_published_version` as the second argument so that
    Streamlit invalidates the cache automatically whenever the published
    artifacts are updated (instead of waiting up to 5 minutes for the old
    TTL-based expiry).

    The ``published_version`` parameter is intentionally included in the
    function signature so that Streamlit's ``@st.cache_data`` incorporates it
    into the cache key.
    """
    base = DATA_DIR / sport.lower()

    meta: Dict[str, Any] = {}
    pool = pd.DataFrame()
    edge_analysis: Dict[str, Any] = {}
    edge_state: Dict[str, Any] = {}
    lineups: Dict[str, pd.DataFrame] = {}

    meta_path = base / "slate_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    pool_path = base / "slate_pool.parquet"
    if pool_path.exists():
        pool = pd.read_parquet(pool_path)
        # Safety net: ensure ownership is valid even if parquet was saved with None values
        if isinstance(pool, (pd.DataFrame, pd.Series)) and not pool.empty:
            try:
                from yak_core.ownership_guard import ensure_ownership
                pool = ensure_ownership(pool, sport=sport)
            except Exception:
                pass  # Non-fatal — downstream code handles missing ownership gracefully

    ea_path = base / "edge_analysis.json"
    if ea_path.exists():
        edge_analysis = json.loads(ea_path.read_text())

    es_path = base / "edge_state.json"
    if es_path.exists():
        edge_state = json.loads(es_path.read_text())

    for lf in base.glob("*_lineups.parquet"):
        contest_slug = lf.stem.replace("_lineups", "")
        lineups[contest_slug] = pd.read_parquet(lf)

    return meta, pool, edge_analysis, edge_state, lineups


def invalidate_published_cache() -> None:
    """Clear the Streamlit cache for load_published_data so the Edge tab picks up new data."""
    load_published_data.clear()


def load_fresh_pool(sport: str) -> pd.DataFrame:
    """Load pool without cache (for Lab tab after writes)."""
    pool_path = DATA_DIR / sport.lower() / "slate_pool.parquet"
    if pool_path.exists():
        pool = pd.read_parquet(pool_path)
        if isinstance(pool, (pd.DataFrame, pd.Series)) and not pool.empty:
            try:
                from yak_core.ownership_guard import ensure_ownership
                pool = ensure_ownership(pool, sport=sport)
            except Exception:
                pass
        return pool
    return pd.DataFrame()


def load_fresh_meta(sport: str) -> Dict[str, Any]:
    """Load slate meta without cache."""
    meta_path = DATA_DIR / sport.lower() / "slate_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def load_calibration_data(sport: str) -> Dict[str, Any]:
    """Load calibration feedback data."""
    base = REPO_ROOT / "data" / "calibration_feedback" / sport.lower()
    result: Dict[str, Any] = {}

    cf_path = base / "correction_factors.json"
    if cf_path.exists():
        result["correction_factors"] = json.loads(cf_path.read_text())

    se_path = base / "slate_errors.json"
    if se_path.exists():
        result["slate_errors"] = json.loads(se_path.read_text())

    return result


def load_signal_history() -> Dict[str, Any]:
    """Load edge signal history."""
    path = REPO_ROOT / "data" / "edge_feedback" / "signal_history.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def published_dir(sport: str) -> Path:
    """Return the published data directory for a sport, creating if needed."""
    d = DATA_DIR / sport.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d
