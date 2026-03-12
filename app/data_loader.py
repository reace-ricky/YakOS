"""Shared data loading functions for the YakOS Streamlit app.

All public-tab data is read from data/published/{sport}/ and cached for 5 minutes.
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


@st.cache_data(ttl=300)
def load_published_data(sport: str) -> Tuple[
    Dict[str, Any],        # slate_meta
    pd.DataFrame,          # slate_pool
    Dict[str, Any],        # edge_analysis
    Dict[str, Any],        # edge_state
    Dict[str, pd.DataFrame],  # lineups by contest slug
]:
    """Load all published data for a sport (cached 5 min)."""
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


def load_fresh_pool(sport: str) -> pd.DataFrame:
    """Load pool without cache (for Lab tab after writes)."""
    pool_path = DATA_DIR / sport.lower() / "slate_pool.parquet"
    if pool_path.exists():
        return pd.read_parquet(pool_path)
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
