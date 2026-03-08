"""yak_core.tank01_ids -- Tank01 player ID resolution for YakOS.

Fetches the full NBA player list from Tank01 once per session and provides
fast name→playerID lookups.  Used by:
  - auto_flag_injuries (injury list match)
  - fetch_player_game_logs (game log fetch)
  - Ricky edge signals (salary drop, minutes trends)

The player list is cached in memory and on disk (24h TTL) to avoid
burning API credits on every page load.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

from .config import YAKOS_ROOT

_CACHE_TTL_SECONDS = 86400  # 24 hours
_CACHE_DIR = os.path.join(YAKOS_ROOT, "data", "tank01_cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "player_list.json")

# In-memory cache (survives across Streamlit reruns within same process)
_mem_cache: Optional[Dict[str, str]] = None
_mem_cache_ts: float = 0.0


def _is_cache_fresh() -> bool:
    """Check if disk cache exists and is within TTL."""
    if not os.path.exists(_CACHE_FILE):
        return False
    age = time.time() - os.path.getmtime(_CACHE_FILE)
    return age < _CACHE_TTL_SECONDS


def _load_disk_cache() -> Optional[Dict[str, str]]:
    """Load name→ID map from disk cache."""
    if not _is_cache_fresh():
        return None
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def _save_disk_cache(name_to_id: Dict[str, str]) -> None:
    """Persist name→ID map to disk."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(name_to_id, f)
    except Exception:
        pass


def fetch_player_id_map(api_key: str, force: bool = False) -> Dict[str, str]:
    """Return a {player_name: tank01_playerID} map for all NBA players.

    Uses getNBAPlayerList (single API call, ~1200 players).
    Cached in memory and on disk (24h TTL).

    Parameters
    ----------
    api_key : str
        Tank01 RapidAPI key.
    force : bool
        Force a fresh fetch even if cache is valid.

    Returns
    -------
    Dict[str, str]
        Mapping of player longName → Tank01 playerID.
    """
    global _mem_cache, _mem_cache_ts

    # Memory cache (fastest)
    if not force and _mem_cache and (time.time() - _mem_cache_ts) < _CACHE_TTL_SECONDS:
        return _mem_cache

    # Disk cache
    if not force:
        disk = _load_disk_cache()
        if disk:
            _mem_cache = disk
            _mem_cache_ts = time.time()
            return _mem_cache

    # Fetch from API
    import requests

    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerList"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        body = data.get("body", data) if isinstance(data, dict) else data

        if not isinstance(body, list):
            print(f"[tank01_ids] Unexpected response type: {type(body)}")
            return _mem_cache or {}

        name_to_id: Dict[str, str] = {}
        for p in body:
            if not isinstance(p, dict):
                continue
            name = str(p.get("longName", "")).strip()
            pid = str(p.get("playerID", "")).strip()
            if name and pid:
                name_to_id[name] = pid

        print(f"[tank01_ids] Fetched {len(name_to_id)} player IDs from Tank01")

        # Cache
        _mem_cache = name_to_id
        _mem_cache_ts = time.time()
        _save_disk_cache(name_to_id)

        return name_to_id

    except Exception as exc:
        print(f"[tank01_ids] API fetch failed: {exc}")
        # Fall back to whatever we have
        return _mem_cache or _load_disk_cache() or {}


def resolve_pool_ids(
    pool_names: list,
    api_key: str,
) -> Dict[str, str]:
    """Resolve a list of pool player names to Tank01 player IDs.

    Uses exact match first, then fuzzy last-name match for remaining.

    Parameters
    ----------
    pool_names : list of str
        Player names from the DK pool.
    api_key : str
        Tank01 RapidAPI key.

    Returns
    -------
    Dict[str, str]
        {pool_name: tank01_playerID} for matched players.
    """
    full_map = fetch_player_id_map(api_key)
    if not full_map:
        return {}

    result: Dict[str, str] = {}
    unmatched: list = []

    # Pass 1: exact match
    for name in pool_names:
        name_str = str(name).strip()
        if name_str in full_map:
            result[name_str] = full_map[name_str]
        else:
            unmatched.append(name_str)

    # Pass 2: fuzzy last-name match for unmatched
    if unmatched:
        # Build reverse index: last_name_lower → [(full_name, id)]
        last_name_index: Dict[str, list] = {}
        for fn, pid in full_map.items():
            parts = fn.split()
            if parts:
                ln = parts[-1].lower()
                last_name_index.setdefault(ln, []).append((fn, pid))

        for name_str in unmatched:
            parts = name_str.split()
            if not parts:
                continue
            ln = parts[-1].lower()
            candidates = last_name_index.get(ln, [])
            if len(candidates) == 1:
                # Unique last name match
                result[name_str] = candidates[0][1]
            elif len(candidates) > 1:
                # Multiple — try first initial match
                first_init = name_str[0].lower() if name_str else ""
                for cand_name, cand_id in candidates:
                    if cand_name and cand_name[0].lower() == first_init:
                        result[name_str] = cand_id
                        break

    matched = len(result)
    total = len(pool_names)
    print(f"[tank01_ids] Resolved {matched}/{total} player IDs")
    return result
