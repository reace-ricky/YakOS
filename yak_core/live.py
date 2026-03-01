"""yak_core.live -- fetch live slate data from Tank01 RapidAPI."""
import os
import requests
import pandas as pd
from typing import Dict, Any
from .config import YAKOS_ROOT

_TANK01_HOST = "tank01-fantasy-stats.p.rapidapi.com"


def _get_rapidapi_key(cfg):
    """Resolve RapidAPI key: cfg > env > raise."""
    key = cfg.get("RAPIDAPI_KEY") or os.environ.get("RAPIDAPI_KEY", "")
    if not key:
        raise ValueError("RAPIDAPI_KEY not found. Set in config or RAPIDAPI_KEY env var.")
    return key


def _headers(api_key):
    return {"x-rapidapi-key": api_key, "x-rapidapi-host": _TANK01_HOST}


def fetch_live_dfs(date_key, cfg):
    """Fetch DK DFS salaries+positions+projections from Tank01 getNBADFS."""
    api_key = _get_rapidapi_key(cfg)
    url = "https://" + _TANK01_HOST + "/getNBADFS"
    resp = requests.get(url, headers=_headers(api_key), params={"date": date_key}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    body = data.get("body", data) if isinstance(data, dict) else data
    if not body:
        raise ValueError("Empty DFS response for " + date_key)
    rows = []
    for entry in body:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("playerID", entry.get("player_id", ""))
        name = entry.get("player", entry.get("longName", entry.get("playerName", "")))
        team = entry.get("team", entry.get("teamAbv", ""))
        opp = entry.get("opponent", entry.get("opp", ""))
        pos = entry.get("pos", entry.get("position", ""))
        sal_raw = entry.get("salary", entry.get("dk_salary", 0))
        proj_raw = entry.get("fantasyPoints", entry.get("fpts", entry.get("proj", 0)))
        own_raw = entry.get("ownership", entry.get("own_proj", entry.get("projectedOwnership", None)))
        try:
            salary = int(float(sal_raw)) if sal_raw else 0
        except (ValueError, TypeError):
            salary = 0
        try:
            proj = float(proj_raw) if proj_raw else 0.0
        except (ValueError, TypeError):
            proj = 0.0
        own_proj = None
        if own_raw is not None:
            try:
                own_proj = float(own_raw)
            except (ValueError, TypeError):
                pass
        rows.append({
            "player_id": str(pid), "player_name": str(name),
            "team": str(team).upper(),
            "opponent": str(opp).upper() if opp else "",
            "pos": str(pos), "salary": salary, "proj": proj,
            "actual_fp": float("nan"), "own_proj": own_proj,
        })
    if not rows:
        raise ValueError("No DFS player rows parsed for " + date_key)
    df = pd.DataFrame(rows)
    print("[fetch_live_dfs] Fetched " + str(len(df)) + " players for " + date_key)
    return df


def fetch_live_opt_pool(slate_date, cfg):
    """Fetch live opt pool via Tank01 API. Live counterpart of load_opt_pool_from_config(."""
    date_key = slate_date.replace("-", "")
    df = fetch_live_dfs(date_key, cfg)
    df = df[df["salary"] > 0].copy()
    if df.empty:
        raise ValueError("No players with salary > 0 for live slate " + slate_date)
    if "player_id" not in df.columns or df["player_id"].isna().all():
        df["player_id"] = df["player_name"].str.lower().str.replace(" ", "_")
    print("[fetch_live_opt_pool] Live pool: " + str(len(df)) + " players for " + slate_date)
    return df


def fetch_actuals_from_api(date_key: str, cfg: dict) -> pd.DataFrame:
    """Fetch actual DraftKings fantasy points for a completed slate from Tank01.

    Calls the ``getNBADFS`` endpoint for a *past* date.  For completed games,
    Tank01 returns the real DK fantasy points scored in the ``fantasyPoints``
    field, which this function exposes as ``actual_fp``.

    Parameters
    ----------
    date_key : str
        Date string in ``YYYYMMDD`` or ``YYYY-MM-DD`` format.
    cfg : dict
        Must contain ``RAPIDAPI_KEY`` or the ``RAPIDAPI_KEY`` env var must be
        set.

    Returns
    -------
    pd.DataFrame
        Columns: ``player_name``, ``actual_fp``.  One row per player.  Players
        with no recorded score (``actual_fp == 0``) are **included** so the
        caller can decide how to handle zero-point performances.

    Raises
    ------
    ValueError
        If the API returns an empty payload or no player rows can be parsed.
    RuntimeError
        If the HTTP request itself fails.
    """
    date_key_clean = date_key.replace("-", "")
    try:
        dfs_df = fetch_live_dfs(date_key_clean, cfg)
    except Exception as exc:
        raise RuntimeError(f"Tank01 actuals API error for {date_key}: {exc}") from exc

    if dfs_df.empty:
        raise ValueError(f"No actuals returned from Tank01 for {date_key}")

    result = dfs_df[["player_name", "proj"]].copy()
    result = result.rename(columns={"proj": "actual_fp"})
    result["actual_fp"] = pd.to_numeric(result["actual_fp"], errors="coerce").fillna(0.0)
    print(f"[fetch_actuals_from_api] {len(result)} player actuals for {date_key}")
    return result.reset_index(drop=True)


def fetch_injury_updates(date_key: str, cfg: dict) -> list:
    """Fetch player injury/news updates from Tank01 API for a given date.

    Calls the ``getNBAInjuryList`` endpoint and maps status codes to the
    format expected by :func:`yak_core.sims.simulate_live_updates`.

    Parameters
    ----------
    date_key : str
        Date string in ``YYYYMMDD`` or ``YYYY-MM-DD`` format.
    cfg : dict
        Must contain ``RAPIDAPI_KEY`` or the ``RAPIDAPI_KEY`` env var must be set.

    Returns
    -------
    list of dict
        Each dict has ``player_name`` and optionally ``status``.
        Empty list if no injuries are reported.

    Raises
    ------
    RuntimeError
        If the API call fails.
    """
    api_key = _get_rapidapi_key(cfg)
    url = "https://" + _TANK01_HOST + "/getNBAInjuryList"
    try:
        resp = requests.get(url, headers=_headers(api_key), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        body = data.get("body", data) if isinstance(data, dict) else data
        if not body:
            return []

        _status_map = {
            "OUT": "OUT",
            "QUESTIONABLE": "QUESTIONABLE",
            "Q": "QUESTIONABLE",
            "GTD": "GTD",
            "GAME TIME DECISION": "GTD",
            "DAY-TO-DAY": "GTD",
            "IN": "IN",
            "ACTIVE": "IN",
        }

        updates = []
        for entry in body:
            if not isinstance(entry, dict):
                continue
            name = (
                entry.get("playerName")
                or entry.get("longName")
                or entry.get("player")
                or ""
            )
            if not name:
                continue
            raw_status = str(
                entry.get("injuryStatus", entry.get("status", ""))
            ).strip().upper()

            # Exact match first; then check if full raw_status starts with a key
            # (avoids "NOT OUT" matching "OUT" via substring)
            mapped = _status_map.get(raw_status)
            if mapped is None:
                for key, val in _status_map.items():
                    if raw_status == key or raw_status.startswith(key + " "):
                        mapped = val
                        break

            if mapped:
                updates.append({"player_name": name, "status": mapped})

        print(f"[fetch_injury_updates] {len(updates)} injury updates for {date_key}")
        return updates

    except Exception as exc:
        raise RuntimeError(f"Tank01 injury API error: {exc}") from exc
