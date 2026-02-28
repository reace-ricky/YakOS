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
