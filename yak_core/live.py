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


def _calc_dk_nba_fp(stats: dict) -> float:
    """Calculate DraftKings NBA classic fantasy points from raw box score stats.

    Scoring: PTS +1, 3PM +0.5, REB +1.25, AST +1.5, STL +2, BLK +2, TOV -0.5,
    double-double bonus +1.5, triple-double bonus +3.
    """

    def _f(*keys: str) -> float:
        for k in keys:
            v = stats.get(k)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return 0.0

    pts = _f("pts", "points", "PTS")
    reb = _f("reb", "rebounds", "REB", "totalReb")
    ast = _f("ast", "assists", "AST")
    stl = _f("stl", "steals", "STL")
    blk = _f("blk", "blocks", "BLK")
    tov = _f("TOV", "to", "turnovers", "tov")
    fg3m = _f("fg3m", "threePM", "three_made", "3PM", "tpm")

    fp = (
        pts
        + (fg3m * 0.5)
        + (reb * 1.25)
        + (ast * 1.5)
        + (stl * 2.0)
        + (blk * 2.0)
        - (tov * 0.5)
    )

    # Double-double (+1.5) / triple-double (+3) bonuses
    cats = sum(1 for c in [pts, reb, ast, stl, blk] if c >= 10)
    if cats >= 3:
        fp += 3.0
    elif cats >= 2:
        fp += 1.5

    return round(fp, 2)


def _fetch_actuals_from_box_scores(date_key: str, cfg: dict) -> pd.DataFrame:
    """Fetch actual DK fantasy points via ``getNBAGamesForDate`` + ``getNBABoxScore``.

    This is the reliable fallback for *historical* dates where the
    ``getNBADFS`` endpoint may not return player data.  It fetches every game
    played on the requested date, pulls each game's box score, and either uses
    the pre-calculated ``fantasyPoints.DraftKings`` value or computes DK FP
    from raw stats via :func:`_calc_dk_nba_fp`.

    Parameters
    ----------
    date_key : str
        Date in ``YYYYMMDD`` or ``YYYY-MM-DD`` format.
    cfg : dict
        Must contain ``RAPIDAPI_KEY``.

    Returns
    -------
    pd.DataFrame
        Columns: ``player_name``, ``actual_fp``.
    """
    api_key = _get_rapidapi_key(cfg)
    clean = date_key.replace("-", "")
    formatted = f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}"

    # Step 1: retrieve game IDs for the date
    games_resp = requests.get(
        "https://" + _TANK01_HOST + "/getNBAGamesForDate",
        headers=_headers(api_key),
        params={"gameDate": formatted},
        timeout=30,
    )
    games_resp.raise_for_status()
    games_data = games_resp.json()
    games_body = games_data.get("body", games_data) if isinstance(games_data, dict) else games_data

    if isinstance(games_body, dict):
        games_list = games_body.get("games", [])
        if not isinstance(games_list, list):
            # Some API versions nest the game list under the first list-typed value
            games_list = next((v for v in games_body.values() if isinstance(v, list)), [])
    elif isinstance(games_body, list):
        games_list = games_body
    else:
        games_list = []

    if not games_list:
        raise ValueError(f"No games found for {date_key}")

    all_players: list = []

    # Step 2: fetch box score for each game
    for game in games_list:
        if not isinstance(game, dict):
            continue
        game_id = (
            game.get("gameID") or game.get("gameId") or game.get("game_id") or ""
        )
        if not game_id:
            continue

        box_resp = requests.get(
            "https://" + _TANK01_HOST + "/getNBABoxScore",
            headers=_headers(api_key),
            params={"gameID": str(game_id)},
            timeout=30,
        )
        box_resp.raise_for_status()
        box_data = box_resp.json()
        box_body = box_data.get("body", box_data) if isinstance(box_data, dict) else box_data

        if isinstance(box_body, dict):
            player_stats = box_body.get("playerStats", box_body.get("players", []))
            if not isinstance(player_stats, list):
                player_stats = []
        elif isinstance(box_body, list):
            player_stats = box_body
        else:
            player_stats = []

        for p in player_stats:
            if not isinstance(p, dict):
                continue
            name = (
                p.get("displayName")
                or p.get("longName")
                or p.get("playerName")
                or p.get("player")
                or ""
            )
            if not name:
                continue

            # Prefer pre-calculated DK FP; fall back to computing from raw stats
            fp_raw = None
            fp_nested = p.get("fantasyPoints")
            if isinstance(fp_nested, dict):
                fp_raw = (
                    fp_nested.get("DraftKings")
                    or fp_nested.get("dkPoints")
                    or fp_nested.get("dk")
                    or fp_nested.get("DK")
                )
            elif fp_nested is not None:
                fp_raw = fp_nested

            try:
                fp = float(fp_raw) if fp_raw is not None else _calc_dk_nba_fp(p)
            except (ValueError, TypeError):
                fp = _calc_dk_nba_fp(p)

            all_players.append({"player_name": str(name), "actual_fp": fp})

    if not all_players:
        raise ValueError(f"No player actuals parsed from box scores for {date_key}")

    df = pd.DataFrame(all_players)
    print(f"[_fetch_actuals_from_box_scores] {len(df)} player actuals from box scores for {date_key}")
    return df.reset_index(drop=True)


def fetch_actuals_from_api(date_key: str, cfg: dict) -> pd.DataFrame:
    """Fetch actual DraftKings fantasy points for a completed slate from Tank01.

    First attempts the ``getNBADFS`` endpoint (works when Tank01 has already
    back-filled final fantasy points into the ``fantasyPoints`` field for the
    requested date).  If that endpoint returns no usable player rows the
    function falls back to :func:`_fetch_actuals_from_box_scores`, which calls
    ``getNBAGamesForDate`` + ``getNBABoxScore`` — a more reliable path for
    *historical* dates.

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
    RuntimeError
        If both the DFS endpoint and the box-score fallback fail.
    """
    date_key_clean = date_key.replace("-", "")

    # Try the DFS endpoint first (works for current/recent slates where
    # Tank01 has back-filled actual fantasy points into "fantasyPoints")
    try:
        dfs_df = fetch_live_dfs(date_key_clean, cfg)
        if not dfs_df.empty:
            result = dfs_df[["player_name", "proj"]].copy()
            result = result.rename(columns={"proj": "actual_fp"})
            result["actual_fp"] = pd.to_numeric(result["actual_fp"], errors="coerce").fillna(0.0)
            print(f"[fetch_actuals_from_api] {len(result)} player actuals via DFS endpoint for {date_key}")
            return result.reset_index(drop=True)
    except Exception:
        pass  # fall through to box-score approach

    # Fallback: box scores are always available for completed games
    try:
        df = _fetch_actuals_from_box_scores(date_key_clean, cfg)
        print(f"[fetch_actuals_from_api] {len(df)} player actuals via box scores for {date_key}")
        return df
    except Exception as exc:
        raise RuntimeError(f"Tank01 actuals API error for {date_key}: {exc}") from exc


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
