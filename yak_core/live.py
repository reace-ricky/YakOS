"""yak_core.live -- fetch live slate data from Tank01 RapidAPI."""
import os
import requests
import pandas as pd
from typing import Any, Dict, List, Optional
from .config import YAKOS_ROOT

_TANK01_HOST = "tank01-fantasy-stats.p.rapidapi.com"
# Known keys under which Tank01 getNBADFS may nest the player list in its body dict.
_TANK01_DFS_PLAYER_KEYS = ("DraftKings", "DK", "dk", "draftkings", "players", "playerList")


# Canonical status mapping from raw Tank01 values
_STATUS_MAP = {
    "ACTIVE": "Active",
    "": "Active",
    "OUT": "OUT",
    "IR": "IR",
    "INJURED RESERVE": "IR",
    "INJ": "IR",
    "SUSPENDED": "Suspended",
    "SUSP": "Suspended",
    "G-LEAGUE": "G-League",
    "G_LEAGUE": "G-League",
    "GLEAGUE": "G-League",
    "DND": "OUT",
    "O": "OUT",
    "QUESTIONABLE": "Questionable",
    "Q": "Questionable",
    "GTD": "GTD",
    "GAME TIME DECISION": "GTD",
    "DAY-TO-DAY": "GTD",
    "PROBABLE": "Probable",
    "P": "Probable",
}


class NoGamesScheduledError(ValueError):
    """Raised when the API confirms no games are scheduled for a given date."""


def load_manual_injury_overrides() -> pd.DataFrame:
    """Load active manual injury overrides from config/manual_injuries.csv.

    Returns a DataFrame with columns ``playerID``, ``player``, ``designation``
    for rows where ``active`` is True.  Returns an empty DataFrame (with those
    columns) if the file is missing or unreadable.
    """
    overrides_path = os.path.join(YAKOS_ROOT, "config", "manual_injuries.csv")
    if not os.path.exists(overrides_path):
        return pd.DataFrame(columns=["playerID", "player", "designation"])
    try:
        df = pd.read_csv(overrides_path)
    except Exception:
        return pd.DataFrame(columns=["playerID", "player", "designation"])
    if df.empty or "active" not in df.columns:
        return pd.DataFrame(columns=["playerID", "player", "designation"])
    active = df[
        df["active"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    ]
    needed = [c for c in ["playerID", "player", "designation"] if c in active.columns]
    return active[needed].reset_index(drop=True)


def apply_manual_injury_overrides_to_pool(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Apply manual injury overrides to the player pool's ``status`` column.

    Matches each override row first by Tank01 ``playerID`` (against the pool's
    ``player_id`` column), then falls back to case-insensitive ``player`` name
    matching (against ``player_name``).  Overrides always win — they force the
    mapped status regardless of what Tank01 reported.

    Designation → status mapping:
      * ``"Out"``        → ``"OUT"``
      * ``"Day-To-Day"`` → ``"GTD"``

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool.  Should have ``player_name`` and optionally ``player_id``.

    Returns
    -------
    pd.DataFrame
        Copy of *pool_df* with updated ``status`` values for matched players.
    """
    overrides = load_manual_injury_overrides()
    if overrides.empty:
        return pool_df
    pool = pool_df.copy()
    if "status" not in pool.columns:
        pool["status"] = "Active"
    _desig_to_status: Dict[str, str] = {
        "Out": "OUT",
        "Day-To-Day": "GTD",
        "GTD": "GTD",
        "OUT": "OUT",
    }
    for _, row in overrides.iterrows():
        pid = str(row.get("playerID", "")).strip()
        pname = str(row.get("player", "")).strip()
        designation = str(row.get("designation", "")).strip()
        mapped_status = _desig_to_status.get(designation, designation.upper())
        matched = False
        # Match by Tank01 playerID first
        if pid and pid.lower() not in ("", "nan") and "player_id" in pool.columns:
            mask = pool["player_id"].astype(str).str.strip() == pid
            if mask.any():
                pool.loc[mask, "status"] = mapped_status
                matched = True
        # Fall back to player name matching
        if not matched and pname and "player_name" in pool.columns:
            mask = pool["player_name"].str.strip().str.lower() == pname.lower()
            if mask.any():
                pool.loc[mask, "status"] = mapped_status
    return pool


def _get_rapidapi_key(cfg):
    """Resolve RapidAPI key: cfg > env > raise."""
    key = cfg.get("RAPIDAPI_KEY") or os.environ.get("RAPIDAPI_KEY", "")
    if not key:
        raise ValueError("RAPIDAPI_KEY not found. Set in config or RAPIDAPI_KEY env var.")
    return key


def _headers(api_key):
    return {"x-rapidapi-key": api_key, "x-rapidapi-host": _TANK01_HOST}


def _map_status(raw: str) -> str:
    """Normalise a raw Tank01 status string to a canonical status label."""
    upper = str(raw).strip().upper()
    return _STATUS_MAP.get(upper, raw.strip() if raw.strip() else "Active")


def fetch_live_dfs(date_key, cfg):
    """Fetch DK DFS salaries+positions+projections from Tank01 getNBADFS."""
    api_key = _get_rapidapi_key(cfg)
    url = "https://" + _TANK01_HOST + "/getNBADFS"
    params = {"date": date_key}
    headers = _headers(api_key)
    print(
        f"[fetch_live_dfs] URL={url} params={params} "
        f"host={headers['x-rapidapi-host']} key=<set>"
    )
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    body = data.get("body", data) if isinstance(data, dict) else data
    if not body:
        raise ValueError("Empty DFS response for " + date_key)
    # Tank01 getNBADFS may return body as {"DraftKings": [...]} or {"DK": [...]}
    # rather than a plain list — unwrap to the player list.
    if isinstance(body, dict):
        for key in _TANK01_DFS_PLAYER_KEYS:
            if key in body and isinstance(body[key], list):
                body = body[key]
                break
        else:
            # Fallback: pick the longest list value in the dict
            list_vals = [v for v in body.values() if isinstance(v, list)]
            if list_vals:
                body = max(list_vals, key=len)
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
        # Extract injury/availability status
        status_raw = entry.get("injuryStatus", entry.get("status", entry.get("active", "")))
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
            "status": _map_status(str(status_raw) if status_raw else ""),
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
    # Preserve Tank01's projection as a named column before any overrides
    df["tank01_proj"] = df["proj"].copy()
    df["proj_source"] = "tank01"
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

    # Step 1: retrieve game IDs for the date
    games_url = "https://" + _TANK01_HOST + "/getNBAGamesForDate"
    games_params = {"gameDate": clean}  # Tank01 expects YYYYMMDD, no dashes
    games_headers = _headers(api_key)
    print(
        f"[_fetch_actuals_from_box_scores] URL={games_url} params={games_params} "
        f"host={games_headers['x-rapidapi-host']} key=<set>"
    )
    games_resp = requests.get(
        games_url,
        headers=games_headers,
        params=games_params,
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
        raise NoGamesScheduledError(f"No games found for {date_key}")

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

        box_url = "https://" + _TANK01_HOST + "/getNBABoxScore"
        box_params = {"gameID": str(game_id), "fantasyPoints": "true"}
        print(f"[_fetch_actuals_from_box_scores] Box score URL={box_url} params={box_params}")
        box_resp = requests.get(
            box_url,
            headers=_headers(api_key),
            params=box_params,
            timeout=30,
        )
        box_resp.raise_for_status()
        box_data = box_resp.json()
        box_body = box_data.get("body", box_data) if isinstance(box_data, dict) else box_data

        if isinstance(box_body, dict):
            raw_ps = box_body.get("playerStats", box_body.get("players", {}))
            # Tank01 returns playerStats as a dict keyed by playerID
            if isinstance(raw_ps, dict):
                player_stats = list(raw_ps.values())
            elif isinstance(raw_ps, list):
                player_stats = raw_ps
            else:
                player_stats = []
        elif isinstance(box_body, list):
            player_stats = box_body
        else:
            player_stats = []
        print(f"[_fetch_actuals_from_box_scores] Parsed {len(player_stats)} players from box score {game_id}")

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
    """Fetch actual DraftKings fantasy points for a completed slate via box scores.

    Always uses ``getNBAGamesForDate`` + ``getNBABoxScore`` to retrieve real
    game statistics.  The ``getNBADFS`` endpoint returns *projected* fantasy
    points, not actuals, and is therefore not used here.

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
        If the box-score API call fails.
    """
    date_key_clean = date_key.replace("-", "")

    try:
        df = _fetch_actuals_from_box_scores(date_key_clean, cfg)
        print(f"[fetch_actuals_from_api] {len(df)} player actuals via box scores for {date_key}")
        return df
    except NoGamesScheduledError:
        # Re-raise as-is so the UI can show "No games scheduled" rather than "API error"
        raise
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


# ---------------------------------------------------------------------------
# NEW: fetch_player_game_logs
# ---------------------------------------------------------------------------

def fetch_player_game_logs(
    player_names: List[str],
    player_id_map: Optional[Dict[str, str]],
    api_key: str,
    max_workers: int = 10,
    timeout_seconds: int = 120,
) -> pd.DataFrame:
    """Fetch rolling game-log stats for a list of players from Tank01.

    Calls Tank01's ``getNBAGamesForPlayer`` endpoint for each player and
    computes rolling averages for DraftKings fantasy points and minutes played
    over the last 5, 10, and 20 games.

    Performance: uses concurrent threads (default 10) with an overall
    timeout cap so the entire batch finishes in ~2 minutes max, even
    with 200+ players.
    """
    if not player_names:
        return pd.DataFrame(columns=[
            "player_name",
            "rolling_fp_5", "rolling_fp_10", "rolling_fp_20",
            "rolling_min_5", "rolling_min_10", "rolling_min_20",
        ])

    _id_map: Dict[str, str] = player_id_map or {}
    url = "https://" + _TANK01_HOST + "/getNBAGamesForPlayer"
    hdrs = _headers(api_key)

    # Filter to only players with Tank01 IDs
    valid_players = [(name, _id_map[name]) for name in player_names if _id_map.get(name)]
    skipped = len(player_names) - len(valid_players)
    if skipped:
        print(f"[fetch_player_game_logs] Skipped {skipped} players with no Tank01 playerID.")
    if not valid_players:
        return pd.DataFrame(columns=[
            "player_name",
            "rolling_fp_5", "rolling_fp_10", "rolling_fp_20",
            "rolling_min_5", "rolling_min_10", "rolling_min_20",
        ])

    def _fetch_one(name_and_id):
        name, player_id = name_and_id
        params = {
            "playerID": player_id,
            "numberOfGames": "20",
            "fantasyPoints": "true",
        }
        try:
            resp = requests.get(url, headers=hdrs, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            body = data.get("body", data) if isinstance(data, dict) else data

            if isinstance(body, dict):
                for list_key in ("games", "gameLog", "stats", "playerStats"):
                    if list_key in body and isinstance(body[list_key], list):
                        body = body[list_key]
                        break
                else:
                    # Tank01 getNBAGamesForPlayer returns a dict of
                    # {gameID: {stats...}} — convert to list of dicts.
                    dict_vals = [v for v in body.values() if isinstance(v, dict)]
                    if dict_vals:
                        body = dict_vals
                    else:
                        list_vals = [v for v in body.values() if isinstance(v, list)]
                        body = list_vals[0] if list_vals else []

            if not isinstance(body, list) or not body:
                return None

            fps: List[float] = []
            mins: List[float] = []

            for game in body:
                if not isinstance(game, dict):
                    continue

                fp_raw = game.get("fantasyPoints")
                if isinstance(fp_raw, dict):
                    fp_val = (
                        fp_raw.get("DraftKings")
                        or fp_raw.get("dk")
                        or fp_raw.get("DK")
                        or fp_raw.get("dkPoints")
                    )
                    try:
                        fp = float(fp_val) if fp_val is not None else _calc_dk_nba_fp(game)
                    except (ValueError, TypeError):
                        fp = _calc_dk_nba_fp(game)
                elif fp_raw is not None:
                    try:
                        fp = float(fp_raw)
                    except (ValueError, TypeError):
                        fp = _calc_dk_nba_fp(game)
                else:
                    fp = _calc_dk_nba_fp(game)

                fps.append(fp)

                mins_raw = (
                    game.get("mins")
                    or game.get("min")
                    or game.get("minutes")
                    or game.get("MP")
                    or game.get("minSeconds")
                )
                try:
                    if mins_raw is not None and ":" in str(mins_raw):
                        parts = str(mins_raw).split(":")
                        m_val = float(parts[0]) + float(parts[1]) / 60.0
                    else:
                        m_val = float(mins_raw) if mins_raw is not None else 0.0
                except (ValueError, TypeError):
                    m_val = 0.0
                mins.append(m_val)

            if not fps:
                return None

            def _rolling_avg(values: List[float], n: int) -> float:
                subset = values[-n:] if len(values) >= n else values
                return round(sum(subset) / len(subset), 2) if subset else 0.0

            return {
                "player_name": name,
                "rolling_fp_5":  _rolling_avg(fps, 5),
                "rolling_fp_10": _rolling_avg(fps, 10),
                "rolling_fp_20": _rolling_avg(fps, 20),
                "rolling_min_5":  _rolling_avg(mins, 5),
                "rolling_min_10": _rolling_avg(mins, 10),
                "rolling_min_20": _rolling_avg(mins, 20),
            }
        except Exception as exc:
            print(f"[fetch_player_game_logs] Error for '{name}': {exc}")
            return None

    # Run concurrently with timeout cap
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows: List[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, vp): vp[0] for vp in valid_players}
            import time
            deadline = time.time() + timeout_seconds
            for future in as_completed(futures, timeout=timeout_seconds):
                if time.time() > deadline:
                    print(f"[fetch_player_game_logs] Timeout ({timeout_seconds}s) reached, stopping.")
                    break
                try:
                    result = future.result(timeout=5)
                    if result:
                        rows.append(result)
                except Exception:
                    pass
    except Exception as exc:
        print(f"[fetch_player_game_logs] Batch error: {exc}")

    if not rows:
        return pd.DataFrame(columns=[
            "player_name",
            "rolling_fp_5", "rolling_fp_10", "rolling_fp_20",
            "rolling_min_5", "rolling_min_10", "rolling_min_20",
        ])

    df = pd.DataFrame(rows)
    print(f"[fetch_player_game_logs] Rolling stats computed for {len(df)} players.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# NEW: fetch_betting_odds
# ---------------------------------------------------------------------------

def fetch_betting_odds(game_date: str, api_key: str) -> pd.DataFrame:
    """Fetch NBA Vegas betting odds from Tank01 for a given date.

    Calls Tank01's ``getNBABettingOdds`` endpoint and returns a tidy DataFrame
    with the over/under total and home-team spread for each game.

    Parameters
    ----------
    game_date : str
        Date in ``YYYYMMDD`` **or** ``YYYY-MM-DD`` format.  Both are accepted
        and normalised internally.
    api_key : str
        Tank01 RapidAPI key.

    Returns
    -------
    pd.DataFrame
        Columns: ``home_team``, ``away_team``, ``vegas_total``, ``spread``.
        ``spread`` is the home-team spread (negative = home favourite).
        Returns an empty DataFrame with those columns on any failure.
    """
    empty = pd.DataFrame(columns=["home_team", "away_team", "vegas_total", "spread"])

    # Normalise date to YYYYMMDD
    date_clean = game_date.replace("-", "")
    if len(date_clean) != 8 or not date_clean.isdigit():
        print(f"[fetch_betting_odds] Invalid date format: '{game_date}'.")
        return empty

    url = "https://" + _TANK01_HOST + "/getNBABettingOdds"
    params = {"gameDate": date_clean}
    hdrs = _headers(api_key)

    try:
        resp = requests.get(url, headers=hdrs, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        body = data.get("body", data) if isinstance(data, dict) else data

        # body may be a list of game-odds dicts, or a dict wrapping one
        if isinstance(body, dict):
            for list_key in ("games", "odds", "bettingOdds", "data"):
                if list_key in body and isinstance(body[list_key], list):
                    body = body[list_key]
                    break
            else:
                list_vals = [v for v in body.values() if isinstance(v, list)]
                body = list_vals[0] if list_vals else []

        if not isinstance(body, list) or not body:
            print(f"[fetch_betting_odds] No odds data returned for {game_date}.")
            return empty

        rows: List[dict] = []
        for game in body:
            if not isinstance(game, dict):
                continue

            # ── Team identifiers ──────────────────────────────────────────
            home = str(
                game.get("homeTeam")
                or game.get("home_team")
                or game.get("home")
                or ""
            ).strip().upper()
            away = str(
                game.get("awayTeam")
                or game.get("away_team")
                or game.get("away")
                or ""
            ).strip().upper()

            if not home and not away:
                continue

            # ── Over/under total ─────────────────────────────────────────
            # Tank01 may nest odds under a bookmaker key or expose them flat.
            total_raw = (
                game.get("overUnder")
                or game.get("total")
                or game.get("ou")
                or game.get("over_under")
            )
            # Dig one level deeper into a nested odds dict if needed
            if total_raw is None and isinstance(game.get("odds"), dict):
                total_raw = game["odds"].get("overUnder") or game["odds"].get("total")
            try:
                vegas_total = float(total_raw) if total_raw is not None else float("nan")
            except (ValueError, TypeError):
                vegas_total = float("nan")

            # ── Home spread ───────────────────────────────────────────────
            spread_raw = (
                game.get("homeSpread")
                or game.get("spread")
                or game.get("home_spread")
                or game.get("pointSpread")
            )
            if spread_raw is None and isinstance(game.get("odds"), dict):
                spread_raw = (
                    game["odds"].get("homeSpread")
                    or game["odds"].get("spread")
                )
            try:
                spread = float(spread_raw) if spread_raw is not None else float("nan")
            except (ValueError, TypeError):
                spread = float("nan")

            rows.append({
                "home_team": home,
                "away_team": away,
                "vegas_total": vegas_total,
                "spread": spread,
            })

        if not rows:
            print(f"[fetch_betting_odds] No parseable odds rows for {game_date}.")
            return empty

        df = pd.DataFrame(rows)
        print(f"[fetch_betting_odds] {len(df)} game odds fetched for {game_date}.")
        return df.reset_index(drop=True)

    except Exception as exc:
        print(f"[fetch_betting_odds] Error fetching odds for {game_date}: {exc}")
        return empty
