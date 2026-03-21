"""yak_core.slate_recap -- Previous-slate projection accuracy recap.

Compares projected FP from the most recent archived slate against actual
DraftKings fantasy points (from archive data or Tank01 box scores).

Used by the Edge Analysis tab to show how yesterday's projections performed.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import YAKOS_ROOT

_ARCHIVE_DIR = Path(YAKOS_ROOT) / "data" / "slate_archive"
_PUBLISHED_DIR = Path(YAKOS_ROOT) / "data" / "published"

# Tank01 API config
_TANK01_HOST = "tank01-fantasy-stats.p.rapidapi.com"
_TANK01_KEY = os.environ.get(
    "RAPIDAPI_KEY",
    os.environ.get(
        "TANK01_RAPIDAPI_KEY",
        "6ecde6fcf4msh272863935e2161fp1ef372jsn7d6df444f886",
    ),
)


def _dk_fantasy_points(stats: dict) -> float:
    """Calculate DraftKings NBA fantasy points from box score stats."""
    pts = float(stats.get("pts", 0) or 0)
    reb = float(stats.get("reb", 0) or 0)
    ast = float(stats.get("ast", 0) or 0)
    stl = float(stats.get("stl", 0) or 0)
    blk = float(stats.get("blk", 0) or 0)
    tov = float(stats.get("TOV", stats.get("tov", 0)) or 0)
    tpm = float(stats.get("tptfgm", 0) or 0)

    fp = pts + reb * 1.25 + ast * 1.5 + stl * 2 + blk * 2 - tov * 0.5 + tpm * 0.5

    # Double-double bonus (1.5 FP)
    cats = [pts, reb, ast, stl, blk]
    doubles = sum(1 for c in cats if c >= 10)
    if doubles >= 2:
        fp += 1.5
    # Triple-double bonus (3 FP, stacks with DD)
    if doubles >= 3:
        fp += 3.0

    return fp


def _fetch_actuals_from_tank01(game_date: str) -> Dict[str, float]:
    """Fetch actual DK fantasy points for all players on a given date.

    Parameters
    ----------
    game_date : str
        Date in YYYYMMDD format (e.g. "20260315").

    Returns
    -------
    Dict[str, float]
        {player_name: dk_fantasy_points}
    """
    try:
        import requests
    except ImportError:
        return {}

    headers = {
        "x-rapidapi-key": _TANK01_KEY,
        "x-rapidapi-host": _TANK01_HOST,
    }

    # Step 1: get all games for the date
    try:
        resp = requests.get(
            f"https://{_TANK01_HOST}/getNBAGamesForDate",
            headers=headers,
            params={"gameDate": game_date},
            timeout=15,
        )
        resp.raise_for_status()
        games = resp.json().get("body", [])
    except Exception as exc:
        print(f"[slate_recap] Failed to fetch games for {game_date}: {exc}")
        return {}

    if not games:
        return {}

    # Step 2: get box scores for each game
    actuals: Dict[str, float] = {}
    for game in games:
        game_id = game.get("gameID", "")
        if not game_id:
            continue
        try:
            resp = requests.get(
                f"https://{_TANK01_HOST}/getNBABoxScore",
                headers=headers,
                params={"gameID": game_id},
                timeout=15,
            )
            resp.raise_for_status()
            body = resp.json().get("body", {})
            player_stats = body.get("playerStats", {})
            for pid, stats in player_stats.items():
                name = str(stats.get("longName", "")).strip()
                if name:
                    actuals[name] = _dk_fantasy_points(stats)
        except Exception as exc:
            print(f"[slate_recap] Failed to fetch box score for {game_id}: {exc}")
            continue

    return actuals


def _find_previous_slate(sport: str, today: date) -> Optional[pd.DataFrame]:
    """Find the most recent slate archive for a sport before today.

    Checks two sources and returns whichever is more recent:
      1. data/slate_archive/ — dedicated archive parquets (up to 7 days back)
      2. data/published/{sport}/ — the most-recently published slate pool
         (only used when its date is strictly before *today*)

    Returns DataFrame with player_name, proj, actual_fp, salary columns,
    or None if no archive found.
    """
    best_df: Optional[pd.DataFrame] = None
    best_date: Optional[date] = None

    # --- Source 1: slate_archive directory ---
    if _ARCHIVE_DIR.exists():
        for days_back in range(1, 8):
            check_date = today - timedelta(days=days_back)
            date_str = check_date.strftime("%Y-%m-%d")

            if sport.upper() == "PGA":
                patterns = [
                    f"{date_str}_pga_gpp.parquet",
                    f"{date_str}_pga_showdown.parquet",
                ]
            else:
                patterns = [
                    f"{date_str}_gpp_main.parquet",
                    f"{date_str}_cash_main.parquet",
                ]
            for pattern in patterns:
                path = _ARCHIVE_DIR / pattern
                if path.exists():
                    try:
                        df = pd.read_parquet(path)
                        if "player_name" in df.columns and "proj" in df.columns:
                            # This source walks backwards, so the first hit is
                            # the most recent archive date — record and stop.
                            best_df = df
                            best_date = check_date
                            break
                    except Exception:
                        continue
            if best_df is not None:
                break  # found the most recent archive file

    # --- Source 2: published slate pool ---
    # The published pool is the *current* working slate. It is only a valid
    # "previous slate" when its date is strictly before today (i.e. today's
    # slate hasn't been published yet, so the published data is yesterday's).
    import json

    pub_dir = _PUBLISHED_DIR / sport.lower()
    meta_path = pub_dir / "slate_meta.json"
    pool_path = pub_dir / "slate_pool.parquet"
    if meta_path.exists() and pool_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            pub_date_str = meta.get("date", "")
            pub_date = date.fromisoformat(pub_date_str) if pub_date_str else None
            if pub_date and pub_date < today:
                # Use published data if it's more recent than what the archive had
                if best_date is None or pub_date > best_date:
                    df = pd.read_parquet(pool_path)
                    if "player_name" in df.columns and "proj" in df.columns:
                        if "slate_date" not in df.columns:
                            df["slate_date"] = pub_date_str
                        best_df = df
        except Exception:
            pass

    return best_df


def _classify_result(projected: float, actual: float) -> Tuple[str, str]:
    """Classify a projection result into hit/miss category.

    Returns (emoji, label) tuple.
    """
    if projected <= 0:
        return ("❓", "no proj")
    ratio = actual / projected
    if ratio >= 1.2:
        return ("🔥", "big hit")
    elif ratio >= 1.0:
        return ("✅", "hit")
    elif ratio >= 0.7:
        return ("❌", "miss")
    else:
        return ("💀", "bust")


def get_previous_slate_recap(
    sport: str,
    today: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """Build a recap of the previous slate's projection accuracy.

    Parameters
    ----------
    sport : str
        Sport code (e.g. "nba").
    today : date, optional
        Reference date (defaults to today).

    Returns
    -------
    dict or None
        {
            "slate_date": "2026-03-15",
            "players": [
                {
                    "player_name": str,
                    "projected": float,
                    "actual": float,
                    "delta": float,
                    "emoji": str,
                    "label": str,
                },
                ...
            ],
            "summary": {
                "total": int,
                "big_hits": int,
                "hits": int,
                "misses": int,
                "busts": int,
                "avg_delta": float,
            },
        }
        or None if no previous slate data is available.
    """
    if today is None:
        today = date.today()

    # Step 1: Find the previous slate archive
    prev_slate = _find_previous_slate(sport, today)
    if prev_slate is None:
        return None

    slate_date = str(prev_slate.get("slate_date", pd.Series(["unknown"])).iloc[0])

    # Step 2: Get actuals — prefer archive data, fall back to Tank01 API
    has_actuals = (
        "actual_fp" in prev_slate.columns
        and prev_slate["actual_fp"].notna().sum() > 0
    )

    if has_actuals:
        # Use actuals from archive (works for both NBA and PGA)
        df = prev_slate[
            prev_slate["actual_fp"].notna()
            & (prev_slate["proj"] > 0)
        ][["player_name", "proj", "actual_fp", "salary"]].copy()
        df = df.rename(columns={"actual_fp": "actual"})
    elif sport.upper() != "PGA":
        # NBA: Fall back to Tank01 API
        try:
            api_date = slate_date.replace("-", "")
        except Exception:
            return None

        actuals_map = _fetch_actuals_from_tank01(api_date)
        if not actuals_map:
            return None

        df = prev_slate[prev_slate["proj"] > 0][
            ["player_name", "proj", "salary"]
        ].copy()
        df["actual"] = df["player_name"].map(actuals_map)
        df = df[df["actual"].notna()].copy()
    else:
        # PGA: no Tank01 fallback — actuals come from DataGolf via calibration
        return None

    if df.empty:
        return None

    # Step 3: Compute deltas and classify
    df["delta"] = df["actual"] - df["proj"]

    players = []
    counts = {"big_hits": 0, "hits": 0, "misses": 0, "busts": 0}
    for _, row in df.iterrows():
        emoji, label = _classify_result(row["proj"], row["actual"])
        players.append({
            "player_name": row["player_name"],
            "projected": round(row["proj"], 1),
            "actual": round(row["actual"], 1),
            "delta": round(row["delta"], 1),
            "emoji": emoji,
            "label": label,
            "salary": int(row.get("salary", 0)),
        })
        if label == "big hit":
            counts["big_hits"] += 1
        elif label == "hit":
            counts["hits"] += 1
        elif label == "miss":
            counts["misses"] += 1
        elif label == "bust":
            counts["busts"] += 1

    # Sort by salary descending (show most relevant players first)
    players.sort(key=lambda p: p["salary"], reverse=True)

    return {
        "slate_date": slate_date,
        "players": players,
        "summary": {
            "total": len(players),
            **counts,
            "avg_delta": round(df["delta"].mean(), 1),
        },
    }
