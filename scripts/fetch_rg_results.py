#!/usr/bin/env python3
"""scripts/fetch_rg_results.py -- Fetch contest results from RotoGrinders ResultsDB.

Scrapes NBA/DraftKings contest results (cash line, winning score, top scores,
entries) and saves them to:
  - data/contest_results/history.json  (contest bands for calibration)
  - data/contest_results/rg_winning_lineups.json  (winning lineup details)

Usage:
  python scripts/fetch_rg_results.py --date 2026-03-14
  python scripts/fetch_rg_results.py --date 2026-03-14 --contest-type gpp

Environment:
  RG_EMAIL      - RotoGrinders login email
  RG_PASSWORD   - RotoGrinders login password
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("YAKOS_ROOT", str(_REPO_ROOT))

from yak_core.config import YAKOS_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fetch_rg_results")

_HISTORY_PATH = Path(YAKOS_ROOT) / "data" / "contest_results" / "history.json"
_WINNING_LINEUPS_PATH = Path(YAKOS_ROOT) / "data" / "contest_results" / "rg_winning_lineups.json"

# RotoGrinders ResultsDB URLs
_RG_BASE = "https://rotogrinders.com"
_RG_RESULTSDB = f"{_RG_BASE}/resultsdb"
_RG_LOGIN = f"{_RG_BASE}/api/users/login"


def _load_history() -> dict:
    if _HISTORY_PATH.exists():
        with open(_HISTORY_PATH) as f:
            return json.load(f)
    return {}


def _save_history(history: dict) -> None:
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def _load_winning_lineups() -> dict:
    if _WINNING_LINEUPS_PATH.exists():
        with open(_WINNING_LINEUPS_PATH) as f:
            return json.load(f)
    return {}


def _save_winning_lineups(data: dict) -> None:
    _WINNING_LINEUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_WINNING_LINEUPS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _get_rg_session() -> "requests.Session":
    """Create an authenticated RotoGrinders session."""
    import requests

    email = os.environ.get("RG_EMAIL", "")
    password = os.environ.get("RG_PASSWORD", "")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; YakOS/1.0)",
        "Accept": "application/json",
    })

    if not email or not password:
        log.warning("RG_EMAIL / RG_PASSWORD not set — attempting unauthenticated access")
        return session

    try:
        resp = session.post(_RG_LOGIN, json={"email": email, "password": password}, timeout=15)
        if resp.status_code == 200:
            log.info("Authenticated with RotoGrinders as %s", email)
        else:
            log.warning("RG login returned %d — continuing unauthenticated", resp.status_code)
    except Exception as e:
        log.warning("RG login failed: %s — continuing unauthenticated", e)

    return session


def _classify_contest(name: str, entry_fee: float, entries: int) -> str:
    """Classify a contest as gpp, cash, or showdown."""
    name_lower = name.lower()
    if "showdown" in name_lower or "captain" in name_lower:
        return "showdown"
    if any(kw in name_lower for kw in ("double up", "50/50", "head to head", "h2h")):
        return "cash"
    # Large field or high entry fee = GPP
    if entries > 500 or entry_fee >= 5:
        return "gpp"
    return "cash"


def fetch_rg_results(
    slate_date: str,
    sport: str = "nba",
    site: str = "draftkings",
    contest_filter: str | None = None,
) -> list[dict]:
    """Fetch contest results from RotoGrinders ResultsDB for a given date.

    Parameters
    ----------
    slate_date : str
        Date in YYYY-MM-DD format.
    sport : str
        Sport slug (nba, pga, nfl, etc.).
    site : str
        DFS site (draftkings, fanduel).
    contest_filter : str | None
        If set, only return contests matching this type (gpp, cash, showdown).

    Returns
    -------
    list of dict
        Each dict has: name, contest_type, entry_fee, entries, cash_line,
        top_15_score, top_1_score, winning_score, winning_lineup (if available).
    """
    import requests

    session = _get_rg_session()

    # ResultsDB API endpoint — fetches contest summaries for a date/sport/site
    api_url = f"{_RG_RESULTSDB}/api/{site}/{sport}/{slate_date}"
    log.info("Fetching RG results: %s", api_url)

    try:
        resp = session.get(api_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        # Try the HTML scraping fallback
        log.warning("RG API returned %s — trying HTML fallback", e)
        data = _scrape_resultsdb_html(session, slate_date, sport, site)
    except Exception as e:
        log.error("RG fetch failed: %s", e)
        return []

    if not data:
        log.warning("No contest data returned from RG for %s %s %s", slate_date, sport, site)
        return []

    contests = []
    raw_contests = data if isinstance(data, list) else data.get("contests", data.get("data", []))

    for c in raw_contests:
        name = c.get("name", c.get("title", ""))
        entry_fee = float(c.get("entry_fee", c.get("entryFee", 0)) or 0)
        entries = int(c.get("entries", c.get("totalEntries", 0)) or 0)
        contest_type = _classify_contest(name, entry_fee, entries)

        if contest_filter and contest_type != contest_filter.lower():
            continue

        cash_line = float(c.get("cash_line", c.get("cashLine", c.get("mincash_score", 0))) or 0)
        winning_score = float(c.get("winning_score", c.get("winningScore", c.get("first_place_score", 0))) or 0)

        # Try to extract percentile scores
        top_scores = c.get("top_scores", c.get("topScores", {}))
        top_15_score = float(top_scores.get("top_15", top_scores.get("15%", 0)) or 0)
        top_1_score = float(top_scores.get("top_1", top_scores.get("1%", 0)) or 0)

        # Winning lineup details
        winning_lineup = c.get("winning_lineup", c.get("winningLineup"))

        contest = {
            "name": name,
            "contest_type": contest_type,
            "entry_fee": entry_fee,
            "entries": entries,
            "cash_line": cash_line,
            "top_15_score": top_15_score,
            "top_1_score": top_1_score,
            "winning_score": winning_score,
        }
        if winning_lineup:
            contest["winning_lineup"] = winning_lineup

        contests.append(contest)

    log.info("Parsed %d contests from RG (%d after filter)", len(raw_contests), len(contests))
    return contests


def _scrape_resultsdb_html(
    session: "requests.Session",
    slate_date: str,
    sport: str,
    site: str,
) -> list:
    """Fallback: scrape contest results from the HTML ResultsDB page."""
    url = f"{_RG_RESULTSDB}/{site}/{sport}/{slate_date}"
    log.info("Scraping RG HTML: %s", url)

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        log.error("HTML scrape failed: %s", e)
        return []

    # Extract contest data from embedded JSON or structured HTML
    # RG often embeds contest data in a script tag
    pattern = r'resultsData\s*=\s*(\[.*?\]);'
    match = re.search(pattern, html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try __NEXT_DATA__ pattern (Next.js)
    next_pattern = r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>'
    match = re.search(next_pattern, html, re.DOTALL)
    if match:
        try:
            next_data = json.loads(match.group(1))
            props = next_data.get("props", {}).get("pageProps", {})
            contests = props.get("contests", props.get("results", []))
            if contests:
                return contests
        except (json.JSONDecodeError, KeyError):
            pass

    log.warning("Could not extract structured data from RG HTML")
    return []


def save_rg_to_history(
    contests: list[dict],
    slate_date: str,
) -> dict:
    """Save fetched RG contest results to history.json and rg_winning_lineups.json.

    For each contest type (gpp, cash, showdown), picks the largest contest
    (by entries) as the representative entry for history.json.

    Returns summary dict.
    """
    if not contests:
        return {"saved": 0, "skipped": "no contests"}

    history = _load_history()
    winning_lineups = _load_winning_lineups()
    saved = 0

    # Group by contest type and pick the largest contest per type
    by_type: dict[str, list[dict]] = {}
    for c in contests:
        ct = c["contest_type"]
        by_type.setdefault(ct, []).append(c)

    for contest_type, group in by_type.items():
        # Pick the contest with the most entries as representative
        best = max(group, key=lambda x: x.get("entries", 0))
        key = f"{slate_date}_{contest_type}"

        # Don't overwrite entries that already have real scores
        existing = history.get(key, {})
        existing_scores = existing.get("scores", {})
        if existing_scores.get("best", 0) > 0:
            log.info("SKIP %s — already has lineup scores", key)
            continue

        entry = {
            "slate_date": slate_date,
            "contest_type": contest_type,
            "cash_line": best.get("cash_line", 0),
            "top_15_score": best.get("top_15_score", 0),
            "top_1_score": best.get("top_1_score", 0),
            "winning_score": best.get("winning_score", 0),
            "num_entries": best.get("entries", 0),
            "notes": f"Auto-ingested from RG: {best.get('name', '')}",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "scores": existing_scores or {
                "n_lineups": 0, "cashed": 0, "top_15": 0, "top_1": 0, "won": 0,
                "cash_rate": 0.0, "top_15_rate": 0.0, "top_1_rate": 0.0,
                "best": 0.0, "avg": 0.0, "median": 0.0,
            },
        }
        history[key] = entry
        saved += 1
        log.info("Saved %s: cash_line=%.1f, winning=%.1f, entries=%d",
                 key, best["cash_line"], best["winning_score"], best["entries"])

        # Save winning lineup if available
        if best.get("winning_lineup"):
            winning_lineups[key] = {
                "slate_date": slate_date,
                "contest_type": contest_type,
                "contest_name": best.get("name", ""),
                "winning_score": best.get("winning_score", 0),
                "lineup": best["winning_lineup"],
            }

    if saved > 0:
        _save_history(history)
        log.info("history.json updated (%d new entries)", saved)

    if winning_lineups:
        _save_winning_lineups(winning_lineups)

    return {"saved": saved, "total_contests": len(contests)}


def fetch_and_save(
    slate_date: str,
    sport: str = "nba",
    site: str = "draftkings",
    contest_filter: str | None = None,
) -> dict:
    """Convenience: fetch from RG and save to history in one call."""
    contests = fetch_rg_results(slate_date, sport=sport, site=site, contest_filter=contest_filter)
    if not contests:
        return {"status": "no_data", "slate_date": slate_date}

    result = save_rg_to_history(contests, slate_date)
    result["status"] = "ok" if result.get("saved", 0) > 0 else "no_new"
    result["slate_date"] = slate_date
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch contest results from RotoGrinders ResultsDB."
    )
    parser.add_argument(
        "--date",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="Slate date (YYYY-MM-DD). Default: yesterday.",
    )
    parser.add_argument(
        "--sport", default="nba",
        help="Sport slug (nba, pga, nfl). Default: nba.",
    )
    parser.add_argument(
        "--site", default="draftkings",
        help="DFS site (draftkings, fanduel). Default: draftkings.",
    )
    parser.add_argument(
        "--contest-type", default=None,
        choices=["gpp", "cash", "showdown"],
        help="Only fetch this contest type.",
    )
    args = parser.parse_args(argv)

    result = fetch_and_save(
        slate_date=args.date,
        sport=args.sport.lower(),
        site=args.site.lower(),
        contest_filter=args.contest_type,
    )

    print(f"\nRG Results: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
