"""
yak_core/dff_ingest.py
DailyFantasyFuel (DFF) fallback pool ingest.

Used when the DraftKings API returns HTTP 403 (blocked from cloud IPs).
DFF publishes the full DK player pool daily; publicly accessible from cloud IPs,
no authentication required.

Public API
----------
fetch_dff_pool(sport, date_str) → pd.DataFrame
    Parses DFF HTML data-* attributes and returns a DataFrame with columns:
        player_name, pos, team, opp, salary, proj, ownership,
        status, vegas_total, vegas_spread

    The returned DataFrame is compatible with the YakOS player pool pipeline:
    columns align with what fetch_dk_draftables() returns after
    _normalize_dk_pool() is applied, so the rest of the Slate Hub pipeline
    works unchanged.

No new third-party dependencies — uses only stdlib html.parser + requests.
"""

from __future__ import annotations

import logging
import re
import time
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DFF URL templates
# ---------------------------------------------------------------------------
_DFF_BASE_URL = "https://www.dailyfantasyfuel.com"
_DFF_SPORT_PATHS: Dict[str, str] = {
    "NBA": "/nba/projections/draftkings/",
    "PGA": "/pga/projections/draftkings/",
    "NFL": "/nfl/projections/draftkings/",
    "MLB": "/mlb/projections/draftkings/",
    "NHL": "/nhl/projections/draftkings/",
}

_REQUEST_TIMEOUT = 30
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2.0  # seconds between retries

# ---------------------------------------------------------------------------
# Column-name aliases: maps DFF data-* attribute names → YakOS column names.
# Multiple possible attribute names are listed in priority order.
# ---------------------------------------------------------------------------
_ATTR_MAP: Dict[str, List[str]] = {
    "player_name": [
        "name", "player-name", "playername", "player_name",
        "fullname", "full-name",
    ],
    "pos": [
        "pos", "position", "positions", "dkpos", "dk-pos",
    ],
    "team": [
        "team", "teamabbrev", "team-abbrev", "teamabv",
    ],
    "opp": [
        "opp", "opponent", "vs", "matchup",
    ],
    "salary": [
        "salary", "sal", "dksalary", "dk-salary", "dk_salary",
    ],
    "proj": [
        "proj", "fpts", "fp", "points", "dkpts", "dk-pts", "projection",
        "projected-points", "projectedpoints",
    ],
    "ownership": [
        "own", "ownership", "proj-own", "projown", "projected-ownership",
    ],
    "status": [
        "status", "injury", "injury-status", "injurystatus",
    ],
    "vegas_total": [
        "ou", "over-under", "total", "vegas-total", "vegastotal",
        "game-total", "gametotal",
    ],
    "vegas_spread": [
        "spread", "vegas-spread", "line", "point-spread",
    ],
}


# ---------------------------------------------------------------------------
# HTML parser — collects elements that have DFF player data-* attributes
# ---------------------------------------------------------------------------

class _DFFHTMLParser(HTMLParser):
    """Minimal HTML parser that collects player rows from DFF.

    DFF renders player data as HTML elements (typically ``<tr>`` or ``<div>``)
    whose ``data-*`` attributes carry salary, projections, etc.

    Strategy:
    1. Look for any element with a ``data-name`` (or aliased) attribute whose
       value looks like a person's name (two+ words, letters/spaces).
    2. Accumulate all ``data-*`` attributes from that element as a dict.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[Dict[str, str]] = []
        # Extract the player-name attribute keys that we recognise
        self._name_attrs = set(_ATTR_MAP["player_name"])

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        # Only examine row-like elements
        if tag not in ("tr", "div", "li", "span", "td"):
            return

        attr_dict: Dict[str, str] = {}
        for key, value in attrs:
            if key.startswith("data-") and value:
                attr_dict[key[5:]] = value  # strip 'data-' prefix

        # Require a recognisable player-name attribute
        name_val = None
        for alias in self._name_attrs:
            if alias in attr_dict:
                name_val = attr_dict[alias].strip()
                break

        if not name_val:
            return

        # Quick sanity check: name must look like a real person
        # (at least two alphabetic words, e.g. "LeBron James")
        if not re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{1,}", name_val):
            return

        self.rows.append(attr_dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_field(row: Dict[str, str], aliases: List[str]) -> Optional[str]:
    """Return the first matching value from ``row`` given a list of alias keys."""
    for alias in aliases:
        if alias in row and row[alias]:
            return row[alias]
    return None


def _parse_rows(rows: List[Dict[str, str]]) -> pd.DataFrame:
    """Convert a list of raw attribute dicts to a normalised YakOS DataFrame."""
    records = []
    for row in rows:
        record: Dict[str, Any] = {}
        for col, aliases in _ATTR_MAP.items():
            record[col] = _extract_field(row, aliases)
        records.append(record)

    if not records:
        return _empty_dff_pool()

    df = pd.DataFrame(records)

    # Type coercions
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0.0)
    df["ownership"] = pd.to_numeric(df["ownership"], errors="coerce").fillna(0.0)
    df["vegas_total"] = pd.to_numeric(df["vegas_total"], errors="coerce")
    df["vegas_spread"] = pd.to_numeric(df["vegas_spread"], errors="coerce")

    # Drop rows without a name or salary
    df = df[df["player_name"].notna() & (df["player_name"] != "")]
    df = df[df["salary"] > 0]

    # Normalise position strings: "SF/PF" → "SF"
    df["pos"] = df["pos"].fillna("UTIL").str.strip().str.upper()
    df["team"] = df["team"].fillna("").str.strip().str.upper()
    df["opp"] = df["opp"].fillna("").str.strip().str.upper()
    df["status"] = df["status"].fillna("Active").str.strip()

    return df.reset_index(drop=True)


def _fetch_html(url: str) -> str:
    """GET *url* with retries; raises ``requests.HTTPError`` on final failure."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            log.warning(
                "[dff_ingest] Attempt %d/%d failed for %s: %s",
                attempt, _RETRY_ATTEMPTS, url, exc,
            )
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_DELAY)
    raise last_exc  # type: ignore[misc]


def _empty_dff_pool() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "player_name", "pos", "team", "opp",
            "salary", "proj", "ownership",
            "status", "vegas_total", "vegas_spread",
        ]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_dff_pool(sport: str = "NBA", date_str: Optional[str] = None) -> pd.DataFrame:
    """Fetch the DFF player pool for *sport* and return a YakOS-compatible DataFrame.

    Parameters
    ----------
    sport:
        Sport abbreviation (``"NBA"``, ``"PGA"``, ``"NFL"``, …).
    date_str:
        Ignored — DFF always serves today's projections at the projections URL.
        Parameter kept for forward-compatibility.

    Returns
    -------
    pd.DataFrame
        Columns: player_name, pos, team, opp, salary, proj, ownership,
                 status, vegas_total, vegas_spread

        Returns an empty DataFrame with the correct columns if the page
        cannot be fetched or no players are found.

    Notes
    -----
    DFF is a **read-only public fallback** — no login, no API key.
    The HTML is fetched and ``data-*`` attributes on player row elements
    are parsed to build the pool.  Attribute name variants are mapped via
    ``_ATTR_MAP``.
    """
    sport_upper = sport.upper()
    path = _DFF_SPORT_PATHS.get(sport_upper)
    if path is None:
        log.warning("[dff_ingest] Unsupported sport: %s", sport)
        return _empty_dff_pool()

    url = _DFF_BASE_URL + path
    log.info("[dff_ingest] Fetching DFF pool from %s", url)

    try:
        html = _fetch_html(url)
    except Exception as exc:
        log.error("[dff_ingest] Failed to fetch DFF page: %s", exc)
        return _empty_dff_pool()

    parser = _DFFHTMLParser()
    parser.feed(html)

    if not parser.rows:
        log.warning("[dff_ingest] No player rows found in DFF HTML for sport=%s", sport)
        return _empty_dff_pool()

    df = _parse_rows(parser.rows)
    log.info(
        "[dff_ingest] Parsed %d players, %d teams from DFF (%s)",
        len(df),
        df["team"].nunique() if not df.empty else 0,
        sport,
    )
    return df
