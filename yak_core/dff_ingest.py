"""yak_core/dff_ingest.py
DailyFantasyFuel (DFF) player pool + projections ingestion.

This module provides a cloud-friendly fallback for the DraftKings draftables
API, which returns HTTP 403 from cloud IPs (Streamlit Cloud, Heroku, AWS, etc.).

DFF publishes the full DK player pool daily at:
    https://www.dailyfantasyfuel.com/nba/projections/draftkings

The HTML contains rich data-* attributes on each <tr> row, including:
    data-name, data-pos, data-pos_alt, data-team, data-opp, data-salary,
    data-ppg_proj, data-value_proj, data-inj, data-l5_avg, data-l10_avg,
    data-ou (over/under), data-spread, data-player_id

Public API
----------
fetch_dff_pool(sport="NBA")  → pd.DataFrame
    Returns the current-day DK player pool sourced from DFF.
    Columns: player_name, pos, team, salary, opp, dff_proj, status,
             l5_avg, l10_avg, vegas_total, spread
"""

from __future__ import annotations

import logging
import re
from typing import List

import pandas as pd
import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DFF endpoint
# ---------------------------------------------------------------------------
_DFF_URL = "https://www.dailyfantasyfuel.com/nba/projections/draftkings"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# DFF injury code → YakOS status mapping
_INJ_STATUS_MAP = {
    "O": "OUT",
    "D": "DTD",
    "Q": "GTD",
    "IR": "IR",
    "SUSP": "Suspended",
    "DND": "DND",
    "GL": "G-League",
}


def _safe_float(val: str, default: float = 0.0) -> float:
    """Safely convert a string to float."""
    if not val or val.strip() in ("", "—", "-", "N/A"):
        return default
    try:
        return float(val.strip())
    except (ValueError, TypeError):
        return default


def _safe_int(val: str, default: int = 0) -> int:
    """Safely convert a string to int."""
    if not val or val.strip() in ("", "—", "-", "N/A"):
        return default
    try:
        return int(float(val.strip()))
    except (ValueError, TypeError):
        return default


def _build_pos(pos: str, pos_alt: str) -> str:
    """Build the DK-style position string (e.g. 'PG/SG')."""
    pos = pos.strip().upper()
    pos_alt = pos_alt.strip().upper()
    if pos_alt and pos_alt != pos:
        return f"{pos}/{pos_alt}"
    return pos


def _parse_data_attrs(html: str) -> List[dict]:
    """Parse player data from <tr> data-* attributes in DFF HTML.

    Each player row has a <tr> like:
        <tr class="projections-listing" data-name="Nikola Jokic"
            data-pos="C" data-pos_alt="" data-team="DEN" data-opp="NY"
            data-salary="12500" data-ppg_proj="59.9" data-inj=""
            data-l5_avg="64.7" data-l10_avg="65.4" data-ou="229.5"
            data-spread="-1" ...>

    OUT players have class="hidden projections-listing".
    """
    rows: List[dict] = []

    # Match all <tr> tags that have data-name attribute
    pattern = re.compile(
        r'<tr\b[^>]*data-name="([^"]*)"[^>]*>',
        re.IGNORECASE,
    )

    for match in pattern.finditer(html):
        tag = match.group(0)

        # Extract all data-* attributes from the <tr> tag
        attrs = {}
        for attr_match in re.finditer(r'data-([\w]+)\s*=\s*"([^"]*)"', tag):
            attrs[attr_match.group(1)] = attr_match.group(2)

        name = attrs.get("name", "").strip()
        if not name:
            continue

        pos = _build_pos(attrs.get("pos", ""), attrs.get("pos_alt", ""))
        team = attrs.get("team", "").strip().upper()
        opp = attrs.get("opp", "").strip().upper()
        salary = _safe_int(attrs.get("salary", "0"))
        proj = _safe_float(attrs.get("ppg_proj", "0"))
        value = _safe_float(attrs.get("value_proj", "0"))
        inj = attrs.get("inj", "").strip().upper()
        l5 = _safe_float(attrs.get("l5_avg", ""))
        l10 = _safe_float(attrs.get("l10_avg", ""))

        # Vegas: data-ou may have a leading space due to HTML (data-ou= "221.5")
        ou_raw = attrs.get("ou", "").strip()
        vegas_total = _safe_float(ou_raw)
        spread = _safe_float(attrs.get("spread", ""))

        # Determine status
        status = _INJ_STATUS_MAP.get(inj, "") if inj else ""

        # Check if row is hidden (OUT players)
        is_hidden = "hidden" in tag.split("data-")[0].lower()

        rows.append({
            "player_name": name,
            "pos": pos,
            "team": team,
            "opp": opp,
            "salary": salary,
            "dff_proj": proj,
            "dff_value": value,
            "status": status,
            "l5_avg": l5,
            "l10_avg": l10,
            "vegas_total": vegas_total,
            "spread": spread,
            "is_hidden": is_hidden,
        })

    return rows


def fetch_dff_pool(sport: str = "NBA") -> pd.DataFrame:
    """Fetch today's DK player pool from DailyFantasyFuel.

    Returns a DataFrame with columns matching the YakOS pool convention:
        player_name, pos, team, opp, salary, dff_proj, dff_value,
        status, l5_avg, l10_avg, vegas_total, spread

    Returns an empty DataFrame on failure.
    """
    if sport.upper() != "NBA":
        log.warning("DFF ingestion only supports NBA; got sport=%s", sport)
        return _empty_dff_df()

    try:
        resp = requests.get(_DFF_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        log.error("DFF fetch failed: %s", exc)
        return _empty_dff_df()

    html = resp.text
    players = _parse_data_attrs(html)

    if not players:
        log.warning("DFF: parsed 0 players from response (%d bytes)", len(html))
        return _empty_dff_df()

    df = pd.DataFrame(players)

    # Normalise team abbreviations to DK/YakOS convention
    _team_normalize = {
        "NYK": "NY", "NOP": "NO", "PHX": "PHO", "SAS": "SA",
        "GS": "GSW", "BK": "BKN", "CHS": "CHA",
    }
    df["team"] = df["team"].replace(_team_normalize)
    df["opp"] = df["opp"].replace(_team_normalize)

    # Deduplicate by player_name (keep first = highest projected)
    df = df.sort_values("dff_proj", ascending=False)
    df = df.drop_duplicates(subset=["player_name"], keep="first").reset_index(drop=True)

    active_count = len(df[df["dff_proj"] > 0])
    out_count = len(df[df["status"] != ""])
    log.info(
        "DFF pool loaded: %d players total, %d active (proj > 0), %d with injury status",
        len(df), active_count, out_count,
    )

    return df


def _empty_dff_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "player_name", "pos", "team", "opp", "salary", "dff_proj",
            "dff_value", "status", "l5_avg", "l10_avg", "vegas_total",
            "spread", "is_hidden",
        ]
    )
