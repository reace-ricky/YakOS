"""
yak_core/dk_ingest.py
DraftKings lobby contest ingestion, Draft Group + player pool ingest,
DK → YakOS player mapping, roster-rule parsing, and contest-scoped pool builder.

Public API
----------
Ingest (server-side):
  fetch_dk_lobby_contests(sport)          → pd.DataFrame  (5.1)
  fetch_dk_draft_group(draft_group_id)    → dict          (5.2)
  fetch_dk_draftables(draft_group_id)     → pd.DataFrame  (5.2)
  save_dk_contests(df)                    → None
  load_dk_contests(sport)                 → pd.DataFrame
  save_dk_slates(df)                      → None
  load_dk_slates()                        → pd.DataFrame
  save_dk_player_pool(df)                 → None
  load_dk_player_pool_for_group(dg_id)    → pd.DataFrame

Mapping (5.3):
  map_dk_players_to_yak(draft_group_id, yak_pool_df) → pd.DataFrame
  save_dk_player_map(df)                  → None
  load_dk_player_map(draft_group_id)      → pd.DataFrame
  get_mapping_diagnostics(draft_group_id) → dict

Roster rules (5.6):
  fetch_game_type_rules(game_type_id)     → dict
  parse_roster_rules(rules_json)          → dict

Contest-scoped pool (5.5):
  build_contest_scoped_pool(draft_group_id, yak_pool_df) → pd.DataFrame

Helpers:
  is_dk_integration_enabled()            → bool
"""

from __future__ import annotations

import os
import re
import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .config import YAKOS_ROOT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
_DK_DIR = Path(YAKOS_ROOT) / "data" / "dk"
_CONTESTS_PATH = _DK_DIR / "dk_contests.parquet"
_SLATES_PATH = _DK_DIR / "dk_slates.parquet"
_PLAYER_POOL_PATH = _DK_DIR / "dk_player_pool.parquet"
_PLAYER_MAP_PATH = _DK_DIR / "dk_player_map.parquet"

# ---------------------------------------------------------------------------
# DK API endpoints
# ---------------------------------------------------------------------------
_DK_LOBBY_URL = "https://www.draftkings.com/lobby/getcontests"
_DK_DRAFT_GROUP_URL = "https://api.draftkings.com/draftgroups/v1/{draft_group_id}"
_DK_DRAFTABLES_URL = (
    "https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group_id}/draftables"
)
_DK_GAME_RULES_URL = (
    "https://api.draftkings.com/lineups/v1/gametypes/{game_type_id}/rules"
)
_RULES_JSON_FALLBACK = Path(YAKOS_ROOT) / "config" / "RulesAndScoring.json"

# Projection/pool columns to carry from YakOS pool into the contest-scoped pool.
_YAK_PROJECTION_COLS = (
    "proj", "floor", "ceil", "ownership", "proj_own", "own_proj",
    "minutes", "proj_minutes", "status", "sim_eligible",
    "injury_bump_fp", "original_proj", "adjusted_proj",
)

# DK game_type_id → human-readable contest label used in the Slate Room panel.
DK_GAME_TYPE_LABELS: Dict[int, str] = {
    0: "Classic",
    1: "Classic",
    65: "Late Night",
    96: "Showdown",
    114: "Showdown",
}

# ---------------------------------------------------------------------------
# Rate-limit state (simple per-process token bucket)
# ---------------------------------------------------------------------------
_last_request_ts: float = 0.0
_MIN_REQUEST_INTERVAL = 0.5  # seconds between DK API calls


def _rate_limited_get(url: str, params: Optional[Dict] = None, timeout: int = 10) -> requests.Response:
    """GET ``url`` with a simple inter-request delay and retries."""
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_ts = time.time()
    headers = {"User-Agent": "YakOS/1.0 (DFS optimizer; https://github.com/reace-ricky/YakOS)"}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def is_dk_integration_enabled() -> bool:
    """Return True when DK integration is turned on via env/config."""
    env = os.environ.get("DK_INTEGRATION_ENABLED", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    # Default: enabled
    return True


def _enabled_sports() -> List[str]:
    raw = os.environ.get("DK_SPORTS_ENABLED", "NBA,PGA")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# 5.1 — DK lobby contest ingest
# ---------------------------------------------------------------------------

def fetch_dk_lobby_contests(sport: str = "NBA") -> pd.DataFrame:
    """Fetch DK lobby contests for ``sport`` and return a normalised DataFrame.

    Columns: contest_id, name, sport, game_type_id, draft_group_id, start_time,
             entry_fee, prize_pool, max_entries, max_entries_per_user,
             current_entries, is_single_entry.

    Raises
    ------
    requests.HTTPError
        When the DK endpoint returns a non-2xx status.
    """
    if not is_dk_integration_enabled():
        log.warning("[dk_ingest] DK integration is disabled; returning empty contests.")
        return _empty_contests_df()

    resp = _rate_limited_get(_DK_LOBBY_URL, params={"sport": sport.upper()})
    data = resp.json()

    contests_raw: List[Dict] = []
    # DK wraps contests under several possible keys
    for key in ("Contests", "contests", "DraftGroups", "draftGroups"):
        if key in data:
            contests_raw = data[key]
            break

    if not contests_raw:
        log.warning("[dk_ingest] No contests found for sport=%s", sport)
        return _empty_contests_df()

    rows = []
    for c in contests_raw:
        rows.append(
            {
                "contest_id": str(c.get("id") or c.get("ContestId") or c.get("contestId") or ""),
                "name": str(c.get("n") or c.get("name") or c.get("Name") or ""),
                "sport": sport.upper(),
                "game_type_id": int(c.get("gameTypeId") or c.get("GameTypeId") or 0),
                "draft_group_id": int(c.get("dg") or c.get("draftGroupId") or c.get("DraftGroupId") or 0),
                "start_time": str(c.get("sd") or c.get("startDate") or c.get("StartDate") or ""),
                "entry_fee": float(c.get("a") or c.get("entryFee") or c.get("EntryFee") or 0),
                "prize_pool": float(c.get("po") or c.get("prizePool") or c.get("PrizePool") or 0),
                "max_entries": int(c.get("m") or c.get("maximumEntries") or c.get("MaximumEntries") or 0),
                "max_entries_per_user": int(
                    c.get("mec") or c.get("maximumEntriesPerUser") or c.get("MaximumEntriesPerUser") or 0
                ),
                "current_entries": int(c.get("nt") or c.get("numberOfEntrants") or c.get("NumberOfEntrants") or 0),
                "is_single_entry": bool(c.get("ise") or c.get("isSingleEntry") or c.get("IsSingleEntry") or False),
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["contest_id"] != ""]
    log.info("[dk_ingest] Fetched %d contests for sport=%s", len(df), sport)
    return df


def _empty_contests_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "contest_id", "name", "sport", "game_type_id", "draft_group_id",
            "start_time", "entry_fee", "prize_pool", "max_entries",
            "max_entries_per_user", "current_entries", "is_single_entry",
        ]
    )


def save_dk_contests(df: pd.DataFrame) -> None:
    """Upsert ``df`` into the persisted dk_contests parquet (keyed by contest_id)."""
    _DK_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_dk_contests()
    if existing.empty:
        combined = df.copy()
    else:
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["contest_id"], keep="last")
    combined.to_parquet(_CONTESTS_PATH, index=False)
    log.info("[dk_ingest] Saved %d contests to %s", len(combined), _CONTESTS_PATH)


def load_dk_contests(sport: Optional[str] = None) -> pd.DataFrame:
    """Load persisted dk_contests, optionally filtered by ``sport``."""
    if not _CONTESTS_PATH.exists():
        return _empty_contests_df()
    try:
        df = pd.read_parquet(_CONTESTS_PATH)
    except Exception as exc:
        log.warning("[dk_ingest] Failed to read %s: %s", _CONTESTS_PATH, exc)
        return _empty_contests_df()
    if sport and "sport" in df.columns:
        df = df[df["sport"].str.upper() == sport.upper()]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5.2 — Draft Group + player pool ingest
# ---------------------------------------------------------------------------

def fetch_dk_draft_group(draft_group_id: int) -> Dict[str, Any]:
    """Fetch draft group metadata from DK API.

    Returns a dict with keys: draft_group_id, game_type_id, sport,
    start_time, games (list), raw.
    """
    if not is_dk_integration_enabled():
        return {"draft_group_id": draft_group_id}

    url = _DK_DRAFT_GROUP_URL.format(draft_group_id=draft_group_id)
    resp = _rate_limited_get(url)
    data = resp.json()

    dg = data.get("draftGroup") or data.get("DraftGroup") or data
    return {
        "draft_group_id": int(dg.get("draftGroupId") or dg.get("DraftGroupId") or draft_group_id),
        "game_type_id": int(dg.get("gameTypeId") or dg.get("GameTypeId") or 0),
        "sport": str(dg.get("sport") or dg.get("sportAbbreviation") or ""),
        "start_time": str(dg.get("startTimeSuffix") or dg.get("startDate") or ""),
        "games": dg.get("games") or dg.get("Games") or [],
        "raw": dg,
    }


def fetch_dk_draftables(draft_group_id: int) -> pd.DataFrame:
    """Fetch DK player pool (draftables) for a draft group.

    Columns: draft_group_id, dk_player_id, name, name_suffix, display_name,
             team, positions, salary, status.
    """
    if not is_dk_integration_enabled():
        return _empty_player_pool_df()

    url = _DK_DRAFTABLES_URL.format(draft_group_id=draft_group_id)
    resp = _rate_limited_get(url)
    data = resp.json()

    draftables_raw = (
        data.get("draftables")
        or data.get("Draftables")
        or (data.get("draftablesResponse") or {}).get("draftables")
        or []
    )

    # Known valid NBA/DFS positions for validation.
    _VALID_POSITIONS = {"PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "CPT", "FLEX"}

    rows = []
    for p in draftables_raw:
        # Primary source: explicit position key on the draftable record.
        primary_pos = str(p.get("position") or p.get("Position") or "").strip()
        if primary_pos and primary_pos.upper() in _VALID_POSITIONS:
            pos_list: List[str] = [primary_pos]
        else:
            # Fallback: scan playerGameAttributes for known position tokens.
            attrs = p.get("playerGameAttributes") or p.get("draftStatAttributes") or []
            pos_list = []
            if isinstance(attrs, list):
                for attr in attrs:
                    if isinstance(attr, dict):
                        v = str(attr.get("value") or attr.get("sortValue") or "").strip()
                        if v.upper() in _VALID_POSITIONS:
                            pos_list.append(v)
            # If attrs yielded nothing, fall back to the raw position field even
            # if it isn't in the known NBA/DFS set (e.g. sport-specific positions).
            if not pos_list:
                pos_list = [primary_pos] if primary_pos else [""]

        # Extract opponent + game info from competition field
        team_abbr = str(p.get("teamAbbreviation") or p.get("teamAbv") or p.get("team") or "")
        comp = p.get("competition") or {}
        game_name = str(comp.get("name", ""))  # e.g. "HOU @ SAS"
        game_time = str(comp.get("startTime", ""))
        opp = ""
        if game_name and "@" in game_name:
            parts = [t.strip() for t in game_name.replace("@", "vs").split("vs")]
            opp_candidates = [t for t in parts if t.upper() != team_abbr.upper()]
            opp = opp_candidates[0] if opp_candidates else ""

        rows.append(
            {
                "draft_group_id": int(draft_group_id),
                "dk_player_id": str(p.get("playerId") or p.get("PlayerID") or p.get("draftableId") or ""),
                "draftable_id": str(p.get("draftableId") or ""),
                "name": str(p.get("displayName") or p.get("shortName") or p.get("playerName") or ""),
                "name_suffix": str(p.get("nameSuffix") or ""),
                "display_name": str(p.get("displayName") or p.get("playerName") or ""),
                "team": team_abbr,
                "opp": opp,
                "game_info": game_name,
                "game_time": game_time,
                "positions": "/".join(pos_list) if pos_list else "",
                "salary": float(p.get("salary") or p.get("Salary") or 0),
                "status": str(p.get("playerGameInfo", {}).get("status") or p.get("status") or "Active"),
            }
        )

    df = pd.DataFrame(rows) if rows else _empty_player_pool_df()

    # Dedup: DK returns one row per roster-slot (PG, SG, UTIL, …) for the
    # same player.  Keep only the first entry per playerId so downstream
    # code sees exactly one row per player.
    if not df.empty and "dk_player_id" in df.columns:
        _before = len(df)
        df = df.drop_duplicates(subset=["dk_player_id"], keep="first").reset_index(drop=True)
        if len(df) < _before:
            log.info(
                "[dk_ingest] Deduped %d roster-slot rows → %d unique players for DG %s",
                _before, len(df), draft_group_id,
            )

    log.info(
        "[dk_ingest] Fetched %d draftables for draft_group_id=%s", len(df), draft_group_id
    )
    return df


def _empty_player_pool_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "draft_group_id", "dk_player_id", "draftable_id", "name",
            "name_suffix", "display_name", "team", "positions", "salary",
            "status",
        ]
    )


def save_dk_slates(df: pd.DataFrame) -> None:
    """Upsert slate info (keyed by draft_group_id)."""
    _DK_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_dk_slates()
    if existing.empty:
        combined = df.copy()
    else:
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["draft_group_id"], keep="last")
    combined.to_parquet(_SLATES_PATH, index=False)


def load_dk_slates() -> pd.DataFrame:
    if not _SLATES_PATH.exists():
        return pd.DataFrame(columns=["draft_group_id", "game_type_id", "sport", "start_time"])
    try:
        return pd.read_parquet(_SLATES_PATH)
    except Exception:
        return pd.DataFrame(columns=["draft_group_id", "game_type_id", "sport", "start_time"])


def save_dk_player_pool(df: pd.DataFrame) -> None:
    """Upsert player pool rows (keyed by draft_group_id + dk_player_id)."""
    _DK_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_dk_player_pool()
    if existing.empty:
        combined = df.copy()
    else:
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["draft_group_id", "dk_player_id"], keep="last"
        )
    combined.to_parquet(_PLAYER_POOL_PATH, index=False)


def load_dk_player_pool() -> pd.DataFrame:
    if not _PLAYER_POOL_PATH.exists():
        return _empty_player_pool_df()
    try:
        return pd.read_parquet(_PLAYER_POOL_PATH)
    except Exception:
        return _empty_player_pool_df()


def load_dk_player_pool_for_group(draft_group_id: int) -> pd.DataFrame:
    """Load persisted player pool filtered to a single draft group."""
    df = load_dk_player_pool()
    if df.empty:
        return _empty_player_pool_df()
    return df[df["draft_group_id"] == int(draft_group_id)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5.3 — DK → YakOS player mapping
# ---------------------------------------------------------------------------

def _normalize_name(s: str) -> str:
    """Lowercase, strip punctuation, remove suffixes for fuzzy matching."""
    s = s.lower().strip()
    s = re.sub(r"[.'`\-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def map_dk_players_to_yak(
    draft_group_id: int,
    yak_pool_df: pd.DataFrame,
    dk_pool_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Map DK draftables to YakOS player IDs.

    Parameters
    ----------
    draft_group_id:
        The DraftKings draft group to map.
    yak_pool_df:
        YakOS player pool.  Must have a ``player_name`` column; optionally
        ``team``, ``pos``, and ``player_id`` columns.
    dk_pool_df:
        Pre-fetched DK draftables.  If None, loaded from persisted storage.

    Returns
    -------
    DataFrame with columns: draft_group_id, dk_player_id, dk_name, dk_team,
    dk_positions, yak_player_id, yak_name, match_quality.
    """
    if dk_pool_df is None:
        dk_pool_df = load_dk_player_pool_for_group(draft_group_id)

    if dk_pool_df.empty or yak_pool_df.empty:
        return _empty_player_map_df()

    yak = yak_pool_df.copy()
    if "player_name" not in yak.columns:
        return _empty_player_map_df()

    # Normalised name lookup for YakOS: norm_name → row index (first match)
    yak_name_lookup: Dict[str, int] = {}
    for idx, row in yak.iterrows():
        n = _normalize_name(str(row.get("player_name", "")))
        if n and n not in yak_name_lookup:
            yak_name_lookup[n] = idx  # type: ignore[arg-type]

    # Optional team lookup
    yak_has_team = "team" in yak.columns

    rows = []
    for _, dk_row in dk_pool_df.iterrows():
        dk_id = str(dk_row.get("dk_player_id", ""))
        dk_name = str(dk_row.get("name", "") or dk_row.get("display_name", ""))
        dk_team = str(dk_row.get("team", ""))
        dk_pos = str(dk_row.get("positions", ""))
        norm_dk = _normalize_name(dk_name)

        yak_idx: Optional[int] = None
        match_quality = "none"

        # 1. Exact normalized name match
        if norm_dk in yak_name_lookup:
            cand_idx = yak_name_lookup[norm_dk]
            if yak_has_team:
                cand_team = str(yak.at[cand_idx, "team"]).upper()  # type: ignore[arg-type]
                if cand_team == dk_team.upper() or not dk_team:
                    yak_idx = cand_idx
                    match_quality = "exact+team"
                else:
                    yak_idx = cand_idx
                    match_quality = "name_only"
            else:
                yak_idx = cand_idx
                match_quality = "name_only"

        # 2. Partial suffix match (first+last token overlap)
        if yak_idx is None:
            dk_tokens = norm_dk.split()
            for norm_yak, cand_idx in yak_name_lookup.items():
                yak_tokens = norm_yak.split()
                if len(dk_tokens) >= 2 and len(yak_tokens) >= 2:
                    if dk_tokens[0] == yak_tokens[0] and dk_tokens[-1] == yak_tokens[-1]:
                        yak_idx = cand_idx
                        match_quality = "fuzzy_name"
                        break

        yak_id = ""
        yak_name = ""
        if yak_idx is not None:
            id_col = "player_id" if "player_id" in yak.columns else "player_name"
            yak_id = str(yak.at[yak_idx, id_col])  # type: ignore[arg-type]
            yak_name = str(yak.at[yak_idx, "player_name"])  # type: ignore[arg-type]

        rows.append(
            {
                "draft_group_id": int(draft_group_id),
                "dk_player_id": dk_id,
                "dk_name": dk_name,
                "dk_team": dk_team,
                "dk_positions": dk_pos,
                "yak_player_id": yak_id,
                "yak_name": yak_name,
                "match_quality": match_quality,
            }
        )

    return pd.DataFrame(rows) if rows else _empty_player_map_df()


def _empty_player_map_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "draft_group_id", "dk_player_id", "dk_name", "dk_team",
            "dk_positions", "yak_player_id", "yak_name", "match_quality",
        ]
    )


def save_dk_player_map(df: pd.DataFrame) -> None:
    """Upsert player map (keyed by draft_group_id + dk_player_id)."""
    _DK_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_dk_player_map()
    if existing.empty:
        combined = df.copy()
    else:
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["draft_group_id", "dk_player_id"], keep="last"
        )
    combined.to_parquet(_PLAYER_MAP_PATH, index=False)


def load_dk_player_map(draft_group_id: Optional[int] = None) -> pd.DataFrame:
    """Load persisted player map, optionally filtered by draft_group_id."""
    if not _PLAYER_MAP_PATH.exists():
        return _empty_player_map_df()
    try:
        df = pd.read_parquet(_PLAYER_MAP_PATH)
    except Exception:
        return _empty_player_map_df()
    if draft_group_id is not None:
        df = df[df["draft_group_id"] == int(draft_group_id)]
    return df.reset_index(drop=True)


def get_mapping_diagnostics(draft_group_id: int) -> Dict[str, Any]:
    """Return a summary of mapping coverage for a draft group.

    Returns
    -------
    dict with keys:
      draft_group_id, total_dk_players, mapped_count, pct_mapped,
      unmapped_players (list of {dk_player_id, dk_name, dk_team}).
    """
    mapping = load_dk_player_map(draft_group_id)
    if mapping.empty:
        return {
            "draft_group_id": draft_group_id,
            "total_dk_players": 0,
            "mapped_count": 0,
            "pct_mapped": 0.0,
            "unmapped_players": [],
        }
    total = len(mapping)
    mapped = mapping["yak_player_id"].astype(bool).sum()
    unmapped_rows = mapping[~mapping["yak_player_id"].astype(bool)]
    unmapped = unmapped_rows[["dk_player_id", "dk_name", "dk_team"]].to_dict("records")
    return {
        "draft_group_id": draft_group_id,
        "total_dk_players": total,
        "mapped_count": int(mapped),
        "pct_mapped": round(mapped / total * 100, 1) if total else 0.0,
        "unmapped_players": unmapped,
    }


# ---------------------------------------------------------------------------
# 5.6 — Game type rules auto-configuration
# ---------------------------------------------------------------------------

def fetch_game_type_rules(game_type_id: int) -> Dict[str, Any]:
    """Fetch roster constraint rules for a DK game type.

    Falls back to ``config/RulesAndScoring.json`` if the API call fails.
    """
    if not is_dk_integration_enabled():
        return _load_fallback_rules(game_type_id)

    url = _DK_GAME_RULES_URL.format(game_type_id=game_type_id)
    try:
        resp = _rate_limited_get(url)
        return resp.json()
    except Exception as exc:
        log.warning("[dk_ingest] game-type rules fetch failed: %s — using fallback", exc)
        return _load_fallback_rules(game_type_id)


def _load_fallback_rules(game_type_id: int) -> Dict[str, Any]:
    if _RULES_JSON_FALLBACK.exists():
        try:
            with open(_RULES_JSON_FALLBACK) as fh:
                data = json.load(fh)
            rules = data.get("gameTypes", {}).get(str(game_type_id), {})
            return rules or {}
        except Exception:
            pass
    return {}


def parse_roster_rules(rules_json: Dict[str, Any]) -> Dict[str, Any]:
    """Parse raw DK game-type rules JSON into YakOS constraint config.

    Returns a dict with keys:
      lineup_size, salary_cap, slots (list of position strings),
      captain_slot (bool), is_showdown (bool),
      team_limits (dict), game_limits (dict).
    """
    out: Dict[str, Any] = {
        "lineup_size": 8,
        "salary_cap": 50000,
        "slots": ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"],
        "captain_slot": False,
        "is_showdown": False,
        "team_limits": {},
        "game_limits": {},
    }

    # Navigate common rule shapes
    lineup_rules = (
        rules_json.get("rosterSlots")
        or rules_json.get("lineupRules")
        or rules_json.get("LineupRules")
        or rules_json.get("gameTypeRules", {}).get("rosterSlots")
        or []
    )

    if isinstance(lineup_rules, list) and lineup_rules:
        slots = []
        is_cpt = False
        for slot in lineup_rules:
            if isinstance(slot, dict):
                label = (
                    slot.get("name")
                    or slot.get("label")
                    or slot.get("slotName")
                    or slot.get("position")
                    or ""
                )
                if str(label).upper() in ("CPT", "CAPTAIN"):
                    is_cpt = True
                slots.append(str(label).upper())
        if slots:
            out["slots"] = slots
            out["lineup_size"] = len(slots)
            out["captain_slot"] = is_cpt
            out["is_showdown"] = is_cpt

    salary_cap = (
        rules_json.get("salaryCap")
        or rules_json.get("SalaryCap")
        or rules_json.get("gameTypeRules", {}).get("salaryCap")
    )
    if salary_cap:
        out["salary_cap"] = int(salary_cap)

    return out


# ---------------------------------------------------------------------------
# 5.5 — Contest-scoped optimizer pool builder
# ---------------------------------------------------------------------------

def build_contest_scoped_pool(
    draft_group_id: int,
    yak_pool_df: pd.DataFrame,
    dk_pool_df: Optional[pd.DataFrame] = None,
    mapping_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build an optimizer pool scoped to a DK draft group.

    Merges DK salary/positions/status from the player pool with YakOS
    projection columns (proj, floor, ceil, ownership, …).  Only players
    present in the DK draft group appear in the result.

    Parameters
    ----------
    draft_group_id:
        DK draft group to scope the pool to.
    yak_pool_df:
        YakOS internal player pool with projections.
    dk_pool_df:
        Pre-fetched DK draftables.  Loaded from storage if None.
    mapping_df:
        Pre-computed mapping table.  Computed on-the-fly if None.

    Returns
    -------
    DataFrame in YakOS pool format with columns:
      player_name, player_id, pos, team, salary (DK), proj (YakOS),
      floor, ceil, ownership, dk_player_id, dk_salary, match_quality.
    Also adds ``_unmapped`` boolean column so callers can surface unmatched
    players in a debug list.
    """
    # Load / compute what we need
    if dk_pool_df is None:
        dk_pool_df = load_dk_player_pool_for_group(draft_group_id)

    if mapping_df is None:
        mapping_df = map_dk_players_to_yak(draft_group_id, yak_pool_df, dk_pool_df)

    if dk_pool_df.empty:
        log.warning("[dk_ingest] DK pool empty for draft_group_id=%s", draft_group_id)
        return yak_pool_df.copy()

    # Merge DK pool + mapping
    dk_merged = dk_pool_df.merge(
        mapping_df[["dk_player_id", "yak_player_id", "yak_name", "match_quality"]],
        on="dk_player_id",
        how="left",
    )

    # Merge YakOS projections via yak_player_id or player_name fallback
    yak = yak_pool_df.copy()
    yak_id_col = "player_id" if "player_id" in yak.columns else None
    if yak_id_col:
        dk_merged = dk_merged.merge(
            yak.rename(columns={"player_id": "yak_player_id"}),
            on="yak_player_id",
            how="left",
            suffixes=("_dk", "_yak"),
        )
    else:
        dk_merged = dk_merged.merge(
            yak.rename(columns={"player_name": "yak_name"}),
            on="yak_name",
            how="left",
            suffixes=("_dk", "_yak"),
        )

    # Build canonical output columns
    out = pd.DataFrame()
    out["player_name"] = dk_merged.get("player_name", dk_merged.get("name", ""))
    # Fall back to DK display name for unmapped players
    no_name_mask = out["player_name"].isna() | (out["player_name"] == "")
    out.loc[no_name_mask, "player_name"] = dk_merged.loc[no_name_mask, "name"].values if "name" in dk_merged.columns else ""

    out["player_id"] = (
        dk_merged.get("yak_player_id", dk_merged.get("dk_player_id", ""))
    )
    out["pos"] = dk_merged.get("pos", dk_merged.get("positions", ""))
    out["team"] = dk_merged.get("team_yak", dk_merged.get("team", dk_merged.get("team_dk", "")))
    # Use DK salary for this run
    out["salary"] = pd.to_numeric(dk_merged.get("salary_dk", dk_merged.get("salary", 0)), errors="coerce").fillna(0)
    out["dk_salary"] = out["salary"]
    out["dk_player_id"] = dk_merged.get("dk_player_id", "")
    # Opponent + game info from DK draftables
    out["opp"] = dk_merged.get("opp", dk_merged.get("opp_dk", ""))
    out["game_info"] = dk_merged.get("game_info", dk_merged.get("game_info_dk", ""))
    out["game_time"] = dk_merged.get("game_time", dk_merged.get("game_time_dk", ""))

    # YakOS projection columns (fill 0 when unmapped)
    for col in _YAK_PROJECTION_COLS:
        if col in dk_merged.columns:
            out[col] = dk_merged[col]
        else:
            col_dk = col + "_dk"
            col_yak = col + "_yak"
            if col_yak in dk_merged.columns:
                out[col] = dk_merged[col_yak]
            elif col_dk in dk_merged.columns:
                out[col] = dk_merged[col_dk]
            else:
                out[col] = 0 if col not in ("status", "sim_eligible") else ("Active" if col == "status" else True)

    out["match_quality"] = dk_merged.get("match_quality", "none")
    out["_unmapped"] = out["match_quality"] == "none"

    # Drop rows with no player name
    out = out[out["player_name"].astype(str).str.strip() != ""].reset_index(drop=True)

    mapped_count = (~out["_unmapped"]).sum()
    log.info(
        "[dk_ingest] Contest-scoped pool: %d players (%d mapped, %d unmapped)",
        len(out),
        mapped_count,
        out["_unmapped"].sum(),
    )
    return out


# ---------------------------------------------------------------------------
# 5.7 — DK Showdown lobby helpers
# ---------------------------------------------------------------------------

def fetch_dk_showdown_matchups(sport: str = "NBA") -> List[Dict[str, Any]]:
    """Fetch live Showdown matchups from the DK lobby.

    Returns a list of dicts, each with:
      draft_group_id, away, home, label ("PHI @ SAC"),
      start_time, start_time_est.

    Only returns matchups that DK is actively offering Showdown
    contests for today.  Returns [] on any failure.
    """
    try:
        resp = _rate_limited_get(_DK_LOBBY_URL, params={"sport": sport.upper()})
        data = resp.json()
    except Exception as exc:
        log.warning("[dk_ingest] Showdown lobby fetch failed: %s", exc)
        return []

    dg_info = data.get("DraftGroups") or data.get("draftGroups") or []

    # game_type 81 = NBA Showdown Captain Mode
    _SD_GAME_TYPES = {81}
    matchups: List[Dict[str, Any]] = []
    seen_dgs: set = set()

    for dg in dg_info:
        gt = dg.get("GameTypeId") or dg.get("gameTypeId") or 0
        if gt not in _SD_GAME_TYPES:
            continue
        dg_id = dg.get("DraftGroupId") or dg.get("draftGroupId") or 0
        if not dg_id or dg_id in seen_dgs:
            continue
        seen_dgs.add(dg_id)

        # Extract matchup from ContestStartTimeSuffix: " (PHI @ SAC)"
        suffix = str(
            dg.get("ContestStartTimeSuffix")
            or dg.get("contestStartTimeSuffix")
            or ""
        ).strip()
        away, home = "", ""
        if "@" in suffix:
            import re as _re
            m = _re.search(r"\(([A-Z]+)\s*@\s*([A-Z]+)\)", suffix.upper())
            if m:
                away, home = m.group(1), m.group(2)
        if not away or not home:
            continue

        matchups.append({
            "draft_group_id": int(dg_id),
            "away": away,
            "home": home,
            "label": f"{away} @ {home}",
            "start_time": str(dg.get("StartDate") or dg.get("startDate") or ""),
            "start_time_est": str(dg.get("StartDateEst") or dg.get("startDateEst") or ""),
        })

    log.info("[dk_ingest] Found %d Showdown matchups for %s", len(matchups), sport)
    return matchups


def fetch_dk_showdown_salaries(draft_group_id: int) -> Dict[str, Any]:
    """Fetch Showdown FLEX salaries for a single draft group.

    Returns a dict with:
      players: list of {name, team, position, salary, dk_player_id}
      salary_map: {normalised_name: salary}  (FLEX / base salary only)
      draft_group_id: int

    DK Showdown returns two rows per player: CPT (rosterSlotId 476,
    salary = 1.5×) and FLEX (rosterSlotId 475, base salary).  We keep
    the FLEX salary since the optimizer applies the CPT multiplier.
    """
    try:
        url = _DK_DRAFTABLES_URL.format(draft_group_id=draft_group_id)
        resp = _rate_limited_get(url)
        raw = resp.json().get("draftables") or []
    except Exception as exc:
        log.warning("[dk_ingest] Showdown draftables fetch failed for DG %s: %s", draft_group_id, exc)
        return {"players": [], "salary_map": {}, "draft_group_id": draft_group_id}

    if not raw:
        return {"players": [], "salary_map": {}, "draft_group_id": draft_group_id}

    # Group by playerId — keep the FLEX (lower) salary for each player.
    _FLEX_SLOT = 475  # rosterSlotId for FLEX
    player_data: Dict[str, Dict] = {}  # playerId -> best record
    for p in raw:
        pid = str(p.get("playerId", ""))
        slot_id = p.get("rosterSlotId", 0)
        sal = float(p.get("salary", 0))
        name = str(p.get("displayName", "")).strip()
        team = str(p.get("teamAbbreviation", "")).upper()
        pos = str(p.get("position", "")).strip()

        if not pid or not name:
            continue

        if pid not in player_data:
            player_data[pid] = {
                "name": name, "team": team, "position": pos,
                "salary": sal, "dk_player_id": pid, "slot_id": slot_id,
            }
        else:
            # Prefer FLEX slot (lower salary = base); if both are FLEX keep lower
            existing = player_data[pid]
            if slot_id == _FLEX_SLOT and existing["slot_id"] != _FLEX_SLOT:
                existing.update(salary=sal, slot_id=slot_id)
            elif slot_id == _FLEX_SLOT and sal < existing["salary"]:
                existing["salary"] = sal
            elif existing["slot_id"] != _FLEX_SLOT and sal < existing["salary"]:
                existing.update(salary=sal, slot_id=slot_id)

    # Build normalised salary map for matching
    salary_map: Dict[str, float] = {}
    players: List[Dict] = []
    for info in player_data.values():
        sal = info["salary"]
        name = info["name"]
        team = info["team"]
        players.append({
            "name": name, "team": team, "position": info["position"],
            "salary": sal, "dk_player_id": info["dk_player_id"],
        })
        # Exact name
        salary_map[name] = sal
        # Normalised name
        norm = _normalize_name(name)
        salary_map[norm] = sal
        # Last-name + team fallback key
        parts = norm.split()
        if len(parts) >= 2:
            salary_map[f"_LN_{parts[-1]}_{team}"] = sal

    sals = [info["salary"] for info in player_data.values()]
    log.info(
        "[dk_ingest] Showdown DG %s: %d players (FLEX), salary range $%.0f-$%.0f",
        draft_group_id, len(player_data),
        min(sals) if sals else 0, max(sals) if sals else 0,
    )
    return {
        "players": players,
        "salary_map": salary_map,
        "draft_group_id": draft_group_id,
    }
