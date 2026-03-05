"""yak_core/salary_history.py
Historical salary lookup via the FantasyLabs and DraftKings APIs.

Public API
----------
SalaryHistoryClient.get_draft_group_ids(date)    → list[dict]
SalaryHistoryClient.get_draftables(draft_group_id) → pd.DataFrame
SalaryHistoryClient.get_historical_salaries(date) → pd.DataFrame
SalaryHistoryClient.load_cached_salaries(date)    → pd.DataFrame | None
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import YAKOS_ROOT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
_SALARY_CACHE_DIR = Path(YAKOS_ROOT) / "data" / "salary_cache"

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
# FantasyLabs NBA (sport 2) / DraftKings (site 4) contest groups
_FL_CONTEST_GROUPS_URL = (
    "https://www.fantasylabs.com/api/ownership-contestgroups/2/4/{date}"
)
# DraftKings draftables endpoint (no auth required)
_DK_DRAFTABLES_URL = (
    "https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group_id}/draftables"
)

# Seconds to sleep between outbound API calls to be a polite client.
_API_SLEEP_SECONDS = 2


def _to_fl_date(date: str) -> str:
    """Convert YYYY-MM-DD to M_D_YYYY (no zero-padding on month/day).

    Examples
    --------
    "2026-02-28" → "2_28_2026"
    "2026-03-05" → "3_5_2026"
    """
    parts = date.split("-")
    if len(parts) != 3:
        raise ValueError(f"Expected YYYY-MM-DD, got: {date!r}")
    yyyy, mm, dd = parts
    return f"{int(mm)}_{int(dd)}_{yyyy}"


class SalaryHistoryClient:
    """Fetches and caches historical DraftKings salaries for a given date.

    The pipeline is:
        1. FantasyLabs API → draft group IDs for the date.
        2. DraftKings draftables API → salary/position/team for each player.
        3. Auto-save to ``data/salary_cache/{YYYY-MM-DD}.parquet``.

    A ``load_cached_salaries`` method is provided to skip API calls on
    subsequent requests for the same date.
    """

    def get_draft_group_ids(self, date: str) -> list[dict]:
        """Return available DraftKings draft groups for *date* (YYYY-MM-DD).

        Calls the FantasyLabs NBA/DK contest groups endpoint and returns a
        list of dicts with keys:
            draft_group_id, game_count, suffix, display_name, start_time

        Returns an empty list when the request fails or no groups are found.
        """
        try:
            fl_date = _to_fl_date(date)
        except ValueError as exc:
            log.warning("get_draft_group_ids: bad date format – %s", exc)
            return []

        url = _FL_CONTEST_GROUPS_URL.format(date=fl_date)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("FantasyLabs contest groups request failed: %s", exc)
            return []

        if not isinstance(data, list):
            log.warning(
                "FantasyLabs contest groups: unexpected response type %s", type(data)
            )
            return []

        groups: list[dict] = []
        for item in data:
            try:
                groups.append(
                    {
                        "draft_group_id": int(item.get("DraftGroupId", 0)),
                        "game_count": int(item.get("GameCount", 0)),
                        "suffix": str(item.get("ContestSuffix", "")),
                        "display_name": str(item.get("DisplayName", "")),
                        "start_time": str(item.get("StartTime", "")),
                    }
                )
            except (TypeError, ValueError) as exc:
                log.debug("Skipping malformed contest group entry: %s – %s", item, exc)

        return groups

    def get_draftables(self, draft_group_id: int) -> pd.DataFrame:
        """Fetch player pool for *draft_group_id* from the DK draftables API.

        Returns a DataFrame with columns:
            player_name, position, team, salary, player_dk_id

        Returns an empty DataFrame when the request fails or no players
        are found.
        """
        url = _DK_DRAFTABLES_URL.format(draft_group_id=draft_group_id)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            log.warning(
                "DK draftables request failed for draft_group_id=%s: %s",
                draft_group_id,
                exc,
            )
            return pd.DataFrame(
                columns=["player_name", "position", "team", "salary", "player_dk_id"]
            )

        draftables = (payload or {}).get("draftables", [])
        if not draftables:
            log.info(
                "No draftables returned for draft_group_id=%s", draft_group_id
            )
            return pd.DataFrame(
                columns=["player_name", "position", "team", "salary", "player_dk_id"]
            )

        rows: list[dict] = []
        for d in draftables:
            try:
                rows.append(
                    {
                        "player_name": str(
                            d.get("displayName") or d.get("playerName") or ""
                        ),
                        "position": str(d.get("position") or ""),
                        "team": str(
                            (d.get("teamAbbreviation") or d.get("team") or "")
                        ).upper(),
                        "salary": int(d.get("salary") or 0),
                        "player_dk_id": int(d.get("playerId") or 0),
                    }
                )
            except (TypeError, ValueError) as exc:
                log.debug("Skipping malformed draftable entry: %s – %s", d, exc)

        if not rows:
            return pd.DataFrame(
                columns=["player_name", "position", "team", "salary", "player_dk_id"]
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Primary selection logic
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_primary_group(groups: list[dict]) -> Optional[dict]:
        """Select the primary (main slate) draft group from a list.

        Preference order:
        1. Groups whose suffix is empty/blank (i.e. the "Main" slate).
        2. If multiple empty-suffix groups exist, pick the one with the
           highest game_count.
        3. If no empty-suffix groups exist (all have "(Night)", "(Early)",
           etc.), fall back to the group with the highest game_count.

        Returns None for an empty input list.
        """
        if not groups:
            return None

        main = [
            g for g in groups
            if g["suffix"].strip() == ""
        ]

        candidates = main if main else groups
        return max(candidates, key=lambda g: g["game_count"])

    def get_historical_salaries(self, date: str) -> pd.DataFrame:
        """Fetch historical salaries for *date* (YYYY-MM-DD) and cache to parquet.

        Steps:
        1. Calls ``get_draft_group_ids`` to discover available groups.
        2. Picks the primary/main group (no suffix, or largest game_count).
        3. Calls ``get_draftables`` (with a polite sleep between calls).
        4. Saves the result to ``data/salary_cache/{date}.parquet``.

        Returns the salary DataFrame (may be empty on failure).
        """
        groups = self.get_draft_group_ids(date)
        if not groups:
            log.info("get_historical_salaries: no draft groups found for %s", date)
            return pd.DataFrame(
                columns=["player_name", "position", "team", "salary", "player_dk_id"]
            )

        primary = self._pick_primary_group(groups)
        if primary is None:
            log.info(
                "get_historical_salaries: could not select a primary group for %s", date
            )
            return pd.DataFrame(
                columns=["player_name", "position", "team", "salary", "player_dk_id"]
            )

        log.info(
            "get_historical_salaries: selected draft_group_id=%s (%s) for %s",
            primary["draft_group_id"],
            primary["display_name"],
            date,
        )

        time.sleep(_API_SLEEP_SECONDS)
        df = self.get_draftables(primary["draft_group_id"])

        if not df.empty:
            self._save_cache(date, df)

        # Attach the selected draft_group_id as metadata so callers can log it.
        df.attrs["draft_group_id"] = primary["draft_group_id"]
        return df

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _save_cache(self, date: str, df: pd.DataFrame) -> None:
        """Persist *df* to ``data/salary_cache/{date}.parquet``."""
        _SALARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _SALARY_CACHE_DIR / f"{date}.parquet"
        try:
            df.to_parquet(path, index=False)
            log.info("Salary cache saved: %s", path)
        except Exception as exc:
            log.warning("Failed to save salary cache for %s: %s", date, exc)

    def load_cached_salaries(self, date: str) -> Optional[pd.DataFrame]:
        """Load cached salaries for *date* (YYYY-MM-DD) from parquet.

        Returns the DataFrame if the cache file exists, otherwise None.
        """
        path = _SALARY_CACHE_DIR / f"{date}.parquet"
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            log.warning("Failed to read salary cache for %s: %s", date, exc)
            return None

    def save_salaries(self, date: str, df: pd.DataFrame) -> None:
        """Public wrapper around ``_save_cache`` (used by Slate Hub live loads)."""
        self._save_cache(date, df)
