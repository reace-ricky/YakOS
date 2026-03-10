"""yak_core.datagolf -- DataGolf API client for PGA DFS data.

Single source of truth for all DataGolf API interactions.
Every endpoint normalises player names to "First Last" format
(DataGolf returns "Last, First").

Endpoints used:
  - /preds/fantasy-projection-defaults  → DFS projections (salary, proj, std_dev, ownership)
  - /preds/pre-tournament               → Win/cut/top-N probabilities
  - /preds/player-decompositions        → Course-fit & SG adjustments
  - /field-updates                      → Field, WD status, tee times, rankings
  - /betting-tools/outrights            → Model vs books odds
  - /preds/approach-skill               → SG by yardage/lie bucket
  - /preds/skill-ratings                → Overall player skill estimates
  - /historical-dfs-data/event-list     → Event IDs for historical lookup
  - /get-player-list                    → Master player list with IDs

Historical endpoints (/historical-dfs-data/points, /historical-raw-data/rounds,
/historical-odds) require a premium subscription.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


_BASE_URL = "https://feeds.datagolf.com"
_TIMEOUT = 30


def _normalise_name(name: str) -> str:
    """Convert 'Last, First' → 'First Last'.  Pass-through if already normal."""
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        return f"{parts[1]} {parts[0]}"
    return name.strip()


class DataGolfClient:
    """Stateless client for the DataGolf REST API."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    # ── internal helpers ────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET request with key injection.  Returns parsed JSON."""
        params = dict(params or {})
        params["key"] = self.api_key
        url = f"{_BASE_URL}{path}"
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _to_df(records: List[Dict], name_col: str = "player_name") -> pd.DataFrame:
        """Convert list of dicts to DataFrame, normalising player names."""
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        if name_col in df.columns:
            df[name_col] = df[name_col].apply(_normalise_name)
        return df

    # ── DFS Projections ─────────────────────────────────────────────

    def get_dfs_projections(
        self,
        site: str = "draftkings",
        slate: str = "main",
    ) -> pd.DataFrame:
        """Current tournament DFS projections.

        Returns
        -------
        DataFrame with columns:
            player_name, dg_id, salary, proj (total FP), proj_scoring,
            proj_finish, std_dev, proj_own, value, early_late_wave,
            r1_teetime, dk_player_id (parsed from site_name_id)
        """
        data = self._get(
            "/preds/fantasy-projection-defaults",
            {"site": site, "slate": slate},
        )
        df = self._to_df(data.get("projections", []))
        if df.empty:
            return df

        # Rename to YakOS pool convention
        rename = {
            "proj_points_total": "proj",
            "proj_points_scoring": "proj_scoring",
            "proj_points_finish": "proj_finish",
            "proj_ownership": "proj_own",
        }
        df = df.rename(columns=rename)

        # Parse DK player ID from site_name_id: "Scottie Scheffler (42166373)"
        if "site_name_id" in df.columns:
            df["dk_player_id"] = df["site_name_id"].apply(_extract_dk_id)
            df = df.drop(columns=["site_name_id"])

        # Store event metadata
        df.attrs["event_name"] = data.get("event_name", "")
        df.attrs["last_updated"] = data.get("last_updated", "")
        df.attrs["tour"] = data.get("tour", "pga")

        return df

    # ── Pre-Tournament Predictions ──────────────────────────────────

    def get_pre_tournament_preds(
        self,
        model: str = "baseline",
    ) -> pd.DataFrame:
        """Win / top-5 / top-10 / top-20 / make-cut probabilities.

        These are the primary edge signals for PGA — analogous to
        DVP/matchup in NBA.
        """
        data = self._get("/preds/pre-tournament")
        records = data.get(model, data.get("baseline", []))
        return self._to_df(records)

    # ── Skill Decompositions (Course Fit) ───────────────────────────

    def get_decompositions(self) -> pd.DataFrame:
        """Per-player course-fit & SG adjustment breakdown.

        Key columns for DFS edge:
          - total_fit_adjustment   → overall course fit (+/- strokes)
          - driving_accuracy_adjustment
          - driving_distance_adjustment
          - cf_approach_comp       → approach composition fit
          - cf_short_comp          → short-game composition fit
          - course_history_adjustment
          - course_experience_adjustment
          - baseline_pred          → baseline SG prediction
          - final_pred             → adjusted SG prediction (baseline + all adjustments)
        """
        data = self._get("/preds/player-decompositions")
        df = self._to_df(data.get("players", []))
        df.attrs["course_name"] = data.get("course_name", "")
        df.attrs["event_name"] = data.get("event_name", "")
        return df

    # ── Field Updates ───────────────────────────────────────────────

    def get_field(self) -> pd.DataFrame:
        """Current tournament field with WD status, rankings, tee times."""
        data = self._get("/field-updates")
        df = self._to_df(data.get("field", []))
        df.attrs["event_name"] = data.get("event_name", "")
        return df

    # ── Betting Odds ────────────────────────────────────────────────

    def get_outright_odds(self, market: str = "win") -> pd.DataFrame:
        """Model-implied vs sportsbook odds.

        The 'datagolf' column is the model's implied probability.
        Other columns are book decimal odds.

        market : 'win', 'top_5', 'top_10', 'top_20', 'make_cut'
        """
        data = self._get("/betting-tools/outrights", {"market": market})
        df = self._to_df(data.get("odds", []))
        df.attrs["books_offering"] = data.get("books_offering", [])
        return df

    # ── Approach Skill ──────────────────────────────────────────────

    def get_approach_skill(self, period: str = "l24") -> pd.DataFrame:
        """Detailed approach stats by yardage/lie bucket.

        period : 'l24' (24 months), 'l12' (12 months), 'ytd'
        """
        data = self._get("/preds/approach-skill", {"period": period})
        # Response shape varies — could be list or dict with players key
        if isinstance(data, list):
            return self._to_df(data)
        return self._to_df(data.get("players", data.get("data", [])))

    # ── Skill Ratings ───────────────────────────────────────────────

    def get_skill_ratings(self) -> pd.DataFrame:
        """Overall player skill estimates and rankings."""
        data = self._get("/preds/skill-ratings")
        if isinstance(data, list):
            return self._to_df(data)
        return self._to_df(data.get("players", data.get("rankings", [])))

    # ── Player List ─────────────────────────────────────────────────

    def get_player_list(self) -> pd.DataFrame:
        """Master player list with IDs, country, amateur status."""
        data = self._get("/get-player-list")
        if isinstance(data, list):
            return self._to_df(data)
        return self._to_df(data.get("players", []))

    # ── Historical DFS (requires premium) ───────────────────────────

    def get_dfs_event_list(self) -> pd.DataFrame:
        """List of events with IDs.  Does NOT require premium."""
        data = self._get("/historical-dfs-data/event-list")
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()

    def get_historical_dfs_points(
        self,
        event_id: int,
        year: int,
        tour: str = "pga",
        site: str = "draftkings",
    ) -> pd.DataFrame:
        """Historical DFS salaries, ownership, actual FP.

        Requires premium subscription.  Returns empty DataFrame with
        warning if access denied.

        Parameters
        ----------
        event_id : int
            DataGolf event ID.
        year : int
            Calendar year of the event.
        tour : str
            Tour ("pga", "euro", "kft", "alt").
        site : str
            DFS site ("draftkings", "fanduel").
        """
        try:
            data = self._get(
                "/historical-dfs-data/points",
                {"site": site, "event_id": event_id, "year": year, "tour": tour},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                print("[datagolf] Historical DFS data requires premium subscription")
                return pd.DataFrame()
            raise
        records = data.get("dfs_points", data.get("players", []))
        return self._to_df(records)

    def get_historical_pre_tournament(
        self,
        event_id: int,
        year: int,
        tour: str = "pga",
    ) -> pd.DataFrame:
        """Historical pre-tournament predictions (win/cut/top-N probs).

        Parameters
        ----------
        event_id : int
            DataGolf event ID.
        year : int
            Calendar year of the event.
        tour : str
            Tour ("pga", "euro", "kft", "alt").

        Returns
        -------
        DataFrame with columns including:
            dg_id, player_name, win_prob, top_5, top_10, top_20, make_cut,
            baseline_pred (predicted SG), etc.
        """
        try:
            data = self._get(
                "/preds/pre-tournament",
                {"event_id": event_id, "year": year, "tour": tour},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (403, 404):
                print(f"[datagolf] Historical pre-tournament data unavailable for event {event_id}/{year}")
                return pd.DataFrame()
            raise
        # Response has a "baseline" or "baseline_history" key with the player list
        records = data.get("baseline_history", data.get("baseline", []))
        if not records:
            # Try top-level if it's a flat list
            if isinstance(data, list):
                records = data
        return self._to_df(records)


# ── Helpers ─────────────────────────────────────────────────────────

def _extract_dk_id(site_name_id: str) -> str:
    """Parse DK player ID from 'Scottie Scheffler (42166373)' format."""
    if not isinstance(site_name_id, str):
        return ""
    m = re.search(r"\((\d+)\)", site_name_id)
    return m.group(1) if m else ""
