"""yak_core.injury_monitor -- Stateful injury monitoring for YakOS.

Tracks player injury status across polls to detect *changes* that matter
for DFS:

  1. **GTD → OUT detection** — player trending toward sitting.
     Tank01 getNBAInjuryList often lags; DK draftables status can flip first.
     The monitor cross-references both sources and flags GTD players likely OUT.

  2. **Late-scratch alerts** — status flip to OUT within the scratch window
     (configurable, default 90 minutes before lock).

  3. **Return watch** — injured player returns → their beneficiaries' inflated
     minutes should deflate.  Feeds injury_cascade to reverse prior bumps.

  4. **Auto-poll** — called on slate load and periodically (default 5 min in
     the last 2 hours before lock, 15 min earlier).  No manual button needed.

Usage
-----
The module stores state in ``InjuryMonitorState`` (dict-backed, serialisable
to session state or disk).  Typical flow:

    state = InjuryMonitorState.load(slate_date)
    alerts = poll_injuries(state, pool_df, cfg)
    # alerts is a list of InjuryAlert dicts
    state.save()

Architecture
-----------
    Tank01 getNBAInjuryList  ──┐
                               ├──► merge_injury_sources() ──► diff vs prior ──► classify
    DK draftables status     ──┘

Backtest context
----------------
From 30-day spike analysis (455 spikes):
  - 67% of minute spikes are teammate_out
  - Late scratches create the biggest DFS edges (no salary adjustment)
  - Return-from-injury causes the most common false positive in cascade
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import YAKOS_ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How close to lock (minutes) counts as "late scratch" window
LATE_SCRATCH_WINDOW_MINUTES: int = 90

# Auto-poll intervals (minutes)
POLL_INTERVAL_FAR: int = 15   # > 2 hours before lock
POLL_INTERVAL_NEAR: int = 5   # ≤ 2 hours before lock

# Confidence thresholds for GTD → OUT trending
GTD_OUT_CONFIDENCE_HIGH: float = 0.85    # very likely OUT
GTD_OUT_CONFIDENCE_MEDIUM: float = 0.65  # probably OUT

# Status hierarchy (higher = more severe)
_STATUS_SEVERITY = {
    "Active": 0,
    "Probable": 1,
    "Questionable": 2,
    "GTD": 3,
    "Doubtful": 4,
    "OUT": 5,
    "IR": 6,
    "Suspended": 7,
}

# Statuses that mean a player is definitively not playing
_DEFINITE_OUT = {"OUT", "IR", "Suspended"}

# Statuses that mean a player might not play (watch list)
_UNCERTAIN = {"GTD", "Questionable", "Doubtful"}

# Canonical status mapping (normalise everything Tank01/DK might send)
_NORMALISE_STATUS = {
    "ACTIVE": "Active",
    "": "Active",
    "HEALTHY": "Active",
    "OUT": "OUT",
    "O": "OUT",
    "IR": "IR",
    "INJURED RESERVE": "IR",
    "INJ": "IR",
    "DND": "OUT",
    "SUSPENDED": "Suspended",
    "SUSP": "Suspended",
    "QUESTIONABLE": "Questionable",
    "Q": "Questionable",
    "GTD": "GTD",
    "GAME TIME DECISION": "GTD",
    "GAME-TIME DECISION": "GTD",
    "DAY-TO-DAY": "GTD",
    "DAY TO DAY": "GTD",
    "PROBABLE": "Probable",
    "P": "Probable",
    "DOUBTFUL": "Doubtful",
    "D": "Doubtful",
}


def normalise_status(raw: str) -> str:
    """Normalise a raw status string to a canonical label."""
    upper = str(raw).strip().upper()
    return _NORMALISE_STATUS.get(upper, raw.strip() if raw.strip() else "Active")


# ---------------------------------------------------------------------------
# Alert types
# ---------------------------------------------------------------------------

class AlertType:
    """Alert classification constants."""
    GTD_TRENDING_OUT = "gtd_trending_out"
    CONFIRMED_OUT = "confirmed_out"
    LATE_SCRATCH = "late_scratch"
    RETURN_WATCH = "return_watch"
    STATUS_UPGRADE = "status_upgrade"
    NEW_INJURY = "new_injury"


@dataclass
class InjuryAlert:
    """A single injury status change alert."""
    player_name: str
    team: str
    alert_type: str
    old_status: str
    new_status: str
    confidence: float  # 0.0–1.0 confidence this is actionable
    timestamp: str     # ISO timestamp of detection
    detail: str        # human-readable explanation
    is_late_scratch: bool = False
    gtd_out_probability: float = 0.0  # for GTD players, P(sitting)
    source: str = ""   # "tank01", "dk", "both"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_name": self.player_name,
            "team": self.team,
            "alert_type": self.alert_type,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "detail": self.detail,
            "is_late_scratch": self.is_late_scratch,
            "gtd_out_probability": self.gtd_out_probability,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Monitor state (persists across polls)
# ---------------------------------------------------------------------------

class InjuryMonitorState:
    """Tracks injury statuses across polls to detect changes.

    Stores:
    - player_statuses: {player_name: {status, team, first_seen, last_seen, source, ...}}
    - alert_history: list of alerts generated
    - last_poll_ts: unix timestamp of last poll
    - return_watch_players: {player_name: {was_out_since, returned_at, team, beneficiaries}}
    """

    def __init__(self, slate_date: str = ""):
        self.slate_date = slate_date
        self.player_statuses: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict] = []
        self.return_watch_players: Dict[str, Dict[str, Any]] = {}
        self.last_poll_ts: float = 0.0
        self._lock_time: Optional[datetime] = None

    @property
    def lock_time(self) -> Optional[datetime]:
        return self._lock_time

    @lock_time.setter
    def lock_time(self, val: Optional[datetime]):
        self._lock_time = val

    def minutes_to_lock(self) -> Optional[float]:
        """Minutes remaining until slate lock. None if lock_time not set."""
        if self._lock_time is None:
            return None
        delta = self._lock_time - datetime.utcnow()
        return max(delta.total_seconds() / 60.0, 0.0)

    def is_late_scratch_window(self) -> bool:
        """Are we within the late-scratch window?"""
        mtl = self.minutes_to_lock()
        if mtl is None:
            return False
        return mtl <= LATE_SCRATCH_WINDOW_MINUTES

    def should_poll(self) -> bool:
        """Should we poll for new data based on time since last poll?"""
        if self.last_poll_ts == 0:
            return True
        elapsed_min = (time.time() - self.last_poll_ts) / 60.0
        mtl = self.minutes_to_lock()
        if mtl is not None and mtl <= 120:
            return elapsed_min >= POLL_INTERVAL_NEAR
        return elapsed_min >= POLL_INTERVAL_FAR

    def poll_interval_seconds(self) -> int:
        """Current poll interval in seconds for UI auto-refresh."""
        mtl = self.minutes_to_lock()
        if mtl is not None and mtl <= 120:
            return POLL_INTERVAL_NEAR * 60
        return POLL_INTERVAL_FAR * 60

    def _state_path(self) -> str:
        """Path to persist state on disk."""
        d = os.path.join(YAKOS_ROOT, "data", "injury_monitor")
        os.makedirs(d, exist_ok=True)
        safe_date = self.slate_date.replace("-", "")
        return os.path.join(d, f"monitor_{safe_date}.json")

    def save(self) -> None:
        """Persist state to disk."""
        data = {
            "slate_date": self.slate_date,
            "player_statuses": self.player_statuses,
            "alert_history": self.alert_history,
            "return_watch_players": self.return_watch_players,
            "last_poll_ts": self.last_poll_ts,
            "lock_time": self._lock_time.isoformat() if self._lock_time else None,
        }
        path = self._state_path()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, slate_date: str) -> "InjuryMonitorState":
        """Load state from disk, or create fresh if not found."""
        state = cls(slate_date)
        path = state._state_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                state.player_statuses = data.get("player_statuses", {})
                state.alert_history = data.get("alert_history", [])
                state.return_watch_players = data.get("return_watch_players", {})
                state.last_poll_ts = data.get("last_poll_ts", 0.0)
                lt = data.get("lock_time")
                if lt:
                    state._lock_time = datetime.fromisoformat(lt)
            except Exception:
                pass
        return state

    def clear(self) -> None:
        """Reset all state."""
        self.player_statuses = {}
        self.alert_history = []
        self.return_watch_players = {}
        self.last_poll_ts = 0.0


# ---------------------------------------------------------------------------
# Core: merge injury sources
# ---------------------------------------------------------------------------

def merge_injury_sources(
    tank01_injuries: List[Dict],
    dk_pool_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, Any]]:
    """Merge Tank01 injury list + DK draftables status into a unified view.

    Returns {player_name: {status, team, source, designation, description}}.

    When both sources report on the same player, the more severe status wins.
    This catches cases where DK flips status before Tank01 updates.
    """
    merged: Dict[str, Dict[str, Any]] = {}

    # Tank01 injury list
    for entry in tank01_injuries:
        if not isinstance(entry, dict):
            continue
        name = str(
            entry.get("playerName")
            or entry.get("longName")
            or entry.get("player")
            or ""
        ).strip()
        if not name:
            continue

        raw_status = str(entry.get("injuryStatus", entry.get("designation", entry.get("status", "")))).strip()
        status = normalise_status(raw_status)
        team = str(entry.get("team", entry.get("teamAbv", ""))).strip().upper()
        desc = str(entry.get("description", "")).strip()
        desig = str(entry.get("designation", "")).strip()

        merged[name] = {
            "status": status,
            "team": team,
            "source": "tank01",
            "designation": desig,
            "description": desc[:120],
        }

    # DK draftables status (often more current for game-day)
    if dk_pool_df is not None and not dk_pool_df.empty:
        name_col = "player_name" if "player_name" in dk_pool_df.columns else "name"
        if name_col in dk_pool_df.columns and "status" in dk_pool_df.columns:
            for _, row in dk_pool_df.iterrows():
                name = str(row.get(name_col, "")).strip()
                if not name:
                    continue
                dk_status = normalise_status(str(row.get("status", "")))
                team = str(row.get("team", "")).strip().upper()

                if name in merged:
                    # Take the more severe status
                    existing_sev = _STATUS_SEVERITY.get(merged[name]["status"], 0)
                    dk_sev = _STATUS_SEVERITY.get(dk_status, 0)
                    if dk_sev > existing_sev:
                        merged[name]["status"] = dk_status
                        merged[name]["source"] = "both"
                    elif dk_sev == existing_sev:
                        merged[name]["source"] = "both"
                elif dk_status != "Active":
                    merged[name] = {
                        "status": dk_status,
                        "team": team,
                        "source": "dk",
                        "designation": "",
                        "description": "",
                    }

    return merged


# ---------------------------------------------------------------------------
# Core: diff and classify changes
# ---------------------------------------------------------------------------

def _gtd_out_probability(
    current_status: str,
    designation: str = "",
    description: str = "",
    minutes_to_lock: Optional[float] = None,
) -> float:
    """Estimate the probability a GTD/Questionable player sits.

    Heuristic based on DFS injury patterns:
    - "Doubtful" ≈ 85% chance of sitting
    - "GTD" with no practice / "did not participate" ≈ 75%
    - "GTD" generic ≈ 50%
    - "Questionable" ≈ 35%
    - Closer to lock without upgrade → higher probability
    """
    base = 0.0

    # Status-based base rate
    if current_status == "Doubtful":
        base = 0.85
    elif current_status == "GTD":
        base = 0.50
    elif current_status == "Questionable":
        base = 0.35
    elif current_status in _DEFINITE_OUT:
        return 1.0
    elif current_status in ("Active", "Probable"):
        return 0.0

    # Description keywords that increase likelihood
    desc_lower = (description + " " + designation).lower()
    if any(kw in desc_lower for kw in ("did not participate", "dnp", "not practice")):
        base = min(base + 0.20, 0.95)
    if any(kw in desc_lower for kw in ("limited", "partial")):
        base = max(base - 0.05, 0.10)
    if any(kw in desc_lower for kw in ("full practice", "full participant", "upgraded")):
        base = max(base - 0.25, 0.05)
    if any(kw in desc_lower for kw in ("ruled out", "will not play", "sidelined")):
        base = 0.98

    # Time-decay: closer to lock without upgrade → more likely out
    if minutes_to_lock is not None and minutes_to_lock <= LATE_SCRATCH_WINDOW_MINUTES:
        # Within 90 min of lock and still GTD → bump probability
        time_factor = 1.0 - (minutes_to_lock / LATE_SCRATCH_WINDOW_MINUTES)
        base = min(base + time_factor * 0.15, 0.95)

    return round(base, 2)


def diff_and_classify(
    state: InjuryMonitorState,
    current_statuses: Dict[str, Dict[str, Any]],
    pool_player_names: Optional[set] = None,
) -> List[InjuryAlert]:
    """Compare current statuses against prior state and generate alerts.

    Only generates alerts for players relevant to the pool (if provided).

    Parameters
    ----------
    state : InjuryMonitorState
        Prior state with historical statuses.
    current_statuses : dict
        Output from merge_injury_sources().
    pool_player_names : set, optional
        Only alert on players in this set + their teammates. If None, alert all.

    Returns
    -------
    List[InjuryAlert]
    """
    now_iso = datetime.utcnow().isoformat()
    mtl = state.minutes_to_lock()
    is_late = state.is_late_scratch_window()
    alerts: List[InjuryAlert] = []

    for name, info in current_statuses.items():
        new_status = info["status"]
        team = info.get("team", "")
        source = info.get("source", "")
        desig = info.get("designation", "")
        desc = info.get("description", "")

        # Check if player is relevant (in pool or teammate of pool player)
        if pool_player_names is not None and name not in pool_player_names:
            # Still track in state, but skip alert unless it's a team-level impact
            pass

        prior = state.player_statuses.get(name, {})
        old_status = prior.get("status", "Active")

        old_sev = _STATUS_SEVERITY.get(old_status, 0)
        new_sev = _STATUS_SEVERITY.get(new_status, 0)

        if old_status == new_status:
            # No change — but if GTD, update probability
            if new_status in _UNCERTAIN:
                prob = _gtd_out_probability(new_status, desig, desc, mtl)
                state.player_statuses.setdefault(name, {})["gtd_out_prob"] = prob
            continue

        # Status changed — classify it
        alert = None

        # CASE 1: Confirmed OUT (Active/GTD/Q → OUT/IR)
        if new_status in _DEFINITE_OUT and old_status not in _DEFINITE_OUT:
            at = AlertType.LATE_SCRATCH if is_late else AlertType.CONFIRMED_OUT
            detail = f"{name} ({team}) confirmed {new_status}"
            if desc:
                detail += f" — {desc}"
            if is_late:
                detail = f"LATE SCRATCH: {detail}"

            alert = InjuryAlert(
                player_name=name,
                team=team,
                alert_type=at,
                old_status=old_status,
                new_status=new_status,
                confidence=0.99,
                timestamp=now_iso,
                detail=detail,
                is_late_scratch=is_late,
                gtd_out_probability=1.0,
                source=source,
            )

        # CASE 2: GTD trending OUT (Active/Probable → GTD/Q/Doubtful)
        elif new_status in _UNCERTAIN and old_sev < new_sev:
            prob = _gtd_out_probability(new_status, desig, desc, mtl)
            conf = prob  # confidence = probability of sitting
            detail = f"{name} ({team}) {old_status} → {new_status}"
            if prob >= GTD_OUT_CONFIDENCE_HIGH:
                detail += f" — very likely OUT ({prob:.0%})"
            elif prob >= GTD_OUT_CONFIDENCE_MEDIUM:
                detail += f" — probably OUT ({prob:.0%})"
            else:
                detail += f" — uncertain ({prob:.0%})"
            if desc:
                detail += f" | {desc}"

            alert = InjuryAlert(
                player_name=name,
                team=team,
                alert_type=AlertType.GTD_TRENDING_OUT,
                old_status=old_status,
                new_status=new_status,
                confidence=conf,
                timestamp=now_iso,
                detail=detail,
                is_late_scratch=False,
                gtd_out_probability=prob,
                source=source,
            )

        # CASE 3: Return watch (OUT/IR → Active/Probable/GTD)
        elif old_status in _DEFINITE_OUT and new_status not in _DEFINITE_OUT:
            detail = f"RETURN WATCH: {name} ({team}) {old_status} → {new_status}"
            if desc:
                detail += f" — {desc}"
            detail += " | Beneficiary minutes may deflate"

            alert = InjuryAlert(
                player_name=name,
                team=team,
                alert_type=AlertType.RETURN_WATCH,
                old_status=old_status,
                new_status=new_status,
                confidence=0.80 if new_status == "Active" else 0.50,
                timestamp=now_iso,
                detail=detail,
                source=source,
            )

            # Track in return_watch for cascade reversal
            state.return_watch_players[name] = {
                "was_out_since": prior.get("first_seen", now_iso),
                "returned_at": now_iso,
                "team": team,
                "new_status": new_status,
            }

        # CASE 4: Status upgrade (Q/GTD → Probable/Active)
        elif new_sev < old_sev and old_status in _UNCERTAIN:
            detail = f"{name} ({team}) upgraded {old_status} → {new_status}"
            if desc:
                detail += f" — {desc}"

            alert = InjuryAlert(
                player_name=name,
                team=team,
                alert_type=AlertType.STATUS_UPGRADE,
                old_status=old_status,
                new_status=new_status,
                confidence=0.85,
                timestamp=now_iso,
                detail=detail,
                source=source,
            )

        # CASE 5: New injury (not previously tracked, appears as non-Active)
        elif name not in state.player_statuses and new_status != "Active":
            prob = _gtd_out_probability(new_status, desig, desc, mtl) if new_status in _UNCERTAIN else (
                1.0 if new_status in _DEFINITE_OUT else 0.0
            )
            at = AlertType.LATE_SCRATCH if (is_late and new_status in _DEFINITE_OUT) else AlertType.NEW_INJURY
            detail = f"NEW: {name} ({team}) listed as {new_status}"
            if desc:
                detail += f" — {desc}"
            if is_late and new_status in _DEFINITE_OUT:
                detail = f"LATE SCRATCH: {detail}"

            alert = InjuryAlert(
                player_name=name,
                team=team,
                alert_type=at,
                old_status="Active",
                new_status=new_status,
                confidence=0.95 if new_status in _DEFINITE_OUT else prob,
                timestamp=now_iso,
                detail=detail,
                is_late_scratch=is_late and new_status in _DEFINITE_OUT,
                gtd_out_probability=prob,
                source=source,
            )

        # Update state
        state.player_statuses[name] = {
            "status": new_status,
            "team": team,
            "source": source,
            "designation": desig,
            "description": desc[:120],
            "first_seen": prior.get("first_seen", now_iso),
            "last_seen": now_iso,
            "gtd_out_prob": alert.gtd_out_probability if alert else 0.0,
        }

        if alert:
            # Filter: only alert on pool-relevant players
            if pool_player_names is None or name in pool_player_names:
                alerts.append(alert)
            # Even if not directly in pool, OUT alerts matter for cascade
            elif alert.alert_type in (
                AlertType.CONFIRMED_OUT, AlertType.LATE_SCRATCH, AlertType.RETURN_WATCH
            ):
                alerts.append(alert)

    return alerts


# ---------------------------------------------------------------------------
# High-level: poll_injuries (the main entry point)
# ---------------------------------------------------------------------------

def poll_injuries(
    state: InjuryMonitorState,
    pool_df: pd.DataFrame,
    cfg: dict,
    dk_pool_df: Optional[pd.DataFrame] = None,
    force: bool = False,
) -> List[InjuryAlert]:
    """Poll Tank01 + DK for injury updates, diff against prior state, classify.

    This is the main entry point. Call from the Lab page on load and on interval.

    Parameters
    ----------
    state : InjuryMonitorState
        Persistent monitor state for this slate.
    pool_df : pd.DataFrame
        Current player pool (used to scope alerts to relevant players).
    cfg : dict
        Must contain RAPIDAPI_KEY for Tank01.
    dk_pool_df : pd.DataFrame, optional
        DK draftables with status column. If None, only Tank01 is used.
    force : bool
        Force a poll even if interval hasn't elapsed.

    Returns
    -------
    List[InjuryAlert]
        New alerts since last poll.
    """
    if not force and not state.should_poll():
        return []

    # Fetch Tank01 injury list
    from .live import fetch_injury_updates, _get_rapidapi_key
    try:
        api_key = _get_rapidapi_key(cfg)
        date_key = state.slate_date.replace("-", "")
        tank01_injuries = fetch_injury_updates(date_key, cfg)
    except Exception as exc:
        print(f"[injury_monitor] Tank01 fetch failed: {exc}")
        tank01_injuries = []

    # Merge sources
    current = merge_injury_sources(tank01_injuries, dk_pool_df)

    # Get pool player names for scoping
    pool_names = set()
    if pool_df is not None and not pool_df.empty and "player_name" in pool_df.columns:
        pool_names = set(pool_df["player_name"].dropna().astype(str).values)

    # Diff and classify
    alerts = diff_and_classify(state, current, pool_names if pool_names else None)

    # Update poll timestamp
    state.last_poll_ts = time.time()

    # Append alerts to history
    for a in alerts:
        state.alert_history.append(a.to_dict())

    # Save state
    state.save()

    return alerts


# ---------------------------------------------------------------------------
# Helpers for cascade integration
# ---------------------------------------------------------------------------

def get_confirmed_outs(state: InjuryMonitorState) -> List[Dict[str, str]]:
    """Return list of {player_name, team, status} for confirmed OUT players."""
    outs = []
    for name, info in state.player_statuses.items():
        if info.get("status") in _DEFINITE_OUT:
            outs.append({
                "player_name": name,
                "team": info.get("team", ""),
                "status": info["status"],
            })
    return outs


def get_high_prob_outs(
    state: InjuryMonitorState,
    threshold: float = GTD_OUT_CONFIDENCE_MEDIUM,
) -> List[Dict[str, Any]]:
    """Return GTD/Q players with P(out) >= threshold.

    These should be treated as likely-OUT for cascade pre-computation,
    with a discount factor = gtd_out_prob.
    """
    candidates = []
    for name, info in state.player_statuses.items():
        status = info.get("status", "Active")
        prob = info.get("gtd_out_prob", 0.0)
        if status in _UNCERTAIN and prob >= threshold:
            candidates.append({
                "player_name": name,
                "team": info.get("team", ""),
                "status": status,
                "gtd_out_prob": prob,
            })
    return sorted(candidates, key=lambda x: x["gtd_out_prob"], reverse=True)


def get_return_watch_players(state: InjuryMonitorState) -> List[Dict[str, Any]]:
    """Return players who were OUT but have returned (minutes should deflate)."""
    return [
        {"player_name": name, **info}
        for name, info in state.return_watch_players.items()
    ]


def apply_monitor_to_pool(
    pool_df: pd.DataFrame,
    state: InjuryMonitorState,
) -> pd.DataFrame:
    """Apply monitor state to pool: update status column, flag GTD probabilities.

    This replaces the manual "Refresh Injuries" flow. Called automatically.

    Adds columns:
    - status: updated from monitor
    - gtd_out_prob: P(sitting) for GTD/Q players (0 for Active/OUT)
    - injury_note: human-readable note from monitor
    - return_watch: True if player was previously a beneficiary of someone
                    who has now returned
    """
    if pool_df is None or pool_df.empty:
        return pool_df

    pool = pool_df.copy()
    if "status" not in pool.columns:
        pool["status"] = "Active"
    if "gtd_out_prob" not in pool.columns:
        pool["gtd_out_prob"] = 0.0
    if "injury_note" not in pool.columns:
        pool["injury_note"] = ""
    if "return_watch" not in pool.columns:
        pool["return_watch"] = False

    name_col = "player_name" if "player_name" in pool.columns else None
    if name_col is None:
        return pool

    for idx, row in pool.iterrows():
        pname = str(row[name_col]).strip()
        if pname in state.player_statuses:
            info = state.player_statuses[pname]
            monitor_status = info.get("status", "Active")
            # Only override if monitor has more severe status
            current_sev = _STATUS_SEVERITY.get(str(row.get("status", "Active")), 0)
            monitor_sev = _STATUS_SEVERITY.get(monitor_status, 0)
            if monitor_sev > current_sev:
                pool.at[idx, "status"] = monitor_status
            pool.at[idx, "gtd_out_prob"] = info.get("gtd_out_prob", 0.0)
            desc = info.get("description", "")
            desig = info.get("designation", "")
            if desc or desig:
                pool.at[idx, "injury_note"] = f"{desig}: {desc}".strip(": ")

    # Mark return-watch players' beneficiaries
    return_teams = {
        info.get("team", ""): name
        for name, info in state.return_watch_players.items()
    }
    if return_teams and "team" in pool.columns:
        for idx, row in pool.iterrows():
            team = str(row.get("team", "")).strip().upper()
            pname = str(row[name_col]).strip()
            if team in return_teams and pname != return_teams[team]:
                # This player is on a team where someone returned from injury
                if row.get("status", "Active") not in _DEFINITE_OUT:
                    pool.at[idx, "return_watch"] = True

    return pool


# ---------------------------------------------------------------------------
# Summary helpers for UI
# ---------------------------------------------------------------------------

def format_alerts_for_ui(alerts: List[InjuryAlert]) -> List[Dict[str, Any]]:
    """Format alerts for Streamlit display.

    Returns list of dicts with: emoji, player, team, detail, severity, type.
    Sorted by severity (late scratch > confirmed out > GTD > return > upgrade).
    """
    _TYPE_PRIORITY = {
        AlertType.LATE_SCRATCH: 0,
        AlertType.CONFIRMED_OUT: 1,
        AlertType.GTD_TRENDING_OUT: 2,
        AlertType.NEW_INJURY: 3,
        AlertType.RETURN_WATCH: 4,
        AlertType.STATUS_UPGRADE: 5,
    }
    _TYPE_EMOJI = {
        AlertType.LATE_SCRATCH: "🚨",
        AlertType.CONFIRMED_OUT: "❌",
        AlertType.GTD_TRENDING_OUT: "⚠️",
        AlertType.NEW_INJURY: "🆕",
        AlertType.RETURN_WATCH: "🔄",
        AlertType.STATUS_UPGRADE: "✅",
    }
    _SEVERITY_LABEL = {
        AlertType.LATE_SCRATCH: "Critical",
        AlertType.CONFIRMED_OUT: "High",
        AlertType.GTD_TRENDING_OUT: "Medium",
        AlertType.NEW_INJURY: "Medium",
        AlertType.RETURN_WATCH: "Info",
        AlertType.STATUS_UPGRADE: "Low",
    }

    formatted = []
    for a in alerts:
        at = a.alert_type
        formatted.append({
            "emoji": _TYPE_EMOJI.get(at, "ℹ️"),
            "player": a.player_name,
            "team": a.team,
            "old_status": a.old_status,
            "new_status": a.new_status,
            "detail": a.detail,
            "severity": _SEVERITY_LABEL.get(at, "Info"),
            "type": at,
            "confidence": a.confidence,
            "is_late_scratch": a.is_late_scratch,
            "gtd_out_prob": a.gtd_out_probability,
            "_priority": _TYPE_PRIORITY.get(at, 99),
        })

    formatted.sort(key=lambda x: (x["_priority"], -x["confidence"]))
    return formatted


def monitor_summary(state: InjuryMonitorState) -> Dict[str, Any]:
    """Quick summary of current monitor state for UI display."""
    total = len(state.player_statuses)
    outs = sum(1 for v in state.player_statuses.values() if v.get("status") in _DEFINITE_OUT)
    gtds = sum(1 for v in state.player_statuses.values() if v.get("status") in _UNCERTAIN)
    high_prob = sum(
        1 for v in state.player_statuses.values()
        if v.get("status") in _UNCERTAIN and v.get("gtd_out_prob", 0) >= GTD_OUT_CONFIDENCE_MEDIUM
    )
    returns = len(state.return_watch_players)
    last_poll = (
        datetime.utcfromtimestamp(state.last_poll_ts).strftime("%H:%M UTC")
        if state.last_poll_ts > 0 else "Never"
    )

    return {
        "total_tracked": total,
        "confirmed_out": outs,
        "gtd_questionable": gtds,
        "likely_out": high_prob,
        "return_watch": returns,
        "last_poll": last_poll,
        "total_alerts": len(state.alert_history),
        "is_late_window": state.is_late_scratch_window(),
    }
