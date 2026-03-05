"""YakOS shared state objects.

This module defines the four canonical shared-state dataclasses:
  - SlateState      : DK contest/slate config, player pool, projections
  - RickyEdgeState  : player tags, game tags, stacks, edge labels, notes
  - LineupSetState  : lineups per contest type, build configs, exposures
  - SimState        : sim parameters, results, calibration profiles

All pages read from and write to these objects via ``st.session_state``.
Never create ad-hoc ``st.session_state`` keys for data that belongs here.

Usage
-----
    from yak_core.state import get_slate_state, get_edge_state, get_lineup_state, get_sim_state

    slate = get_slate_state()
    slate.sport = "NBA"
    set_slate_state(slate)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# SlateState
# ---------------------------------------------------------------------------

@dataclass
class SlateState:
    """Holds everything about the current slate selection.

    Fields
    ------
    sport               : "NBA" or "PGA"
    site                : "DK" (only DK supported for now)
    slate_date          : ISO date string e.g. "2026-03-01"
    contest_id          : DK contest ID (int)
    draft_group_id      : DK draft group ID (int)
    game_type_id        : DK game type ID (int)
    contest_name        : human-readable contest name
    contest_type        : "Classic" or "Showdown Captain"
    is_showdown         : True when contest_type == "Showdown Captain"
    roster_slots        : list of position slot strings e.g. ["PG","SG","SF","PF","C","G","F","UTIL"]
    lineup_size         : number of players per lineup
    salary_cap          : integer salary cap
    captain_multiplier  : float (1.5 for Showdown Captain, 1.0 otherwise)
    scoring_rules       : dict of scoring rule overrides
    player_pool         : full player pool DataFrame
    edge_df             : computed edge metrics DataFrame (output of compute_edge_metrics)
    calibration_state   : dict of active calibration adjustments keyed by contest type
    proj_source         : active projection source label
    published           : True once Publish Slate has been confirmed
    published_at        : ISO datetime string of last publish
    active_layers       : list of active layer names e.g. ["Base","Calibration","Edge","Sims"]
    """

    sport: str = "NBA"
    site: str = "DK"
    slate_date: str = ""
    contest_id: Optional[int] = None
    draft_group_id: Optional[int] = None
    game_type_id: Optional[int] = None
    contest_name: str = ""
    contest_type: str = "Classic"
    is_showdown: bool = False
    roster_slots: List[str] = field(default_factory=lambda: ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"])
    lineup_size: int = 8
    salary_cap: int = 50000
    captain_multiplier: float = 1.0
    scoring_rules: Dict[str, Any] = field(default_factory=dict)
    player_pool: Optional[pd.DataFrame] = None
    edge_df: Optional[pd.DataFrame] = None
    calibration_state: Dict[str, Any] = field(default_factory=dict)
    proj_source: str = "salary_implied"
    published: bool = False
    published_at: str = ""
    active_layers: List[str] = field(default_factory=lambda: ["Base"])

    def is_ready(self) -> bool:
        """Return True when the slate is fully configured and published."""
        return (
            self.published
            and self.player_pool is not None
            and not self.player_pool.empty
        )

    def apply_roster_rules(self, rules: Dict[str, Any]) -> None:
        """Apply parsed DK roster rules dict into this state object."""
        self.roster_slots = rules.get("slots", self.roster_slots)
        self.lineup_size = rules.get("lineup_size", self.lineup_size)
        self.salary_cap = rules.get("salary_cap", self.salary_cap)
        self.is_showdown = rules.get("is_showdown", self.is_showdown)
        self.contest_type = "Showdown Captain" if self.is_showdown else "Classic"
        self.captain_multiplier = 1.5 if self.is_showdown else 1.0


# ---------------------------------------------------------------------------
# RickyEdgeState
# ---------------------------------------------------------------------------

@dataclass
class RickyEdgeState:
    """Holds Ricky's manual tags and edge analysis for the current slate.

    Fields
    ------
    player_tags         : {player_name: {"tag": str, "conviction": int}}
                          tag is one of "core"/"secondary"/"value"/"punt"/"fade"
                          conviction is 1-5
    game_tags           : {game_key: {"pace": str, "total": float, "environment": str}}
    stacks              : list of {team, players, rationale}
    edge_labels         : list of auto-generated edge label strings
    slate_notes         : free-text notes for the slate
    ricky_edge_check    : True when Ricky has approved this slate for publishing
    edge_check_ts       : ISO datetime of last edge check approval
    approved_not_with_pairs : list of approved minute-cannibal "not together" pairs,
                              each dict has keys player_a, player_b, team, position_group
    """

    player_tags: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    game_tags: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stacks: List[Dict[str, Any]] = field(default_factory=list)
    edge_labels: List[str] = field(default_factory=list)
    slate_notes: str = ""
    ricky_edge_check: bool = False
    edge_check_ts: str = ""
    approved_not_with_pairs: List[Dict[str, str]] = field(default_factory=list)

    def tag_player(self, player_name: str, tag: str, conviction: int = 3) -> None:
        """Set tag and conviction for a player."""
        self.player_tags[player_name] = {"tag": tag, "conviction": max(1, min(5, conviction))}

    def remove_tag(self, player_name: str) -> None:
        """Remove tag for a player."""
        self.player_tags.pop(player_name, None)

    def get_tagged(self, tag: str) -> List[str]:
        """Return list of player names with the given tag."""
        return [p for p, v in self.player_tags.items() if v.get("tag") == tag]

    def add_stack(self, team: str, players: List[str], rationale: str = "") -> None:
        """Add a new stack definition."""
        self.stacks.append({"team": team, "players": players, "rationale": rationale})

    def approve_edge_check(self, ts: str) -> None:
        """Mark the Ricky Edge Check as approved at the given ISO timestamp."""
        self.ricky_edge_check = True
        self.edge_check_ts = ts

    def revoke_edge_check(self) -> None:
        """Revoke the Ricky Edge Check approval."""
        self.ricky_edge_check = False
        self.edge_check_ts = ""


# ---------------------------------------------------------------------------
# LineupSetState
# ---------------------------------------------------------------------------

@dataclass
class LineupSetState:
    """Holds built lineups and configs per contest type.

    Fields
    ------
    lineups             : {contest_label: pd.DataFrame of lineups}
    build_configs       : {contest_label: dict of build config}
    exposures           : {contest_label: pd.DataFrame of per-player exposure}
    published_sets      : {contest_label: {"lineups_df": ..., "published_at": str}}
    snapshot_times      : {contest_label: ISO datetime string}
    """

    lineups: Dict[str, Optional[pd.DataFrame]] = field(default_factory=dict)
    build_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    exposures: Dict[str, Optional[pd.DataFrame]] = field(default_factory=dict)
    published_sets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    snapshot_times: Dict[str, str] = field(default_factory=dict)

    def set_lineups(self, contest_label: str, lineups_df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """Store built lineups and config for a contest type."""
        self.lineups[contest_label] = lineups_df
        self.build_configs[contest_label] = config

    def publish(self, contest_label: str, ts: str) -> None:
        """Publish lineups for a contest type to Edge Share."""
        df = self.lineups.get(contest_label)
        if df is not None:
            self.published_sets[contest_label] = {
                "lineups_df": df.copy(),
                "config": self.build_configs.get(contest_label, {}),
                "published_at": ts,
            }
            self.snapshot_times[contest_label] = ts

    def get_published_labels(self) -> List[str]:
        """Return list of contest labels that have published lineups."""
        return list(self.published_sets.keys())


# ---------------------------------------------------------------------------
# SimState
# ---------------------------------------------------------------------------

@dataclass
class SimState:
    """Holds simulation parameters, results, and calibration data.

    Fields
    ------
    sim_mode            : "Live" or "Historical"
    draft_group_id      : DK draft group ID used in last sim
    variance            : sim variance multiplier (0.5 – 2.0)
    n_sims              : number of Monte Carlo iterations
    sim_results         : {contest_label: pd.DataFrame of lineup-level sim results}
    player_results      : pd.DataFrame of player-level sim metrics (smash/bust/leverage)
    sim_learnings       : {player_name: {"boost": float, "reason": str}} – non-destructive layer
    calibration_profiles: {profile_name: dict} – versioned calibration profiles
    active_profile      : name of active calibration profile (or None)
    contest_gauges      : {contest_label: {"score": float, "label": str}}
    """

    sim_mode: str = "Live"
    draft_group_id: Optional[int] = None
    variance: float = 1.0
    n_sims: int = 10000
    sim_results: Dict[str, Optional[pd.DataFrame]] = field(default_factory=dict)
    player_results: Optional[pd.DataFrame] = None
    sim_learnings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    calibration_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_profile: Optional[str] = None
    contest_gauges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pipeline_output: Dict[str, Optional[pd.DataFrame]] = field(default_factory=dict)

    def apply_learning(self, player_name: str, boost: float, reason: str = "") -> None:
        """Write a sim learning boost for a player (capped at ±15%)."""
        capped = max(-0.15, min(0.15, boost))
        self.sim_learnings[player_name] = {"boost": capped, "reason": reason}

    def clear_learnings(self) -> None:
        """Remove all sim learning boosts."""
        self.sim_learnings.clear()

    def save_calibration_profile(self, name: str, profile: Dict[str, Any]) -> None:
        """Save a versioned calibration profile."""
        self.calibration_profiles[name] = profile

    def clone_profile(self, source_name: str, new_name: str) -> bool:
        """Clone a calibration profile under a new name. Returns True on success."""
        if source_name in self.calibration_profiles:
            self.calibration_profiles[new_name] = dict(self.calibration_profiles[source_name])
            return True
        return False


# ---------------------------------------------------------------------------
# Session state accessors
# ---------------------------------------------------------------------------

# Key names used in st.session_state to store the shared objects.
_KEY_SLATE = "_yakos_slate_state"
_KEY_EDGE = "_yakos_edge_state"
_KEY_LINEUP = "_yakos_lineup_state"
_KEY_SIM = "_yakos_sim_state"


def _ss():
    """Thin wrapper – returns st.session_state, imported lazily to avoid
    importing streamlit at module load time (breaks unit tests)."""
    import streamlit as st  # noqa: PLC0415
    return st.session_state


def get_slate_state() -> SlateState:
    """Return the current SlateState from session_state, creating if absent."""
    ss = _ss()
    if _KEY_SLATE not in ss:
        ss[_KEY_SLATE] = SlateState()
    return ss[_KEY_SLATE]


def set_slate_state(state: SlateState) -> None:
    """Write a SlateState back to session_state."""
    _ss()[_KEY_SLATE] = state


def get_edge_state() -> RickyEdgeState:
    """Return the current RickyEdgeState from session_state, creating if absent."""
    ss = _ss()
    if _KEY_EDGE not in ss:
        ss[_KEY_EDGE] = RickyEdgeState()
    return ss[_KEY_EDGE]


def set_edge_state(state: RickyEdgeState) -> None:
    """Write a RickyEdgeState back to session_state."""
    _ss()[_KEY_EDGE] = state


def get_lineup_state() -> LineupSetState:
    """Return the current LineupSetState from session_state, creating if absent."""
    ss = _ss()
    if _KEY_LINEUP not in ss:
        ss[_KEY_LINEUP] = LineupSetState()
    return ss[_KEY_LINEUP]


def set_lineup_state(state: LineupSetState) -> None:
    """Write a LineupSetState back to session_state."""
    _ss()[_KEY_LINEUP] = state


def get_sim_state() -> SimState:
    """Return the current SimState from session_state, creating if absent."""
    ss = _ss()
    if _KEY_SIM not in ss:
        ss[_KEY_SIM] = SimState()
    return ss[_KEY_SIM]


def set_sim_state(state: SimState) -> None:
    """Write a SimState back to session_state."""
    _ss()[_KEY_SIM] = state
