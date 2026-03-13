"""Tests for GPP correlation stacking: game stack, team stack, and bring-back.

Verifies that:
- GPP lineups satisfy game stack constraints (3+ from same game)
- GPP lineups satisfy team stack constraints (2+ from same team)
- GPP lineups satisfy bring-back constraints (1 opponent player)
- Cash lineups are NOT affected by any stacking constraints
- The optimizer remains feasible with all stacking constraints enabled
"""

import pandas as pd
import pytest
from yak_core.lineups import build_multiple_lineups_with_exposure


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_GAMES = [
    ("BOS", "LAL"),
    ("MIA", "DEN"),
    ("PHX", "GSW"),
    ("MIL", "DAL"),
]


def _make_stacking_pool(n_per_team: int = 8) -> pd.DataFrame:
    """Pool with 4 games × 2 teams × 8 players = 64 players.

    Enough positional coverage and salary spread for stacking tests.
    """
    positions = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "PF/C"]
    rows = []
    pid = 0
    for home, away in _GAMES:
        for team in (home, away):
            opp = away if team == home else home
            for i in range(n_per_team):
                rows.append({
                    "player_id": str(pid),
                    "player_name": f"{team}_{i}",
                    "team": team,
                    "opponent": opp,
                    "pos": positions[i % len(positions)],
                    "salary": 4000 + (pid % 20) * 300,
                    "proj": 15.0 + (pid % 15) * 2.0,
                    "ceil": 20.0 + (pid % 15) * 3.0,
                    "ownership": 5.0 + (pid % 10),  # 5-15% range
                })
                pid += 1
    return pd.DataFrame(rows)


_GPP_CFG = {
    "NUM_LINEUPS": 5,
    "SALARY_CAP": 50000,
    "MIN_SALARY_USED": 40000,
    "MAX_EXPOSURE": 1.0,
    "SOLVER_TIME_LIMIT": 30,
    "CONTEST_TYPE": "gpp",
    "GPP_FORCE_GAME_STACK": True,
    "GPP_MIN_GAME_STACK": 3,
    "GPP_MIN_TEAM_STACK": 2,
    "GPP_FORCE_BRING_BACK": True,
    "GPP_MAX_PUNT_PLAYERS": 8,
    "GPP_MIN_MID_PLAYERS": 0,
    "GPP_OWN_CAP": 20.0,
    "GPP_MIN_LOW_OWN_PLAYERS": 0,
    "GPP_LOW_OWN_THRESHOLD": 0.01,
}

_CASH_CFG = {
    "NUM_LINEUPS": 3,
    "SALARY_CAP": 50000,
    "MIN_SALARY_USED": 40000,
    "MAX_EXPOSURE": 1.0,
    "SOLVER_TIME_LIMIT": 30,
    "CONTEST_TYPE": "cash",
    "GPP_FORCE_GAME_STACK": False,
    "GPP_MIN_TEAM_STACK": 0,
    "GPP_FORCE_BRING_BACK": False,
}


def _get_lineup_players(lu_df: pd.DataFrame, lineup_idx: int) -> pd.DataFrame:
    return lu_df[lu_df["lineup_index"] == lineup_idx]


# ---------------------------------------------------------------------------
# Game stack tests
# ---------------------------------------------------------------------------

class TestGameStack:
    def test_game_stack_enforced(self):
        """Every GPP lineup must have >= 3 players from the same game."""
        pool = _make_stacking_pool()
        lu_df, _ = build_multiple_lineups_with_exposure(pool, _GPP_CFG)
        assert lu_df["lineup_index"].nunique() == 5

        for lu_idx in lu_df["lineup_index"].unique():
            lu = _get_lineup_players(lu_df, lu_idx)
            # Build game keys (sorted team+opp tuple) and count
            game_counts = {}
            for _, row in lu.iterrows():
                gk = tuple(sorted([row["team"], row["opponent"]]))
                game_counts[gk] = game_counts.get(gk, 0) + 1
            max_from_game = max(game_counts.values())
            assert max_from_game >= 3, (
                f"Lineup {lu_idx}: max players from any game = {max_from_game}, "
                f"expected >= 3. Game counts: {game_counts}"
            )

    def test_configurable_min_game_stack(self):
        """GPP_MIN_GAME_STACK=4 should require 4+ from one game."""
        pool = _make_stacking_pool()
        cfg = dict(_GPP_CFG, GPP_MIN_GAME_STACK=4, GPP_MIN_TEAM_STACK=0,
                   GPP_FORCE_BRING_BACK=False, NUM_LINEUPS=3)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)

        for lu_idx in lu_df["lineup_index"].unique():
            lu = _get_lineup_players(lu_df, lu_idx)
            game_counts = {}
            for _, row in lu.iterrows():
                gk = tuple(sorted([row["team"], row["opponent"]]))
                game_counts[gk] = game_counts.get(gk, 0) + 1
            assert max(game_counts.values()) >= 4


# ---------------------------------------------------------------------------
# Team stack tests
# ---------------------------------------------------------------------------

class TestTeamStack:
    def test_team_stack_enforced(self):
        """Every GPP lineup must have >= 2 players from the same team."""
        pool = _make_stacking_pool()
        lu_df, _ = build_multiple_lineups_with_exposure(pool, _GPP_CFG)

        for lu_idx in lu_df["lineup_index"].unique():
            lu = _get_lineup_players(lu_df, lu_idx)
            team_counts = lu["team"].value_counts()
            assert team_counts.max() >= 2, (
                f"Lineup {lu_idx}: max players from any team = {team_counts.max()}, "
                f"expected >= 2. Team counts: {team_counts.to_dict()}"
            )

    def test_team_stack_disabled_when_zero(self):
        """GPP_MIN_TEAM_STACK=0 should not enforce team stacking."""
        pool = _make_stacking_pool()
        cfg = dict(_GPP_CFG, GPP_MIN_TEAM_STACK=0, GPP_FORCE_BRING_BACK=False)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        # Should still produce lineups (no infeasibility)
        assert lu_df["lineup_index"].nunique() == 5


# ---------------------------------------------------------------------------
# Bring-back tests
# ---------------------------------------------------------------------------

class TestBringBack:
    def test_bring_back_enforced(self):
        """When team stack is active with bring-back, at least one stacked team
        (2+ players) must have an opponent player in the lineup."""
        pool = _make_stacking_pool()
        lu_df, _ = build_multiple_lineups_with_exposure(pool, _GPP_CFG)

        for lu_idx in lu_df["lineup_index"].unique():
            lu = _get_lineup_players(lu_df, lu_idx)
            team_counts = lu["team"].value_counts()
            teams_in_lineup = set(team_counts.index)

            # Find all teams with 2+ players (potential stacks)
            stacked_teams = [t for t, c in team_counts.items() if c >= 2]
            assert stacked_teams, f"Lineup {lu_idx}: no team stack found"

            # At least one stacked team must have its opponent in the lineup
            has_bringback = False
            for st in stacked_teams:
                opp = lu[lu["team"] == st].iloc[0]["opponent"]
                if opp in teams_in_lineup:
                    has_bringback = True
                    break

            assert has_bringback, (
                f"Lineup {lu_idx}: stacked teams {stacked_teams} but none have "
                f"an opponent player in the lineup. Teams: {team_counts.to_dict()}"
            )

    def test_bring_back_disabled(self):
        """GPP_FORCE_BRING_BACK=False should not require opponent players."""
        pool = _make_stacking_pool()
        cfg = dict(_GPP_CFG, GPP_FORCE_BRING_BACK=False)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        # Should produce lineups without error
        assert lu_df["lineup_index"].nunique() == 5


# ---------------------------------------------------------------------------
# Cash lineups not affected
# ---------------------------------------------------------------------------

class TestCashNotAffected:
    def test_cash_no_stacking_constraints(self):
        """Cash lineups should build without any stacking constraints."""
        pool = _make_stacking_pool()
        lu_df, _ = build_multiple_lineups_with_exposure(pool, _CASH_CFG)
        assert lu_df["lineup_index"].nunique() == 3
        # Cash lineups might happen to have stacks naturally, but the
        # constraint should not be enforced — just verify feasibility.

    def test_cash_ignores_gpp_stacking_keys(self):
        """Even if GPP stacking keys are in config, cash should ignore them."""
        pool = _make_stacking_pool()
        cfg = dict(
            _CASH_CFG,
            GPP_FORCE_GAME_STACK=True,
            GPP_MIN_TEAM_STACK=2,
            GPP_FORCE_BRING_BACK=True,
        )
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        assert lu_df["lineup_index"].nunique() == 3


# ---------------------------------------------------------------------------
# Feasibility on small slates
# ---------------------------------------------------------------------------

class TestSmallSlateFeasibility:
    def test_two_game_slate(self):
        """A 2-game slate (4 teams) should still produce lineups with stacking."""
        pool = _make_stacking_pool(n_per_team=10)
        # Filter to just 2 games
        two_game_teams = {"BOS", "LAL", "MIA", "DEN"}
        small_pool = pool[pool["team"].isin(two_game_teams)].reset_index(drop=True)
        cfg = dict(_GPP_CFG, NUM_LINEUPS=3)
        lu_df, _ = build_multiple_lineups_with_exposure(small_pool, cfg)
        assert lu_df["lineup_index"].nunique() == 3

    def test_blank_opponent_skips_stacking(self):
        """If any player has blank opponent data, stacking should be skipped."""
        pool = _make_stacking_pool()
        pool.loc[0, "opponent"] = ""  # blank opponent
        cfg = dict(_GPP_CFG, NUM_LINEUPS=2)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        # Should still produce lineups (stacking skipped, not infeasible)
        assert lu_df["lineup_index"].nunique() == 2
