"""Tests for optimizer session cancellation fixes.

PR #9 fix: multi-position strings (e.g. "PF/C") must not be stripped to the
            first token before passing to the optimizer.

This PR fix: when exposure-cap exhaustion makes a lineup infeasible, the
             optimizer should fall back to a relaxed-cap solve rather than
             cancelling the lineup entirely.
"""

import pandas as pd
import pytest
from yak_core.lineups import _eligible_slots, build_multiple_lineups_with_exposure


def _make_pool(c_players: int = 20, pfc_players: int = 9, other_players: int = 82) -> pd.DataFrame:
    """Return a minimal pool with the given position mix."""
    rows = []
    pid = 0

    def _add_player(pos, salary, team="T1", opp="T2"):
        nonlocal pid
        rows.append({
            "player_id": str(pid),
            "player_name": f"Player_{pid}",
            "team": team,
            "opponent": opp,
            "pos": pos,
            "salary": salary,
            "proj": salary / 1000.0 * 4.0,
        })
        pid += 1

    for i in range(c_players):
        _add_player("C", 5000 + i * 150)

    for i in range(pfc_players):
        _add_player("PF/C", 5000 + i * 167)

    positions = ["PG", "SG", "SF", "PF", "PG/SG", "SG/SF", "SF/PF"]
    for i in range(other_players):
        _add_player(positions[i % len(positions)], 4000 + (i % 20) * 250)

    return pd.DataFrame(rows)


_BASE_CFG = {
    "SITE": "dk",
    "SPORT": "nba",
    "NUM_LINEUPS": 20,
    "MIN_SALARY_USED": 46000,
    "PROJ_COL": "proj",
    "SOLVER_TIME_LIMIT": 30,
}


class TestMultiPositionEligibility:
    def test_pfc_eligible_for_c_slot(self):
        slots = _eligible_slots("PF/C")
        assert "C" in slots
        assert "PF" in slots
        assert "F" in slots
        assert "UTIL" in slots

    def test_pgsg_eligible_for_g_slot(self):
        slots = _eligible_slots("PG/SG")
        assert "PG" in slots
        assert "SG" in slots
        assert "G" in slots
        assert "UTIL" in slots

    def test_pure_c_not_eligible_for_f_slot(self):
        slots = _eligible_slots("C")
        assert "C" in slots
        assert "UTIL" in slots
        assert "F" not in slots
        assert "PF" not in slots

    def test_multi_pos_has_more_slots_than_first_only(self):
        full_slots = _eligible_slots("SF/PF")
        first_only_slots = _eligible_slots("SF")
        assert len(full_slots) > len(first_only_slots)
        assert "PF" in full_slots
        assert "PF" not in first_only_slots


class TestExposureFallback:
    def test_default_exposure_all_lineups(self):
        pool = _make_pool()
        cfg = dict(_BASE_CFG, MAX_EXPOSURE=0.35)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        assert lu_df["lineup_index"].nunique() == 20

    def test_tight_exposure_all_lineups_with_fallback(self):
        pool = _make_pool()
        cfg = dict(_BASE_CFG, MAX_EXPOSURE=0.05)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        assert lu_df["lineup_index"].nunique() == 20

    def test_medium_tight_exposure_all_lineups(self):
        pool = _make_pool()
        cfg = dict(_BASE_CFG, MAX_EXPOSURE=0.10)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        assert lu_df["lineup_index"].nunique() == 20

    def test_lineups_valid_salary(self):
        pool = _make_pool()
        cfg = dict(_BASE_CFG, MAX_EXPOSURE=0.05, SALARY_CAP=50000)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        for lu_idx in lu_df["lineup_index"].unique():
            lu = lu_df[lu_df["lineup_index"] == lu_idx]
            assert lu["salary"].sum() <= 50000

    def test_each_lineup_has_correct_size(self):
        from yak_core.config import DK_LINEUP_SIZE
        pool = _make_pool()
        cfg = dict(_BASE_CFG, MAX_EXPOSURE=0.05)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        for lu_idx in lu_df["lineup_index"].unique():
            lu = lu_df[lu_df["lineup_index"] == lu_idx]
            assert len(lu) == DK_LINEUP_SIZE

    def test_pfc_used_in_c_slot_when_pure_c_capped(self):
        # Use fewer lineups to keep test fast; tight cap forces PF/C into C slot
        pool = _make_pool(c_players=2, pfc_players=9, other_players=82)
        cfg = dict(_BASE_CFG, MAX_EXPOSURE=0.50, NUM_LINEUPS=5)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        assert lu_df["lineup_index"].nunique() == 5
        # Verify PF/C players are used in C slot
        c_slot_players = lu_df[lu_df["slot"] == "C"]
        pfc_in_c = c_slot_players[c_slot_players["pos"] == "PF/C"]
        assert len(pfc_in_c) > 0, "PF/C players must be eligible for and fill the C slot"
