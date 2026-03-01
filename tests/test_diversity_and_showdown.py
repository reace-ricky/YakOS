"""Tests for R6 (lineup diversity / pair-fade) and R7 (Showdown Captain optimizer)."""

import pandas as pd
import pytest
from yak_core.lineups import (
    build_multiple_lineups_with_exposure,
    build_showdown_lineups,
    to_dk_showdown_upload_format,
)
from yak_core.config import (
    DK_SHOWDOWN_LINEUP_SIZE,
    DK_SHOWDOWN_SLOTS,
    DK_SHOWDOWN_CAPTAIN_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_classic_pool(n: int = 120) -> pd.DataFrame:
    """Minimal Classic pool with enough players to build 20 lineups."""
    positions = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "PF/C"]
    rows = []
    for i in range(n):
        rows.append({
            "player_id": str(i),
            "player_name": f"Player_{i}",
            "team": f"T{i % 6}",
            "opponent": f"T{(i + 3) % 6}",
            "pos": positions[i % len(positions)],
            "salary": 4000 + (i % 30) * 200,
            "proj": 10.0 + (i % 20) * 1.5,
        })
    return pd.DataFrame(rows)


def _make_showdown_pool(n: int = 20) -> pd.DataFrame:
    """Minimal Showdown pool (two teams, n players each)."""
    rows = []
    for i in range(n):
        team = "HOM" if i < n // 2 else "AWY"
        opp = "AWY" if team == "HOM" else "HOM"
        rows.append({
            "player_id": str(i),
            "player_name": f"Player_{i}",
            "team": team,
            "opponent": opp,
            "pos": ["PG", "SG", "SF", "PF", "C"][i % 5],
            "salary": 6000 + i * 300,
            "proj": 20.0 + i * 2.0,
        })
    return pd.DataFrame(rows)


_BASE_CFG = {
    "NUM_LINEUPS": 5,
    "SALARY_CAP": 50000,
    "MIN_SALARY_USED": 40000,
    "MAX_EXPOSURE": 1.0,
    "PROJ_COL": "proj",
    "SOLVER_TIME_LIMIT": 30,
}

_SHOWDOWN_CFG = {
    "NUM_LINEUPS": 5,
    "SALARY_CAP": 50000,
    "MIN_SALARY_USED": 0,
    "MAX_EXPOSURE": 1.0,
    "PROJ_COL": "proj",
    "SOLVER_TIME_LIMIT": 30,
}


# ===========================================================================
# R6: Lineup diversity / pair-fade controls
# ===========================================================================

class TestMaxPairAppearances:
    def test_disabled_by_default(self):
        """MAX_PAIR_APPEARANCES=0 should behave exactly like before (no constraint)."""
        pool = _make_classic_pool()
        cfg = dict(_BASE_CFG, NUM_LINEUPS=10, MAX_PAIR_APPEARANCES=0)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        assert lu_df["lineup_index"].nunique() == 10

    def test_pair_cap_respected(self):
        """No player pair should appear together more times than MAX_PAIR_APPEARANCES."""
        pool = _make_classic_pool()
        max_pair = 2
        cfg = dict(_BASE_CFG, NUM_LINEUPS=8, MAX_PAIR_APPEARANCES=max_pair, MAX_EXPOSURE=1.0)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)

        # Build (player_id, lineup_index) presence set
        presence = lu_df.set_index("lineup_index")["player_id"]
        lineup_sets = {
            lu: set(lu_df[lu_df["lineup_index"] == lu]["player_id"])
            for lu in lu_df["lineup_index"].unique()
        }
        lineups = list(lineup_sets.values())
        from itertools import combinations
        for a, b in combinations(range(len(lineups)), 2):
            pair_together = len(lineups[a] & lineups[b]) > 0  # count doesn't matter for pairs
        # Count occurrences of each player pair
        pair_count: dict = {}
        for lu_set in lineups:
            players = sorted(lu_set)
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    key = (players[i], players[j])
                    pair_count[key] = pair_count.get(key, 0) + 1
        for pair, count in pair_count.items():
            assert count <= max_pair, (
                f"Pair {pair} appeared together {count} times, exceeding max_pair={max_pair}"
            )

    def test_pair_cap_1_no_pair_repeats(self):
        """With MAX_PAIR_APPEARANCES=1, no player pair can appear together more than once."""
        pool = _make_classic_pool(150)
        cfg = dict(_BASE_CFG, NUM_LINEUPS=5, MAX_PAIR_APPEARANCES=1, MAX_EXPOSURE=1.0)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        lineup_sets = {
            lu: set(lu_df[lu_df["lineup_index"] == lu]["player_id"])
            for lu in sorted(lu_df["lineup_index"].unique())
        }
        lineups = list(lineup_sets.values())
        pair_count: dict = {}
        for lu_set in lineups:
            players_in_lu = sorted(lu_set)
            for i in range(len(players_in_lu)):
                for j in range(i + 1, len(players_in_lu)):
                    key = (players_in_lu[i], players_in_lu[j])
                    pair_count[key] = pair_count.get(key, 0) + 1
        for pair, count in pair_count.items():
            assert count <= 1, (
                f"Pair {pair} appeared together {count} times with MAX_PAIR_APPEARANCES=1"
            )

    def test_pair_cap_config_key_present(self):
        """MAX_PAIR_APPEARANCES key should be accepted without error."""
        pool = _make_classic_pool()
        cfg = dict(_BASE_CFG, NUM_LINEUPS=3, MAX_PAIR_APPEARANCES=5)
        lu_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
        assert lu_df["lineup_index"].nunique() == 3


# ===========================================================================
# R7: Showdown Captain optimizer
# ===========================================================================

class TestShowdownConfig:
    def test_showdown_lineup_size(self):
        assert DK_SHOWDOWN_LINEUP_SIZE == 6

    def test_showdown_slots_count(self):
        assert len(DK_SHOWDOWN_SLOTS) == 6

    def test_showdown_slots_contains_cpt(self):
        assert "CPT" in DK_SHOWDOWN_SLOTS

    def test_showdown_slots_contains_five_flex(self):
        assert DK_SHOWDOWN_SLOTS.count("FLEX") == 5

    def test_captain_multiplier_is_1_5(self):
        assert DK_SHOWDOWN_CAPTAIN_MULTIPLIER == 1.5


class TestBuildShowdownLineups:
    def test_returns_correct_lineup_count(self):
        pool = _make_showdown_pool()
        cfg = dict(_SHOWDOWN_CFG, NUM_LINEUPS=3)
        lu_df, _ = build_showdown_lineups(pool, cfg)
        assert lu_df["lineup_index"].nunique() == 3

    def test_each_lineup_has_six_players(self):
        pool = _make_showdown_pool()
        lu_df, _ = build_showdown_lineups(pool, _SHOWDOWN_CFG)
        for lu_idx in lu_df["lineup_index"].unique():
            lu = lu_df[lu_df["lineup_index"] == lu_idx]
            assert len(lu) == DK_SHOWDOWN_LINEUP_SIZE, (
                f"Lineup {lu_idx} has {len(lu)} players, expected {DK_SHOWDOWN_LINEUP_SIZE}"
            )

    def test_each_lineup_has_exactly_one_cpt(self):
        pool = _make_showdown_pool()
        lu_df, _ = build_showdown_lineups(pool, _SHOWDOWN_CFG)
        for lu_idx in lu_df["lineup_index"].unique():
            lu = lu_df[lu_df["lineup_index"] == lu_idx]
            cpt_count = (lu["slot"] == "CPT").sum()
            assert cpt_count == 1, f"Lineup {lu_idx}: expected 1 CPT, got {cpt_count}"

    def test_each_lineup_has_exactly_five_flex(self):
        pool = _make_showdown_pool()
        lu_df, _ = build_showdown_lineups(pool, _SHOWDOWN_CFG)
        for lu_idx in lu_df["lineup_index"].unique():
            lu = lu_df[lu_df["lineup_index"] == lu_idx]
            flex_count = (lu["slot"] == "FLEX").sum()
            assert flex_count == 5, f"Lineup {lu_idx}: expected 5 FLEX, got {flex_count}"

    def test_no_player_appears_as_both_cpt_and_flex(self):
        pool = _make_showdown_pool()
        lu_df, _ = build_showdown_lineups(pool, _SHOWDOWN_CFG)
        for lu_idx in lu_df["lineup_index"].unique():
            lu = lu_df[lu_df["lineup_index"] == lu_idx]
            assert lu["player_id"].nunique() == DK_SHOWDOWN_LINEUP_SIZE, (
                f"Lineup {lu_idx}: duplicate player (player used in both CPT and FLEX)"
            )

    def test_cpt_salary_is_1_5x(self):
        """The captain row's salary should be 1.5× the base salary."""
        pool = _make_showdown_pool()
        lu_df, _ = build_showdown_lineups(pool, _SHOWDOWN_CFG)

        # Build base salary lookup
        base_salary = {row["player_id"]: row["salary"] for _, row in pool.iterrows()}

        for _, row in lu_df[lu_df["slot"] == "CPT"].iterrows():
            pid = row["player_id"]
            expected_cpt_salary = round(base_salary[pid] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER)
            assert row["salary"] == expected_cpt_salary, (
                f"CPT {pid}: salary {row['salary']} != expected {expected_cpt_salary}"
            )

    def test_cpt_proj_is_1_5x(self):
        """The captain row's proj should be 1.5× the base proj."""
        pool = _make_showdown_pool()
        lu_df, _ = build_showdown_lineups(pool, _SHOWDOWN_CFG)

        base_proj = {row["player_id"]: row["proj"] for _, row in pool.iterrows()}

        for _, row in lu_df[lu_df["slot"] == "CPT"].iterrows():
            pid = row["player_id"]
            expected_cpt_proj = base_proj[pid] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER
            assert abs(row["proj"] - expected_cpt_proj) < 1e-6, (
                f"CPT {pid}: proj {row['proj']} != expected {expected_cpt_proj}"
            )

    def test_salary_cap_respected(self):
        pool = _make_showdown_pool()
        lu_df, _ = build_showdown_lineups(pool, _SHOWDOWN_CFG)
        for lu_idx in lu_df["lineup_index"].unique():
            lu = lu_df[lu_df["lineup_index"] == lu_idx]
            total_salary = lu["salary"].sum()
            assert total_salary <= _SHOWDOWN_CFG["SALARY_CAP"], (
                f"Lineup {lu_idx} salary {total_salary} exceeds cap {_SHOWDOWN_CFG['SALARY_CAP']}"
            )

    def test_returns_exposure_df(self):
        pool = _make_showdown_pool()
        _, exp_df = build_showdown_lineups(pool, _SHOWDOWN_CFG)
        assert "player_id" in exp_df.columns
        assert "exposure" in exp_df.columns
        assert (exp_df["exposure"] >= 0).all()
        assert (exp_df["exposure"] <= 1.0).all()

    def test_too_small_pool_raises(self):
        pool = _make_showdown_pool(4)  # only 4 players, need 6
        with pytest.raises(ValueError, match="Showdown pool"):
            build_showdown_lineups(pool, _SHOWDOWN_CFG)


class TestToDkShowdownUploadFormat:
    def _build_lineups(self) -> pd.DataFrame:
        pool = _make_showdown_pool()
        lu_df, _ = build_showdown_lineups(pool, dict(_SHOWDOWN_CFG, NUM_LINEUPS=3))
        return lu_df

    def test_returns_dataframe(self):
        lu_df = self._build_lineups()
        result = to_dk_showdown_upload_format(lu_df)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_lineup_count(self):
        lu_df = self._build_lineups()
        result = to_dk_showdown_upload_format(lu_df)
        assert len(result) == lu_df["lineup_index"].nunique()

    def test_has_cpt_column(self):
        lu_df = self._build_lineups()
        result = to_dk_showdown_upload_format(lu_df)
        assert "CPT" in result.columns

    def test_has_meta_columns(self):
        lu_df = self._build_lineups()
        result = to_dk_showdown_upload_format(lu_df)
        for col in ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]:
            assert col in result.columns

    def test_empty_input_returns_empty_df(self):
        result = to_dk_showdown_upload_format(pd.DataFrame())
        assert result.empty

    def test_cpt_cell_contains_player_name(self):
        lu_df = self._build_lineups()
        result = to_dk_showdown_upload_format(lu_df)
        # Every CPT cell should contain a non-empty string
        assert result["CPT"].notna().all()
        assert (result["CPT"] != "").all()
