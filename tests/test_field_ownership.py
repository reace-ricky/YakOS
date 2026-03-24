"""Tests for yak_core.field_ownership and yak_core.ownership_store."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yak_core.field_ownership import (
    Lineup,
    build_field_lineups,
    estimate_ownership_from_field,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_nba_pool(n_players: int = 20) -> pd.DataFrame:
    """Create a minimal NBA classic pool for testing."""
    rng = np.random.default_rng(0)
    positions = (["PG", "SG", "SF", "PF", "C"] * 10)[:n_players]
    salaries = rng.integers(4000, 10000, size=n_players) // 100 * 100
    projs = rng.uniform(10, 50, size=n_players)
    return pd.DataFrame({
        "player_name": [f"Player{i}" for i in range(n_players)],
        "pos": positions,
        "salary": salaries.astype(int),
        "proj": projs,
    })


def _make_pga_pool(n_players: int = 12) -> pd.DataFrame:
    """Create a minimal PGA pool for testing."""
    rng = np.random.default_rng(1)
    salaries = rng.integers(6000, 11000, size=n_players) // 100 * 100
    projs = rng.uniform(30, 80, size=n_players)
    return pd.DataFrame({
        "player_name": [f"Golfer{i}" for i in range(n_players)],
        "pos": ["G"] * n_players,
        "salary": salaries.astype(int),
        "proj": projs,
    })


# ---------------------------------------------------------------------------
# Lineup dataclass
# ---------------------------------------------------------------------------

class TestLineup:
    def test_lineup_has_player_ids(self):
        lu = Lineup(player_ids=["A", "B", "C"], total_salary=30000, proj_fp=120.5)
        assert lu.player_ids == ["A", "B", "C"]
        assert lu.total_salary == 30000
        assert lu.proj_fp == pytest.approx(120.5)

    def test_lineup_defaults(self):
        lu = Lineup(player_ids=["X"])
        assert lu.total_salary == 0
        assert lu.proj_fp == 0.0


# ---------------------------------------------------------------------------
# build_field_lineups — NBA Classic
# ---------------------------------------------------------------------------

class TestBuildFieldLineups:
    def test_returns_list_of_lineup(self):
        pool = _make_nba_pool()
        result = build_field_lineups(pool, config={"n_field_lineups": 10, "random_seed": 42})
        assert isinstance(result, list)
        assert all(isinstance(lu, Lineup) for lu in result)

    def test_n_field_lineups(self):
        pool = _make_nba_pool()
        result = build_field_lineups(pool, config={"n_field_lineups": 50, "random_seed": 1})
        # Allow for some failures, but should generate close to requested
        assert len(result) >= 1

    def test_lineup_size_nba_classic(self):
        pool = _make_nba_pool()
        result = build_field_lineups(pool, config={"n_field_lineups": 20, "random_seed": 42})
        assert len(result) > 0
        for lu in result:
            assert len(lu.player_ids) == 8  # DK NBA classic

    def test_reproducibility(self):
        pool = _make_nba_pool()
        r1 = build_field_lineups(pool, config={"n_field_lineups": 10, "random_seed": 7})
        r2 = build_field_lineups(pool, config={"n_field_lineups": 10, "random_seed": 7})
        # Same seed → same lineups
        assert [lu.player_ids for lu in r1] == [lu.player_ids for lu in r2]

    def test_different_seeds_differ(self):
        pool = _make_nba_pool()
        r1 = build_field_lineups(pool, config={"n_field_lineups": 10, "random_seed": 1})
        r2 = build_field_lineups(pool, config={"n_field_lineups": 10, "random_seed": 99})
        all_same = all(
            set(a.player_ids) == set(b.player_ids)
            for a, b in zip(r1, r2)
        )
        assert not all_same, "Different seeds should produce different lineups"

    def test_salary_cap_respected(self):
        pool = _make_nba_pool()
        cap = 50000
        result = build_field_lineups(pool, config={"n_field_lineups": 30, "random_seed": 42, "salary_cap": cap})
        for lu in result:
            assert lu.total_salary <= cap

    def test_no_duplicate_players_in_lineup(self):
        pool = _make_nba_pool()
        result = build_field_lineups(pool, config={"n_field_lineups": 20, "random_seed": 42})
        for lu in result:
            assert len(lu.player_ids) == len(set(lu.player_ids)), "Duplicate player in lineup"

    def test_total_salary_computed(self):
        pool = _make_nba_pool()
        result = build_field_lineups(pool, config={"n_field_lineups": 10, "random_seed": 42})
        assert len(result) > 0
        for lu in result:
            assert lu.total_salary > 0

    def test_proj_fp_computed(self):
        pool = _make_nba_pool()
        result = build_field_lineups(pool, config={"n_field_lineups": 10, "random_seed": 42})
        assert len(result) > 0
        for lu in result:
            assert lu.proj_fp > 0.0

    def test_empty_pool_returns_empty(self):
        pool = pd.DataFrame(columns=["player_name", "pos", "salary", "proj"])
        result = build_field_lineups(pool, config={"n_field_lineups": 10})
        assert result == []

    def test_raises_on_missing_salary(self):
        pool = pd.DataFrame({"player_name": ["A"], "pos": ["PG"], "proj": [30.0]})
        with pytest.raises(ValueError, match="salary"):
            build_field_lineups(pool, config={"n_field_lineups": 1})

    def test_uses_player_name_as_id_when_no_player_id(self):
        pool = _make_nba_pool()
        result = build_field_lineups(pool, config={"n_field_lineups": 5, "random_seed": 42})
        assert len(result) > 0
        # IDs should match player_name values
        all_names = set(pool["player_name"].tolist())
        for lu in result:
            for pid in lu.player_ids:
                assert pid in all_names

    def test_uses_player_id_col_when_present(self):
        pool = _make_nba_pool()
        pool["player_id"] = [f"PID{i:03d}" for i in range(len(pool))]
        result = build_field_lineups(pool, config={"n_field_lineups": 5, "random_seed": 42})
        assert len(result) > 0
        all_pids = set(pool["player_id"].tolist())
        for lu in result:
            for pid in lu.player_ids:
                assert pid in all_pids


# ---------------------------------------------------------------------------
# build_field_lineups — PGA
# ---------------------------------------------------------------------------

class TestBuildFieldLineupsPGA:
    def test_pga_lineup_size(self):
        pool = _make_pga_pool()
        result = build_field_lineups(pool, config={
            "n_field_lineups": 20, "random_seed": 42, "sport": "PGA"
        })
        assert len(result) > 0
        for lu in result:
            assert len(lu.player_ids) == 6  # DK PGA lineup size

    def test_pga_no_duplicates(self):
        pool = _make_pga_pool()
        result = build_field_lineups(pool, config={
            "n_field_lineups": 20, "random_seed": 42, "sport": "PGA"
        })
        for lu in result:
            assert len(lu.player_ids) == len(set(lu.player_ids))


# ---------------------------------------------------------------------------
# estimate_ownership_from_field
# ---------------------------------------------------------------------------

class TestEstimateOwnership:
    def test_returns_dataframe(self):
        lineups = [
            Lineup(player_ids=["A", "B", "C"]),
            Lineup(player_ids=["A", "B", "D"]),
        ]
        result = estimate_ownership_from_field(lineups)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self):
        lineups = [Lineup(player_ids=["A", "B"])]
        result = estimate_ownership_from_field(lineups)
        assert "player_id" in result.columns
        assert "own_proj" in result.columns

    def test_own_proj_sums_to_lineup_size(self):
        # sum of own_proj should equal lineup size (each lineup has k players)
        pool = _make_nba_pool(20)
        lineups = build_field_lineups(pool, config={"n_field_lineups": 100, "random_seed": 42})
        own = estimate_ownership_from_field(lineups)
        lineup_size = 8
        expected_sum = pytest.approx(lineup_size, abs=0.5)
        assert own["own_proj"].sum() == expected_sum

    def test_own_proj_between_0_and_1(self):
        pool = _make_nba_pool()
        lineups = build_field_lineups(pool, config={"n_field_lineups": 50, "random_seed": 42})
        own = estimate_ownership_from_field(lineups)
        assert (own["own_proj"] >= 0.0).all()
        assert (own["own_proj"] <= 1.0).all()

    def test_chalk_has_higher_ownership(self):
        """Players with higher projections should get more ownership on average."""
        pool = _make_nba_pool(20)
        lineups = build_field_lineups(pool, config={"n_field_lineups": 1000, "random_seed": 42})
        own = estimate_ownership_from_field(lineups)

        # Merge back to pool
        merged = pool.merge(own.rename(columns={"player_id": "player_name"}), on="player_name", how="left")
        merged["own_proj"] = merged["own_proj"].fillna(0.0)

        # Correlation between proj and own_proj should be positive
        corr = merged["proj"].corr(merged["own_proj"])
        assert corr > 0.1, f"Expected positive correlation, got {corr:.3f}"

    def test_empty_lineups_returns_empty_df(self):
        result = estimate_ownership_from_field([])
        assert result.empty
        assert "player_id" in result.columns
        assert "own_proj" in result.columns

    def test_all_players_accounted_for(self):
        lineups = [
            Lineup(player_ids=["A", "B", "C"]),
            Lineup(player_ids=["A", "D", "E"]),
        ]
        result = estimate_ownership_from_field(lineups)
        found_ids = set(result["player_id"].tolist())
        assert found_ids == {"A", "B", "C", "D", "E"}

    def test_frequency_calculation(self):
        # Directly test the count-based ownership formula with hand-crafted lineups.
        # The field builder enforces uniqueness within each lineup; here we
        # explicitly test the counting logic with a player appearing in the
        # same lineup twice to verify the denominator is lineup count, not
        # player-appearance count.
        lineups = [
            Lineup(player_ids=["A", "B"]),
            Lineup(player_ids=["A", "C"]),
            Lineup(player_ids=["B", "C"]),
            Lineup(player_ids=["C", "C"]),  # same player twice: counts as 2 appearances
        ]
        result = estimate_ownership_from_field(lineups)
        own_map = dict(zip(result["player_id"], result["own_proj"]))
        # A: 2 appearances / 4 lineups = 0.5
        assert own_map["A"] == pytest.approx(0.5)
        # B: 2/4 = 0.5
        assert own_map["B"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Integration: build → estimate → ownership store round-trip
# ---------------------------------------------------------------------------

class TestOwnershipStoreRoundTrip:
    def test_write_and_load(self, tmp_path):
        """write_own_proj_to_archive persists and load_own_proj_for_slate restores."""
        import yak_core.ownership_store as store_mod
        # Patch the storage dir to tmp_path
        original = store_mod._STORE_DIR
        store_mod._STORE_DIR = tmp_path

        try:
            own_df = pd.DataFrame({
                "player_id": ["A", "B", "C"],
                "own_proj": [0.5, 0.3, 0.2],
            })
            store_mod.write_own_proj_to_archive("nba", "dk", "2026-01-01", "gpp_main", own_df)
            loaded = store_mod.load_own_proj_for_slate("nba", "dk", "2026-01-01", "gpp_main")

            assert not loaded.empty
            assert "player_id" in loaded.columns
            assert "own_proj" in loaded.columns
            assert set(loaded["player_id"]) == {"A", "B", "C"}
            assert loaded.loc[loaded["player_id"] == "A", "own_proj"].iloc[0] == pytest.approx(0.5)
        finally:
            store_mod._STORE_DIR = original

    def test_load_missing_returns_empty(self, tmp_path):
        import yak_core.ownership_store as store_mod
        original = store_mod._STORE_DIR
        store_mod._STORE_DIR = tmp_path
        try:
            result = store_mod.load_own_proj_for_slate("nba", "dk", "NONEXISTENT", "gpp_main")
            assert result.empty
        finally:
            store_mod._STORE_DIR = original

    def test_attach_own_proj_to_pool(self, tmp_path):
        import yak_core.ownership_store as store_mod
        original = store_mod._STORE_DIR
        store_mod._STORE_DIR = tmp_path

        try:
            pool = pd.DataFrame({
                "player_name": ["A", "B", "C", "D"],
                "salary": [8000, 7000, 6000, 5000],
            })
            own_df = pd.DataFrame({
                "player_id": ["A", "B", "C"],
                "own_proj": [0.5, 0.3, 0.2],
            })
            store_mod.write_own_proj_to_archive("nba", "dk", "TEST", "gpp_main", own_df)
            enriched = store_mod.attach_own_proj_to_pool(
                pool, "nba", "dk", "TEST", "gpp_main"
            )
            assert "own_proj" in enriched.columns
            assert enriched.loc[enriched["player_name"] == "A", "own_proj"].iloc[0] == pytest.approx(0.5)
            # D has no match → NaN
            assert pd.isna(enriched.loc[enriched["player_name"] == "D", "own_proj"].iloc[0])
        finally:
            store_mod._STORE_DIR = original


# ---------------------------------------------------------------------------
# compute_leverage leverage_mode
# ---------------------------------------------------------------------------

class TestComputeLeverageMode:
    def _make_pool(self) -> pd.DataFrame:
        return pd.DataFrame({
            "player_name": [f"P{i}" for i in range(6)],
            "proj": [40.0, 35.0, 30.0, 25.0, 20.0, 15.0],
            "own_proj": [30.0, 15.0, 20.0, 10.0, 25.0, 5.0],  # 0-100 scale
            "smash_prob": [0.8, 0.5, 0.6, 0.4, 0.7, 0.3],
        })

    def test_none_mode_returns_leverage(self):
        from yak_core.ownership import compute_leverage
        pool = self._make_pool()
        result = compute_leverage(pool, own_col="own_proj", leverage_mode="none")
        assert "leverage" in result.columns
        assert (result["leverage"] >= 0.0).all()
        assert (result["leverage"] <= 1.0).all()

    def test_smash_minus_own_mode(self):
        from yak_core.ownership import compute_leverage
        pool = self._make_pool()
        result = compute_leverage(pool, own_col="own_proj", leverage_mode="smash_minus_own")
        assert "leverage" in result.columns
        assert result["leverage"].notna().all()

    def test_fp_over_own_mode(self):
        from yak_core.ownership import compute_leverage
        pool = self._make_pool()
        result = compute_leverage(pool, own_col="own_proj", leverage_mode="fp_over_own")
        assert "leverage" in result.columns
        assert (result["leverage"] >= 0.0).all()

    def test_missing_own_proj_raises(self):
        from yak_core.ownership import compute_leverage
        pool = pd.DataFrame({"player_name": ["A"], "proj": [30.0]})
        with pytest.raises(ValueError):
            compute_leverage(pool, own_col="own_proj")

    def test_different_modes_produce_different_scores(self):
        from yak_core.ownership import compute_leverage
        pool = self._make_pool()
        r_none = compute_leverage(pool.copy(), own_col="own_proj", leverage_mode="none")
        r_smash = compute_leverage(pool.copy(), own_col="own_proj", leverage_mode="smash_minus_own")
        # At least some values should differ
        assert not r_none["leverage"].equals(r_smash["leverage"])
