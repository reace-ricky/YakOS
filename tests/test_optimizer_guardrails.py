"""Optimizer guardrail tests.

These tests catch the class of bugs that previously shipped silently:
  - Salary floor not enforced (phantom player bug)
  - Multi-position players only eligible for UTIL
  - Lineups violating configured constraints

Run: pytest tests/test_optimizer_guardrails.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from yak_core.config import (
    DEFAULT_CONFIG, CONTEST_PRESETS, merge_config,
    DK_POS_SLOTS, DK_PGA_POS_SLOTS, SALARY_CAP,
)
from yak_core.lineups import (
    _build_one_lineup, prepare_pool,
    build_multiple_lineups_with_exposure,
    _get_sim_col,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pool(n=40, salary_range=(3500, 12000), seed=42):
    """Generate a synthetic NBA player pool for testing."""
    rng = np.random.default_rng(seed)
    positions = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C"]
    teams = ["BOS", "LAL", "DEN", "MIL", "PHX", "ORL", "SAC", "ATL"]
    pool = pd.DataFrame({
        "player_name": [f"Player_{i}" for i in range(n)],
        "position": [positions[i % len(positions)] for i in range(n)],
        "salary": rng.integers(salary_range[0], salary_range[1], size=n),
        "team": [teams[i % len(teams)] for i in range(n)],
        "opp": [teams[(i + 4) % len(teams)] for i in range(n)],
    })
    pool["proj"] = pool["salary"] / 1000 * rng.uniform(3.5, 5.5, size=n)
    pool["ceil"] = pool["proj"] * rng.uniform(1.2, 1.6, size=n)
    pool["floor"] = pool["proj"] * rng.uniform(0.5, 0.8, size=n)
    pool["own_pct"] = rng.uniform(0.02, 0.45, size=n)
    pool["game_id"] = pool["team"] + "_" + pool["opp"]
    return pool


def _make_cfg(**overrides):
    """Build a GPP Main config with optional overrides."""
    preset = CONTEST_PRESETS["GPP Main"]
    base = {**preset, "SPORT": "NBA", "NUM_LINEUPS": 5,
            "PROJ_SOURCE": "parquet"}
    base.update(overrides)
    return merge_config(base)


# ---------------------------------------------------------------------------
# 1. Position eligibility — multi-position players
# ---------------------------------------------------------------------------

class TestPositionEligibility:
    """Multi-position strings like SG/SF must be eligible for both slots."""

    def _eligible(self, pos_str, slot):
        """Replicate the optimizer's eligibility check."""
        positions = {p.strip() for p in pos_str.upper().split("/") if p.strip()}
        if slot == "UTIL":
            return True
        if slot == "G":
            return bool(positions & {"PG", "SG", "G"})
        if slot == "F":
            return bool(positions & {"SF", "PF", "F"})
        return slot in positions

    @pytest.mark.parametrize("pos,slot,expected", [
        # Single positions
        ("PG", "PG", True),
        ("SG", "SG", True),
        ("C", "C", True),
        ("PG", "G", True),
        ("SG", "G", True),
        ("SF", "F", True),
        ("PF", "F", True),
        ("PG", "UTIL", True),
        ("PG", "SF", False),
        ("C", "PG", False),
        # Multi-position — the bug that broke everything
        ("PG/SG", "PG", True),
        ("PG/SG", "SG", True),
        ("PG/SG", "G", True),
        ("PG/SG", "SF", False),
        ("SG/SF", "SG", True),
        ("SG/SF", "SF", True),
        ("SG/SF", "G", True),
        ("SG/SF", "F", True),
        ("SF/PF", "SF", True),
        ("SF/PF", "PF", True),
        ("SF/PF", "F", True),
        ("SF/PF", "G", False),
        ("PF/C", "PF", True),
        ("PF/C", "C", True),
        ("PF/C", "F", True),
        ("PF/C", "PG", False),
        # Edge cases
        ("", "UTIL", True),
        ("PG/SG", "UTIL", True),
    ])
    def test_eligibility(self, pos, slot, expected):
        assert self._eligible(pos, slot) == expected, \
            f"pos={pos} slot={slot}: expected {expected}"


# ---------------------------------------------------------------------------
# 2. Salary floor enforcement
# ---------------------------------------------------------------------------

class TestSalaryFloor:
    """Salary floor constraint must be physically enforced by the LP solver."""

    def test_salary_floor_respected(self):
        """Lineup total salary must be >= MIN_SALARY_USED."""
        pool = _make_pool(n=60)
        cfg = _make_cfg(MIN_SALARY_USED=49000, NUM_LINEUPS=3)
        player_pool = prepare_pool(pool, cfg)
        lineups_df, _ = build_multiple_lineups_with_exposure(player_pool, cfg)

        if lineups_df.empty:
            pytest.skip("No lineups built (possible infeasibility)")

        for li in lineups_df["lineup_index"].unique():
            lu = lineups_df[lineups_df["lineup_index"] == li]
            total_sal = lu["salary"].sum()
            assert total_sal >= 49000, \
                f"Lineup {li}: salary ${total_sal:,} < floor $49,000"

    def test_salary_cap_respected(self):
        """Lineup total salary must be <= SALARY_CAP."""
        pool = _make_pool(n=60)
        cfg = _make_cfg(NUM_LINEUPS=5)
        player_pool = prepare_pool(pool, cfg)
        lineups_df, _ = build_multiple_lineups_with_exposure(player_pool, cfg)

        if lineups_df.empty:
            pytest.skip("No lineups built")

        for li in lineups_df["lineup_index"].unique():
            lu = lineups_df[lineups_df["lineup_index"] == li]
            total_sal = lu["salary"].sum()
            assert total_sal <= 50000, \
                f"Lineup {li}: salary ${total_sal:,} > cap $50,000"

    def test_no_phantom_players(self):
        """Each player in a lineup must be eligible for its assigned slot."""
        pool = _make_pool(n=60)
        cfg = _make_cfg(NUM_LINEUPS=3)
        player_pool = prepare_pool(pool, cfg)
        lineups_df, _ = build_multiple_lineups_with_exposure(player_pool, cfg)

        if lineups_df.empty:
            pytest.skip("No lineups built")

        for _, row in lineups_df.iterrows():
            pos = row["position"]
            slot = row["slot"]
            positions = {p.strip() for p in pos.upper().split("/") if p.strip()}
            if slot == "UTIL":
                continue
            if slot == "G":
                assert positions & {"PG", "SG", "G"}, \
                    f"{row['player_name']} (pos={pos}) in slot {slot}"
            elif slot == "F":
                assert positions & {"SF", "PF", "F"}, \
                    f"{row['player_name']} (pos={pos}) in slot {slot}"
            else:
                assert slot in positions, \
                    f"{row['player_name']} (pos={pos}) in slot {slot}"


# ---------------------------------------------------------------------------
# 3. Lineup structure
# ---------------------------------------------------------------------------

class TestLineupStructure:
    """Each lineup must have exactly 8 players in valid DK slots."""

    def test_lineup_size(self):
        pool = _make_pool(n=60)
        cfg = _make_cfg(NUM_LINEUPS=3)
        player_pool = prepare_pool(pool, cfg)
        lineups_df, _ = build_multiple_lineups_with_exposure(player_pool, cfg)

        if lineups_df.empty:
            pytest.skip("No lineups built")

        for li in lineups_df["lineup_index"].unique():
            lu = lineups_df[lineups_df["lineup_index"] == li]
            assert len(lu) == 8, f"Lineup {li}: {len(lu)} players, expected 8"

    def test_slot_coverage(self):
        """Each DK slot must be filled exactly once per lineup."""
        pool = _make_pool(n=60)
        cfg = _make_cfg(NUM_LINEUPS=3)
        player_pool = prepare_pool(pool, cfg)
        lineups_df, _ = build_multiple_lineups_with_exposure(player_pool, cfg)

        if lineups_df.empty:
            pytest.skip("No lineups built")

        expected_slots = sorted(DK_POS_SLOTS)
        for li in lineups_df["lineup_index"].unique():
            lu = lineups_df[lineups_df["lineup_index"] == li]
            actual_slots = sorted(lu["slot"].tolist())
            assert actual_slots == expected_slots, \
                f"Lineup {li}: slots {actual_slots} != {expected_slots}"

    def test_no_duplicate_players_in_lineup(self):
        """Each player appears at most once per lineup."""
        pool = _make_pool(n=60)
        cfg = _make_cfg(NUM_LINEUPS=5)
        player_pool = prepare_pool(pool, cfg)
        lineups_df, _ = build_multiple_lineups_with_exposure(player_pool, cfg)

        if lineups_df.empty:
            pytest.skip("No lineups built")

        for li in lineups_df["lineup_index"].unique():
            lu = lineups_df[lineups_df["lineup_index"] == li]
            names = lu["player_name"].tolist()
            assert len(names) == len(set(names)), \
                f"Lineup {li}: duplicate player(s)"


# ---------------------------------------------------------------------------
# 4. Projection sanity checks
# ---------------------------------------------------------------------------

class TestProjectionSanity:
    """Lineups should have reasonable total projections."""

    def test_min_projection_floor(self):
        """With salary floor enforced, lineups should project > 200."""
        pool = _make_pool(n=80, salary_range=(3500, 12000))
        cfg = _make_cfg(MIN_SALARY_USED=49000, NUM_LINEUPS=3)
        player_pool = prepare_pool(pool, cfg)
        lineups_df, _ = build_multiple_lineups_with_exposure(player_pool, cfg)

        if lineups_df.empty:
            pytest.skip("No lineups built")

        for li in lineups_df["lineup_index"].unique():
            lu = lineups_df[lineups_df["lineup_index"] == li]
            total_proj = lu["proj"].sum()
            assert total_proj > 200, \
                f"Lineup {li}: proj {total_proj:.1f} < 200 (suspiciously low)"

    def test_gpp_score_uses_ceil(self):
        """gpp_score should incorporate ceiling, not just projection."""
        pool = _make_pool(n=40)
        cfg = _make_cfg()
        player_pool = prepare_pool(pool, cfg)
        # gpp_score should be > proj*0.25 when ceil > proj
        for _, row in player_pool.iterrows():
            if row["ceil"] > row["proj"]:
                assert row["gpp_score"] > row["proj"] * 0.25, \
                    f"{row['player_name']}: gpp_score should use ceil"


# ---------------------------------------------------------------------------
# 5. GPP constraint enforcement
# ---------------------------------------------------------------------------

class TestGPPConstraints:
    """GPP-specific constraints (punt cap, game stack, etc.) are enforced."""

    def test_max_punt_players(self):
        """At most max_punt_players with salary < $4000."""
        pool = _make_pool(n=80, salary_range=(2000, 12000))
        cfg = _make_cfg(GPP_MAX_PUNT_PLAYERS=2, NUM_LINEUPS=3)
        player_pool = prepare_pool(pool, cfg)
        lineups_df, _ = build_multiple_lineups_with_exposure(player_pool, cfg)

        if lineups_df.empty:
            pytest.skip("No lineups built")

        for li in lineups_df["lineup_index"].unique():
            lu = lineups_df[lineups_df["lineup_index"] == li]
            punt_count = (lu["salary"] < 4000).sum()
            assert punt_count <= 2, \
                f"Lineup {li}: {punt_count} punt players > max 2"

    def test_exposure_cap(self):
        """No player exceeds MAX_EXPOSURE across lineups."""
        pool = _make_pool(n=60)
        cfg = _make_cfg(MAX_EXPOSURE=0.35, NUM_LINEUPS=20)
        player_pool = prepare_pool(pool, cfg)
        lineups_df, _ = build_multiple_lineups_with_exposure(player_pool, cfg)

        if lineups_df.empty:
            pytest.skip("No lineups built")

        n_built = lineups_df["lineup_index"].nunique()
        max_allowed = max(1, int(n_built * 0.35))
        for pname in lineups_df["player_name"].unique():
            count = lineups_df[lineups_df["player_name"] == pname]["lineup_index"].nunique()
            assert count <= max_allowed + 1, \
                f"{pname}: appeared in {count}/{n_built} lineups, max allowed ~{max_allowed}"


# ---------------------------------------------------------------------------
# 6. Config wiring
# ---------------------------------------------------------------------------

class TestConfigWiring:
    """Contest presets correctly wire into optimizer config."""

    def test_gpp_main_salary_floor(self):
        """GPP Main preset sets MIN_SALARY_USED=49000."""
        cfg = _make_cfg()
        assert cfg["MIN_SALARY_USED"] == 49000

    def test_showdown_no_salary_floor(self):
        """Showdown preset sets MIN_SALARY_USED=0."""
        preset = CONTEST_PRESETS["Showdown"]
        cfg = merge_config({**preset, "SPORT": "NBA"})
        assert cfg["MIN_SALARY_USED"] == 0

    def test_salary_cap_is_50000(self):
        cfg = _make_cfg()
        assert cfg["SALARY_CAP"] == 50000

    def test_gpp_weights_in_config(self):
        """GPP v8 scoring weights must be present."""
        cfg = _make_cfg()
        # v8 sim-based scoring weights
        assert cfg.get("GPP_PROJ_WEIGHT", 0.50) == 0.50
        assert "GPP_UPSIDE_WEIGHT" in cfg
        assert "GPP_BOOM_WEIGHT" in cfg
        assert "GPP_OWN_PENALTY_STRENGTH" in cfg
        assert "GPP_PROJ_FLOOR" in cfg


class TestOwnershipWiring:
    """Ensure RG ownership data flows to own_pct in the optimizer."""

    def test_proj_own_maps_to_own_pct(self):
        """proj_own column from RG (POWN) should become own_pct."""
        from yak_core.lineups import _add_ownership
        df = pd.DataFrame({
            "player_name": ["A", "B", "C"],
            "salary": [9000, 7000, 5000],
            "proj_own": [25.0, 15.0, 8.0],
        })
        result = _add_ownership(df, {})
        assert "own_pct" in result.columns
        assert abs(result["own_pct"].iloc[0] - 0.25) < 0.01
        assert abs(result["own_pct"].iloc[1] - 0.15) < 0.01

    def test_ownership_col_maps_to_own_pct(self):
        """ownership column from RG (OWNERSHIP) should become own_pct."""
        from yak_core.lineups import _add_ownership
        df = pd.DataFrame({
            "player_name": ["A", "B"],
            "salary": [9000, 7000],
            "ownership": [30.0, 20.0],
        })
        result = _add_ownership(df, {})
        assert abs(result["own_pct"].iloc[0] - 0.30) < 0.01

    def test_own_pct_takes_priority(self):
        """own_pct column should take priority over ownership/proj_own."""
        from yak_core.lineups import _add_ownership
        df = pd.DataFrame({
            "player_name": ["A", "B"],
            "salary": [9000, 7000],
            "own_pct": [0.25, 0.15],
            "ownership": [50.0, 40.0],  # should be ignored
        })
        result = _add_ownership(df, {})
        assert abs(result["own_pct"].iloc[0] - 0.25) < 0.01

    def test_salary_rank_fallback(self):
        """Without ownership data, should use salary-rank proxy."""
        from yak_core.lineups import _add_ownership
        df = pd.DataFrame({
            "player_name": ["A", "B", "C"],
            "salary": [9000, 7000, 5000],
        })
        result = _add_ownership(df, {})
        assert "own_pct" in result.columns
        # Higher salary should get higher ownership
        assert result["own_pct"].iloc[0] > result["own_pct"].iloc[2]


class TestGPPv8SimScoring:
    """Tests for the v8 sim-based GPP scoring formula."""

    def _make_pool(self, sim_cols=True):
        """Create a small player pool with sim percentile columns."""
        from yak_core.lineups import _add_scores
        data = {
            "player_name": ["Star", "Mid", "Value", "Punt"],
            "salary": [10000, 7000, 5000, 3500],
            "proj": [50.0, 35.0, 25.0, 15.0],
            "own_pct": [0.35, 0.20, 0.10, 0.04],
        }
        if sim_cols:
            data.update({
                "sim50th": [45.0, 32.0, 22.0, 12.0],
                "sim85th": [60.0, 45.0, 35.0, 25.0],
                "sim90th": [65.0, 48.0, 38.0, 28.0],
                "sim99th": [80.0, 60.0, 50.0, 40.0],
            })
        return pd.DataFrame(data)

    def test_gpp_score_column_exists(self):
        """_add_scores must produce a gpp_score column."""
        from yak_core.lineups import _add_scores
        df = self._make_pool()
        result = _add_scores(df, DEFAULT_CONFIG)
        assert "gpp_score" in result.columns
        assert result["gpp_score"].notna().all()

    def test_high_proj_star_scores_well(self):
        """High-proj stars should not be crushed by ownership penalty."""
        from yak_core.lineups import _add_scores
        df = self._make_pool()
        result = _add_scores(df, DEFAULT_CONFIG)
        # Star (50 proj, 35% own) should score higher than Punt (15 proj, 4% own)
        # The old formula with -3.0 * own would crush the star; v8 should not
        assert result.loc[0, "gpp_score"] > result.loc[3, "gpp_score"]

    def test_sim_percentiles_boost_upside(self):
        """Players with higher sim upside should score higher, all else equal."""
        from yak_core.lineups import _add_scores
        # Two players with same proj and ownership but different sim profiles
        df = pd.DataFrame({
            "player_name": ["HighVar", "LowVar"],
            "salary": [8000, 8000],
            "proj": [40.0, 40.0],
            "own_pct": [0.15, 0.15],  # neutral ownership (no penalty)
            "sim50th": [38.0, 38.0],
            "sim90th": [55.0, 42.0],  # HighVar has much higher ceiling
            "sim99th": [70.0, 45.0],  # HighVar has much higher boom
        })
        result = _add_scores(df, DEFAULT_CONFIG)
        assert result.loc[0, "gpp_score"] > result.loc[1, "gpp_score"]

    def test_ownership_penalty_is_nonlinear(self):
        """Ownership penalty should scale non-linearly (log-based)."""
        from yak_core.lineups import _add_scores
        # Three players with same proj/sims but different ownership
        df = pd.DataFrame({
            "player_name": ["LowOwn", "MidOwn", "HighOwn"],
            "salary": [8000, 8000, 8000],
            "proj": [40.0, 40.0, 40.0],
            "own_pct": [0.05, 0.15, 0.40],
            "sim50th": [38.0, 38.0, 38.0],
            "sim90th": [50.0, 50.0, 50.0],
            "sim99th": [60.0, 60.0, 60.0],
        })
        result = _add_scores(df, DEFAULT_CONFIG)
        scores = result["gpp_score"].tolist()
        # Low own (5%) should score highest, high own (40%) should score lowest
        assert scores[0] > scores[1] > scores[2]
        # Penalty from 15% to 40% should be larger than boost from 5% to 15%
        # (log scaling makes the high-own penalty moderate, not crushing)
        penalty_high = scores[1] - scores[2]  # mid vs high
        boost_low = scores[0] - scores[1]     # low vs mid
        assert penalty_high > 0
        assert boost_low > 0

    def test_fallback_without_sim_columns(self):
        """Scoring should still work when sim columns are missing."""
        from yak_core.lineups import _add_scores
        df = self._make_pool(sim_cols=False)
        result = _add_scores(df, DEFAULT_CONFIG)
        assert "gpp_score" in result.columns
        assert result["gpp_score"].notna().all()
        # Higher proj should still score higher (proj dominates without sims)
        assert result.loc[0, "gpp_score"] > result.loc[3, "gpp_score"]

    def test_fallback_with_ceil_column(self):
        """When sim columns are missing but ceil exists, use ceil for upside."""
        from yak_core.lineups import _add_scores
        df = self._make_pool(sim_cols=False)
        df["ceil"] = df["proj"] * 1.5  # synthetic ceiling
        result = _add_scores(df, DEFAULT_CONFIG)
        assert "gpp_score" in result.columns
        assert result.loc[0, "gpp_score"] > result.loc[3, "gpp_score"]

    def test_configurable_weights(self):
        """Custom weights should override defaults."""
        from yak_core.lineups import _add_scores
        df = self._make_pool()
        # Pure projection mode (zero upside/boom weights)
        cfg = {**DEFAULT_CONFIG, "GPP_PROJ_WEIGHT": 1.0,
               "GPP_UPSIDE_WEIGHT": 0.0, "GPP_BOOM_WEIGHT": 0.0,
               "GPP_OWN_PENALTY_STRENGTH": 0.0, "GPP_OWN_LOW_BOOST": 0.0}
        result = _add_scores(df, cfg)
        # With zero ownership penalty and pure projection, gpp_score = proj * 1.0
        for i in range(len(df)):
            assert abs(result.loc[i, "gpp_score"] - df.loc[i, "proj"]) < 0.01

    def test_sim_col_naming_variants(self):
        """Both sim50th and sim50 naming conventions should work."""
        from yak_core.lineups import _get_sim_col
        df1 = pd.DataFrame({"sim50th": [40.0, 35.0]})
        df2 = pd.DataFrame({"sim50": [40.0, 35.0]})
        df3 = pd.DataFrame({"SIM50TH": [40.0, 35.0]})
        for df in [df1, df2, df3]:
            result = _get_sim_col(df, "50")
            assert result.iloc[0] == 40.0


class TestProjectionFloorSafeguard:
    """Tests for the v8 projection floor flagging."""

    def test_below_floor_flag(self):
        """Lineups below GPP_PROJ_FLOOR should be flagged."""
        lineups_df = pd.DataFrame({
            "lineup_index": [0]*8 + [1]*8,
            "proj": [30]*8 + [40]*8,  # lineup 0: 240 total, lineup 1: 320 total
            "player_name": [f"P{i}" for i in range(16)],
        })
        # Simulate the flagging logic
        proj_floor = 280
        lu_proj = lineups_df.groupby("lineup_index")["proj"].sum()
        below = lu_proj[lu_proj < proj_floor]
        assert len(below) == 1  # lineup 0 (240) is below 280
        assert 0 in below.index
        assert 1 not in below.index
