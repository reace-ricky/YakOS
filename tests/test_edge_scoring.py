"""Tests for yak_core/edge_scoring.py -- FadeScorer and classify_plays.

Verifies:
- FadeScorer produces fade_score and reasoning columns.
- Fade score targets high-owned, low-ceiling players (not just cheap ones).
- classify_plays() returns the expected 4-box structure.
- Fade candidates always include required fields including 'reasoning'.
- High-value cheap players are NOT auto-faded.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from yak_core.edge_scoring import FadeScorer, classify_plays


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pool(n: int = 20) -> pd.DataFrame:
    """Build a realistic player pool with all standard columns."""
    return pd.DataFrame({
        "player_name": [f"Player{i}" for i in range(n)],
        "pos": ["PG", "SG", "SF", "PF", "C"] * (n // 5) + ["PG"] * (n % 5),
        "team": [f"T{i % 4}" for i in range(n)],
        "salary": [4000 + i * 500 for i in range(n)],
        "proj": [10.0 + i * 2.0 for i in range(n)],
        "ownership": [3.0 + i * 1.5 for i in range(n)],
        "ceil": [15.0 + i * 3.0 for i in range(n)],
        "floor": [6.0 + i * 1.0 for i in range(n)],
        "proj_minutes": [20.0 + i * 0.5 for i in range(n)],
        "sim90th": [18.0 + i * 2.5 for i in range(n)],
        "rolling_fp_5": [9.0 + i * 1.8 for i in range(n)],
        "risk_score": [20.0 + i * 2.0 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# FadeScorer tests
# ---------------------------------------------------------------------------

class TestFadeScorer:

    def test_returns_dataframe_with_fade_score(self):
        pool = _make_pool(10)
        scorer = FadeScorer()
        result = scorer.compute_fade_scores(pool)
        assert isinstance(result, pd.DataFrame)
        assert "fade_score" in result.columns

    def test_returns_dataframe_with_reasoning(self):
        pool = _make_pool(10)
        scorer = FadeScorer()
        result = scorer.compute_fade_scores(pool)
        assert "reasoning" in result.columns
        for reason in result["reasoning"]:
            assert isinstance(reason, str)
            assert len(reason) > 0

    def test_empty_pool_returns_empty_df_with_columns(self):
        scorer = FadeScorer()
        result = scorer.compute_fade_scores(pd.DataFrame())
        assert result.empty
        assert "fade_score" in result.columns
        assert "reasoning" in result.columns

    def test_fade_score_is_float(self):
        pool = _make_pool(10)
        scorer = FadeScorer()
        result = scorer.compute_fade_scores(pool)
        for score in result["fade_score"]:
            assert isinstance(float(score), float)

    def test_high_own_low_ceil_scores_high(self):
        """A chalk trap (high ownership, low ceiling) should have the highest fade score."""
        pool = pd.DataFrame({
            "player_name": ["ChalkTrap", "LowOwn_HighCeil", "Control"],
            "salary":      [7000,         5000,              6000],
            "proj":        [30.0,          20.0,              25.0],
            "ownership":   [30.0,           3.0,              10.0],
            "ceil":        [32.0,          45.0,              40.0],  # chalk trap has very low ceil
            "sim90th":     [33.0,          46.0,              41.0],
        })
        scorer = FadeScorer()
        result = scorer.compute_fade_scores(pool)
        chalk_score = float(result.loc[result["player_name"] == "ChalkTrap", "fade_score"].iloc[0])
        low_own_score = float(result.loc[result["player_name"] == "LowOwn_HighCeil", "fade_score"].iloc[0])
        assert chalk_score > low_own_score, (
            f"ChalkTrap ({chalk_score:.3f}) should score higher than LowOwn_HighCeil ({low_own_score:.3f})"
        )

    def test_high_value_penalises_fade_score(self):
        """High pts/salary value should reduce a player's fade score.

        We isolate the value component by holding ownership and ceiling gap
        constant across the two players; only salary (and therefore pts/salary)
        differs.
        """
        pool = pd.DataFrame({
            "player_name": ["HighValue", "LowValue", "Filler"],
            # Same proj (30) and same ceiling (40) → same ceiling gap
            "salary":      [4000,         8000,        6000],
            "proj":        [30.0,          30.0,        25.0],
            "ownership":   [15.0,          15.0,        10.0],
            "ceil":        [40.0,          40.0,        35.0],
            "sim90th":     [41.0,          41.0,        36.0],
        })
        scorer = FadeScorer()
        result = scorer.compute_fade_scores(pool)
        high_val_score = float(result.loc[result["player_name"] == "HighValue", "fade_score"].iloc[0])
        low_val_score  = float(result.loc[result["player_name"] == "LowValue",  "fade_score"].iloc[0])
        assert high_val_score < low_val_score, (
            f"HighValue ({high_val_score:.3f}) should fade score lower than "
            f"LowValue ({low_val_score:.3f}) because pts/salary penalises fade score"
        )

    def test_reasoning_mentions_ownership_for_chalk_trap(self):
        pool = pd.DataFrame({
            "player_name": ["ChalkTrap"],
            "salary":      [7000],
            "proj":        [30.0],
            "ownership":   [35.0],   # well above threshold
            "ceil":        [32.0],
            "sim90th":     [33.0],
        })
        scorer = FadeScorer()
        result = scorer.compute_fade_scores(pool)
        reason = result["reasoning"].iloc[0]
        # Should mention ownership
        assert "owned" in reason or "ownership" in reason, f"Expected ownership mention: {reason}"

    def test_no_temp_columns_leaked(self):
        """Helper columns used during scoring must not appear in the output."""
        pool = _make_pool(5)
        scorer = FadeScorer()
        result = scorer.compute_fade_scores(pool)
        for col in ["_own_used", "_ceil_used", "_val_used", "_own_z", "_ceil_z", "_value_z"]:
            assert col not in result.columns, f"Temp column '{col}' leaked into output"

    def test_original_columns_preserved(self):
        pool = _make_pool(5)
        scorer = FadeScorer()
        result = scorer.compute_fade_scores(pool)
        for col in pool.columns:
            assert col in result.columns, f"Original column '{col}' missing from result"

    def test_configurable_weights(self):
        """Custom weights should change scores but not crash."""
        pool = _make_pool(10)
        scorer = FadeScorer(w_own=0.6, w_ceil=0.3, w_val=0.1)
        result = scorer.compute_fade_scores(pool)
        assert "fade_score" in result.columns
        assert len(result) == 10


# ---------------------------------------------------------------------------
# classify_plays tests
# ---------------------------------------------------------------------------

class TestClassifyPlays:

    def test_returns_four_buckets(self):
        pool = _make_pool(20)
        classified = classify_plays(pool, sport="NBA")
        for key in ("core_plays", "leverage_plays", "value_plays", "fade_candidates"):
            assert key in classified, f"Missing key: {key}"

    def test_fade_candidates_have_reasoning(self):
        """Every fade candidate must have a 'reasoning' field."""
        pool = _make_pool(20)
        classified = classify_plays(pool, sport="NBA")
        fades = classified.get("fade_candidates", [])
        for f in fades:
            assert "reasoning" in f, f"fade candidate missing 'reasoning': {f}"
            assert isinstance(f["reasoning"], str)
            assert len(f["reasoning"]) > 0

    def test_fade_candidates_have_fade_score(self):
        """Every algorithmic fade candidate must include fade_score."""
        pool = _make_pool(20)
        classified = classify_plays(pool, sport="NBA")
        fades = classified.get("fade_candidates", [])
        for f in fades:
            assert "fade_score" in f, f"fade candidate missing 'fade_score': {f}"

    def test_fade_candidates_have_required_fields(self):
        required = ["player_name", "salary", "proj", "ownership", "own_pct",
                    "ceil", "edge", "value", "risk_score", "reasoning"]
        pool = _make_pool(20)
        classified = classify_plays(pool, sport="NBA")
        for f in classified.get("fade_candidates", []):
            for field in required:
                assert field in f, f"fade candidate missing '{field}': {f}"

    def test_high_value_cheap_player_not_faded(self):
        """A cheap player with exceptional value should NOT be a fade candidate."""
        # Build a pool where the cheap player has outstanding pts/salary
        # and very low ownership — should be value, not fade
        players = []
        for i in range(18):
            players.append({
                "player_name": f"Player{i}",
                "salary": 7000 + i * 200,
                "proj": 30.0 + i,
                "ownership": 10.0 + i,
                "ceil": 45.0 + i * 2,
                "sim90th": 46.0 + i * 2,
                "rolling_fp_5": 28.0 + i,
            })
        # Value gem: cheap, high proj/salary, low ownership, great ceiling
        players.append({
            "player_name": "ValueGem",
            "salary": 4000,
            "proj": 35.0,
            "ownership": 2.0,   # very low ownership — should NOT be faded
            "ceil": 55.0,
            "sim90th": 56.0,
            "rolling_fp_5": 34.0,
        })
        # Chalk trap: expensive, high ownership, low ceiling
        players.append({
            "player_name": "ChalkTrap",
            "salary": 8500,
            "proj": 38.0,
            "ownership": 35.0,  # very high ownership
            "ceil": 40.0,       # barely above projection — low upside
            "sim90th": 41.0,
            "rolling_fp_5": 36.0,
        })
        pool = pd.DataFrame(players)
        classified = classify_plays(pool, sport="NBA")
        fade_names = {f["player_name"] for f in classified["fade_candidates"]}
        assert "ValueGem" not in fade_names, (
            "ValueGem (cheap, high-value, low-ownership) should not be faded"
        )

    def test_chalk_trap_is_faded(self):
        """A high-owned player with minimal upside should appear as a fade."""
        players = []
        for i in range(15):
            players.append({
                "player_name": f"Filler{i}",
                "salary": 5000 + i * 300,
                "proj": 20.0 + i,
                "ownership": 5.0 + i * 0.5,
                "ceil": 30.0 + i * 2,
                "sim90th": 31.0 + i * 2,
            })
        players.append({
            "player_name": "ChalkTrap",
            "salary": 8000,
            "proj": 32.0,
            "ownership": 40.0,   # highest ownership
            "ceil": 33.0,        # almost no upside over projection
            "sim90th": 34.0,
        })
        pool = pd.DataFrame(players)
        classified = classify_plays(pool, sport="NBA")
        fade_names = {f["player_name"] for f in classified["fade_candidates"]}
        assert "ChalkTrap" in fade_names, (
            "ChalkTrap (high-owned, no upside) should be a fade candidate"
        )

    def test_no_overlap_between_tiers(self):
        """A player should not appear in multiple tiers."""
        pool = _make_pool(20)
        classified = classify_plays(pool, sport="NBA")
        all_names: list[str] = []
        for key in ("core_plays", "leverage_plays", "value_plays", "fade_candidates"):
            for p in classified[key]:
                all_names.append(p["player_name"])
        assert len(all_names) == len(set(all_names)), "Player appears in multiple tiers"

    def test_all_players_have_player_name(self):
        pool = _make_pool(20)
        classified = classify_plays(pool, sport="NBA")
        for key in ("core_plays", "leverage_plays", "value_plays", "fade_candidates"):
            for p in classified[key]:
                assert p.get("player_name", ""), f"Empty player_name in {key}: {p}"

    def test_ownership_and_own_pct_both_present(self):
        pool = _make_pool(20)
        classified = classify_plays(pool, sport="NBA")
        for key in ("core_plays", "leverage_plays", "value_plays", "fade_candidates"):
            for p in classified[key]:
                assert "ownership" in p, f"'ownership' missing in {key}: {p}"
                assert "own_pct" in p, f"'own_pct' missing in {key}: {p}"
                assert p["ownership"] == p["own_pct"], (
                    f"ownership != own_pct in {key}: {p}"
                )

    def test_proj_nonzero_when_pool_has_projections(self):
        pool = _make_pool(20)
        classified = classify_plays(pool, sport="NBA")
        for key in ("core_plays", "leverage_plays", "value_plays"):
            for p in classified[key]:
                assert p["proj"] > 0, f"proj is zero in {key}: {p}"

    def test_empty_pool_returns_empty_buckets(self):
        classified = classify_plays(pd.DataFrame(), sport="NBA")
        for key in ("core_plays", "leverage_plays", "value_plays", "fade_candidates"):
            assert classified[key] == [], f"Expected empty list for {key}"
