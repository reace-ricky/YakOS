"""Tests for player pool deduplication and numeric rounding in slate_hub.

Validates the two pool-cleanup steps added to pages/1_slate_hub.py:
  1. Deduplication by dk_player_id (or player_name fallback), keeping highest proj.
  2. Float columns rounded to 1 decimal place before display.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers that mirror the logic in 1_slate_hub.py so we can unit-test it.
# ---------------------------------------------------------------------------

def _deduplicate_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Dedup by dk_player_id (or player_name), keeping highest-proj row."""
    dedup_key = "dk_player_id" if "dk_player_id" in pool.columns else "player_name"
    pool = pool.sort_values("proj", ascending=False)
    pool = pool.drop_duplicates(subset=[dedup_key], keep="first")
    return pool.reset_index(drop=True)


_FLOAT_PREVIEW_COLS = ["proj", "floor", "ceil", "proj_minutes", "ownership", "actual_fp"]


def _round_preview(df: pd.DataFrame) -> pd.DataFrame:
    """Round float preview columns to 1 decimal place."""
    df = df.copy()
    float_cols = [c for c in _FLOAT_PREVIEW_COLS if c in df.columns]
    df[float_cols] = df[float_cols].round(1)
    return df


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplicatePool:
    def test_no_duplicates_unchanged(self):
        pool = pd.DataFrame({
            "dk_player_id": ["1", "2", "3"],
            "player_name": ["A", "B", "C"],
            "proj": [20.0, 18.0, 15.0],
        })
        result = _deduplicate_pool(pool)
        assert len(result) == 3

    def test_dedup_keeps_highest_proj(self):
        pool = pd.DataFrame({
            "dk_player_id": ["1", "1", "2"],
            "player_name": ["A", "A", "B"],
            "proj": [18.0, 22.0, 15.0],  # second row for player 1 has higher proj
        })
        result = _deduplicate_pool(pool)
        assert len(result) == 2
        player1_row = result[result["dk_player_id"] == "1"]
        assert player1_row["proj"].iloc[0] == 22.0

    def test_dedup_fallback_to_player_name(self):
        """When dk_player_id is absent, dedup should use player_name."""
        pool = pd.DataFrame({
            "player_name": ["A", "A", "B"],
            "proj": [10.0, 14.0, 8.0],
        })
        result = _deduplicate_pool(pool)
        assert len(result) == 2
        a_row = result[result["player_name"] == "A"]
        assert a_row["proj"].iloc[0] == 14.0

    def test_dedup_result_is_reset_index(self):
        pool = pd.DataFrame({
            "dk_player_id": ["1", "1"],
            "player_name": ["A", "A"],
            "proj": [5.0, 9.0],
        })
        result = _deduplicate_pool(pool)
        assert list(result.index) == [0]

    def test_empty_pool_returns_empty(self):
        pool = pd.DataFrame(columns=["dk_player_id", "player_name", "proj"])
        result = _deduplicate_pool(pool)
        assert result.empty


# ---------------------------------------------------------------------------
# Rounding tests
# ---------------------------------------------------------------------------

class TestRoundPreview:
    def test_rounds_to_one_decimal(self):
        df = pd.DataFrame({
            "player_name": ["A"],
            "proj": [24.5678],
            "floor": [18.3333],
            "ceil": [31.999],
            "proj_minutes": [34.12345],
            "ownership": [0.2567],
        })
        result = _round_preview(df)
        assert result["proj"].iloc[0] == 24.6
        assert result["floor"].iloc[0] == 18.3
        assert result["ceil"].iloc[0] == 32.0
        assert result["proj_minutes"].iloc[0] == 34.1
        assert result["ownership"].iloc[0] == 0.3

    def test_skips_missing_float_cols(self):
        """Only present float columns are rounded; no KeyError for absent ones."""
        df = pd.DataFrame({
            "player_name": ["A"],
            "proj": [20.555],
        })
        result = _round_preview(df)
        assert result["proj"].iloc[0] == 20.6
        # actual_fp, floor, ceil, etc. were not in df — no error

    def test_non_float_cols_unchanged(self):
        df = pd.DataFrame({
            "player_name": ["A"],
            "salary": [7500],
            "proj": [22.345],
        })
        result = _round_preview(df)
        assert result["salary"].iloc[0] == 7500
        assert result["proj"].iloc[0] == 22.3

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"proj": [19.999]})
        _ = _round_preview(df)
        assert df["proj"].iloc[0] == 19.999  # original unchanged
