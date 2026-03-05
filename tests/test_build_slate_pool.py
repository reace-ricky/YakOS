"""Tests for yak_core/lineups.py – build_slate_pool."""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.lineups import build_slate_pool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_slate_players(ids: list) -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": [str(i) for i in ids],
        "player_name": [f"Player{i}" for i in ids],
        "pos": ["PG"] * len(ids),
        "salary": [5000 + i * 100 for i in range(len(ids))],
        "team": ["LAL"] * len(ids),
    })


def _make_opt_pool(ids: list) -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": [str(i) for i in ids],
        "player_name": [f"Player{i}" for i in ids],
        "pos": ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"][:len(ids)]
                + ["PG"] * max(0, len(ids) - 8),
        "salary": [5000 + i * 100 for i in range(len(ids))],
        "team": ["LAL"] * len(ids),
        "proj": [20.0 + i for i in range(len(ids))],
        "opponent": ["BOS"] * len(ids),
    })


# ---------------------------------------------------------------------------
# build_slate_pool
# ---------------------------------------------------------------------------

class TestBuildSlatePool:
    _BASE_CFG = {
        "PROJ_COL": "proj",
        "SLATE_DATE": "2026-03-05",
        "SITE": "DK",
        "SPORT": "NBA",
        "CONTEST_TYPE": "gpp",
    }

    def test_only_slate_players_included(self):
        """Players not in slate_players must be excluded from the pool."""
        slate_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        extra_ids = [9, 10]  # NOT on slate
        opt_pool = _make_opt_pool(slate_ids + extra_ids)
        slate_players = _make_slate_players(slate_ids)

        result = build_slate_pool(opt_pool, slate_players, self._BASE_CFG)
        result_ids = set(result["player_id"].astype(str))
        slate_id_strs = {str(i) for i in slate_ids}
        assert result_ids.issubset(slate_id_strs), (
            f"Players not on slate found in pool: {result_ids - slate_id_strs}"
        )

    def test_off_day_player_excluded(self):
        """The 'Shai on off day' scenario: player in opt_pool but not in slate_players."""
        slate_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        shai_id = 99  # Off-day player
        opt_pool = _make_opt_pool(slate_ids + [shai_id])
        slate_players = _make_slate_players(slate_ids)

        result = build_slate_pool(opt_pool, slate_players, self._BASE_CFG)
        assert str(shai_id) not in result["player_id"].astype(str).tolist()

    def test_inner_join_assertion_passes_when_clean(self):
        """No AssertionError when pool only contains slate players."""
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        opt_pool = _make_opt_pool(ids)
        slate_players = _make_slate_players(ids)
        # Should not raise
        result = build_slate_pool(opt_pool, slate_players, self._BASE_CFG)
        assert len(result) > 0

    def test_missing_player_id_in_opt_pool_raises(self):
        opt_pool = pd.DataFrame({"player_name": ["X"], "proj": [20.0]})
        slate_players = _make_slate_players([1])
        with pytest.raises(ValueError, match="player_id"):
            build_slate_pool(opt_pool, slate_players, self._BASE_CFG)

    def test_missing_player_id_in_slate_raises(self):
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        opt_pool = _make_opt_pool(ids)
        slate_players = pd.DataFrame({"player_name": ["X"]})  # No player_id
        with pytest.raises(ValueError, match="player_id"):
            build_slate_pool(opt_pool, slate_players, self._BASE_CFG)

    def test_no_minutes_filter_applied(self):
        """Pre-slate: proj_minutes should NOT be used to filter players."""
        ids = list(range(1, 9))
        opt_pool = _make_opt_pool(ids)
        opt_pool["proj_minutes"] = 0.0  # Would filter everyone if minutes-filter were applied
        slate_players = _make_slate_players(ids)

        result = build_slate_pool(opt_pool, slate_players, self._BASE_CFG)
        # All slate players should still be in pool (proj_minutes not used here)
        assert len(result) == len(ids)

    def test_delegated_normalisation_from_build_player_pool(self):
        """build_slate_pool delegates normalisation to build_player_pool."""
        ids = list(range(1, 9))
        opt_pool = _make_opt_pool(ids)
        slate_players = _make_slate_players(ids)

        result = build_slate_pool(opt_pool, slate_players, self._BASE_CFG)
        # Standard columns from build_player_pool should be present
        assert "player_id" in result.columns
        assert "player_name" in result.columns
        assert "proj" in result.columns
        assert "salary" in result.columns
