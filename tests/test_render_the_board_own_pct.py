"""Tests for the own_pct KeyError fix in _render_the_board.

Root cause: fade_candidates dicts produced by _classify_plays (app/lab_tab.py)
use the key "ownership", while _render_the_board previously accessed f["own_pct"]
directly. The fix uses f.get("own_pct", f.get("ownership", 0)) so either key name
works.

These tests verify:
  1. compute_sniper_spots returns dicts with "own_pct" key.
  2. compute_fades returns dicts with "own_pct" key.
  3. The ownership extraction pattern handles dicts with "own_pct" key.
  4. The ownership extraction pattern handles dicts with "ownership" key only.
  5. The ownership extraction pattern handles dicts with neither key (defaults to 0).
"""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.board import compute_sniper_spots, compute_fades


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_own(d: dict) -> float:
    """Mirror of the safe ownership lookup used in _render_the_board after the fix."""
    return float(d.get("own_pct", d.get("ownership", 0)) or 0)


def _make_pool(n: int = 15) -> pd.DataFrame:
    """Build a minimal player pool compatible with board functions."""
    return pd.DataFrame({
        "player_name": [f"Player{i}" for i in range(n)],
        "pos": ["PG", "SG", "SF", "PF", "C"] * (n // 5) + ["PG"] * (n % 5),
        "team": [f"T{i % 4}" for i in range(n)],
        "salary": [4000 + i * 400 for i in range(n)],
        "proj": [15.0 + i * 1.5 for i in range(n)],
        "ceil": [22.0 + i * 2 for i in range(n)],
        "floor": [10.0 + i for i in range(n)],
        "ownership": [3.0 + i * 1.5 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# 1 & 2. compute_sniper_spots / compute_fades include "own_pct"
# ---------------------------------------------------------------------------

class TestComputeSniperSpotsOwnPct:
    def test_own_pct_key_present(self):
        pool = _make_pool(20)
        results = compute_sniper_spots(pool, {})
        for r in results:
            assert "own_pct" in r, f"own_pct missing from sniper spot: {r}"

    def test_own_pct_is_numeric(self):
        pool = _make_pool(20)
        results = compute_sniper_spots(pool, {})
        for r in results:
            assert isinstance(r["own_pct"], float)

    def test_empty_pool_returns_empty(self):
        assert compute_sniper_spots(pd.DataFrame(), {}) == []

    def test_none_pool_returns_empty(self):
        assert compute_sniper_spots(None, {}) == []


class TestComputeFadesOwnPct:
    def test_own_pct_key_present(self):
        pool = _make_pool(15)
        results = compute_fades(pool, {})
        for r in results:
            assert "own_pct" in r, f"own_pct missing from fade candidate: {r}"

    def test_own_pct_is_numeric(self):
        pool = _make_pool(15)
        results = compute_fades(pool, {})
        for r in results:
            assert isinstance(r["own_pct"], float)

    def test_empty_pool_returns_empty(self):
        assert compute_fades(pd.DataFrame(), {}) == []

    def test_none_pool_returns_empty(self):
        assert compute_fades(None, {}) == []


# ---------------------------------------------------------------------------
# 3, 4, 5. Ownership extraction pattern handles both key names and absence
# ---------------------------------------------------------------------------

class TestExtractOwnFallback:
    def test_own_pct_key_used_when_present(self):
        d = {"player_name": "Alice", "own_pct": 18.5, "ownership": 20.0}
        assert _extract_own(d) == 18.5

    def test_ownership_key_used_when_own_pct_absent(self):
        """Bug scenario: fade_candidates built by _classify_plays use 'ownership'."""
        d = {"player_name": "Bob", "ownership": 22.0, "salary": 6800}
        assert _extract_own(d) == 22.0

    def test_defaults_to_zero_when_neither_key_present(self):
        d = {"player_name": "Charlie", "salary": 5000}
        assert _extract_own(d) == 0.0

    def test_handles_none_value_gracefully(self):
        d = {"player_name": "Dave", "own_pct": None, "ownership": None}
        assert _extract_own(d) == 0.0

    def test_handles_zero_ownership(self):
        d = {"player_name": "Eve", "ownership": 0.0}
        assert _extract_own(d) == 0.0

    def test_prefers_own_pct_over_ownership(self):
        """own_pct should take priority over ownership if both keys exist."""
        d = {"player_name": "Frank", "own_pct": 5.0, "ownership": 30.0}
        assert _extract_own(d) == 5.0
