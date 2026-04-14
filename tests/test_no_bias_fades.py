"""Tests verifying that fades are never derived from ricky_bias.json max_exposure.

Requirements tested:
  1. classify_plays() in edge_scoring.py does NOT inject bias max_exposure=0.0 entries
     into fade_candidates — core players stay in core regardless of bias state.
  2. A player with max_exposure=0.0 in bias but high projection is NOT in fade_candidates
     when classify_plays() runs.
  3. A player classified as core never appears in fade_candidates from classify_plays().
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from yak_core.edge_scoring import classify_plays


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(n: int = 20, core_name: str = "LaMelo Ball") -> pd.DataFrame:
    """Build a minimal player pool where ``core_name`` is the top projection."""
    names = [core_name] + [f"Player{i}" for i in range(n - 1)]
    projs = [55.0] + [15.0 + i * 0.8 for i in range(n - 1)]
    ownerships = [18.0] + [5.0 + i * 1.2 for i in range(n - 1)]
    salaries = [9800] + [4000 + i * 200 for i in range(n - 1)]
    # Vary rolling_fp_5 so risk scores differ (core player has good recent form)
    rolling = [projs[0] * 0.95] + [projs[i + 1] * (0.5 + 0.05 * i) for i in range(n - 1)]
    return pd.DataFrame({
        "player_name": names,
        "pos": (["PG", "SG", "SF", "PF", "C"] * ((n // 5) + 1))[:n],
        "team": [f"T{i % 4}" for i in range(n)],
        "salary": salaries,
        "proj": projs,
        "ceil": [p * 1.3 for p in projs],
        "floor": [p * 0.7 for p in projs],
        "ownership": ownerships,
        "own_pct": ownerships,
        "value": [p / s * 1000 for p, s in zip(projs, salaries)],
        "edge_score": [p - 20 for p in projs],
        "proj_minutes": [35.0] + [25.0] * (n - 1),
        "sim90th": [p * 1.4 for p in projs],
        "risk_score": [1.0] * n,
        "rolling_fp_5": rolling,
        "spread": [0.0] * n,
        "blowout_risk": [0.0] * n,
        "dvp_rank": [10.0] * n,
    })


# ---------------------------------------------------------------------------
# 1. classify_plays does not use bias max_exposure=0.0
# ---------------------------------------------------------------------------

class TestClassifyPlaysNoBiasFades:
    def test_core_player_not_in_fades_even_with_bias_max_exposure_zero(self, tmp_path):
        """Player with max_exposure=0.0 in bias must NOT appear in fade_candidates
        if classify_plays() ranks them as a core play."""
        core_name = "LaMelo Ball"
        pool = _make_pool(core_name=core_name)

        # Write a bias file with max_exposure=0.0 for the core player
        bias_data = {core_name: {"max_exposure": 0.0}}
        bias_file = tmp_path / "ricky_bias.json"
        bias_file.write_text(json.dumps(bias_data))

        # Patch load_bias to return our polluted bias (simulates old persisted state)
        def _fake_load_bias():
            return bias_data

        with patch("yak_core.bias.load_bias", side_effect=_fake_load_bias):
            result = classify_plays(pool, sport="NBA")

        fade_names = {f["player_name"] for f in result.get("fade_candidates", [])}
        assert core_name not in fade_names, (
            f"{core_name} should not be in fade_candidates even though "
            f"bias has max_exposure=0.0 — fades must not be derived from bias"
        )

    def test_core_player_in_core_plays_regardless_of_bias(self, tmp_path):
        """The top-projected player must appear in core_plays even if bias fades them."""
        core_name = "LaMelo Ball"
        pool = _make_pool(core_name=core_name)
        bias_data = {core_name: {"max_exposure": 0.0}}

        def _fake_load_bias():
            return bias_data

        with patch("yak_core.bias.load_bias", side_effect=_fake_load_bias):
            result = classify_plays(pool, sport="NBA")

        core_names = {p["player_name"] for p in result.get("core_plays", [])}
        assert core_name in core_names, (
            f"{core_name} must appear in core_plays regardless of bias max_exposure"
        )

    def test_fade_candidates_all_algorithmic(self):
        """All fade_candidates must have reasoning NOT equal to 'Manual fade'
        (manual fades are injected at render time, not in classify_plays)."""
        pool = _make_pool()
        result = classify_plays(pool, sport="NBA")
        for fade in result.get("fade_candidates", []):
            assert fade.get("reasoning") != "Manual fade", (
                f"classify_plays should not produce 'Manual fade' entries — "
                f"found: {fade}"
            )

    def test_no_player_in_both_core_and_fade(self):
        """No player should appear in both core_plays and fade_candidates."""
        pool = _make_pool()
        result = classify_plays(pool, sport="NBA")
        core_names = {p["player_name"] for p in result.get("core_plays", [])}
        fade_names = {f["player_name"] for f in result.get("fade_candidates", [])}
        overlap = core_names & fade_names
        assert not overlap, (
            f"Players in both core and fade: {overlap} — "
            f"this causes them to be stripped from the Core box on The Board"
        )
