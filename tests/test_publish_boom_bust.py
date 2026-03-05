"""Tests for LineupSetState.publish() with boom/bust and exposure data.

Validates that boom_bust_df and exposure_df are correctly included (or
excluded) when lineups are published to Edge Share.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import LineupSetState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lineup_df() -> pd.DataFrame:
    rows = []
    for i, slot in enumerate(["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]):
        rows.append({"lineup_index": 0, "slot": slot, "player_name": f"P{i}", "salary": 5000})
    return pd.DataFrame(rows)


def _make_boom_bust_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"lineup_index": 0, "boom_score": 80.0, "bust_risk": 20.0,
         "lineup_grade": "A", "total_ceil": 250.0, "total_floor": 180.0,
         "avg_smash_prob": 0.7, "avg_bust_prob": 0.15},
        {"lineup_index": 1, "boom_score": 60.0, "bust_risk": 40.0,
         "lineup_grade": "C", "total_ceil": 220.0, "total_floor": 160.0,
         "avg_smash_prob": 0.5, "avg_bust_prob": 0.30},
    ])


def _make_exposure_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"player": "Alice", "team": "LAL", "salary": 9000,
         "your_exposure_pct": 0.6, "field_own_pct": 0.3, "delta": 0.3, "leverage_ratio": 2.0},
        {"player": "Bob", "team": "BOS", "salary": 7500,
         "your_exposure_pct": 0.4, "field_own_pct": 0.1, "delta": 0.3, "leverage_ratio": 4.0},
    ])


# ---------------------------------------------------------------------------
# publish() with boom_bust_rankings populated
# ---------------------------------------------------------------------------

class TestPublishWithBoomBust:
    def test_published_set_has_boom_bust_df_key(self):
        lu = LineupSetState()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {})
        lu.set_boom_bust("GPP - 20 Max", _make_boom_bust_df())
        lu.publish("GPP - 20 Max", "2026-03-05T10:00:00Z")

        pub = lu.published_sets["GPP - 20 Max"]
        assert "boom_bust_df" in pub

    def test_boom_bust_df_is_not_none(self):
        lu = LineupSetState()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {})
        lu.set_boom_bust("GPP - 20 Max", _make_boom_bust_df())
        lu.publish("GPP - 20 Max", "2026-03-05T10:00:00Z")

        assert lu.published_sets["GPP - 20 Max"]["boom_bust_df"] is not None

    def test_boom_bust_df_is_a_copy(self):
        """Modifying the original boom_bust DataFrame should not affect published copy."""
        lu = LineupSetState()
        original = _make_boom_bust_df()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {})
        lu.set_boom_bust("GPP - 20 Max", original)
        lu.publish("GPP - 20 Max", "2026-03-05T10:00:00Z")

        # Mutate the original
        original.loc[0, "boom_score"] = 999.0

        published_score = lu.published_sets["GPP - 20 Max"]["boom_bust_df"].loc[0, "boom_score"]
        assert published_score != 999.0

    def test_boom_bust_df_content_matches(self):
        lu = LineupSetState()
        bb = _make_boom_bust_df()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {})
        lu.set_boom_bust("GPP - 20 Max", bb)
        lu.publish("GPP - 20 Max", "2026-03-05T10:00:00Z")

        pub_bb = lu.published_sets["GPP - 20 Max"]["boom_bust_df"]
        assert len(pub_bb) == 2
        assert list(pub_bb["lineup_grade"]) == ["A", "C"]


# ---------------------------------------------------------------------------
# publish() without boom_bust_rankings
# ---------------------------------------------------------------------------

class TestPublishWithoutBoomBust:
    def test_boom_bust_df_is_none_when_no_rankings(self):
        lu = LineupSetState()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {})
        lu.publish("GPP - 20 Max", "2026-03-05T10:00:00Z")

        assert lu.published_sets["GPP - 20 Max"]["boom_bust_df"] is None

    def test_boom_bust_df_key_present_even_without_rankings(self):
        lu = LineupSetState()
        lu.set_lineups("Cash", _make_lineup_df(), {})
        lu.publish("Cash", "2026-03-05T10:00:00Z")

        assert "boom_bust_df" in lu.published_sets["Cash"]

    def test_boom_bust_none_when_rankings_key_absent(self):
        """Contest with no boom_bust_rankings entry → published boom_bust_df is None."""
        lu = LineupSetState()
        lu.set_lineups("Showdown", _make_lineup_df(), {})
        # Set boom_bust for a different contest, not Showdown
        lu.set_boom_bust("GPP - 20 Max", _make_boom_bust_df())
        lu.publish("Showdown", "2026-03-05T10:00:00Z")

        assert lu.published_sets["Showdown"]["boom_bust_df"] is None


# ---------------------------------------------------------------------------
# publish() with exposures populated
# ---------------------------------------------------------------------------

class TestPublishWithExposures:
    def test_published_set_has_exposure_df_key(self):
        lu = LineupSetState()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {})
        lu.exposures["GPP - 20 Max"] = _make_exposure_df()
        lu.publish("GPP - 20 Max", "2026-03-05T10:00:00Z")

        pub = lu.published_sets["GPP - 20 Max"]
        assert "exposure_df" in pub

    def test_exposure_df_is_not_none(self):
        lu = LineupSetState()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {})
        lu.exposures["GPP - 20 Max"] = _make_exposure_df()
        lu.publish("GPP - 20 Max", "2026-03-05T10:00:00Z")

        assert lu.published_sets["GPP - 20 Max"]["exposure_df"] is not None

    def test_exposure_df_is_a_copy(self):
        """Modifying the original exposure DataFrame should not affect published copy."""
        lu = LineupSetState()
        original = _make_exposure_df()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {})
        lu.exposures["GPP - 20 Max"] = original
        lu.publish("GPP - 20 Max", "2026-03-05T10:00:00Z")

        original.loc[0, "your_exposure_pct"] = 0.99

        pub_val = lu.published_sets["GPP - 20 Max"]["exposure_df"].loc[0, "your_exposure_pct"]
        assert pub_val != 0.99


# ---------------------------------------------------------------------------
# publish() without exposures
# ---------------------------------------------------------------------------

class TestPublishWithoutExposures:
    def test_exposure_df_is_none_when_no_exposures(self):
        lu = LineupSetState()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {})
        lu.publish("GPP - 20 Max", "2026-03-05T10:00:00Z")

        assert lu.published_sets["GPP - 20 Max"]["exposure_df"] is None

    def test_exposure_df_key_present_even_without_exposures(self):
        lu = LineupSetState()
        lu.set_lineups("Cash", _make_lineup_df(), {})
        lu.publish("Cash", "2026-03-05T10:00:00Z")

        assert "exposure_df" in lu.published_sets["Cash"]
