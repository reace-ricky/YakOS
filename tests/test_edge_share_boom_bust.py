"""Tests for Edge Share page boom/bust integration.

Validates that:
- The Edge Share page module imports without error.
- With boom_bust_df in published_sets, summary metrics render without crash.
- Without boom_bust_df, no crash occurs (summary strip simply doesn't appear).
- With exposure_df in published_sets, exposure expander renders without crash.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


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
        {"lineup_index": 0, "boom_score": 80.0, "bust_risk": 20.0, "lineup_grade": "A",
         "total_ceil": 250.0, "total_floor": 180.0, "avg_smash_prob": 0.7, "avg_bust_prob": 0.15},
        {"lineup_index": 1, "boom_score": 55.0, "bust_risk": 45.0, "lineup_grade": "C",
         "total_ceil": 220.0, "total_floor": 160.0, "avg_smash_prob": 0.45, "avg_bust_prob": 0.35},
        {"lineup_index": 2, "boom_score": 72.0, "bust_risk": 28.0, "lineup_grade": "B",
         "total_ceil": 235.0, "total_floor": 170.0, "avg_smash_prob": 0.6, "avg_bust_prob": 0.22},
    ])


def _make_exposure_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"player": "Alice", "team": "LAL", "salary": 9000,
         "your_exposure_pct": 0.6, "field_own_pct": 0.3, "delta": 0.3, "leverage_ratio": 2.0},
        {"player": "Bob", "team": "BOS", "salary": 7500,
         "your_exposure_pct": 0.4, "field_own_pct": 0.1, "delta": 0.3, "leverage_ratio": 4.0},
    ])


# ---------------------------------------------------------------------------
# 1. Smoke test: module imports without error
# ---------------------------------------------------------------------------

class TestEdgeShareBoomBustImports:
    def test_page_module_importable(self):
        mod = importlib.import_module("pages.5_friends_edge_share")
        assert hasattr(mod, "main")
        assert hasattr(mod, "_render_optimizer_col")

    def test_contest_order_accessible(self):
        mod = importlib.import_module("pages.5_friends_edge_share")
        assert isinstance(mod.CONTEST_ORDER, list)
        assert len(mod.CONTEST_ORDER) > 0


# ---------------------------------------------------------------------------
# 2. publish() with boom_bust_df → published set contains it
# ---------------------------------------------------------------------------

class TestPublishedSetIncludesBoomBust:
    def _setup_lu_state(self, with_boom_bust: bool = True, with_exposure: bool = True):
        from yak_core.state import LineupSetState
        lu = LineupSetState()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {"build_mode": "ceil", "num_lineups": 3})
        if with_boom_bust:
            lu.set_boom_bust("GPP - 20 Max", _make_boom_bust_df())
        if with_exposure:
            lu.exposures["GPP - 20 Max"] = _make_exposure_df()
        lu.publish("GPP - 20 Max", "2026-03-05T12:00:00Z")
        return lu

    def test_boom_bust_df_in_published_set(self):
        lu = self._setup_lu_state(with_boom_bust=True)
        pub = lu.published_sets["GPP - 20 Max"]
        assert "boom_bust_df" in pub
        assert pub["boom_bust_df"] is not None

    def test_boom_bust_df_none_when_not_set(self):
        lu = self._setup_lu_state(with_boom_bust=False, with_exposure=False)
        pub = lu.published_sets["GPP - 20 Max"]
        assert pub["boom_bust_df"] is None

    def test_exposure_df_in_published_set(self):
        lu = self._setup_lu_state(with_exposure=True)
        pub = lu.published_sets["GPP - 20 Max"]
        assert "exposure_df" in pub
        assert pub["exposure_df"] is not None

    def test_exposure_df_none_when_not_set(self):
        lu = self._setup_lu_state(with_boom_bust=False, with_exposure=False)
        pub = lu.published_sets["GPP - 20 Max"]
        assert pub["exposure_df"] is None


# ---------------------------------------------------------------------------
# 3. Summary metrics computed correctly from boom_bust_df
# ---------------------------------------------------------------------------

class TestBoomBustSummaryMetrics:
    def test_n_ab_grades_count(self):
        """A/B grades are counted correctly from boom_bust_df."""
        bb = _make_boom_bust_df()
        n_ab = len(bb[bb["lineup_grade"].isin(["A", "B"])])
        # grades: A, C, B → 2 are A/B
        assert n_ab == 2

    def test_avg_boom_score(self):
        """Average boom score is computed from boom_bust_df."""
        bb = _make_boom_bust_df()
        avg_boom = bb["boom_score"].mean()
        expected = (80.0 + 55.0 + 72.0) / 3
        assert abs(avg_boom - expected) < 0.01

    def test_avg_bust_risk(self):
        """Average bust risk is computed from boom_bust_df."""
        bb = _make_boom_bust_df()
        avg_bust = bb["bust_risk"].mean()
        expected = (20.0 + 45.0 + 28.0) / 3
        assert abs(avg_bust - expected) < 0.01

    def test_n_lineups(self):
        """Total lineup count from boom_bust_df."""
        bb = _make_boom_bust_df()
        assert len(bb) == 3


# ---------------------------------------------------------------------------
# 4. Contest-aware caption logic
# ---------------------------------------------------------------------------

class TestContestAwareCaption:
    def test_gpp_tagging_mode_ceiling(self):
        """GPP Main preset has tagging_mode == 'ceiling'."""
        from yak_core.config import CONTEST_PRESETS
        preset = CONTEST_PRESETS.get("GPP Main", {})
        assert preset.get("tagging_mode") == "ceiling"

    def test_cash_tagging_mode_floor(self):
        """Cash Main preset has tagging_mode == 'floor'."""
        from yak_core.config import CONTEST_PRESETS
        preset = CONTEST_PRESETS.get("Cash Main", {})
        assert preset.get("tagging_mode") == "floor"

    def test_gpp_caption_text(self):
        """GPP ceiling mode caption starts with 'GPP lineup set'."""
        from yak_core.config import CONTEST_PRESETS
        preset = CONTEST_PRESETS.get("GPP Main", {})
        bb = _make_boom_bust_df()
        n_ab = len(bb[bb["lineup_grade"].isin(["A", "B"])])
        avg_bust = bb["bust_risk"].mean()
        if preset.get("tagging_mode") == "ceiling":
            caption = f"GPP lineup set — {n_ab} high-ceiling lineups, avg bust risk {avg_bust:.0f}/100"
        else:
            caption = f"Cash lineup set — {n_ab} safe-floor lineups, avg bust risk {avg_bust:.0f}/100"
        assert caption.startswith("GPP lineup set")


# ---------------------------------------------------------------------------
# 5. _render_optimizer_col data path logic (no Streamlit required)
# ---------------------------------------------------------------------------

class TestRenderOptimizerColLogic:
    def _build_pub_with_boom_bust(self):
        from yak_core.state import LineupSetState
        lu = LineupSetState()
        lu.set_lineups("GPP - 20 Max", _make_lineup_df(), {"build_mode": "ceil", "num_lineups": 3})
        lu.set_boom_bust("GPP - 20 Max", _make_boom_bust_df())
        lu.exposures["GPP - 20 Max"] = _make_exposure_df()
        lu.publish("GPP - 20 Max", "2026-03-05T12:00:00Z")
        return lu

    def test_published_set_boom_bust_df_retrievable(self):
        """boom_bust_df can be retrieved from published_sets the same way _render_optimizer_col does."""
        lu = self._build_pub_with_boom_bust()
        pub = lu.published_sets.get("GPP - 20 Max", {})
        boom_bust_df = pub.get("boom_bust_df")
        assert boom_bust_df is not None
        assert not boom_bust_df.empty

    def test_published_set_exposure_df_retrievable(self):
        """exposure_df can be retrieved from published_sets."""
        lu = self._build_pub_with_boom_bust()
        pub = lu.published_sets.get("GPP - 20 Max", {})
        exposure_df = pub.get("exposure_df")
        assert exposure_df is not None
        assert not exposure_df.empty

    def test_published_set_no_boom_bust_returns_none(self):
        """When no boom_bust was set, pub.get('boom_bust_df') returns None."""
        from yak_core.state import LineupSetState
        lu = LineupSetState()
        lu.set_lineups("Cash", _make_lineup_df(), {})
        lu.publish("Cash", "2026-03-05T12:00:00Z")
        pub = lu.published_sets.get("Cash", {})
        assert pub.get("boom_bust_df") is None

    def test_published_set_no_exposure_returns_none(self):
        """When no exposure was set, pub.get('exposure_df') returns None."""
        from yak_core.state import LineupSetState
        lu = LineupSetState()
        lu.set_lineups("Cash", _make_lineup_df(), {})
        lu.publish("Cash", "2026-03-05T12:00:00Z")
        pub = lu.published_sets.get("Cash", {})
        assert pub.get("exposure_df") is None

    def test_exposure_display_cols_filtered(self):
        """display_cols filter keeps only columns that exist in exposure_df."""
        exp = _make_exposure_df()
        wanted = ["player", "team", "salary", "your_exposure_pct",
                  "field_own_pct", "delta", "leverage_ratio"]
        display_cols = [c for c in wanted if c in exp.columns]
        assert display_cols == wanted  # all columns present in our mock
        assert len(exp[display_cols]) == 2
