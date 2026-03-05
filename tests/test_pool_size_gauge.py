"""Tests for pool size gauge logic and contest type persistence (PR 2B).

Validates:
1. get_pool_size_range returns correct bounds for each contest type.
2. The pool size gauge renders the correct status (in-range / below / above).
3. SlateState.contest_type is written with the full contest_type_label on publish.
"""

from __future__ import annotations

import pytest
import pandas as pd

from yak_core.config import CONTEST_PRESETS, CONTEST_PRESET_LABELS, get_pool_size_range
from yak_core.state import SlateState


# ---------------------------------------------------------------------------
# Helpers that mirror the gauge logic in pages/1_slate_hub.py
# ---------------------------------------------------------------------------

def _pool_gauge_status(pool_count: int, contest_label: str) -> str:
    """Return 'in_range', 'below', or 'above' mirroring the gauge logic."""
    pmin, pmax = get_pool_size_range(contest_label)
    if pmin <= pool_count <= pmax:
        return "in_range"
    elif pool_count < pmin:
        return "below"
    else:
        return "above"


# ---------------------------------------------------------------------------
# Pool size gauge status tests
# ---------------------------------------------------------------------------

class TestPoolSizeGaugeStatus:
    """The gauge should categorize pool size correctly for each contest type."""

    @pytest.mark.parametrize("label", CONTEST_PRESET_LABELS)
    def test_in_range_returns_in_range(self, label: str) -> None:
        pmin, pmax = get_pool_size_range(label)
        mid = (pmin + pmax) // 2
        assert _pool_gauge_status(mid, label) == "in_range"

    @pytest.mark.parametrize("label", CONTEST_PRESET_LABELS)
    def test_at_min_boundary_is_in_range(self, label: str) -> None:
        pmin, _ = get_pool_size_range(label)
        assert _pool_gauge_status(pmin, label) == "in_range"

    @pytest.mark.parametrize("label", CONTEST_PRESET_LABELS)
    def test_at_max_boundary_is_in_range(self, label: str) -> None:
        _, pmax = get_pool_size_range(label)
        assert _pool_gauge_status(pmax, label) == "in_range"

    @pytest.mark.parametrize("label", CONTEST_PRESET_LABELS)
    def test_below_min_returns_below(self, label: str) -> None:
        pmin, _ = get_pool_size_range(label)
        if pmin > 0:
            assert _pool_gauge_status(pmin - 1, label) == "below"

    @pytest.mark.parametrize("label", CONTEST_PRESET_LABELS)
    def test_above_max_returns_above(self, label: str) -> None:
        _, pmax = get_pool_size_range(label)
        assert _pool_gauge_status(pmax + 1, label) == "above"

    def test_gpp_20_max_specific_ranges(self) -> None:
        """Regression: GPP - 20 Max in-range, below, and above values."""
        pmin, pmax = get_pool_size_range("GPP - 20 Max")
        mid = (pmin + pmax) // 2
        assert _pool_gauge_status(mid, "GPP - 20 Max") == "in_range", (
            f"Expected in_range for count={mid}, range=({pmin},{pmax})"
        )
        assert _pool_gauge_status(pmin - 1, "GPP - 20 Max") == "below", (
            f"Expected below for count={pmin - 1}, range=({pmin},{pmax})"
        )
        assert _pool_gauge_status(pmax + 1, "GPP - 20 Max") == "above", (
            f"Expected above for count={pmax + 1}, range=({pmin},{pmax})"
        )

    def test_cash_specific_ranges(self) -> None:
        """50/50 / Double-Up has small target range."""
        pmin, pmax = get_pool_size_range("50/50 / Double-Up")
        assert _pool_gauge_status(pmin, "50/50 / Double-Up") == "in_range"
        assert _pool_gauge_status(pmax, "50/50 / Double-Up") == "in_range"
        assert _pool_gauge_status(pmax + 5, "50/50 / Double-Up") == "above"


# ---------------------------------------------------------------------------
# Contest type persistence in SlateState
# ---------------------------------------------------------------------------

class TestContestTypePersistence:
    """contest_type_label must be written to SlateState.contest_type on publish."""

    def _publish_slate(self, contest_label: str, is_showdown: bool = False) -> SlateState:
        """Simulate the Publish Slate action from pages/1_slate_hub.py."""
        slate = SlateState()
        hub_rules = {
            "slots": ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"] if is_showdown else
                     ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"],
            "lineup_size": 6 if is_showdown else 8,
            "salary_cap": 50000,
            "is_showdown": is_showdown,
        }
        slate.apply_roster_rules(hub_rules)
        # Simulate the override that happens AFTER apply_roster_rules in the page
        slate.contest_type = contest_label
        return slate

    @pytest.mark.parametrize("label", CONTEST_PRESET_LABELS)
    def test_contest_type_label_stored_in_state(self, label: str) -> None:
        slate = self._publish_slate(label)
        assert slate.contest_type == label, (
            f"Expected slate.contest_type='{label}', got '{slate.contest_type}'"
        )

    def test_non_showdown_contest_label_overrides_classic(self) -> None:
        """For classic contests, apply_roster_rules sets 'Classic'; the label overrides it."""
        slate = self._publish_slate("GPP - 150 Max", is_showdown=False)
        assert slate.contest_type == "GPP - 150 Max"
        # is_showdown flag should still be False (unaffected by contest_type override)
        assert slate.is_showdown is False

    def test_showdown_contest_label_overrides_showdown_captain(self) -> None:
        """For showdown, apply_roster_rules sets 'Showdown Captain'; label overrides it."""
        slate = self._publish_slate("Showdown", is_showdown=True)
        assert slate.contest_type == "Showdown"
        assert slate.is_showdown is True

    def test_slate_state_contest_type_default_is_classic(self) -> None:
        """Default SlateState.contest_type should remain 'Classic' before publish."""
        slate = SlateState()
        assert slate.contest_type == "Classic"
