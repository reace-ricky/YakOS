"""Tests for CONTEST_PRESETS pool sizing + methodology rules (PR 2A)."""
import pytest
from yak_core.config import (
    CONTEST_PRESETS,
    get_pool_size_range,
    get_methodology_rules,
)

# Keys that every preset must expose after PR 2A
_REQUIRED_METHODOLOGY_KEYS = [
    "pool_size_min",
    "pool_size_max",
    "tagging_mode",
    "show_leverage",
    "eat_chalk",
    "target_avg_ownership_min",
    "target_avg_ownership_max",
    "ownership_caps_by_tier",
    "not_with_auto",
    "max_per_team",
    "exposure_rules",
]


class TestAllPresetsHaveNewKeys:
    """Every preset must contain all new methodology keys — no KeyError."""

    @pytest.mark.parametrize("label", list(CONTEST_PRESETS.keys()))
    def test_preset_has_all_methodology_keys(self, label: str) -> None:
        preset = CONTEST_PRESETS[label]
        for key in _REQUIRED_METHODOLOGY_KEYS:
            assert key in preset, f"Preset '{label}' is missing key '{key}'"


class TestGetPoolSizeRange:
    """get_pool_size_range returns a tuple of two ints with min < max."""

    @pytest.mark.parametrize("label", list(CONTEST_PRESETS.keys()))
    def test_returns_two_int_tuple(self, label: str) -> None:
        result = get_pool_size_range(label)
        assert isinstance(result, tuple), f"Expected tuple for '{label}', got {type(result)}"
        assert len(result) == 2, f"Expected 2-element tuple for '{label}', got {len(result)}"
        lo, hi = result
        assert isinstance(lo, int), f"pool_size_min must be int for '{label}'"
        assert isinstance(hi, int), f"pool_size_max must be int for '{label}'"

    @pytest.mark.parametrize("label", list(CONTEST_PRESETS.keys()))
    def test_min_less_than_max(self, label: str) -> None:
        lo, hi = get_pool_size_range(label)
        assert lo < hi, f"pool_size_min ({lo}) must be < pool_size_max ({hi}) for '{label}'"

    def test_gpp_20_max_values(self) -> None:
        assert get_pool_size_range("GPP Early") == (25, 45)

    def test_invalid_label_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown contest label"):
            get_pool_size_range("Not A Real Contest")


class TestCashPresetRules:
    """50/50 / Double-Up cash preset has correct methodology values."""

    CASH_LABEL = "Cash Main"

    def test_eat_chalk_true(self) -> None:
        assert CONTEST_PRESETS[self.CASH_LABEL]["eat_chalk"] is True

    def test_show_leverage_false(self) -> None:
        assert CONTEST_PRESETS[self.CASH_LABEL]["show_leverage"] is False

    def test_exposure_rules_false(self) -> None:
        assert CONTEST_PRESETS[self.CASH_LABEL]["exposure_rules"] is False

    def test_tagging_mode_floor(self) -> None:
        assert CONTEST_PRESETS[self.CASH_LABEL]["tagging_mode"] == "floor"


class TestGpp150MaxPresetRules:
    """GPP - 150 Max preset has correct pool sizing and methodology values."""

    LABEL = "GPP Main"

    def test_pool_size_min(self) -> None:
        assert CONTEST_PRESETS[self.LABEL]["pool_size_min"] == 40

    def test_exposure_rules_true(self) -> None:
        assert CONTEST_PRESETS[self.LABEL]["exposure_rules"] is True

    def test_not_with_auto_true(self) -> None:
        assert CONTEST_PRESETS[self.LABEL]["not_with_auto"] is True


class TestShowdownPreset:
    """Showdown preset has captain_aware=True."""

    LABEL = "Showdown"

    def test_captain_aware(self) -> None:
        assert CONTEST_PRESETS[self.LABEL].get("captain_aware") is True


class TestGetMethodologyRules:
    """get_methodology_rules returns a dict with all expected keys."""

    @pytest.mark.parametrize("label", list(CONTEST_PRESETS.keys()))
    def test_returns_all_keys(self, label: str) -> None:
        rules = get_methodology_rules(label)
        for key in _REQUIRED_METHODOLOGY_KEYS:
            assert key in rules, f"get_methodology_rules('{label}') missing key '{key}'"

    def test_cash_eat_chalk(self) -> None:
        rules = get_methodology_rules("Cash Main")
        assert rules["eat_chalk"] is True

    def test_gpp_150_exposure_rules(self) -> None:
        rules = get_methodology_rules("GPP Main")
        assert rules["exposure_rules"] is True

    def test_invalid_label_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown contest label"):
            get_methodology_rules("Not A Real Contest")
