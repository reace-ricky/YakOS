"""Tests for bias write durability (atomic writes) and cache invalidation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bias_path(tmp_path, monkeypatch):
    """Redirect BIAS_PATH to a temp directory so tests never touch real data."""
    p = tmp_path / "ricky_bias.json"
    import yak_core.bias as bias_mod
    monkeypatch.setattr(bias_mod, "BIAS_PATH", p)
    return p


# ---------------------------------------------------------------------------
# save_bias – atomic write behaviour
# ---------------------------------------------------------------------------

class TestSaveBiasAtomic:
    def test_save_creates_file(self, bias_path):
        from yak_core.bias import save_bias
        save_bias({"LeBron James": {"proj_adj": 2.0}})
        assert bias_path.exists()

    def test_saved_content_is_valid_json(self, bias_path):
        from yak_core.bias import save_bias
        payload = {"LeBron James": {"proj_adj": 2.0, "max_exposure": 0.4}}
        save_bias(payload)
        loaded = json.loads(bias_path.read_text())
        assert loaded == payload

    def test_no_temp_files_left_after_success(self, bias_path):
        from yak_core.bias import save_bias
        save_bias({"A": {"proj_adj": 1.0}})
        tmp_files = list(bias_path.parent.glob("*.tmp"))
        assert tmp_files == [], f"Orphaned temp file(s): {tmp_files}"

    def test_overwrite_preserves_full_content(self, bias_path):
        from yak_core.bias import save_bias
        save_bias({"Player1": {"proj_adj": 1.0}})
        save_bias({"Player1": {"proj_adj": 1.0}, "Player2": {"max_exposure": 0.0}})
        loaded = json.loads(bias_path.read_text())
        assert "Player1" in loaded
        assert "Player2" in loaded

    def test_write_error_cleans_up_temp_file(self, bias_path, monkeypatch):
        """If os.replace fails, the temp file must not be left on disk."""
        import yak_core.bias as bias_mod

        original_replace = os.replace

        def failing_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(OSError, match="disk full"):
            bias_mod.save_bias({"X": {}})

        tmp_files = list(bias_path.parent.glob("*.tmp"))
        assert tmp_files == [], f"Orphaned temp file after error: {tmp_files}"

    def test_write_error_does_not_corrupt_existing_file(self, bias_path, monkeypatch):
        """A failed write must leave the original file intact."""
        from yak_core.bias import save_bias
        import yak_core.bias as bias_mod

        original = {"Good": {"proj_adj": 5.0}}
        save_bias(original)

        def failing_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            bias_mod.save_bias({"Bad": {}})

        # Original file must be untouched
        loaded = json.loads(bias_path.read_text())
        assert loaded == original

    def test_save_empty_dict(self, bias_path):
        from yak_core.bias import save_bias
        save_bias({})
        assert json.loads(bias_path.read_text()) == {}

    def test_parent_directory_created_if_missing(self, tmp_path, monkeypatch):
        import yak_core.bias as bias_mod
        nested = tmp_path / "deep" / "nested" / "ricky_bias.json"
        monkeypatch.setattr(bias_mod, "BIAS_PATH", nested)
        bias_mod.save_bias({"X": {"proj_adj": 0.5}})
        assert nested.exists()


# ---------------------------------------------------------------------------
# load_bias
# ---------------------------------------------------------------------------

class TestLoadBias:
    def test_returns_empty_dict_when_file_missing(self, bias_path):
        from yak_core.bias import load_bias
        assert load_bias() == {}

    def test_returns_empty_dict_on_corrupt_json(self, bias_path):
        bias_path.write_text("not valid json {{{")
        from yak_core.bias import load_bias
        assert load_bias() == {}

    def test_round_trip(self, bias_path):
        from yak_core.bias import load_bias, save_bias
        data = {
            "Luka Doncic": {"proj_adj": -1.5, "max_exposure": 0.3},
            "Anthony Davis": {"min_exposure": 0.5},
        }
        save_bias(data)
        assert load_bias() == data


# ---------------------------------------------------------------------------
# GitHub sync is optional – failures must not propagate
# ---------------------------------------------------------------------------

class TestSaveBiasGitHubSync:
    def test_github_sync_failure_does_not_raise(self, bias_path, monkeypatch):
        import yak_core.bias as bias_mod

        def boom(*a, **kw):
            raise RuntimeError("network down")

        monkeypatch.setattr(bias_mod, "save_bias", bias_mod.save_bias)
        # Patch the sync function inside the module
        with patch("yak_core.github_persistence.sync_feedback_async", side_effect=boom):
            # Should not raise
            bias_mod.save_bias({"Z": {}})

        assert bias_path.exists()
