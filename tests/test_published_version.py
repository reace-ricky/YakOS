"""Tests for app.data_loader cache-busting helpers.

Validates that:
- get_published_version() changes when tracked published files change and stays
  stable when nothing changes.
- invalidate_published_cache() clears load_published_data's cache without error.
- Edge cases (missing dirs, PGA sport, sport-name casing) are handled.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_missing_directory_returns_empty_string(tmp_path, monkeypatch):
    """get_published_version returns '' when the sport directory does not exist."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    assert dl.get_published_version("nba") == ""


def test_stable_when_files_unchanged(tmp_path, monkeypatch):
    """Version is stable across two calls with no file changes."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    sport_dir = tmp_path / "nba"
    sport_dir.mkdir()
    (sport_dir / "slate_meta.json").write_text("{}")

    v1 = dl.get_published_version("nba")
    v2 = dl.get_published_version("nba")
    assert v1 == v2


def test_version_changes_when_file_updated(tmp_path, monkeypatch):
    """Version changes after a tracked file is re-written."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    sport_dir = tmp_path / "nba"
    sport_dir.mkdir()
    meta = sport_dir / "slate_meta.json"
    meta.write_text("{}")

    v1 = dl.get_published_version("nba")

    # Force a different mtime by nudging the modification time by 1 second.
    current = meta.stat().st_mtime
    os.utime(meta, (current + 1, current + 1))

    v2 = dl.get_published_version("nba")
    assert v1 != v2


def test_version_changes_when_lineup_file_added(tmp_path, monkeypatch):
    """Version changes when a new *_lineups.parquet file appears."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    sport_dir = tmp_path / "nba"
    sport_dir.mkdir()

    v1 = dl.get_published_version("nba")

    (sport_dir / "classic_gpp_main_lineups.parquet").write_bytes(b"")
    v2 = dl.get_published_version("nba")

    assert v1 != v2


def test_independent_across_sports(tmp_path, monkeypatch):
    """Versions for different sports are independent."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    (tmp_path / "nba").mkdir()
    (tmp_path / "pga").mkdir()
    (tmp_path / "nba" / "slate_meta.json").write_text("{}")
    (tmp_path / "pga" / "slate_meta.json").write_text("{}")

    v_nba = dl.get_published_version("nba")
    v_pga = dl.get_published_version("pga")

    # Updating NBA file must not affect PGA version
    meta_nba = tmp_path / "nba" / "slate_meta.json"
    current = meta_nba.stat().st_mtime
    os.utime(meta_nba, (current + 1, current + 1))

    assert dl.get_published_version("nba") != v_nba
    assert dl.get_published_version("pga") == v_pga


def test_sport_name_is_case_insensitive(tmp_path, monkeypatch):
    """get_published_version treats 'PGA' and 'pga' as the same sport directory."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    pga_dir = tmp_path / "pga"
    pga_dir.mkdir()
    (pga_dir / "slate_meta.json").write_text("{}")

    assert dl.get_published_version("PGA") == dl.get_published_version("pga")


def test_version_string_contains_tracked_filenames(tmp_path, monkeypatch):
    """Version string embeds the names of all tracked files."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    sport_dir = tmp_path / "nba"
    sport_dir.mkdir()
    for fname in dl._VERSIONED_FILES:
        (sport_dir / fname).write_text("{}")

    version = dl.get_published_version("nba")
    for fname in dl._VERSIONED_FILES:
        assert fname in version, f"Expected '{fname}' in version string"


def test_version_string_contains_lineup_filename(tmp_path, monkeypatch):
    """Version string embeds the name of a dynamic *_lineups.parquet file."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    sport_dir = tmp_path / "nba"
    sport_dir.mkdir()
    lineup_file = "classic_gpp_main_lineups.parquet"
    (sport_dir / lineup_file).write_bytes(b"")

    version = dl.get_published_version("nba")
    assert lineup_file in version


def test_invalidate_published_cache_does_not_raise():
    """invalidate_published_cache() can be called without error."""
    import app.data_loader as dl

    # Should not raise even if cache is already empty
    dl.invalidate_published_cache()
    dl.invalidate_published_cache()  # second call is also safe


def test_different_versions_produce_distinct_cache_entries(tmp_path, monkeypatch):
    """Calling load_published_data with two different version strings uses separate cache entries.

    This confirms that Streamlit's cache key incorporates published_version so
    that stale results from an old version are never returned after files change.
    """
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    sport_dir = tmp_path / "nba"
    sport_dir.mkdir()

    # Write minimal valid slate_meta.json for the first "version"
    meta_path = sport_dir / "slate_meta.json"
    meta_path.write_text('{"slate": "v1"}')

    v1 = dl.get_published_version("nba")
    meta_v1, *_ = dl.load_published_data("nba", v1)

    # Simulate a publish: overwrite the meta with new content
    meta_path.write_text('{"slate": "v2"}')
    current = meta_path.stat().st_mtime
    os.utime(meta_path, (current + 1, current + 1))

    v2 = dl.get_published_version("nba")
    assert v1 != v2, "Version must change when file mtime changes"

    # Load with the new version – must get the updated content, not the cached v1
    meta_v2, *_ = dl.load_published_data("nba", v2)

    assert meta_v1.get("slate") == "v1"
    assert meta_v2.get("slate") == "v2", (
        "load_published_data returned stale v1 data when called with v2 version key"
    )
