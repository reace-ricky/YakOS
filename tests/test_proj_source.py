"""Tests for the _merge_external_proj helper in streamlit_app.py.

The pure-logic core (_merge_external_proj) is imported directly so that the
tests can run without a Streamlit context or st.session_state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the repo root importable
_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import only the pure function — NOT the full app (avoids Streamlit bootstrap)
# We pull it out of the module's AST-evaluated namespace to avoid executing
# Streamlit top-level code.
import importlib.util
import types
import ast
import textwrap

_APP_PATH = Path(__file__).parent.parent / "streamlit_app.py"


def _load_merge_fn():
    """Extract _merge_external_proj without executing the Streamlit app."""
    src = _APP_PATH.read_text()
    tree = ast.parse(src)

    # Find the function def
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_merge_external_proj":
            func_src = ast.get_source_segment(src, node)
            break
    else:
        raise RuntimeError("_merge_external_proj not found in streamlit_app.py")

    # Build a minimal module with only the imports needed by the function
    minimal = textwrap.dedent("""
from __future__ import annotations
from typing import Optional
import pandas as pd
""")
    minimal += "\n" + func_src
    ns: dict = {}
    exec(compile(minimal, "<test_proj_source>", "exec"), ns)
    return ns["_merge_external_proj"]


_merge_external_proj = _load_merge_fn()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _pool(n: int = 5) -> pd.DataFrame:
    names = [f"Player_{i}" for i in range(n)]
    return pd.DataFrame({
        "player_name": names,
        "salary": [5000 + i * 300 for i in range(n)],
        "proj": [20.0 + i * 2 for i in range(n)],
    })


def _rg_proj(pool: pd.DataFrame, multiplier: float = 1.1) -> pd.DataFrame:
    return pd.DataFrame({
        "player_name": pool["player_name"],
        "proj": (pool["proj"] * multiplier).round(2),
    })


def _fp_proj(pool: pd.DataFrame, multiplier: float = 0.9) -> pd.DataFrame:
    return pd.DataFrame({
        "player_name": pool["player_name"],
        "proj": (pool["proj"] * multiplier).round(2),
    })


# ---------------------------------------------------------------------------
# Tests — no source / fallback
# ---------------------------------------------------------------------------

class TestNoSource:
    def test_proj_fp_equals_proj_when_source_is_none(self):
        pool = _pool()
        result = _merge_external_proj(pool, None, None, None)
        assert "proj_fp" in result.columns
        pd.testing.assert_series_equal(
            result["proj_fp"].reset_index(drop=True),
            pool["proj"].reset_index(drop=True),
            check_names=False,
        )

    def test_proj_fp_equals_proj_when_no_dfs_provided(self):
        pool = _pool()
        result = _merge_external_proj(pool, "RotoGrinders", None, None)
        pd.testing.assert_series_equal(
            result["proj_fp"].reset_index(drop=True),
            pool["proj"].reset_index(drop=True),
            check_names=False,
        )

    def test_original_pool_is_not_mutated(self):
        pool = _pool()
        _merge_external_proj(pool, None, None, None)
        assert "proj_fp" not in pool.columns


# ---------------------------------------------------------------------------
# Tests — RotoGrinders source
# ---------------------------------------------------------------------------

class TestRotoGrindersSource:
    def test_proj_fp_uses_rg_projections(self):
        pool = _pool()
        rg = _rg_proj(pool, multiplier=1.1)
        result = _merge_external_proj(pool, "RotoGrinders", rg, None)
        expected = rg.set_index("player_name")["proj"]
        for _, row in result.iterrows():
            assert row["proj_fp"] == pytest.approx(expected[row["player_name"]], rel=1e-4)

    def test_proj_fp_falls_back_to_base_for_unmatched_players(self):
        pool = _pool(n=5)
        rg = _rg_proj(pool.head(3), multiplier=1.1)  # only first 3 players
        result = _merge_external_proj(pool, "RotoGrinders", rg, None)
        # First 3 should use RG values
        for i in range(3):
            assert result.iloc[i]["proj_fp"] == pytest.approx(rg.iloc[i]["proj"], rel=1e-4)
        # Last 2 should fall back to base proj
        for i in range(3, 5):
            assert result.iloc[i]["proj_fp"] == pytest.approx(pool.iloc[i]["proj"], rel=1e-4)

    def test_fantasypros_df_is_ignored_when_rg_selected(self):
        pool = _pool()
        rg = _rg_proj(pool, multiplier=1.5)
        fp = _fp_proj(pool, multiplier=0.5)
        result = _merge_external_proj(pool, "RotoGrinders", rg, fp)
        expected = rg.set_index("player_name")["proj"]
        for _, row in result.iterrows():
            assert row["proj_fp"] == pytest.approx(expected[row["player_name"]], rel=1e-4)


# ---------------------------------------------------------------------------
# Tests — FantasyPros source
# ---------------------------------------------------------------------------

class TestFantasyProsSource:
    def test_proj_fp_uses_fp_projections(self):
        pool = _pool()
        fp = _fp_proj(pool, multiplier=0.9)
        result = _merge_external_proj(pool, "FantasyPros", None, fp)
        expected = fp.set_index("player_name")["proj"]
        for _, row in result.iterrows():
            assert row["proj_fp"] == pytest.approx(expected[row["player_name"]], rel=1e-4)

    def test_proj_fp_falls_back_for_missing_fp_source(self):
        pool = _pool()
        result = _merge_external_proj(pool, "FantasyPros", None, None)
        pd.testing.assert_series_equal(
            result["proj_fp"].reset_index(drop=True),
            pool["proj"].reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Tests — Blended source
# ---------------------------------------------------------------------------

class TestBlendedSource:
    def test_blended_averages_both_sources(self):
        pool = _pool()
        rg = _rg_proj(pool, multiplier=1.2)
        fp = _fp_proj(pool, multiplier=0.8)
        result = _merge_external_proj(pool, "Blended", rg, fp)
        rg_vals = rg.set_index("player_name")["proj"]
        fp_vals = fp.set_index("player_name")["proj"]
        for _, row in result.iterrows():
            expected_blend = (rg_vals[row["player_name"]] + fp_vals[row["player_name"]]) / 2
            assert row["proj_fp"] == pytest.approx(expected_blend, rel=1e-4)

    def test_blended_uses_rg_only_when_fp_missing_for_player(self):
        pool = _pool(n=4)
        rg = _rg_proj(pool, multiplier=1.2)
        # FP only covers first 2 players
        fp = _fp_proj(pool.head(2), multiplier=0.8)
        result = _merge_external_proj(pool, "Blended", rg, fp)
        rg_vals = rg.set_index("player_name")["proj"]
        fp_vals = fp.set_index("player_name")["proj"]
        # First 2: blended
        for i in range(2):
            name = pool.iloc[i]["player_name"]
            expected = (rg_vals[name] + fp_vals[name]) / 2
            assert result.iloc[i]["proj_fp"] == pytest.approx(expected, rel=1e-4)
        # Last 2: RG only
        for i in range(2, 4):
            name = pool.iloc[i]["player_name"]
            assert result.iloc[i]["proj_fp"] == pytest.approx(rg_vals[name], rel=1e-4)

    def test_blended_falls_back_when_only_one_source_uploaded(self):
        pool = _pool()
        rg = _rg_proj(pool, multiplier=1.2)
        # Blended requires both — missing fp_df → fallback to base proj
        result = _merge_external_proj(pool, "Blended", rg, None)
        pd.testing.assert_series_equal(
            result["proj_fp"].reset_index(drop=True),
            pool["proj"].reset_index(drop=True),
            check_names=False,
        )

    def test_blended_no_nan_in_proj_fp(self):
        pool = _pool()
        rg = _rg_proj(pool, multiplier=1.1)
        fp = _fp_proj(pool, multiplier=0.9)
        result = _merge_external_proj(pool, "Blended", rg, fp)
        assert result["proj_fp"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_pool_does_not_crash(self):
        pool = pd.DataFrame({"player_name": [], "proj": []})
        result = _merge_external_proj(pool, "RotoGrinders", None, None)
        assert "proj_fp" in result.columns
        assert len(result) == 0

    def test_proj_fp_no_nan_when_base_proj_present(self):
        pool = _pool()
        rg = _rg_proj(pool.head(2))  # partial match
        result = _merge_external_proj(pool, "RotoGrinders", rg, None)
        assert result["proj_fp"].isna().sum() == 0

    def test_returns_copy_not_inplace(self):
        pool = _pool()
        rg = _rg_proj(pool)
        result = _merge_external_proj(pool, "RotoGrinders", rg, None)
        assert "proj_fp" not in pool.columns
        assert result is not pool
