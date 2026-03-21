"""Tests for batch comparison completeness filtering.

Verifies that the sim_lab.py source code includes:
1. completeness_pct in _run_pipeline return dict
2. skip_incomplete / completeness tracking in _run_batch
3. completeness columns in _append_batch_history
4. Visual indicators + filter in _render_comparison_table

These tests use AST parsing since app/sim_lab.py requires streamlit
which is not available in the test environment.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SIM_LAB_PATH = Path(__file__).resolve().parent.parent / "app" / "sim_lab.py"
_SIM_LAB_SRC = _SIM_LAB_PATH.read_text()
_SIM_LAB_TREE = ast.parse(_SIM_LAB_SRC)


def _get_function_source(func_name: str) -> str:
    """Extract the source of a top-level function from sim_lab.py."""
    for node in ast.walk(_SIM_LAB_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            start = node.lineno - 1
            end = node.end_lineno
            lines = _SIM_LAB_SRC.splitlines()[start:end]
            return "\n".join(lines)
    raise ValueError(f"Function {func_name} not found in sim_lab.py")


def _function_has_param(func_name: str, param_name: str) -> bool:
    """Check if a function has a specific parameter."""
    for node in ast.walk(_SIM_LAB_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            all_params = (
                [a.arg for a in node.args.args]
                + [a.arg for a in node.args.kwonlyargs]
            )
            return param_name in all_params
    return False


def _function_return_has_key(func_name: str, key: str) -> bool:
    """Check if a function's return dict literal includes a given string key."""
    src = _get_function_source(func_name)
    # Simple heuristic: look for the key in a return dict literal
    return f'"{key}"' in src or f"'{key}'" in src


# ---------------------------------------------------------------------------
# _run_pipeline completeness
# ---------------------------------------------------------------------------


class TestRunPipelineCompleteness:
    def test_completeness_pct_in_return(self) -> None:
        assert _function_return_has_key("_run_pipeline", "completeness_pct")

    def test_completeness_calculation_present(self) -> None:
        src = _get_function_source("_run_pipeline")
        # Should compute completeness from actuals
        assert "_completeness_pct" in src
        assert "_n_with_actuals" in src or "_actual_fp_col" in src


# ---------------------------------------------------------------------------
# _run_batch completeness
# ---------------------------------------------------------------------------


class TestRunBatchCompleteness:
    def test_skip_incomplete_param(self) -> None:
        assert _function_has_param("_run_batch", "skip_incomplete")

    def test_completeness_threshold_param(self) -> None:
        assert _function_has_param("_run_batch", "completeness_threshold")

    def test_return_has_incomplete_dates(self) -> None:
        assert _function_return_has_key("_run_batch", "incomplete_dates")

    def test_return_has_min_completeness(self) -> None:
        assert _function_return_has_key("_run_batch", "min_completeness_pct")

    def test_return_has_incomplete_flag(self) -> None:
        assert _function_return_has_key("_run_batch", "has_incomplete_dates")

    def test_skip_logic_present(self) -> None:
        """Should skip dates when skip_incomplete is True and pct is low."""
        src = _get_function_source("_run_batch")
        assert "skip_incomplete" in src
        assert "completeness_threshold" in src
        assert "incomplete_dates" in src


# ---------------------------------------------------------------------------
# _append_batch_history completeness columns
# ---------------------------------------------------------------------------


class TestAppendBatchHistoryCompleteness:
    def test_stores_completeness_columns(self) -> None:
        src = _get_function_source("_append_batch_history")
        assert "min_completeness_pct" in src
        assert "has_incomplete_dates" in src
        assert "incomplete_date_count" in src


# ---------------------------------------------------------------------------
# _render_comparison_table completeness UI
# ---------------------------------------------------------------------------


class TestComparisonTableCompleteness:
    def test_hide_incomplete_toggle(self) -> None:
        src = _get_function_source("_render_comparison_table")
        assert "hide_incomplete" in src
        assert "Hide runs with incomplete slate data" in src

    def test_warning_icon_for_incomplete(self) -> None:
        src = _get_function_source("_render_comparison_table")
        # Should use warning emoji for incomplete rows
        assert "\u26a0\ufe0f" in src or "\\u26a0" in src

    def test_data_quality_column(self) -> None:
        src = _get_function_source("_render_comparison_table")
        # Should have a "Data" column showing completeness status
        assert '"Data"' in src or "'Data'" in src

    def test_legacy_unknown_indicator(self) -> None:
        src = _get_function_source("_render_comparison_table")
        # Legacy rows with unknown completeness should show ? icon
        assert "\u2753" in src or "\\u2753" in src


# ---------------------------------------------------------------------------
# _render_history_table completeness column
# ---------------------------------------------------------------------------


class TestHistoryTableCompleteness:
    def test_data_quality_column(self) -> None:
        src = _get_function_source("_render_history_table")
        assert "data_quality" in src

    def test_legacy_defaults(self) -> None:
        src = _get_function_source("_render_history_table")
        assert "min_completeness_pct" in src
        assert "has_incomplete_dates" in src


# ---------------------------------------------------------------------------
# Batch run UI wires skip_incomplete
# ---------------------------------------------------------------------------


class TestBatchRunUIWiring:
    def test_skip_incomplete_checkbox_exists(self) -> None:
        """The main render function should have a skip_incomplete checkbox."""
        assert "sim_lab_batch_skip_incomplete" in _SIM_LAB_SRC

    def test_skip_incomplete_passed_to_run_batch(self) -> None:
        """The _run_batch calls should pass skip_incomplete."""
        # Count occurrences of skip_incomplete in _run_batch calls
        # Should appear at least twice (baseline + user batch)
        matches = re.findall(r"skip_incomplete\s*=", _SIM_LAB_SRC)
        # At least: 1 in function def, 2 in checkbox, 3+ in calls
        assert len(matches) >= 4, (
            f"Expected skip_incomplete= at least 4 times "
            f"(def + checkbox + 2 calls), found {len(matches)}"
        )
