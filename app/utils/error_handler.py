"""Error containment and structured logging for YakOS Streamlit tabs.

Purpose
-------
Provides a ``safe_render()`` decorator that wraps any tab render function
with a try/except block, displays a user-friendly error UI (with Retry and
Clear State recovery buttons), and writes structured error logs to the
``logs/`` directory.

Also exposes lightweight validation helpers used by tab renderers to surface
data problems early:

* ``validate_pool()``       – checks that a player-pool DataFrame is usable.
* ``validate_edge_analysis()`` – checks that an edge-analysis dict is usable.

Inputs / Outputs
----------------
- Inputs : the wrapped render function and its arguments.
- Outputs: structured JSON-lines to ``logs/yakos.log`` and a sport-specific
  ``logs/<sport>_errors.log`` file; user-visible error UI in the Streamlit app.

Usage example
-------------
    from app.utils.error_handler import safe_render

    with tab_edge:
        safe_render(render_edge_tab, sport)
"""
from __future__ import annotations

import functools
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import streamlit as st

# ── Logging setup ──────────────────────────────────────────────────────────
# Derive the project root as the directory containing streamlit_app.py so the
# log directory stays correct even if this module is moved within the tree.
_PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "streamlit_app.py").exists()),
    Path(__file__).resolve().parents[2],
)
_LOG_DIR = _PROJECT_ROOT / "logs"


def _ensure_log_dir() -> Path:
    """Create ``logs/`` directory if it does not exist and return its path."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def _get_file_logger(name: str, filepath: Path) -> logging.Logger:
    """Return a Logger that appends JSON-lines to *filepath*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(filepath, encoding="utf-8")
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    return logger


# ── Public helpers ─────────────────────────────────────────────────────────

def log_error_event(
    component: str,
    sport: str,
    exc: BaseException,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a structured error record to the log files.

    Parameters
    ----------
    component:
        Name of the tab / function where the error occurred (e.g. ``"edge_tab"``).
    sport:
        Active sport context (e.g. ``"NBA"`` or ``"PGA"``).
    exc:
        The caught exception.
    extra:
        Optional dict of additional context key/value pairs.
    """
    log_dir = _ensure_log_dir()
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "sport": sport,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "stack_trace": traceback.format_exc(),
    }
    if extra:
        record["extra"] = extra

    line = json.dumps(record)

    # Write to the shared yakos.log
    main_logger = _get_file_logger("yakos.main", log_dir / "yakos.log")
    main_logger.error(line)

    # Write to a sport-specific error file
    sport_key = sport.lower() if sport else "unknown"
    sport_logger = _get_file_logger(
        f"yakos.{sport_key}", log_dir / f"{sport_key}_errors.log"
    )
    sport_logger.error(line)


def validate_pool(pool: Any, sport: str = "") -> tuple[bool, str]:
    """Validate that *pool* is a non-empty pandas DataFrame with required columns.

    Parameters
    ----------
    pool:
        The player-pool object to validate (expected: ``pd.DataFrame``).
    sport:
        Active sport for context in the returned message.

    Returns
    -------
    (ok, message):
        ``ok`` is ``True`` when the pool is usable.
        ``message`` describes the problem when ``ok`` is ``False``.
    """
    try:
        import pandas as pd
    except ImportError:
        return False, "pandas is not installed."

    if pool is None:
        return False, f"Player pool is None{' for ' + sport if sport else ''}."
    if not isinstance(pool, pd.DataFrame):
        return False, f"Player pool is not a DataFrame (got {type(pool).__name__})."
    if pool.empty:
        return False, f"Player pool is empty{' for ' + sport if sport else ''}."

    required_cols = {"player_name"}
    missing = required_cols - set(pool.columns)
    if missing:
        return False, f"Player pool is missing required columns: {sorted(missing)}."

    return True, ""


def validate_edge_analysis(edge: Any) -> tuple[bool, str]:
    """Validate that *edge* is a non-empty dict.

    Parameters
    ----------
    edge:
        The edge-analysis object to validate (expected: ``dict``).

    Returns
    -------
    (ok, message):
        ``ok`` is ``True`` when the edge dict is usable.
        ``message`` describes the problem when ``ok`` is ``False``.
    """
    if edge is None:
        return False, "Edge analysis data is None."
    if not isinstance(edge, dict):
        return False, f"Edge analysis data is not a dict (got {type(edge).__name__})."
    if not edge:
        return False, "Edge analysis data is empty."
    return True, ""


def safe_render(fn: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """Invoke *fn* with the given arguments inside a try/except block.

    On success, the wrapped function renders normally.  On failure:
    1. A user-friendly error card is displayed with recovery buttons
       ("Retry render" and "Clear state").
    2. The full error details are written to the log files via
       :func:`log_error_event`.

    Parameters
    ----------
    fn:
        The render function to call (e.g. ``render_edge_tab``).
    *args:
        Positional arguments forwarded to *fn*.
    **kwargs:
        Keyword arguments forwarded to *fn*.

    Example
    -------
    ::

        safe_render(render_edge_tab, sport)
        safe_render(render_optimizer_tab, sport, is_admin=True)
    """
    # Derive a human-readable component name from the function
    component = getattr(fn, "__name__", repr(fn))

    # Infer sport from the first positional arg when it is a string
    sport: str = ""
    if args and isinstance(args[0], str):
        sport = args[0]
    elif "sport" in kwargs and isinstance(kwargs["sport"], str):
        sport = kwargs["sport"]

    try:
        fn(*args, **kwargs)
    except Exception as exc:  # KeyboardInterrupt/SystemExit inherit from BaseException, not Exception,
        # so they propagate normally and are not swallowed here.
        log_error_event(component=component, sport=sport, exc=exc)

        # ── User-facing error UI ───────────────────────────────────────────
        st.error(
            f"**{component} encountered an error.**\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Use the buttons below to recover, or contact support."
        )

        with st.expander("Show full traceback", expanded=False):
            st.code(traceback.format_exc(), language="python")

        col_retry, col_clear = st.columns(2)
        with col_retry:
            if st.button("🔄 Retry render", key=f"retry_{component}"):
                st.rerun()
        with col_clear:
            if st.button("🗑️ Clear state", key=f"clear_{component}"):
                # Preserve authentication and preference keys across error recovery.
                _preserve = {
                    k: v
                    for k, v in st.session_state.items()
                    if k.startswith(("admin_", "auth_", "sport_"))
                }
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state.update(_preserve)
                st.rerun()
