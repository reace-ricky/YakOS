"""Edge integrity helpers — validate and clean stale edge artifacts.

These functions ensure that ``edge_state.json``, ``edge_analysis.json`` and
``signals.parquet`` in a published directory are always consistent with the
canonical slate date from ``slate_meta.json`` and the player pool in
``slate_pool.parquet``.

They are intentionally dependency-light (only stdlib + pandas) so they can be
imported and tested without pulling in the full YakOS stack.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# Files that constitute an "edge run" and must match the current slate date.
EDGE_ARTIFACT_FILES: tuple[str, ...] = ("edge_state.json", "edge_analysis.json", "signals.parquet")

# Keys in edge_analysis.json that contain player classification lists.
EDGE_PLAY_KEYS: tuple[str, ...] = ("core_plays", "leverage_plays", "value_plays", "fade_candidates")


def validate_edge_artifacts(out_dir: Path) -> Dict[str, Any]:
    """Validate that edge artifacts are consistent with ``slate_meta.json`` and the pool.

    Checks:
    1. ``edge_state.json["date"]`` must equal ``slate_meta.json["date"]``.
    2. Every player name in ``edge_analysis.json`` classification lists must exist
       in ``slate_pool.parquet``.

    Parameters
    ----------
    out_dir:
        Directory containing the published artifacts (e.g. ``data/published/nba/``).

    Returns
    -------
    dict with keys:
        - ``valid`` (bool): True when all checks pass or edge files don't exist.
        - ``meta_date`` (str | None): canonical date from ``slate_meta.json``.
        - ``edge_date`` (str | None): date recorded in ``edge_state.json``.
        - ``phantom_players`` (list[str]): players in edge analysis not in the pool.
        - ``reason`` (str): human-readable explanation when not valid.
    """
    result: Dict[str, Any] = {
        "valid": True,
        "meta_date": None,
        "edge_date": None,
        "phantom_players": [],
        "reason": "",
    }

    # ── Read canonical slate date ──────────────────────────────────────────
    meta_path = out_dir / "slate_meta.json"
    if not meta_path.exists():
        return result
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return result
    meta_date: str = meta.get("date") or ""
    result["meta_date"] = meta_date

    edge_state_path = out_dir / "edge_state.json"
    edge_analysis_path = out_dir / "edge_analysis.json"
    pool_path = out_dir / "slate_pool.parquet"

    # If no edge files are present there is nothing to validate.
    if not edge_state_path.exists() and not edge_analysis_path.exists():
        return result

    # ── Check edge_state date matches meta_date ────────────────────────────
    if edge_state_path.exists() and meta_date:
        try:
            edge_state = json.loads(edge_state_path.read_text())
        except Exception:
            edge_state = {}
        edge_date: str = str(edge_state.get("date") or "")
        result["edge_date"] = edge_date
        if edge_date and edge_date != meta_date:
            result["valid"] = False
            result["reason"] = (
                f"Edge analysis is for {edge_date} but pool/meta is for {meta_date}."
            )
            return result

    # ── Check all edge play names exist in the pool ────────────────────────
    if edge_analysis_path.exists() and pool_path.exists():
        try:
            ea = json.loads(edge_analysis_path.read_text())
        except Exception:
            ea = {}
        try:
            pool_df = pd.read_parquet(pool_path, columns=["player_name"])
            pool_names: set = set(pool_df["player_name"].dropna().astype(str).tolist())
        except Exception:
            pool_names = set()

        if pool_names:
            phantoms: List[str] = []
            for key in EDGE_PLAY_KEYS:
                for entry in ea.get(key, []):
                    pname = (
                        entry.get("player_name", "")
                        if isinstance(entry, dict)
                        else str(entry)
                    )
                    if pname and pname not in pool_names:
                        phantoms.append(pname)
            if phantoms:
                # Deduplicate while preserving first-seen order.
                seen: set = set()
                deduped: List[str] = []
                for p in phantoms:
                    if p not in seen:
                        seen.add(p)
                        deduped.append(p)
                result["valid"] = False
                result["phantom_players"] = deduped
                result["reason"] = (
                    f"Edge analysis contains {len(deduped)} player(s) "
                    f"not in the current pool: {', '.join(deduped[:5])}"
                    + (" …" if len(deduped) > 5 else "")
                )

    return result


def clean_stale_edge_artifacts(out_dir: Path) -> List[str]:
    """Remove stale edge artifact files from *out_dir*.

    Removes ``edge_state.json``, ``edge_analysis.json``, and ``signals.parquet``
    if they are present.

    Returns
    -------
    list[str]
        Names of files that were successfully removed.
    """
    removed: List[str] = []
    for fname in EDGE_ARTIFACT_FILES:
        fpath = out_dir / fname
        if fpath.exists():
            try:
                fpath.unlink()
                removed.append(fname)
            except Exception as exc:
                print(f"[edge_integrity] Could not remove {fname}: {exc}")
    return removed
