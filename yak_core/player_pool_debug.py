"""yak_core.player_pool_debug -- Debug helper for tracing a player through the
YakOS player pool pipeline.

Usage example (run from repo root)::

    from yak_core.player_pool_debug import debug_player_pipeline
    debug_player_pipeline("20260303", "Shai Gilgeous-Alexander")

Or from the command line::

    python -m yak_core.player_pool_debug 20260303 "Shai Gilgeous-Alexander"

The function prints each intermediate stage for the named player:

  Stage 0 – Raw DK draftables row (status, salary, positions)
  Stage 1 – After Tank01 merge (status updated to date-specific value)
  Stage 2 – After apply_projections (proj column set)
  Stage 3 – After _enrich_pool (proj_minutes, floor, ceil, sim_eligible)
  Stage 4 – After _filter_ineligible_players (player present or removed)

This makes it trivial to verify that a player who was OUT on a given date
is correctly excluded from the final pool.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Ensure repo root on sys.path when run directly
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Ineligible status set shared across the YakOS pipeline.
from yak_core.sims import _INELIGIBLE_STATUSES  # noqa: F401 (re-exported for local use)


def _find_player(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Return the first row matching *name* (case-insensitive), or None."""
    if df is None or df.empty:
        return None
    col = "player_name" if "player_name" in df.columns else (
        "name" if "name" in df.columns else None
    )
    if col is None:
        return None
    mask = df[col].fillna("").str.strip().str.lower() == name.strip().lower()
    rows = df[mask]
    return rows.iloc[0] if not rows.empty else None


def _print_stage(stage_num: int, label: str, row: Optional[pd.Series]) -> None:
    """Pretty-print a pipeline stage row."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Stage {stage_num}: {label}")
    print(sep)
    if row is None:
        print("  ⚠️  Player NOT FOUND at this stage.")
    else:
        for k, v in row.items():
            print(f"  {k:<24} {v}")


def debug_player_pipeline(
    slate_date: str,
    player_name: str,
    draft_group_id: Optional[int] = None,
    rapidapi_key: str = "",
    proj_source: str = "model",
) -> Dict[str, Any]:
    """Trace *player_name* through every stage of the YakOS player pool pipeline.

    Parameters
    ----------
    slate_date : str
        Date string in ``YYYYMMDD`` or ``YYYY-MM-DD`` format.
    player_name : str
        Full display name to look up, e.g. ``"Shai Gilgeous-Alexander"``.
    draft_group_id : int, optional
        DK draft group ID.  When omitted the function skips the DK-draftables
        stage and starts from an empty placeholder pool, which still allows
        the Tank01 stages to be exercised.
    rapidapi_key : str
        Tank01 RapidAPI key.  When empty the Tank01 stage is skipped.
    proj_source : str
        Projection source passed to ``apply_projections`` (default ``"model"``).

    Returns
    -------
    dict
        Mapping of stage label → pandas Series (or None when player absent).
    """
    from yak_core.dk_ingest import fetch_dk_draftables
    from yak_core.projections import apply_projections
    from yak_core.config import merge_config
    from yak_core.ownership import apply_ownership
    from yak_core.sims import compute_sim_eligible
    from yak_core.projections import yakos_fp_projection, yakos_minutes_projection

    # Normalise date
    date_iso = slate_date.replace("-", "")
    date_str = f"{date_iso[:4]}-{date_iso[4:6]}-{date_iso[6:]}"

    results: Dict[str, Any] = {}

    # ── Stage 0: DK draftables ───────────────────────────────────────────
    pool_dk: pd.DataFrame
    if draft_group_id:
        try:
            pool_dk = fetch_dk_draftables(int(draft_group_id))
            # Normalize column names (mirrors _normalize_dk_pool)
            if "name" in pool_dk.columns and "player_name" not in pool_dk.columns:
                pool_dk = pool_dk.rename(columns={"name": "player_name"})
            if "positions" in pool_dk.columns and "pos" not in pool_dk.columns:
                pool_dk = pool_dk.rename(columns={"positions": "pos"})
        except Exception as exc:
            print(f"[debug] DK draftables fetch failed: {exc}")
            pool_dk = pd.DataFrame()
    else:
        pool_dk = pd.DataFrame()
        print("[debug] No draft_group_id provided; skipping DK draftables stage.")

    row_s0 = _find_player(pool_dk, player_name)
    results["Stage 0 – DK draftables"] = row_s0
    _print_stage(0, "DK draftables (raw status from DK API)", row_s0)

    # Work with a copy so each stage builds on the previous
    pool = pool_dk.copy() if not pool_dk.empty else pd.DataFrame()

    # ── Stage 1: Tank01 merge ────────────────────────────────────────────
    if rapidapi_key:
        try:
            from yak_core.live import fetch_live_opt_pool
            tank01_pool = fetch_live_opt_pool(date_str, {"RAPIDAPI_KEY": rapidapi_key})
            if not tank01_pool.empty:
                if "proj" in tank01_pool.columns and "tank01_proj" not in tank01_pool.columns:
                    tank01_pool = tank01_pool.rename(columns={"proj": "tank01_proj"})
                merge_cols = ["player_name"]
                for col in ("opp", "opponent", "tank01_proj", "own_proj", "actual_fp", "status"):
                    if col in tank01_pool.columns:
                        merge_cols.append(col)

                if pool.empty:
                    # No DK pool: use Tank01 as the base so all stages can run
                    pool = tank01_pool.copy()
                    if "tank01_proj" in pool.columns:
                        pool = pool.rename(columns={"tank01_proj": "proj"})
                else:
                    pool = pool.merge(
                        tank01_pool[merge_cols],
                        on="player_name",
                        how="left",
                        suffixes=("", "_tank01"),
                    )
                    # Prefer Tank01 status (date-specific)
                    if "status_tank01" in pool.columns:
                        tank01_norm = (
                            pool["status_tank01"].fillna("").astype(str).str.strip().str.upper()
                        )
                        has_t01 = tank01_norm != ""
                        pool.loc[has_t01, "status"] = pool.loc[has_t01, "status_tank01"]
                        pool = pool.drop(columns=["status_tank01"])

            row_s1 = _find_player(pool, player_name)
            results["Stage 1 – after Tank01 merge"] = row_s1
            _print_stage(1, "After Tank01 merge (status now date-specific)", row_s1)
        except Exception as exc:
            print(f"[debug] Tank01 fetch failed: {exc}")
            row_s1 = _find_player(pool, player_name)
            results["Stage 1 – after Tank01 merge"] = row_s1
            _print_stage(1, "After Tank01 merge (fetch failed – DK status kept)", row_s1)
    else:
        row_s1 = _find_player(pool, player_name)
        results["Stage 1 – after Tank01 merge"] = row_s1
        _print_stage(1, "Tank01 stage SKIPPED (no API key)", row_s1)

    # ── Stage 2: apply_projections ───────────────────────────────────────
    if not pool.empty:
        try:
            cfg = merge_config({"PROJ_SOURCE": proj_source, "SLATE_DATE": date_str})
            pool = apply_projections(pool, cfg)
        except Exception as exc:
            print(f"[debug] apply_projections failed: {exc}")
    row_s2 = _find_player(pool, player_name)
    results["Stage 2 – after apply_projections"] = row_s2
    _print_stage(2, "After apply_projections (proj column set)", row_s2)

    # ── Stage 3: _enrich_pool equivalent ────────────────────────────────
    if not pool.empty:
        try:
            floors, ceils, mins_proj = [], [], []
            for _, row in pool.iterrows():
                feats = {"salary": float(row.get("salary", 0) or 0)}
                fp_res = yakos_fp_projection(feats)
                min_res = yakos_minutes_projection(feats)
                floors.append(fp_res.get("floor", fp_res["proj"] * 0.7))
                ceils.append(fp_res.get("ceil", fp_res["proj"] * 1.4))
                mins_proj.append(min_res.get("proj_minutes", 0.0))

            if "floor" not in pool.columns or not pool["floor"].notna().any():
                pool["floor"] = floors
            if "ceil" not in pool.columns or not pool["ceil"].notna().any():
                pool["ceil"] = ceils
            pool["proj_minutes"] = mins_proj

            # Zero out proj_minutes for ineligible players
            if "status" in pool.columns:
                inelig_mask = (
                    pool["status"].fillna("").astype(str).str.strip().str.upper()
                    .isin(_INELIGIBLE_STATUSES)
                )
                pool.loc[inelig_mask, "proj_minutes"] = 0.0

            pool = apply_ownership(pool)
            pool = compute_sim_eligible(pool)
        except Exception as exc:
            print(f"[debug] _enrich_pool equivalent failed: {exc}")

    row_s3 = _find_player(pool, player_name)
    results["Stage 3 – after _enrich_pool"] = row_s3
    _print_stage(3, "After _enrich_pool (proj_minutes / sim_eligible set)", row_s3)

    # ── Stage 4: _filter_ineligible_players ──────────────────────────────
    if not pool.empty:
        # Status filter
        if "status" in pool.columns:
            inelig_mask = (
                pool["status"].fillna("").astype(str).str.strip().str.upper()
                .isin(_INELIGIBLE_STATUSES)
            )
            pool = pool[~inelig_mask]
        # Minutes filter
        mins_col = "proj_minutes" if "proj_minutes" in pool.columns else (
            "minutes" if "minutes" in pool.columns else None
        )
        if mins_col is not None:
            mins = pd.to_numeric(pool[mins_col], errors="coerce").fillna(0)
            pool = pool[mins > 0]
        pool = pool.reset_index(drop=True)

    row_s4 = _find_player(pool, player_name)
    results["Stage 4 – after _filter_ineligible_players"] = row_s4
    _print_stage(4, "After _filter_ineligible_players (final displayable pool)", row_s4)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  SUMMARY for '{player_name}' on {date_str}")
    print("═" * 60)
    for stage, row in results.items():
        if row is None:
            verdict = "❌ ABSENT"
        else:
            status_val = row.get("status", "?")
            mins_val = row.get("proj_minutes", row.get("minutes", "?"))
            eligible_val = row.get("sim_eligible", "?")
            verdict = f"✅ PRESENT  status={status_val}  proj_minutes={mins_val}  sim_eligible={eligible_val}"
        print(f"  {stage:<45} {verdict}")
    print()

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trace a player through the YakOS player pool pipeline."
    )
    parser.add_argument("date", help="Slate date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("player", help="Full player display name")
    parser.add_argument("--draft-group-id", type=int, default=None, dest="dg_id")
    parser.add_argument("--rapidapi-key", default="", dest="api_key")
    parser.add_argument("--proj-source", default="model", dest="proj_source")
    args = parser.parse_args()

    debug_player_pipeline(
        slate_date=args.date,
        player_name=args.player,
        draft_group_id=args.dg_id,
        rapidapi_key=args.api_key,
        proj_source=args.proj_source,
    )
