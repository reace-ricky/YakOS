#!/usr/bin/env python3
"""scripts/run_edge.py -- Run edge analysis on a loaded pool.

Reads the pool from data/published/{sport}/slate_pool.parquet,
computes edge metrics and 4-box classification (Core/Leverage/Value/Fades),
and writes results.

Outputs:
  data/published/{sport}/edge_state.json
  data/published/{sport}/edge_analysis.json
  data/published/{sport}/signals.parquet
"""
from __future__ import annotations

import argparse
import json
import sys

from _env import published_dir, today_str  # noqa: E402

import pandas as pd


def _classify_plays(sdf: pd.DataFrame, sport: str = "NBA") -> dict:
    """Classify players into Core / Leverage / Value / Fade buckets.

    Delegates to :func:`yak_core.edge_scoring.classify_plays` which uses
    the multi-factor :class:`~yak_core.edge_scoring.FadeScorer` for fades
    (ownership + ceiling gap + value) rather than the old salary-biased
    heuristic.
    """
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from yak_core.edge_scoring import classify_plays
    return classify_plays(sdf, sport=sport)


def _build_bullets(classified: dict, edge_df: pd.DataFrame, sport: str = "NBA") -> list[str]:
    """Generate human-readable analysis bullet points."""
    bullets = []
    n_core = len(classified["core_plays"])
    n_leverage = len(classified["leverage_plays"])
    n_value = len(classified["value_plays"])
    n_fades = len(classified["fade_candidates"])

    if n_core:
        top_core = ", ".join(p["player_name"] for p in classified["core_plays"][:5])
        bullets.append(f"Anchor studs ({n_core}): {top_core}")
    if n_value:
        top_val = ", ".join(p["player_name"] for p in classified["value_plays"][:5])
        bullets.append(f"Value plays ({n_value}): {top_val}")
    if n_leverage:
        top_lev = ", ".join(p["player_name"] for p in classified["leverage_plays"][:5])
        bullets.append(f"Leverage plays ({n_leverage}): {top_lev}")
    if n_fades:
        bullets.append(f"Fade candidates: {n_fades} players below edge threshold")

    # PGA wave analysis
    if sport.upper() == "PGA" and "early_late_wave" in edge_df.columns:
        early = edge_df[edge_df["early_late_wave"].isin([0, "Early"])]
        late = edge_df[edge_df["early_late_wave"].isin([1, "Late"])]
        if len(early) > 0 and len(late) > 0:
            early_avg = early["proj"].mean()
            late_avg = late["proj"].mean()
            diff = abs(early_avg - late_avg)
            favored = "Early" if early_avg > late_avg else "Late"
            bullets.append(
                f"Wave split: {favored} wave projects +{diff:.1f} pts avg "
                f"(Early {early_avg:.1f} vs Late {late_avg:.1f})"
            )
            # Core plays wave breakdown
            core_waves = [p.get("wave", "?") for p in classified["core_plays"]]
            n_early = core_waves.count("Early")
            n_late = core_waves.count("Late")
            if n_early > 0 or n_late > 0:
                bullets.append(f"Core wave mix: {n_early} Early / {n_late} Late")

    # Signal convergence summary
    if "edge_score" in edge_df.columns:
        strong = (edge_df["edge_score"] >= 2.0).sum()
        if strong > 0:
            bullets.append(f"Strong edge night — {strong} players with 2+ converging signals.")

    return bullets


def run_edge(sport: str, slate_date: str) -> pd.DataFrame:
    """Run edge analysis and write outputs. Returns the edge DataFrame."""
    from yak_core.edge import compute_edge_metrics
    from yak_core.calibration_feedback import get_correction_factors
    from yak_core.ownership_guard import ensure_ownership

    out_dir = published_dir(sport)
    pool_path = f"{out_dir}/slate_pool.parquet"

    try:
        pool = pd.read_parquet(pool_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Pool not found at {pool_path}. Run load_pool.py first.")

    print(f"[run_edge] Loaded {len(pool)} players from {pool_path}")

    # Ensure valid ownership before edge computation
    pool = ensure_ownership(pool, sport=sport)

    # Filter out excluded players (checkbox exclusions from Lab)
    import os
    _excl_path = os.path.join(out_dir, "excluded_players.json")
    if os.path.exists(_excl_path):
        with open(_excl_path) as _ef:
            _excl_names = json.load(_ef)
        if _excl_names:
            pre = len(pool)
            pool = pool[~pool["player_name"].isin(_excl_names)].reset_index(drop=True)
            print(f"[run_edge] Excluded {pre - len(pool)} player(s): {_excl_names}")

    # Pop Catalyst: score situational upside signals before edge computation
    try:
        from yak_core.pop_catalyst import compute_pop_catalyst
        pool = compute_pop_catalyst(pool)
    except Exception as exc:
        print(f"[run_edge] compute_pop_catalyst failed (non-fatal): {exc}")

    # Load calibration state for edge computation
    calibration_state = get_correction_factors(sport=sport.upper())

    # Compute edge metrics
    print("[run_edge] Computing edge metrics ...")
    edge_df = compute_edge_metrics(
        pool,
        calibration_state=calibration_state if calibration_state.get("n_slates", 0) > 0 else None,
        sport=sport.upper(),
    )

    # Classify into 4-box
    classified = _classify_plays(edge_df, sport=sport)
    bullets = _build_bullets(classified, edge_df, sport=sport)

    # Recommendation summary — actually useful context
    n_total = len(edge_df)
    n_core = len(classified["core_plays"])
    n_value = len(classified["value_plays"])
    n_leverage = len(classified["leverage_plays"])

    # Top core salary range
    core_sals = [p["salary"] for p in classified["core_plays"]]
    core_sal_range = f"${min(core_sals):,}–${max(core_sals):,}" if core_sals else ""

    # Value avg pts/$1K
    val_rates = [p["value"] for p in classified["value_plays"] if p["value"] > 0]
    val_avg = f"{sum(val_rates)/len(val_rates):.1f}" if val_rates else "0"

    # Leverage ownership range
    lev_owns = [p["ownership"] for p in classified["leverage_plays"]]
    lev_own_range = f"{min(lev_owns):.0f}–{max(lev_owns):.0f}%" if lev_owns else ""

    rec_parts = []
    rec_parts.append(f"{n_core} core plays anchored at {core_sal_range}")
    if n_value:
        rec_parts.append(f"{n_value} value plays averaging {val_avg} pts/$1K")
    if n_leverage and lev_own_range:
        rec_parts.append(f"{n_leverage} leverage plays at {lev_own_range} ownership")
    recommendation = ". ".join(rec_parts) + "."

    # Edge state (tags and stacks info for lineup builder)
    edge_state = {
        "sport": sport.upper(),
        "date": slate_date,
        "core_names": [p["player_name"] for p in classified["core_plays"]],
        "leverage_names": [p["player_name"] for p in classified["leverage_plays"]],
        "value_names": [p["player_name"] for p in classified["value_plays"]],
        "fade_names": [p["player_name"] for p in classified["fade_candidates"]],
        "calibration_slates": calibration_state.get("n_slates", 0),
    }

    # PGA wave breakdown in edge state (for lineup builder wave-aware builds)
    if sport.upper() == "PGA" and "early_late_wave" in edge_df.columns:
        early_df = edge_df[edge_df["early_late_wave"].isin([0, "Early"])]
        late_df = edge_df[edge_df["early_late_wave"].isin([1, "Late"])]
        edge_state["wave_split"] = {
            "early_count": int(len(early_df)),
            "late_count": int(len(late_df)),
            "early_avg_proj": round(float(early_df["proj"].mean()), 1) if len(early_df) > 0 else 0,
            "late_avg_proj": round(float(late_df["proj"].mean()), 1) if len(late_df) > 0 else 0,
            "early_players": early_df.nlargest(5, "proj")["player_name"].tolist(),
            "late_players": late_df.nlargest(5, "proj")["player_name"].tolist(),
        }

    # Edge analysis (pre-computed for consumers)
    edge_analysis = {
        "bullets": bullets,
        "recommendation": recommendation,
        **classified,
        "signals_df_path": "signals.parquet",
    }

    # Write outputs
    edge_state_path = f"{out_dir}/edge_state.json"
    edge_analysis_path = f"{out_dir}/edge_analysis.json"
    signals_path = f"{out_dir}/signals.parquet"

    with open(edge_state_path, "w") as f:
        json.dump(edge_state, f, indent=2, default=str)
    with open(edge_analysis_path, "w") as f:
        json.dump(edge_analysis, f, indent=2, default=str)
    edge_df.to_parquet(signals_path, index=False)

    print(f"[run_edge] Edge state → {edge_state_path}")
    print(f"[run_edge] Edge analysis → {edge_analysis_path}")
    print(f"[run_edge] Signals ({len(edge_df)} rows) → {signals_path}")
    print(f"[run_edge] Summary: {recommendation}")
    return edge_df


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Run edge analysis on a loaded pool.")
    parser.add_argument("--sport", required=True, choices=["NBA", "PGA"],
                        help="Sport to analyse.")
    parser.add_argument("--date", default=None,
                        help="Slate date (YYYY-MM-DD). Default: today.")
    args = parser.parse_args(argv)

    slate_date = args.date or today_str()
    return run_edge(args.sport.upper(), slate_date)


if __name__ == "__main__":
    main()
