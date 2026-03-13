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

    Uses the same logic as pages/4_right_angle_ricky.py:
    - Core (Chalk): $7K+ salary, top 5 by projection
    - Leverage (GPP Gold): ownership < 15%, top 5 by edge, not in core
    - Value (Salary Savers): salary < $6.5K, top 5 by pts/$1K, not used
    - Fades: high ownership with weakest edge
    """
    import numpy as np
    # Ensure valid ownership data before classification — fixes the None/all-zeros bug
    try:
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from yak_core.ownership_guard import ensure_ownership
        sdf = ensure_ownership(sdf, sport=sport)
    except Exception as _eg:
        print(f"[_classify_plays] ownership_guard unavailable: {_eg}")

    def _safe_col(frame, name, default=0):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index)

    df = sdf.copy()
    _sal = _safe_col(df, "salary")
    _proj = _safe_col(df, "proj")
    _own_col = "ownership" if "ownership" in df.columns and df["ownership"].notna().any() else "own_pct"
    _own = _safe_col(df, _own_col)
    # Normalise ownership to 0-100 range
    if _own.max() <= 1.0 and _own.max() > 0:
        _own = _own * 100
    _edge = _safe_col(df, "edge_composite") if "edge_composite" in df.columns else _safe_col(df, "edge_score")
    _val = np.where(_sal > 0, _proj / (_sal / 1000), 0)
    df["_sal"] = _sal
    df["_proj"] = _proj
    df["_own"] = _own
    df["_edge"] = _edge
    df["_val"] = _val

    is_pga = sport.upper() == "PGA"

    def _to_list(frame, tag: str = ""):
        out = []
        for _, row in frame.iterrows():
            entry = {
                "player_name": row.get("player_name", ""),
                "tag": tag,
                "proj": round(float(row.get("proj", 0)), 1),
                "salary": int(row.get("salary", 0)),
                "ownership": round(float(row.get("_own", 0)), 1),
                "edge": round(float(row.get("_edge", 0)), 2),
                "value": round(float(row.get("_val", 0)), 2),
            }
            # PGA wave data
            if is_pga:
                wave = row.get("early_late_wave")
                entry["wave"] = "Early" if wave in (0, "Early") else "Late" if wave in (1, "Late") else "Unknown"
                teetime = row.get("r1_teetime", "")
                entry["r1_teetime"] = str(teetime) if pd.notna(teetime) else ""
            out.append(entry)
        return out

    # Core (Chalk): top projected players, $7K+ salary
    core = df[df["_sal"] >= 7000].nlargest(5, "_proj")
    _used = set(core["player_name"].tolist())

    # Leverage (GPP Gold): best edge, low ownership (<15%), not in core
    _lev_pool = df[(df["_own"] < 15) & (~df["player_name"].isin(_used))]
    leverage = _lev_pool.nlargest(5, "_edge")
    _used.update(leverage["player_name"].tolist())

    # Value (Salary Savers): best pts/$1K, under $6.5K, not already used
    _val_pool = df[(df["_sal"] < 6500) & (df["_sal"] > 0) & (~df["player_name"].isin(_used))]
    value = _val_pool.nlargest(5, "_val")
    _used.update(value["player_name"].tolist())

    # Fades: high ownership with weakest edge
    _fade_pool = df[~df["player_name"].isin(_used)].copy()
    _fade_high_own = _fade_pool[_fade_pool["_own"] >= 10]
    if len(_fade_high_own) >= 3:
        fades = _fade_high_own.nsmallest(5, "_edge")
    else:
        _fade_sal = _fade_pool[_fade_pool["_sal"] >= 5000]
        fades = _fade_sal.nsmallest(5, "_edge") if not _fade_sal.empty else _fade_pool.nsmallest(5, "_edge")

    return {
        "core_plays": _to_list(core, tag="core"),
        "leverage_plays": _to_list(leverage, tag="leverage"),
        "value_plays": _to_list(value, tag="value"),
        "fade_candidates": _to_list(fades, tag="fade"),
    }


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
