#!/usr/bin/env python3
"""Test: verify bust/risk signals are wired into tier classifier.

Ensures no player appears in BOTH a positive tier (core/leverage/value)
AND as the bust call.
"""
import sys
import os
import numpy as np
import pandas as pd

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_synthetic_pool(n=50):
    """Create a synthetic test DataFrame with ~50 players."""
    rng = np.random.RandomState(42)
    names = [f"Player_{i:02d}" for i in range(n)]
    salaries = rng.choice(range(3500, 11000, 100), size=n)
    projs = rng.uniform(10, 45, size=n).round(1)
    ownership = rng.uniform(1, 35, size=n).round(1)
    edge_scores = rng.uniform(-1.5, 3.5, size=n).round(2)
    ceil = projs * rng.uniform(1.1, 1.6, size=n)
    floor = projs * rng.uniform(0.4, 0.8, size=n)
    sim90th = projs * rng.uniform(1.2, 1.5, size=n)
    sim_leverage = rng.uniform(-25, 40, size=n).round(1)
    injury_bump = rng.choice([0, 0, 0, 0, 2, 5, 8], size=n).astype(float)
    proj_minutes = rng.uniform(12, 38, size=n).round(1)
    original_proj = projs - injury_bump

    # Bust-related signals
    rolling_fp_5 = projs * rng.uniform(0.6, 1.1, size=n)  # some players overpriced
    spread = rng.uniform(-8, 12, size=n).round(1)
    blowout_risk = rng.choice([0.0, 0.0, 0.0, 0.3, 0.6, 0.9], size=n)
    dvp_rank = rng.choice(range(1, 31), size=n)

    # Make a few players clearly high-risk (should become fades/bust)
    # Player_00: high ownership, bad form, tough matchup, blowout
    rolling_fp_5[0] = projs[0] * 0.55  # way below proj
    dvp_rank[0] = 28  # tough matchup
    spread[0] = 10  # wrong side
    blowout_risk[0] = 0.9
    ownership[0] = 30  # high owned
    salaries[0] = 8500
    projs[0] = 35

    return pd.DataFrame({
        "player_name": names,
        "salary": salaries,
        "proj": projs,
        "ownership": ownership,
        "edge_score": edge_scores,
        "ceil": ceil.round(1),
        "floor": floor.round(1),
        "sim90th": sim90th.round(1),
        "sim_leverage": sim_leverage,
        "injury_bump_fp": injury_bump,
        "proj_minutes": proj_minutes,
        "original_proj": original_proj.round(1),
        "rolling_fp_5": rolling_fp_5.round(1),
        "spread": spread,
        "blowout_risk": blowout_risk,
        "dvp_rank": dvp_rank,
        "status": "Active",
    })


def main():
    # Try real parquet first
    parquet_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "published", "nba", "slate_pool.parquet",
    )
    if os.path.exists(parquet_path):
        print(f"[test] Loading real slate from {parquet_path}")
        pool = pd.read_parquet(parquet_path)
        print(f"[test] Loaded {len(pool)} players, columns: {list(pool.columns)}")
        # Ensure edge_score exists for classification
        if "edge_score" not in pool.columns and "edge" not in pool.columns:
            pool["edge_score"] = np.random.uniform(-1, 3, len(pool)).round(2)
    else:
        print("[test] No real parquet found, using synthetic data")
        pool = _make_synthetic_pool(50)

    # Import _classify_plays from app/tabs/lab_tab.py
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_proj_root, "app", "tabs"))
    sys.path.insert(0, os.path.join(_proj_root, "app"))
    sys.path.insert(0, _proj_root)

    from app.tabs.lab_tab import _classify_plays

    print("\n" + "=" * 60)
    print("Running _classify_plays()...")
    print("=" * 60)
    result = _classify_plays(pool, sport="NBA")

    core_names = {p["player_name"] for p in result.get("core_plays", [])}
    leverage_names = {p["player_name"] for p in result.get("leverage_plays", [])}
    value_names = {p["player_name"] for p in result.get("value_plays", [])}
    fade_names = {p["player_name"] for p in result.get("fade_candidates", [])}
    positive_names = core_names | leverage_names | value_names

    print(f"\nCore ({len(core_names)}): {sorted(core_names)}")
    print(f"Leverage ({len(leverage_names)}): {sorted(leverage_names)}")
    print(f"Value ({len(value_names)}): {sorted(value_names)}")
    print(f"Fade ({len(fade_names)}): {sorted(fade_names)}")

    # Print risk_score stats from the pool (we need to re-run to get the df with risk_score)
    # The function modifies a copy, so let's compute risk_score here for stats
    if "risk_score" not in pool.columns:
        # Run classify on a copy to get risk_score populated
        _tmp = pool.copy()
        _classify_plays(_tmp, sport="NBA")

    # Look for risk_score in fade_candidates records
    fade_risks = [p.get("risk_score", "N/A") for p in result.get("fade_candidates", [])]
    core_risks = [p.get("risk_score", "N/A") for p in result.get("core_plays", [])]
    print(f"\nCore risk_scores: {core_risks}")
    print(f"Fade risk_scores: {fade_risks}")

    # Import generate_bust_call
    from yak_core.rickys_take import generate_bust_call

    print("\n" + "=" * 60)
    print("Running generate_bust_call()...")
    print("=" * 60)
    bust = generate_bust_call(
        pool,
        fade_candidates=result.get("fade_candidates", []),
        positive_tier_names=positive_names,
    )

    if bust:
        bust_name = bust["name"]
        print(f"\nBust call: {bust_name} (${bust['salary']})")
        print(f"Explanation: {bust['explanation']}")
    else:
        bust_name = None
        print("\nNo bust call generated")

    # ── VERIFICATION ──
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    passed = True

    # Check 1: No player in both positive tier AND bust call
    if bust_name and bust_name in positive_names:
        print(f"FAIL: {bust_name} is BOTH in a positive tier and the bust call!")
        passed = False
    else:
        print(f"PASS: Bust call ({bust_name}) does NOT appear in any positive tier")

    # Check 2: Verify risk_score distribution
    # Re-compute risk_score on the SAME filtered set (excluding OUT/IR/WD/Suspended)
    def _safe_col(frame, name, default=0):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index)

    _filtered_pool = pool.copy()
    _REMOVE_STATUSES = {"OUT", "IR", "SUSPENDED", "WD"}
    if "status" in _filtered_pool.columns:
        _filtered_pool = _filtered_pool[
            ~_filtered_pool["status"].fillna("").str.strip().str.upper().isin(_REMOVE_STATUSES)
        ].reset_index(drop=True)

    _proj = _safe_col(_filtered_pool, "proj")
    _rolling_fp_5 = _safe_col(_filtered_pool, "rolling_fp_5")
    _spread = _safe_col(_filtered_pool, "spread")
    _blowout_risk = _safe_col(_filtered_pool, "blowout_risk")
    _dvp_rank = _safe_col(_filtered_pool, "dvp_rank")

    _risk_score = pd.Series(0.0, index=_filtered_pool.index)
    _form_gap = _proj - _rolling_fp_5.where(_rolling_fp_5 > 0, _proj)
    _form_gap_norm = (_form_gap / _proj.clip(lower=1)).clip(lower=0)
    _risk_score += _form_gap_norm * 30
    _dvp_filled = _dvp_rank.where(_dvp_rank > 0, 15)
    _risk_score += (_dvp_filled / 30) * 25
    _spread_risk = _spread.clip(lower=0) / 10
    _risk_score += _spread_risk * 15
    _risk_score += _blowout_risk * 10
    _risk_max = _risk_score.max()
    if _risk_max > 0:
        _risk_score = (_risk_score / _risk_max) * 100

    p80 = np.percentile(_risk_score.dropna(), 80)
    p85 = np.percentile(_risk_score.dropna(), 85)
    print(f"\nRisk score distribution:")
    print(f"  Min:    {_risk_score.min():.1f}")
    print(f"  Median: {_risk_score.median():.1f}")
    print(f"  P80:    {p80:.1f}")
    print(f"  P85:    {p85:.1f}")
    print(f"  Max:    {_risk_score.max():.1f}")

    # Check 3: No positive-tier player has risk >= p80
    for tier_name, tier_players in [("core", result["core_plays"]),
                                     ("leverage", result["leverage_plays"]),
                                     ("value", result["value_plays"])]:
        for p in tier_players:
            rs = p.get("risk_score")
            if rs is not None and rs >= p80:
                print(f"FAIL: {p['player_name']} in {tier_name} has risk_score={rs} >= p80={p80:.1f}")
                passed = False

    if passed:
        print("\nALL CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
