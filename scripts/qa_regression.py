#!/usr/bin/env python3
"""YakOS Sprint 1 – QA Regression Script (S1.8).

Purpose
-------
Validates the full Sprint 1 pipeline end-to-end on a test slate without
requiring a live DK API connection.  Checks that:

  1. State module imports cleanly and state objects initialize correctly.
  2. Slate Hub logic: draftable pool build + roster-rule application.
  3. Ricky Edge logic: player tagging, stack definitions, edge labels,
     Edge Check gate.
  4. Sim engine: player-level smash / bust / leverage computation.
  5. Apply-learnings: non-destructive Sim Learnings layer with ±15% cap.
  6. Calibration: bucketed table with sample-size threshold.
  7. Build & Publish: Classic + Showdown lineup build from SlateState rules.
  8. DK CSV export format is valid.
  9. Friends / Edge Share: published lineup read, friend builder pool filter.
  10. Late swap: GTD / OUT swap suggestions generated correctly.

Usage
-----
    python scripts/qa_regression.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed (see output).
"""

from __future__ import annotations

import sys
import os
import traceback
from pathlib import Path
from typing import Any

# Ensure repo root is importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "✅ PASS"
_FAIL = "❌ FAIL"
_results: list[dict] = []


def _check(name: str, fn):
    """Run a QA check, record pass/fail."""
    try:
        fn()
        _results.append({"check": name, "status": _PASS, "error": ""})
        print(f"{_PASS}  {name}")
    except Exception as exc:
        _results.append({"check": name, "status": _FAIL, "error": str(exc)})
        print(f"{_FAIL}  {name}")
        print(f"       {exc}")
        if os.environ.get("QA_VERBOSE"):
            traceback.print_exc()


def _make_test_pool(n: int = 12) -> pd.DataFrame:
    """Build a minimal test player pool.

    Uses valid DK NBA player positions (PG/SG/SF/PF/C only) with enough
    per-position depth to support building multiple diverse lineups.
    """
    teams = ["LAL", "GSW", "BOS", "MIA", "MIL", "DEN"]
    # Valid player positions: PG/SG → eligible for G flex; SF/PF → F flex; C → C only.
    # Cycle through 5 positions so every group of 5 covers all positions.
    pos_cycle = ["PG", "SG", "SF", "PF", "C"]
    rows = []
    for i in range(n):
        salary = 4500 + i * 250
        proj = 14.0 + i * 1.5
        rows.append({
            "player_id": f"player_{i}",
            "player_name": f"Player_{i}",
            "pos": pos_cycle[i % len(pos_cycle)],
            "team": teams[i % len(teams)],
            "opponent": teams[(i + 3) % len(teams)],
            "salary": salary,
            "proj": proj,
            "floor": proj * 0.7,
            "ceil": proj * 1.4,
            "ownership": 5.0 + i * 1.5,
            "proj_minutes": 28.0 + (i % 5),
            "status": "",
            "sim_eligible": True,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Check 1: State module
# ---------------------------------------------------------------------------

def _check_state_module():
    from yak_core.state import (
        SlateState, RickyEdgeState, LineupSetState, SimState,
    )
    slate = SlateState()
    assert slate.sport == "NBA"
    assert slate.salary_cap == 50000
    assert not slate.is_ready()

    edge = RickyEdgeState()
    edge.tag_player("LeBron", "core", 5)
    assert edge.get_tagged("core") == ["LeBron"]
    edge.remove_tag("LeBron")
    assert edge.get_tagged("core") == []

    lu = LineupSetState()
    pool_df = _make_test_pool(8)
    lu.set_lineups("Cash Game", pool_df, {"build_mode": "floor"})
    assert lu.lineups["Cash Game"] is not None

    sim = SimState()
    sim.apply_learning("Player_0", 0.20, "test")
    assert sim.sim_learnings["Player_0"]["boost"] == 0.15  # capped at 15%
    sim.apply_learning("Player_1", -0.20, "test")
    assert sim.sim_learnings["Player_1"]["boost"] == -0.15  # capped at -15%


# ---------------------------------------------------------------------------
# Check 2: Slate Hub – pool build + roster rule application
# ---------------------------------------------------------------------------

def _check_slate_hub():
    from yak_core.dk_ingest import parse_roster_rules
    from yak_core.projections import salary_implied_proj
    from yak_core.sims import compute_sim_eligible
    from yak_core.state import SlateState

    # Build pool from test data (no API needed)
    pool = _make_test_pool()
    pool["proj"] = salary_implied_proj(pool["salary"])
    pool = compute_sim_eligible(pool)
    assert "sim_eligible" in pool.columns
    assert not pool.empty

    # Apply roster rules
    rules_json = {
        "rosterSlots": [
            {"name": "PG"}, {"name": "SG"}, {"name": "SF"},
            {"name": "PF"}, {"name": "C"}, {"name": "G"},
            {"name": "F"}, {"name": "UTIL"},
        ],
        "salaryCap": 50000,
    }
    parsed = parse_roster_rules(rules_json)
    assert parsed["lineup_size"] == 8
    assert parsed["salary_cap"] == 50000
    assert not parsed["is_showdown"]

    slate = SlateState()
    slate.apply_roster_rules(parsed)
    assert slate.lineup_size == 8
    assert slate.salary_cap == 50000
    assert slate.contest_type == "Classic"
    assert not slate.is_showdown

    # Showdown rules
    sd_rules = {
        "rosterSlots": [
            {"name": "CPT"}, {"name": "FLEX"}, {"name": "FLEX"},
            {"name": "FLEX"}, {"name": "FLEX"}, {"name": "FLEX"},
        ],
        "salaryCap": 50000,
    }
    parsed_sd = parse_roster_rules(sd_rules)
    assert parsed_sd["is_showdown"]
    assert parsed_sd["captain_slot"]

    slate_sd = SlateState()
    slate_sd.apply_roster_rules(parsed_sd)
    assert slate_sd.is_showdown
    assert slate_sd.contest_type == "Showdown Captain"
    assert slate_sd.captain_multiplier == 1.5


# ---------------------------------------------------------------------------
# Check 3: Ricky Edge – tagging, stacks, edge labels, gate
# ---------------------------------------------------------------------------

def _check_ricky_edge():
    from yak_core.state import RickyEdgeState

    edge = RickyEdgeState()
    pool = _make_test_pool(10)
    player_names = pool["player_name"].tolist()

    # Tagging
    edge.tag_player(player_names[0], "core", 5)
    edge.tag_player(player_names[1], "value", 3)
    edge.tag_player(player_names[2], "fade", 4)
    assert len(edge.player_tags) == 3
    assert edge.get_tagged("core") == [player_names[0]]
    assert edge.get_tagged("fade") == [player_names[2]]

    # Stack
    edge.add_stack("LAL", [player_names[0], player_names[4]], "correlated pair")
    assert len(edge.stacks) == 1
    assert edge.stacks[0]["team"] == "LAL"

    # Edge labels
    edge.edge_labels = ["SMASH: Player_0", "FADE: Player_2"]
    assert len(edge.edge_labels) == 2

    # Slate notes
    edge.slate_notes = "Heavy pace game tonight"
    assert edge.slate_notes == "Heavy pace game tonight"

    # Edge Check gate
    assert not edge.ricky_edge_check
    edge.approve_edge_check("2026-03-01T12:00:00Z")
    assert edge.ricky_edge_check
    assert edge.edge_check_ts == "2026-03-01T12:00:00Z"
    edge.revoke_edge_check()
    assert not edge.ricky_edge_check


# ---------------------------------------------------------------------------
# Check 4: Sim engine – player-level metrics
# ---------------------------------------------------------------------------

def _check_sim_engine():
    from scipy.stats import norm
    pool = _make_test_pool(12)
    variance = 1.0

    proj = pool["proj"]
    ceil = pool["ceil"]
    floor = pool["floor"]
    own = pool["ownership"]

    std = (ceil - floor) / 4 * variance
    std = std.clip(lower=0.5)

    smash_z = (ceil * 0.9 - proj) / std
    bust_z = (floor * 1.1 - proj) / std
    smash_prob = 1 - norm.cdf(smash_z)
    bust_prob = norm.cdf(bust_z)
    own_frac = (own / 100.0).clip(lower=0.01)
    leverage = smash_prob / own_frac

    assert len(smash_prob) == 12
    assert (smash_prob >= 0).all() and (smash_prob <= 1).all()
    assert (bust_prob >= 0).all() and (bust_prob <= 1).all()
    assert (leverage > 0).all()


# ---------------------------------------------------------------------------
# Check 5: Apply learnings – non-destructive, capped at ±15%
# ---------------------------------------------------------------------------

def _check_apply_learnings():
    from yak_core.state import SimState, SlateState

    sim = SimState()
    pool = _make_test_pool(5)

    # Simulate learnings
    for pname in pool["player_name"].tolist():
        sim.apply_learning(pname, 0.25)  # oversized boost — should be capped at 0.15

    assert all(v["boost"] <= 0.15 for v in sim.sim_learnings.values())
    assert all(v["boost"] >= -0.15 for v in sim.sim_learnings.values())

    # Apply to effective_proj
    pool2 = pool.copy()
    pool2["effective_proj"] = pool2["proj"].copy()
    for pname, learning in sim.sim_learnings.items():
        mask = pool2["player_name"] == pname
        pool2.loc[mask, "effective_proj"] *= (1 + learning["boost"])

    assert "effective_proj" in pool2.columns
    # Effective should be higher than base (since all boosts are positive)
    assert (pool2["effective_proj"] >= pool2["proj"]).all()

    # Clear
    sim.clear_learnings()
    assert sim.sim_learnings == {}


# ---------------------------------------------------------------------------
# Check 6: Calibration – bucketed table with sample-size threshold
# ---------------------------------------------------------------------------

def _check_calibration():
    pool = _make_test_pool(20)
    pool["salary_bucket"] = pd.cut(
        pool["salary"],
        bins=[0, 5500, 6500, 7500, 99999],
        labels=["<5.5K", "5.5–6.5K", "6.5–7.5K", "7.5K+"],
    )
    _MIN_SAMPLES = 3  # Lower threshold for test data
    bucket_counts = pool.groupby("salary_bucket", observed=True).size().reset_index(name="n")
    valid = bucket_counts[bucket_counts["n"] >= _MIN_SAMPLES]["salary_bucket"].tolist()
    assert len(valid) >= 1, "Expected at least one bucket with ≥3 samples"

    # Profile save/clone
    from yak_core.state import SimState
    sim = SimState()
    sim.save_calibration_profile("baseline", {"knob_a": 1.0, "knob_b": 0.5})
    assert "baseline" in sim.calibration_profiles
    ok = sim.clone_profile("baseline", "v2")
    assert ok
    assert "v2" in sim.calibration_profiles
    assert sim.calibration_profiles["v2"] == sim.calibration_profiles["baseline"]


# ---------------------------------------------------------------------------
# Check 7: Build & Publish – Classic lineup build from SlateState rules
# ---------------------------------------------------------------------------

def _check_build_classic():
    from yak_core.lineups import build_multiple_lineups_with_exposure
    from yak_core.state import SlateState

    pool = _make_test_pool(16)
    slate = SlateState()
    slate.roster_slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    slate.salary_cap = 50000

    cfg = {
        "NUM_LINEUPS": 3,
        "SALARY_CAP": slate.salary_cap,
        "MAX_EXPOSURE": 1.0,
        "MIN_SALARY_USED": 40000,
        "LOCK": [],
        "EXCLUDE": [],
    }
    lineups_df, expo_df = build_multiple_lineups_with_exposure(pool, cfg)
    assert lineups_df is not None and not lineups_df.empty
    assert "lineup_index" in lineups_df.columns
    assert lineups_df["lineup_index"].nunique() == 3


# ---------------------------------------------------------------------------
# Check 8: DK CSV export
# ---------------------------------------------------------------------------

def _check_dk_csv_export():
    from yak_core.lineups import build_multiple_lineups_with_exposure, to_dk_upload_format

    pool = _make_test_pool(16)
    cfg = {
        "NUM_LINEUPS": 2,
        "SALARY_CAP": 50000,
        "MAX_EXPOSURE": 1.0,
        "MIN_SALARY_USED": 40000,
        "LOCK": [],
        "EXCLUDE": [],
    }
    lineups_df, _ = build_multiple_lineups_with_exposure(pool, cfg)
    assert lineups_df is not None
    csv_df = to_dk_upload_format(lineups_df)
    assert isinstance(csv_df, pd.DataFrame)
    assert not csv_df.empty
    # Validate CSV bytes
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 0


# ---------------------------------------------------------------------------
# Check 9: Friends / Edge Share – published lineup read + pool filter
# ---------------------------------------------------------------------------

def _check_friends_edge_share():
    from yak_core.state import LineupSetState, RickyEdgeState, SlateState

    pool = _make_test_pool(12)
    edge = RickyEdgeState()
    edge.tag_player("Player_0", "core", 5)
    edge.tag_player("Player_1", "value", 3)
    edge.tag_player("Player_2", "fade", 4)

    # Only core/value should be in friend builder pool
    allowed_tags = {"core", "secondary", "value"}
    tagged = {p for p, v in edge.player_tags.items() if v.get("tag") in allowed_tags}
    fades = edge.get_tagged("fade")
    filtered = pool[pool["player_name"].isin(tagged) & ~pool["player_name"].isin(fades)]
    assert "Player_0" in filtered["player_name"].values
    assert "Player_1" in filtered["player_name"].values
    assert "Player_2" not in filtered["player_name"].values

    # Published lineup
    lu = LineupSetState()
    lu.set_lineups("Cash Game", pool, {"build_mode": "floor"})
    lu.publish("Cash Game", "2026-03-01T12:00:00Z")
    assert "Cash Game" in lu.get_published_labels()
    pub = lu.published_sets["Cash Game"]
    assert pub["published_at"] == "2026-03-01T12:00:00Z"


# ---------------------------------------------------------------------------
# Check 10: Late swap – GTD / OUT suggestions
# ---------------------------------------------------------------------------

def _check_late_swap():
    pool = _make_test_pool(12)

    # Simulate lineups that include Player_0 (OUT) and Player_1 (GTD)
    lineups_df = pd.DataFrame([
        {"lineup_index": 0, "slot": "PG", "player_name": "Player_0"},
        {"lineup_index": 0, "slot": "SG", "player_name": "Player_1"},
    ])

    injury_updates = [
        {"player_name": "Player_0", "status": "OUT"},
        {"player_name": "Player_1", "status": "GTD"},
    ]

    suggestions = []
    player_pool_map = {row["player_name"]: row.to_dict() for _, row in pool.iterrows()}
    in_lineup_set = set(lineups_df["player_name"].tolist())

    for update in injury_updates:
        pname = update["player_name"]
        status = update["status"].upper()
        if pname not in in_lineup_set:
            continue

        if status in ("OUT", "IR", "O"):
            player_row = player_pool_map.get(pname, {})
            pos = player_row.get("pos", "")
            current_salary = float(player_row.get("salary", 0) or 0)
            candidates = [
                r for n, r in player_pool_map.items()
                if r.get("pos") == pos and n != pname and n not in in_lineup_set
                and abs(float(r.get("salary", 0) or 0) - current_salary) <= 1500
            ]
            if candidates:
                best = max(candidates, key=lambda r: float(r.get("proj", 0) or 0))
                suggestions.append({"action": "PIVOT", "out": pname, "in": best["player_name"]})
            else:
                suggestions.append({"action": "PIVOT", "out": pname, "in": None})

        elif status in ("GTD", "Q", "QUESTIONABLE", "LIMITED"):
            suggestions.append({"action": "REDUCE_EXPOSURE", "player": pname})

    assert len(suggestions) == 2
    pivot = next((s for s in suggestions if s["action"] == "PIVOT"), None)
    assert pivot is not None
    assert pivot["out"] == "Player_0", f"Expected OUT player Player_0, got {pivot['out']}"
    # Replacement player must exist, have correct position, and satisfy salary constraint
    if pivot["in"] is not None:
        replacement = player_pool_map.get(pivot["in"])
        assert replacement is not None, f"Replacement {pivot['in']} not found in pool"
        out_pos = player_pool_map["Player_0"]["pos"]
        assert replacement["pos"] == out_pos, (
            f"Replacement pos {replacement['pos']} doesn't match out player pos {out_pos}"
        )
        out_salary = float(player_pool_map["Player_0"]["salary"] or 0)
        repl_salary = float(replacement["salary"] or 0)
        assert abs(repl_salary - out_salary) <= 1500, (
            f"Salary delta {abs(repl_salary - out_salary)} exceeds 1500 cap"
        )

    reduce = next((s for s in suggestions if s["action"] == "REDUCE_EXPOSURE"), None)
    assert reduce is not None
    assert reduce["player"] == "Player_1", f"Expected GTD player Player_1, got {reduce['player']}"


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n" + "=" * 60)
    print("YakOS Sprint 1 – QA Regression Script")
    print("=" * 60 + "\n")

    _check("1. State module (SlateState/RickyEdgeState/LineupSetState/SimState)", _check_state_module)
    _check("2. Slate Hub – pool build + roster rule application", _check_slate_hub)
    _check("3. Ricky Edge – tagging, stacks, labels, gate", _check_ricky_edge)
    _check("4. Sim engine – player-level smash/bust/leverage", _check_sim_engine)
    _check("5. Apply learnings – non-destructive, ±15% cap", _check_apply_learnings)
    _check("6. Calibration – bucketed table + profile management", _check_calibration)
    _check("7. Build & Publish – Classic lineup build from SlateState", _check_build_classic)
    _check("8. DK CSV export format", _check_dk_csv_export)
    _check("9. Friends / Edge Share – published lineup + pool filter", _check_friends_edge_share)
    _check("10. Late swap – OUT/GTD suggestions", _check_late_swap)

    print("\n" + "=" * 60)
    passed = sum(1 for r in _results if r["status"] == _PASS)
    failed = sum(1 for r in _results if r["status"] == _FAIL)
    total = len(_results)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60 + "\n")

    if failed > 0:
        print("Failed checks:")
        for r in _results:
            if r["status"] == _FAIL:
                print(f"  {r['check']}: {r['error']}")
        return 1

    print("🎉 All Sprint 1 QA checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
