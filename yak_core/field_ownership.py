"""Heuristic field builder for YakOS ownership projection.

Generates plausible field lineups via weighted random sampling (no LP solver),
then computes own_proj as the fraction of generated lineups containing each player.

Key design decisions:
- No PuLP / LP optimizer — pure Python weighted random sampling for speed.
- Contest-type aware: NBA Classic (8-player), NBA Showdown (6-player), PGA (6-golfer).
- Reproducible via random_seed in config.
- Does NOT call or modify the main optimizer.
- Does NOT re-load projections; uses pool exactly as passed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from yak_core.config import (
    DK_LINEUP_SIZE,
    DK_POS_SLOTS,
    DK_PGA_LINEUP_SIZE,
    DK_SHOWDOWN_LINEUP_SIZE,
    DK_SHOWDOWN_SLOTS,
    SALARY_CAP,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Lineup:
    """A single generated field lineup."""
    player_ids: List[str]
    total_salary: int = 0
    proj_fp: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# DK NBA Classic slot eligibility (mirrors ownership.py _eligible_slots)
_POS_TO_SLOTS: Dict[str, tuple] = {}


def _eligible_slots_classic(pos_str: str) -> tuple:
    """Map a position string to eligible DK Classic slots."""
    if not isinstance(pos_str, str) or not pos_str.strip():
        return ("UTIL",)
    parts = [p.strip().upper() for p in pos_str.split("/")]
    slots: set = set()
    for p in parts:
        if p in ("PG", "SG"):
            slots.add(p)
            slots.add("G")
        elif p in ("SF", "PF"):
            slots.add(p)
            slots.add("F")
        elif p == "C":
            slots.add("C")
    slots.add("UTIL")
    return tuple(sorted(slots))


def _detect_sport(pool: pd.DataFrame, config: Dict[str, Any]) -> str:
    """Auto-detect sport from pool or config."""
    sport = str(config.get("sport", config.get("SPORT", ""))).upper()
    if sport in ("NBA", "PGA"):
        return sport
    # Heuristic: PGA pools typically have no multi-position eligibility
    if "pos" in pool.columns:
        unique_pos = set(str(p).upper().strip() for p in pool["pos"].dropna())
        if unique_pos and unique_pos.issubset({"G", ""}):
            return "PGA"
    return "NBA"


def _is_showdown(contest_type: str) -> bool:
    """Return True if this is a Showdown contest type."""
    ct = contest_type.lower()
    return "showdown" in ct


def _player_id_col(pool: pd.DataFrame) -> str:
    """Return the best available player identifier column."""
    for col in ("player_id", "dk_player_id", "player_name"):
        if col in pool.columns:
            return col
    raise KeyError(
        "Player pool must have a 'player_id', 'dk_player_id', or 'player_name' column."
    )


def _proj_col(pool: pd.DataFrame) -> str:
    """Return the projection column to use."""
    for col in ("proj_fp", "proj", "ceil", "median_proj"):
        if col in pool.columns:
            vals = pd.to_numeric(pool[col], errors="coerce").fillna(0)
            if (vals > 0).any():
                return col
    return "salary"   # absolute last resort


def _build_nba_classic_lineup(
    players: List[Dict[str, Any]],
    scores: np.ndarray,
    salary_cap: int,
    rng: np.random.Generator,
    max_attempts: int = 20,
) -> Optional[List[int]]:
    """Sample one valid NBA Classic lineup (8 players, salary-capped) via greedy random.

    Uses weighted sampling with positional filling. Returns list of player indices
    or None if a valid lineup couldn't be built within max_attempts.
    """
    slots_needed = list(DK_POS_SLOTS)  # ["PG","SG","SF","PF","C","G","F","UTIL"]

    for _ in range(max_attempts):
        selected_indices: List[int] = []
        remaining_slots = slots_needed.copy()
        remaining_salary = salary_cap
        used: set = set()

        # Fill position-specific slots first, then UTIL/G/F as catch-alls
        prio_slots = ["PG", "SG", "SF", "PF", "C"]
        flex_slots = ["G", "F", "UTIL"]

        all_slot_order = prio_slots + flex_slots

        ok = True
        for slot in all_slot_order:
            if slot not in remaining_slots:
                continue
            candidates_mask = np.array([
                i not in used
                and slot in players[i]["_slots"]
                and players[i]["salary"] <= remaining_salary
                for i in range(len(players))
            ])
            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) == 0:
                ok = False
                break

            cand_scores = scores[candidate_indices]
            total = cand_scores.sum()
            if total <= 0:
                probs = np.ones(len(candidate_indices)) / len(candidate_indices)
            else:
                probs = cand_scores / total

            chosen_local = rng.choice(len(candidate_indices), p=probs)
            chosen = candidate_indices[chosen_local]
            selected_indices.append(chosen)
            remaining_slots.remove(slot)
            remaining_salary -= players[chosen]["salary"]
            used.add(chosen)

        if ok and len(selected_indices) == len(slots_needed):
            return selected_indices

    return None


def _build_nba_showdown_lineup(
    players: List[Dict[str, Any]],
    scores: np.ndarray,
    salary_cap: int,
    rng: np.random.Generator,
    captain_multiplier: float = 1.5,
    max_attempts: int = 20,
) -> Optional[List[int]]:
    """Sample one valid NBA Showdown lineup (1 CPT + 5 FLEX) via greedy random.

    Returns list of player indices; no player may appear in more than one slot.
    CPT slot costs captain_multiplier × base salary.
    """
    # CPT slot costs captain_multiplier × salary
    for _ in range(max_attempts):
        selected_indices: List[int] = []
        remaining_salary = salary_cap
        used: set = set()

        # Pick CPT first
        cpt_eligible = np.array([
            players[i]["salary"] * captain_multiplier <= remaining_salary
            for i in range(len(players))
        ])
        cpt_idx_pool = np.where(cpt_eligible)[0]
        if len(cpt_idx_pool) == 0:
            continue

        cpt_scores = scores[cpt_idx_pool]
        total = cpt_scores.sum()
        if total <= 0:
            probs = np.ones(len(cpt_idx_pool)) / len(cpt_idx_pool)
        else:
            probs = cpt_scores / total

        chosen_local = rng.choice(len(cpt_idx_pool), p=probs)
        cpt_chosen = cpt_idx_pool[chosen_local]
        selected_indices.append(cpt_chosen)
        remaining_salary -= int(players[cpt_chosen]["salary"] * captain_multiplier)
        used.add(cpt_chosen)

        # Fill 5 FLEX slots
        ok = True
        for _ in range(DK_SHOWDOWN_LINEUP_SIZE - 1):
            flex_eligible = np.array([
                i not in used and players[i]["salary"] <= remaining_salary
                for i in range(len(players))
            ])
            flex_pool = np.where(flex_eligible)[0]
            if len(flex_pool) == 0:
                ok = False
                break
            flex_scores = scores[flex_pool]
            total = flex_scores.sum()
            if total <= 0:
                probs = np.ones(len(flex_pool)) / len(flex_pool)
            else:
                probs = flex_scores / total
            chosen_local = rng.choice(len(flex_pool), p=probs)
            chosen = flex_pool[chosen_local]
            selected_indices.append(chosen)
            remaining_salary -= players[chosen]["salary"]
            used.add(chosen)

        if ok and len(selected_indices) == DK_SHOWDOWN_LINEUP_SIZE:
            return selected_indices

    return None


def _build_pga_lineup(
    players: List[Dict[str, Any]],
    scores: np.ndarray,
    salary_cap: int,
    rng: np.random.Generator,
    lineup_size: int = 6,
    max_attempts: int = 20,
) -> Optional[List[int]]:
    """Sample one valid PGA lineup (6 golfers, no position constraints) via greedy random."""
    for _ in range(max_attempts):
        selected_indices: List[int] = []
        remaining_salary = salary_cap
        used: set = set()
        ok = True

        for _ in range(lineup_size):
            eligible = np.array([
                i not in used and players[i]["salary"] <= remaining_salary
                for i in range(len(players))
            ])
            pool_idx = np.where(eligible)[0]
            if len(pool_idx) == 0:
                ok = False
                break
            pool_scores = scores[pool_idx]
            total = pool_scores.sum()
            if total <= 0:
                probs = np.ones(len(pool_idx)) / len(pool_idx)
            else:
                probs = pool_scores / total

            chosen_local = rng.choice(len(pool_idx), p=probs)
            chosen = pool_idx[chosen_local]
            selected_indices.append(chosen)
            remaining_salary -= players[chosen]["salary"]
            used.add(chosen)

        if ok and len(selected_indices) == lineup_size:
            return selected_indices

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_field_lineups(
    pool: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> List[Lineup]:
    """Build heuristic field lineups via weighted random sampling.

    Parameters
    ----------
    pool : pd.DataFrame
        Player pool for the slate.  Must have at minimum: salary, position/pos.
        Better results when proj / proj_fp are present.  Columns supported:
        player_id, dk_player_id, player_name, salary, proj_fp, proj, pos,
        position, ceil, minutes.
    config : dict, optional
        Configuration keys:
        - n_field_lineups (int): number of lineups to generate.  Default 5000.
        - random_seed (int): reproducibility seed.  Default 42.
        - contest_type (str): e.g. "gpp_main", "showdown_gpp".  Default "gpp_main".
        - salary_cap (int): override DK cap.  Default from config.py.
        - alpha (float): value-score boost exponent.  Default 0.3.
        - sport (str): "NBA" or "PGA" (auto-detected if absent).

    Returns
    -------
    list[Lineup]
        Each Lineup has player_ids (list of str), total_salary (int), proj_fp (float).
    """
    if config is None:
        config = {}

    n_field_lineups: int = int(config.get("n_field_lineups", 5000))
    random_seed: int = int(config.get("random_seed", 42))
    contest_type: str = str(config.get("contest_type", "gpp_main"))
    alpha: float = float(config.get("alpha", 0.3))

    sport = _detect_sport(pool, config)
    showdown = _is_showdown(contest_type)

    # ── Defaults from config ──────────────────────────────────────────────
    from yak_core.config import DK_PGA_SALARY_CAP, DK_PGA_LINEUP_SIZE
    if sport == "PGA":
        default_cap = DK_PGA_SALARY_CAP
        default_lineup_size = DK_PGA_LINEUP_SIZE
    else:
        default_cap = SALARY_CAP
        default_lineup_size = DK_LINEUP_SIZE

    salary_cap: int = int(config.get("salary_cap", default_cap))
    lineup_size: int = int(config.get("lineup_size", default_lineup_size))

    # ── Validate pool ─────────────────────────────────────────────────────
    if pool.empty:
        warnings.warn("[field_ownership] Empty pool passed to build_field_lineups.")
        return []

    if "salary" not in pool.columns:
        raise ValueError("Player pool must have a 'salary' column.")

    # ── Resolve identifier column ──────────────────────────────────────────
    id_col = _player_id_col(pool)

    # ── Resolve projection column ─────────────────────────────────────────
    proj_col = _proj_col(pool)

    # ── Build player records ──────────────────────────────────────────────
    df = pool.copy().reset_index(drop=True)

    # Normalise position column
    pos_col = "pos" if "pos" in df.columns else ("position" if "position" in df.columns else None)

    # Ensure salary is numeric
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    df[proj_col] = pd.to_numeric(df[proj_col], errors="coerce").fillna(0.0)

    # Pre-compute eligible slots for NBA Classic
    if sport == "NBA" and not showdown:
        df["_slots"] = df[pos_col].apply(_eligible_slots_classic) if pos_col else [("UTIL",)] * len(df)

    # Compute player scores: proj_fp weighted by value
    proj_vals = df[proj_col].values.astype(float)
    sal_vals = df["salary"].values.astype(float)

    # Avoid division by zero
    sal_safe = np.where(sal_vals > 0, sal_vals, 1.0)
    value_ratio = proj_vals / (sal_safe / 1000.0)  # proj per $1K

    # score = proj * (value_ratio ** alpha), clipped to avoid zero
    raw_scores = proj_vals * (np.maximum(value_ratio, 0.001) ** alpha)
    raw_scores = np.maximum(raw_scores, 0.001)  # ensure all positive

    players_list = df.to_dict("records")

    # ── Random sampling ───────────────────────────────────────────────────
    rng = np.random.default_rng(random_seed)

    from yak_core.config import DK_SHOWDOWN_CAPTAIN_MULTIPLIER
    captain_multiplier = float(config.get("captain_multiplier", DK_SHOWDOWN_CAPTAIN_MULTIPLIER))

    field_lineups: List[Lineup] = []
    max_global_attempts = n_field_lineups * 5  # stop if too many failures

    attempt = 0
    while len(field_lineups) < n_field_lineups and attempt < max_global_attempts:
        attempt += 1

        if sport == "PGA":
            selected = _build_pga_lineup(
                players_list, raw_scores, salary_cap, rng, lineup_size=lineup_size
            )
        elif showdown:
            selected = _build_nba_showdown_lineup(
                players_list, raw_scores, salary_cap, rng,
                captain_multiplier=captain_multiplier,
            )
        else:
            selected = _build_nba_classic_lineup(
                players_list, raw_scores, salary_cap, rng
            )

        if selected is None:
            continue

        pids = [str(players_list[i][id_col]) for i in selected]
        total_sal = sum(int(players_list[i]["salary"]) for i in selected)
        total_proj = sum(float(players_list[i].get(proj_col, 0.0)) for i in selected)
        field_lineups.append(Lineup(player_ids=pids, total_salary=total_sal, proj_fp=total_proj))

    if len(field_lineups) < n_field_lineups:
        warnings.warn(
            f"[field_ownership] Only generated {len(field_lineups)}/{n_field_lineups} "
            f"valid lineups after {attempt} attempts."
        )

    print(
        f"[field_ownership] Built {len(field_lineups)} lineups "
        f"({sport}/{contest_type}, salary_cap={salary_cap})"
    )
    return field_lineups


def estimate_ownership_from_field(
    field_lineups: List[Lineup],
) -> pd.DataFrame:
    """Compute own_proj as fraction of field lineups containing each player.

    Parameters
    ----------
    field_lineups : list[Lineup]
        Output of build_field_lineups().

    Returns
    -------
    pd.DataFrame with columns:
        - player_id (str)
        - own_proj  (float in [0, 1])
    """
    if not field_lineups:
        return pd.DataFrame(columns=["player_id", "own_proj"])

    n = len(field_lineups)

    from collections import Counter
    counts: Counter = Counter()
    for lu in field_lineups:
        for pid in lu.player_ids:
            counts[pid] += 1

    rows = [
        {"player_id": pid, "own_proj": count / n}
        for pid, count in counts.items()
    ]
    own_df = pd.DataFrame(rows).sort_values("own_proj", ascending=False).reset_index(drop=True)
    return own_df
