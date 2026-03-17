"""yak_core.injury_cascade -- Injury cascade projections for YakOS.

When a key player is OUT or IR, their projected minutes are redistributed to
eligible teammates.  This raises teammate projections so the optimizer and sim
automatically account for the opportunity.

Algorithm (Sprint 4 — primary-backup-focused redistribution):
  3.1  Find key injuries: OUT/IR players with proj_minutes >= 20.
  3.2  Redistribute minutes using HEADROOM model with primary backup boost:
       - Each teammate's weight = headroom × position_boost × rotation_tier
       - headroom = MAX_PLAYER_MINUTES - current proj_minutes (room to grow)
       - position_boost: same position group = 2.5×, adjacent = 1.5×, else 1.0×
       - rotation_tier: mid-rotation (12-22 min) = 1.5×, low-rotation = 0.8×,
         starter-lite = 1.0×, starters = 0.4×, deep bench = 0.3×
       - primary_backup_boost: the same-pos player in the 12–28 min range with
         the highest projected minutes gets a 2.0× weight multiplier, capped at
         _PRIMARY_BACKUP_MAX_EXTRA_MINS (12) extra minutes per injury; overflow
         is redistributed proportionally to remaining teammates.
       - Capped at MAX_PLAYER_MINUTES per player.
  3.3  Recalculate: adjusted_proj = original_proj + extra_mins × fp_per_minute.
       Store original_proj, adjusted_proj, injury_bump_fp.
  3.4  Update proj = adjusted_proj so all downstream consumers are unaffected.
  3.5  Return cascade report: list of {out_player, team, out_proj_mins,
       beneficiaries: [{name, original_proj, adjusted_proj, bump, salary,
       new_value_multiple}]}.

Backtest results (30-day, 134 events, 303 spikes):
  - Top-3 beneficiary accuracy: 40%  (was 22% with old baseline-weighted model)
  - Top-1 hit rate:             37% → targeting 50%+ with primary backup boost
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Position grouping constants
# ---------------------------------------------------------------------------
_GUARDS = {"PG", "SG", "G"}
_WINGS = {"SF", "SG", "G", "F"}
_BIGS = {"PF", "C", "F"}
_BACKCOURT = {"PG", "SG", "G"}
_FRONTCOURT = {"SF", "PF", "C", "F"}

# Statuses that mark a player as OUT for cascade purposes
_OUT_STATUSES = {"OUT", "IR"}

# Threshold: only players projected for >= this many minutes are "key injuries"
KEY_INJURY_MIN_MINUTES: float = 15.0

# Hard cap on any player's total projected minutes after all bumps
MAX_PLAYER_MINUTES: float = 40.0

# Max bump multiplier: cascade bump cannot exceed this × original projection.
# A player projected for 20 FP can get at most +10 FP (0.5×20), totaling 30 FP.
_MAX_BUMP_MULTIPLIER: float = 0.50  # Max injury bump = 50% of base projection
_MAX_TOTAL_BUMP_MULTIPLIER: float = 0.65  # Max total bump (injury + minutes gap combined)

# ---------------------------------------------------------------------------
# Position matching
# ---------------------------------------------------------------------------

_POSITION_BOOST_SAME: float = 2.5
_POSITION_BOOST_ADJ: float = 1.5

# Rotation tier multipliers (backtested against 303 teammate_out spikes)
_TIER_DEEP_BENCH: float = 0.3    # < 5 min baseline
_TIER_LOW_ROTATION: float = 0.8  # 5-12 min
_TIER_MID_ROTATION: float = 1.5  # 12-22 min  ← primary beneficiaries
_TIER_STARTER_LITE: float = 1.0  # 22-28 min
_TIER_STARTER: float = 0.4       # 28+ min    ← near ceiling, limited upside

# Primary backup boost: concentrate redistribution on the most likely direct
# replacement (same position, 12–28 min projected, highest minute count).
_PRIMARY_BACKUP_BOOST_MULT: float = 2.0   # weight multiplier for primary backup
_PRIMARY_BACKUP_MAX_EXTRA_MINS: float = 12.0  # hard cap on extra mins per injury

# ---------------------------------------------------------------------------
# Position-based FP/min floor rates (NBA)
# ---------------------------------------------------------------------------
# Cheap players ($3.5K-$5.5K) have low projected FP/min (~1.15) because their
# projections are already low.  But cascade beneficiaries get starter-quality
# minutes and can produce well above their baseline rate.  These floors prevent
# the cascade bump from being unrealistically small for value-priced players.
_POS_FPMIN_FLOOR: Dict[str, float] = {
    "PG": 1.3,
    "SG": 1.2,
    "SF": 1.25,
    "PF": 1.35,
    "C":  1.4,
}

# Cascade beneficiaries get starter-quality minutes — apply a small multiplier
# to the floor rate to reflect higher-quality opportunity.
_CASCADE_OPPORTUNITY_MULT: float = 1.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _primary_pos(pos: str) -> str:
    """Return the primary (first) position token, upper-cased."""
    return str(pos).split("/")[0].strip().upper()


def _pos_group(pos: str) -> str:
    """Return 'backcourt', 'frontcourt', or 'other' for a position string."""
    p = _primary_pos(pos)
    if p in _BACKCOURT:
        return "backcourt"
    if p in _FRONTCOURT:
        return "frontcourt"
    return "other"


def _same_pos_group(pos1: str, pos2: str) -> bool:
    """Check if two positions are in the same functional group."""
    p1 = _primary_pos(pos1)
    p2 = _primary_pos(pos2)
    if p1 in _GUARDS and p2 in _GUARDS:
        return True
    if p1 in _WINGS and p2 in _WINGS:
        return True
    if p1 in _BIGS and p2 in _BIGS:
        return True
    if p1 == p2:
        return True
    return False


def _adjacent_pos_group(pos1: str, pos2: str) -> bool:
    """Check if two positions are in adjacent groups (guard/wing or wing/big)."""
    p1 = _primary_pos(pos1)
    p2 = _primary_pos(pos2)
    if (p1 in _GUARDS and p2 in _WINGS) or (p1 in _WINGS and p2 in _GUARDS):
        return True
    if (p1 in _WINGS and p2 in _BIGS) or (p1 in _BIGS and p2 in _WINGS):
        return True
    return False


def _position_boost(player_pos: str, out_pos: str) -> float:
    """Return the positional boost multiplier for a beneficiary."""
    if _same_pos_group(player_pos, out_pos):
        return _POSITION_BOOST_SAME
    if _adjacent_pos_group(player_pos, out_pos):
        return _POSITION_BOOST_ADJ
    return 1.0


def _rotation_tier(proj_minutes: float) -> float:
    """Return the rotation tier multiplier based on current projected minutes."""
    if proj_minutes < 5:
        return _TIER_DEEP_BENCH
    if proj_minutes < 12:
        return _TIER_LOW_ROTATION
    if proj_minutes < 22:
        return _TIER_MID_ROTATION
    if proj_minutes < 28:
        return _TIER_STARTER_LITE
    return _TIER_STARTER


def _realistic_ceiling(rolling_min_10: Optional[float], proj_minutes: float) -> float:
    """Return a player-specific minutes ceiling based on recent usage.

    Uses rolling_min_10 (10-game rolling avg minutes) to set a realistic cap.
    A deep bench player (1.4 min avg) gets ceiling 20, not 40.
    A rotation player (15 min avg) gets ceiling 37.5.
    Falls back to proj_minutes * 2.5 if rolling_min_10 is unavailable.
    """
    if rolling_min_10 is not None and rolling_min_10 > 0:
        base = rolling_min_10
    elif proj_minutes > 0:
        base = proj_minutes
    else:
        return MAX_PLAYER_MINUTES
    return min(MAX_PLAYER_MINUTES, max(base * 2.5, 20.0))


def _headroom(proj_minutes: float, ceiling: Optional[float] = None) -> float:
    """Return the minutes headroom (room to grow before hitting ceiling)."""
    cap = ceiling if ceiling is not None else MAX_PLAYER_MINUTES
    return max(cap - proj_minutes, 0.0)


def _effective_fp_per_min(orig_proj: float, orig_mins: float, pos: str) -> float:
    """Return the FP/min rate for cascade bump calculation.

    Uses the higher of the player's own rate and the position-based floor
    (with the opportunity multiplier applied).  This prevents cheap players
    from getting unrealistically small cascade bumps.
    """
    player_rate = orig_proj / orig_mins if orig_mins > 0 else 0.0
    primary = _primary_pos(pos)
    floor_rate = _POS_FPMIN_FLOOR.get(primary, 1.2) * _CASCADE_OPPORTUNITY_MULT
    return max(player_rate, floor_rate)


def _find_primary_backup_idx(
    eligible_weights: Dict[int, float],
    eligible_df: pd.DataFrame,
    out_pos: str,
) -> Optional[int]:
    """Return the DataFrame index of the most likely direct replacement, or None.

    The primary backup is the same-pos teammate in the 12–28 min range with
    the highest projected minutes.  Returns None when no candidate exists.
    """
    candidates: Dict[int, float] = {}
    for idx in eligible_weights:
        player_pos = _primary_pos(str(eligible_df.at[idx, "pos"]))
        player_mins = float(
            pd.to_numeric(eligible_df.at[idx, "proj_minutes"], errors="coerce") or 0
        )
        if player_pos == out_pos and 12.0 <= player_mins < 28.0:
            candidates[idx] = player_mins
    if not candidates:
        return None
    return max(candidates, key=lambda i: candidates[i])


def _primary_backup_boost(
    eligible_weights: Dict[int, float],
    eligible_df: pd.DataFrame,
    out_pos: str,
    out_minutes: float,
) -> Dict[int, float]:
    """Concentrate redistribution on the most likely direct replacement.

    Heuristic: The primary backup is the teammate with:
    1. Same primary position as the OUT player
    2. Currently in mid-rotation or starter-lite tier (12-28 min)
    3. Highest current projected minutes among same-position candidates

    The primary backup gets a 2.0x weight multiplier. If no clear
    primary backup exists (e.g., position is covered by committee),
    weights are returned unchanged.
    """
    primary_idx = _find_primary_backup_idx(eligible_weights, eligible_df, out_pos)
    if primary_idx is None:
        return eligible_weights
    boosted = dict(eligible_weights)
    boosted[primary_idx] = boosted[primary_idx] * _PRIMARY_BACKUP_BOOST_MULT
    return boosted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_key_injuries(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows for OUT/IR players with proj_minutes >= KEY_INJURY_MIN_MINUTES.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool.  Must contain ``status`` and ``proj_minutes`` columns.

    Returns
    -------
    pd.DataFrame
        Subset of *pool_df* rows that qualify as key injuries, or an empty
        DataFrame when there are none.
    """
    if pool_df is None or pool_df.empty:
        return pd.DataFrame()
    if "status" not in pool_df.columns or "proj_minutes" not in pool_df.columns:
        return pd.DataFrame()

    out_mask = pool_df["status"].fillna("").str.upper().isin(_OUT_STATUSES)
    mins_mask = (
        pd.to_numeric(pool_df["proj_minutes"], errors="coerce").fillna(0)
        >= KEY_INJURY_MIN_MINUTES
    )
    return pool_df[out_mask & mins_mask].copy()


def apply_injury_cascade(
    pool_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Apply headroom-weighted injury-cascade projections to a player pool.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool after standard projections have been applied.  Should
        contain columns: ``player_name``, ``team``, ``pos``, ``salary``,
        ``proj``, ``proj_minutes``, ``status``.

    Returns
    -------
    updated_pool : pd.DataFrame
        Pool with three new / updated columns:

        * ``original_proj`` – projection before any cascade adjustments.
        * ``adjusted_proj`` – projection after cascade (= original + bump).
        * ``injury_bump_fp`` – the FP delta attributable to injury minutes.
        * ``proj`` – updated to equal ``adjusted_proj`` so all downstream
          consumers (optimizer, sim, display) pick up the change
          transparently.

    cascade_report : list of dict
        One entry per key-injured player::

            {
                "out_player": str,
                "team": str,
                "out_proj_mins": float,
                "beneficiaries": [
                    {
                        "name": str,
                        "original_proj": float,
                        "adjusted_proj": float,
                        "bump": float,
                        "salary": int,
                        "new_value_multiple": float,
                        "headroom": float,
                        "position_boost": float,
                        "rotation_tier": float,
                    },
                    ...
                ]
            }
    """
    if pool_df is None or pool_df.empty:
        return pool_df, []

    df = pool_df.copy()

    # Ensure required columns exist with safe defaults
    if "proj_minutes" not in df.columns:
        if "minutes" in df.columns:
            df["proj_minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
        else:
            df["proj_minutes"] = 0.0
    if "proj" not in df.columns:
        df["proj"] = 0.0
    if "status" not in df.columns:
        df["status"] = "Active"
    if "pos" not in df.columns:
        df["pos"] = ""

    # Snapshot original projections before any cascade changes
    df["original_proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0.0)
    df["adjusted_proj"] = df["original_proj"].copy()
    df["injury_bump_fp"] = 0.0

    key_injuries = find_key_injuries(df)
    if key_injuries.empty:
        return df, []

    # Cumulative extra minutes accumulated per player index (for cap enforcement)
    cum_extra: Dict[int, float] = {idx: 0.0 for idx in df.index}
    # Per-player realistic minutes ceiling (populated during redistribution)
    player_ceilings: Dict[int, float] = {}

    cascade_report: List[Dict] = []

    for _, out_row in key_injuries.iterrows():
        out_name = str(out_row.get("player_name", ""))
        out_team = str(out_row.get("team", "")).upper()
        out_pos = _primary_pos(str(out_row.get("pos", "")))
        out_mins = float(
            pd.to_numeric(out_row.get("proj_minutes", 0), errors="coerce") or 0
        )

        if out_mins <= 0:
            continue

        # ── Eligible teammates ──────────────────────────────────────────
        team_mask = df["team"].fillna("").str.upper() == out_team
        not_out_mask = ~df["status"].fillna("").str.upper().isin(_OUT_STATUSES)
        not_self_mask = df["player_name"] != out_name
        eligible = df[team_mask & not_out_mask & not_self_mask]

        if eligible.empty:
            continue

        # ── Compute headroom-weighted redistribution ────────────────────
        weights: Dict[int, float] = {}
        weight_details: Dict[int, Dict] = {}

        for idx2 in eligible.index:
            player_mins = float(
                pd.to_numeric(df.at[idx2, "proj_minutes"], errors="coerce") or 0
            )
            player_pos = _primary_pos(str(df.at[idx2, "pos"]))
            already = cum_extra.get(idx2, 0.0)

            # Player-specific realistic minutes ceiling
            r10 = None
            if "rolling_min_10" in df.columns:
                r10_raw = pd.to_numeric(df.at[idx2, "rolling_min_10"], errors="coerce")
                if pd.notna(r10_raw):
                    r10 = float(r10_raw)
            ceiling = _realistic_ceiling(r10, player_mins)
            player_ceilings[idx2] = ceiling

            # Headroom = space to grow (using realistic ceiling)
            hr = max(ceiling - player_mins - already, 0.0)
            if hr <= 0:
                continue

            # Position boost
            pb = _position_boost(player_pos, out_pos)

            # Rotation tier
            rt = _rotation_tier(player_mins)

            # Combined weight
            w = hr * pb * rt
            weights[idx2] = max(w, 0.1)
            weight_details[idx2] = {"headroom": round(hr, 1), "pos_boost": pb, "tier": rt}

        # Apply primary backup boost before normalizing
        weights = _primary_backup_boost(weights, eligible, out_pos, out_mins)

        # Identify primary backup index for the single-injury cap (reuses helper)
        primary_backup_idx: Optional[int] = _find_primary_backup_idx(weights, eligible, out_pos)

        total_weight = sum(weights.values())
        if total_weight <= 0:
            continue

        # ── Distribute minutes proportionally ───────────────────────────
        injury_bumps: Dict[int, float] = {}
        overflow_mins: float = 0.0

        for idx2, w in weights.items():
            share = w / total_weight
            extra = out_mins * share

            # Cap at remaining headroom (using realistic ceiling)
            player_mins = float(
                pd.to_numeric(df.at[idx2, "proj_minutes"], errors="coerce") or 0
            )
            already = cum_extra.get(idx2, 0.0)
            ceiling = player_ceilings.get(idx2, MAX_PLAYER_MINUTES)
            space = max(ceiling - player_mins - already, 0.0)
            extra = min(extra, space)

            # Primary backup: cap extra minutes from this single injury at 12
            if idx2 == primary_backup_idx and extra > _PRIMARY_BACKUP_MAX_EXTRA_MINS:
                overflow_mins += extra - _PRIMARY_BACKUP_MAX_EXTRA_MINS
                extra = _PRIMARY_BACKUP_MAX_EXTRA_MINS

            if extra > 0:
                injury_bumps[idx2] = extra

        # Redistribute overflow to secondary players (proportional to their weights)
        if overflow_mins > 0 and primary_backup_idx is not None:
            sec_weights = {idx: w for idx, w in weights.items() if idx != primary_backup_idx}
            sec_total = sum(sec_weights.values())
            if sec_total > 0:
                for idx2, w in sec_weights.items():
                    additional = overflow_mins * (w / sec_total)
                    player_mins = float(
                        pd.to_numeric(df.at[idx2, "proj_minutes"], errors="coerce") or 0
                    )
                    already = cum_extra.get(idx2, 0.0)
                    existing = injury_bumps.get(idx2, 0.0)
                    ceiling = player_ceilings.get(idx2, MAX_PLAYER_MINUTES)
                    space = max(ceiling - player_mins - already - existing, 0.0)
                    additional = min(additional, space)
                    if additional > 0:
                        injury_bumps[idx2] = injury_bumps.get(idx2, 0.0) + additional

        # Update cumulative tracker
        for idx2, extra in injury_bumps.items():
            cum_extra[idx2] = cum_extra.get(idx2, 0.0) + extra

        # ── Build cascade report entry ───────────────────────────────────
        beneficiaries: List[Dict] = []
        for idx2, extra in injury_bumps.items():
            if extra <= 0:
                continue
            orig_proj = float(df.at[idx2, "original_proj"])
            orig_mins = float(
                pd.to_numeric(df.at[idx2, "proj_minutes"], errors="coerce") or 0
            )
            player_pos = str(df.at[idx2, "pos"])
            fp_per_min = _effective_fp_per_min(orig_proj, orig_mins, player_pos)
            bump_fp = round(extra * fp_per_min, 2)

            # Cap per-injury bump by remaining bump room (_MAX_BUMP_MULTIPLIER)
            already_bumped = float(df.at[idx2, "injury_bump_fp"])
            remaining_bump_room = max(orig_proj * _MAX_BUMP_MULTIPLIER - already_bumped, 0.0)
            bump_fp = min(bump_fp, round(remaining_bump_room, 2))

            # Track accumulated bump so next injury iteration sees it
            df.at[idx2, "injury_bump_fp"] = already_bumped + bump_fp

            adj_proj = round(orig_proj + already_bumped + bump_fp, 2)
            sal = float(df.at[idx2, "salary"])
            new_val = round(adj_proj / (sal / 1000.0), 2) if sal > 0 else 0.0
            details = weight_details.get(idx2, {})
            beneficiaries.append(
                {
                    "name": str(df.at[idx2, "player_name"]),
                    "original_proj": round(orig_proj, 2),
                    "adjusted_proj": adj_proj,
                    "bump": bump_fp,
                    "extra_minutes": round(extra, 1),
                    "salary": int(sal),
                    "new_value_multiple": new_val,
                    "headroom": details.get("headroom", 0),
                    "position_boost": details.get("pos_boost", 1.0),
                    "rotation_tier": details.get("tier", 1.0),
                }
            )

        beneficiaries.sort(key=lambda x: x["bump"], reverse=True)

        if beneficiaries:
            out_proj_fp = round(
                float(pd.to_numeric(out_row.get("original_proj", out_row.get("proj", 0)), errors="coerce") or 0),
                2,
            )
            cascade_report.append(
                {
                    "out_player": out_name,
                    "team": out_team,
                    "out_proj_mins": round(out_mins, 1),
                    "out_proj_fp": out_proj_fp,
                    "beneficiaries": beneficiaries,
                }
            )

    # ── Final projection update using total accumulated extra minutes ───
    for idx in df.index:
        total_extra = cum_extra.get(idx, 0.0)
        if total_extra <= 0:
            continue
        orig_proj = float(df.at[idx, "original_proj"])
        orig_mins = float(
            pd.to_numeric(df.at[idx, "proj_minutes"], errors="coerce") or 0
        )
        player_pos = str(df.at[idx, "pos"])
        fp_per_min = _effective_fp_per_min(orig_proj, orig_mins, player_pos)
        total_bump = round(total_extra * fp_per_min, 2)

        # Safety net: cap bump at _MAX_BUMP_MULTIPLIER × original projection
        max_bump = orig_proj * _MAX_BUMP_MULTIPLIER
        total_bump = min(total_bump, round(max_bump, 2))

        df.at[idx, "adjusted_proj"] = round(orig_proj + total_bump, 2)
        df.at[idx, "injury_bump_fp"] = total_bump

    # Make proj = adjusted_proj so all downstream consumers pick up the change
    df["proj"] = df["adjusted_proj"]

    # Scale ceil and floor proportionally so proj never exceeds ceil.
    # If original proj was 20 and ceil was 35, that's a 1.75× ratio.
    # After a +10 cascade bump (proj→30), ceil should be 30 × 1.75 = 52.5.
    _orig = pd.to_numeric(df["original_proj"], errors="coerce").fillna(0)
    _adj = pd.to_numeric(df["adjusted_proj"], errors="coerce").fillna(0)
    _bumped = _adj > _orig  # only touch rows that actually got bumped
    if _bumped.any() and "ceil" in df.columns:
        _ceil = pd.to_numeric(df["ceil"], errors="coerce").fillna(0)
        # Ratio of ceil-to-proj (before cascade); clamp to >= 1.0
        _ratio = (_ceil / _orig.clip(lower=0.1)).clip(lower=1.0)
        df.loc[_bumped, "ceil"] = (_adj[_bumped] * _ratio[_bumped]).round(2)
    if _bumped.any() and "floor" in df.columns:
        _floor = pd.to_numeric(df["floor"], errors="coerce").fillna(0)
        _floor_ratio = (_floor / _orig.clip(lower=0.1)).clip(lower=0.0, upper=1.0)
        df.loc[_bumped, "floor"] = (_adj[_bumped] * _floor_ratio[_bumped]).round(2)

    return df, cascade_report


# ---------------------------------------------------------------------------
# Minutes Gap Detection & Redistribution
# ---------------------------------------------------------------------------
# When multiple players are injured/removed from the DK slate entirely
# (long-term injuries not in the pool), the cascade can't see them.
# This detects teams whose active players project well below 240 total
# minutes and redistributes the gap using headroom/rotation_tier weighting.

_TEAM_TOTAL_MINUTES: float = 240.0
_MINUTES_GAP_THRESHOLD: float = 30.0


def apply_minutes_gap_redistribution(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Redistribute unaccounted team minutes to active players.

    Runs AFTER ``apply_injury_cascade``.  For each team whose active players
    project to well below 240 total minutes (gap > 30), the missing minutes
    are distributed proportionally using headroom × rotation_tier weighting.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool (post-cascade).  Must contain ``proj_minutes``, ``proj``,
        ``team``, ``status`` columns.

    Returns
    -------
    pd.DataFrame
        Pool with updated ``proj_minutes`` and ``proj``, plus new columns
        ``minutes_gap_bump_min`` and ``minutes_gap_bump_fp``.
    """
    if pool_df is None or pool_df.empty:
        return pool_df

    df = pool_df.copy()

    # Ensure required columns
    if "proj_minutes" not in df.columns:
        return df
    if "proj" not in df.columns:
        return df

    df["proj_minutes"] = pd.to_numeric(df["proj_minutes"], errors="coerce").fillna(0.0)
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0.0)

    # Initialise tracking columns
    df["minutes_gap_bump_min"] = 0.0
    df["minutes_gap_bump_fp"] = 0.0

    # Identify active players (not OUT/IR)
    if "status" in df.columns:
        active_mask = ~df["status"].fillna("").str.strip().str.upper().isin(_OUT_STATUSES)
    else:
        active_mask = pd.Series(True, index=df.index)

    teams = df.loc[active_mask, "team"].fillna("").str.upper().unique()

    for team in teams:
        if not team:
            continue

        team_active_mask = (
            active_mask
            & (df["team"].fillna("").str.upper() == team)
        )
        team_idx = df.index[team_active_mask]
        if len(team_idx) == 0:
            continue

        total_proj_min = float(df.loc[team_idx, "proj_minutes"].sum())
        gap = _TEAM_TOTAL_MINUTES - total_proj_min

        if gap <= _MINUTES_GAP_THRESHOLD:
            continue

        # Compute weights: headroom × rotation_tier for each active player
        weights: Dict[int, float] = {}
        for idx in team_idx:
            pm = float(df.at[idx, "proj_minutes"])
            hr = max(MAX_PLAYER_MINUTES - pm, 0.0)
            if hr <= 0:
                continue
            rt = _rotation_tier(pm)
            w = hr * rt
            weights[idx] = max(w, 0.1)

        total_weight = sum(weights.values())
        if total_weight <= 0:
            continue

        # Distribute gap minutes proportionally
        for idx, w in weights.items():
            share = w / total_weight
            extra_min = gap * share

            # Cap at MAX_PLAYER_MINUTES
            current_min = float(df.at[idx, "proj_minutes"])
            space = max(MAX_PLAYER_MINUTES - current_min, 0.0)
            extra_min = min(extra_min, space)

            if extra_min <= 0:
                continue

            # FP bump: use player's fp_per_min rate
            current_proj = float(df.at[idx, "proj"])
            if current_min > 0:
                fp_per_min = current_proj / current_min
            else:
                fp_per_min = 1.0
            bump_fp = round(extra_min * fp_per_min, 2)

            # Cap: total bumps (injury + minutes gap) cannot exceed
            # _MAX_TOTAL_BUMP_MULTIPLIER × original projection.
            orig_proj = float(df.at[idx, "original_proj"]) if "original_proj" in df.columns else current_proj
            existing_injury_bump = float(df.at[idx, "injury_bump_fp"]) if "injury_bump_fp" in df.columns else 0.0
            total_bump_so_far = existing_injury_bump
            max_total_bump = orig_proj * _MAX_TOTAL_BUMP_MULTIPLIER
            room = max(max_total_bump - total_bump_so_far, 0.0)
            bump_fp = min(bump_fp, round(room, 2))

            if bump_fp <= 0:
                continue

            df.at[idx, "proj_minutes"] = round(current_min + extra_min, 1)
            df.at[idx, "proj"] = round(current_proj + bump_fp, 2)
            df.at[idx, "minutes_gap_bump_min"] = round(extra_min, 1)
            df.at[idx, "minutes_gap_bump_fp"] = bump_fp

            # Update adjusted_proj if it exists (keep in sync)
            if "adjusted_proj" in df.columns:
                df.at[idx, "adjusted_proj"] = df.at[idx, "proj"]

    return df


# ---------------------------------------------------------------------------
# Return-watch deflation (Sprint 3.5 — injury speed & coverage)
# ---------------------------------------------------------------------------

# When a previously-OUT player returns, their teammates' cascade bumps should
# be reversed (partially or fully depending on return confidence).
# This prevents stale cascade projections from inflating beneficiaries who
# no longer have extra minutes.

RETURN_DEFLATION_FULL: float = 1.0    # Active / Probable → full reversal
RETURN_DEFLATION_PARTIAL: float = 0.5  # GTD → half reversal (may still sit)


def apply_return_watch_deflation(
    pool_df: pd.DataFrame,
    return_players: list,
) -> tuple:
    """Reverse cascade bumps for teammates of returning players.

    When a player was OUT and has returned (or upgraded to Active/Probable),
    their teammates' ``injury_bump_fp`` should be deflated because the minutes
    that were redistributed are now reclaimed.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool that has already had ``apply_injury_cascade()`` applied.
        Must contain ``player_name``, ``team``, ``injury_bump_fp``,
        ``original_proj``, ``adjusted_proj``.
    return_players : list of dict
        Each dict: ``player_name``, ``team``, ``new_status``.
        From ``injury_monitor.get_return_watch_players()``.

    Returns
    -------
    (pool_df, deflation_report)
        pool_df : pd.DataFrame with deflated projections.
        deflation_report : list of dict with deflation details.
    """
    if pool_df is None or pool_df.empty or not return_players:
        return pool_df, []

    df = pool_df.copy()
    if "injury_bump_fp" not in df.columns:
        return df, []

    report = []

    for rp in return_players:
        ret_name = rp.get("player_name", "")
        ret_team = str(rp.get("team", "")).upper()
        ret_status = rp.get("new_status", "Active")

        if not ret_name or not ret_team:
            continue

        # Determine deflation factor based on return confidence
        if ret_status in ("Active", "Probable"):
            deflation = RETURN_DEFLATION_FULL
        elif ret_status in ("GTD", "Questionable"):
            deflation = RETURN_DEFLATION_PARTIAL
        else:
            deflation = RETURN_DEFLATION_FULL

        # Find teammates with cascade bumps
        team_mask = df["team"].fillna("").str.upper() == ret_team
        bump_mask = df["injury_bump_fp"].fillna(0) > 0
        not_self = df["player_name"] != ret_name
        affected = df[team_mask & bump_mask & not_self]

        if affected.empty:
            continue

        entry = {
            "returning_player": ret_name,
            "team": ret_team,
            "new_status": ret_status,
            "deflation_factor": deflation,
            "affected_players": [],
        }

        for idx in affected.index:
            old_bump = float(df.at[idx, "injury_bump_fp"])
            reduction = round(old_bump * deflation, 2)
            new_bump = round(old_bump - reduction, 2)

            df.at[idx, "injury_bump_fp"] = new_bump
            orig = float(df.at[idx, "original_proj"])
            df.at[idx, "adjusted_proj"] = round(orig + new_bump, 2)
            df.at[idx, "proj"] = df.at[idx, "adjusted_proj"]

            entry["affected_players"].append({
                "name": str(df.at[idx, "player_name"]),
                "old_bump": old_bump,
                "new_bump": new_bump,
                "reduction": reduction,
            })

        report.append(entry)

    return df, report
