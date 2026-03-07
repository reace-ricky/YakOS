"""yak_core.injury_cascade -- Injury cascade projections for YakOS.

When a key player is OUT or IR, their projected minutes are redistributed to
eligible teammates.  This raises teammate projections so the optimizer and sim
automatically account for the opportunity.

Algorithm (Sprint 3 — headroom-weighted):
  3.1  Find key injuries: OUT/IR players with proj_minutes >= 20.
  3.2  Redistribute minutes using HEADROOM model:
       - Each teammate's weight = headroom × position_boost × rotation_tier
       - headroom = MAX_PLAYER_MINUTES - current proj_minutes (room to grow)
       - position_boost: same position group = 2.5×, adjacent = 1.5×, else 1.0×
       - rotation_tier: mid-rotation (12-22 min) = 1.5×, low-rotation = 0.8×,
         starter-lite = 1.0×, starters = 0.4×, deep bench = 0.3×
       - Capped at MAX_PLAYER_MINUTES per player.
  3.3  Recalculate: adjusted_proj = original_proj + extra_mins × fp_per_minute.
       Store original_proj, adjusted_proj, injury_bump_fp.
  3.4  Update proj = adjusted_proj so all downstream consumers are unaffected.
  3.5  Return cascade report: list of {out_player, team, out_proj_mins,
       beneficiaries: [{name, original_proj, adjusted_proj, bump, salary,
       new_value_multiple}]}.

Backtest results (30-day, 134 events, 303 spikes):
  - Top-3 beneficiary accuracy: 40%  (was 22% with old baseline-weighted model)
  - Top-1 hit rate:             37%  (was  9%)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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
KEY_INJURY_MIN_MINUTES: float = 20.0

# Hard cap on any player's total projected minutes after all bumps
MAX_PLAYER_MINUTES: float = 40.0

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


def _headroom(proj_minutes: float) -> float:
    """Return the minutes headroom (room to grow before hitting ceiling)."""
    return max(MAX_PLAYER_MINUTES - proj_minutes, 0.0)


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

            # Headroom = space to grow
            hr = max(MAX_PLAYER_MINUTES - player_mins - already, 0.0)
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

        total_weight = sum(weights.values())
        if total_weight <= 0:
            continue

        # ── Distribute minutes proportionally ───────────────────────────
        injury_bumps: Dict[int, float] = {}

        for idx2, w in weights.items():
            share = w / total_weight
            extra = out_mins * share

            # Cap at remaining headroom
            player_mins = float(
                pd.to_numeric(df.at[idx2, "proj_minutes"], errors="coerce") or 0
            )
            already = cum_extra.get(idx2, 0.0)
            space = max(MAX_PLAYER_MINUTES - player_mins - already, 0.0)
            extra = min(extra, space)

            if extra > 0:
                injury_bumps[idx2] = extra

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
            fp_per_min = orig_proj / orig_mins if orig_mins > 0 else 0.0
            bump_fp = round(extra * fp_per_min, 2)
            adj_proj = round(orig_proj + bump_fp, 2)
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
        fp_per_min = orig_proj / orig_mins if orig_mins > 0 else 0.0
        total_bump = round(total_extra * fp_per_min, 2)
        df.at[idx, "adjusted_proj"] = round(orig_proj + total_bump, 2)
        df.at[idx, "injury_bump_fp"] = total_bump

    # Make proj = adjusted_proj so all downstream consumers pick up the change
    df["proj"] = df["adjusted_proj"]

    return df, cascade_report


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
