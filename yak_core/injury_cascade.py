"""yak_core.injury_cascade -- Injury cascade projections for YakOS.

When a key player is OUT or IR, their projected minutes are redistributed to
eligible teammates.  This raises teammate projections so the optimizer and sim
automatically account for the opportunity.

Algorithm (Sprint 2):
  2.1  Find key injuries: OUT/IR players with proj_minutes >= 20.
  2.2  Redistribute minutes: same-position teammates get 60%, adjacent-group
       (backcourt ↔ frontcourt) get 40%, weighted by existing proj_minutes,
       capped at 40 min per player.
  2.3  Recalculate: adjusted_proj = original_proj + extra_mins × fp_per_minute.
       Store original_proj, adjusted_proj, injury_bump_fp.
  2.4  Update proj = adjusted_proj so all downstream consumers are unaffected.
  2.5  Return cascade report: list of {out_player, team, out_proj_mins,
       beneficiaries: [{name, original_proj, adjusted_proj, bump, salary,
       new_value_multiple}]}.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Position grouping constants
# ---------------------------------------------------------------------------
_BACKCOURT = {"PG", "SG", "G"}
_FRONTCOURT = {"SF", "PF", "C", "F"}

# Statuses that mark a player as OUT for cascade purposes
_OUT_STATUSES = {"OUT", "IR"}

# Threshold: only players projected for >= this many minutes are "key injuries"
KEY_INJURY_MIN_MINUTES: float = 20.0

# Share of redistributed minutes going to same-pos / adjacent-group teammates
_SAME_POS_SHARE: float = 0.60
_ADJ_POS_SHARE: float = 0.40

# Hard cap on any player's total projected minutes after all bumps
MAX_PLAYER_MINUTES: float = 40.0


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
    """Apply injury-cascade projections to a player pool.

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
        df["proj_minutes"] = 0.0
    if "proj" not in df.columns:
        df["proj"] = 0.0
    if "status" not in df.columns:
        df["status"] = "Active"

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
        out_group = _pos_group(out_pos)
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

        # ── Position buckets ────────────────────────────────────────────
        same_pos = eligible[eligible["pos"].apply(_primary_pos) == out_pos]

        if out_group in ("backcourt", "frontcourt"):
            adj_group = "frontcourt" if out_group == "backcourt" else "backcourt"
            adj = eligible[eligible["pos"].apply(_pos_group) == adj_group]
        else:
            adj = pd.DataFrame()

        # ── Per-injury minute distribution ──────────────────────────────
        injury_bumps: Dict[int, float] = {}

        def _distribute(players: pd.DataFrame, share: float) -> None:
            if players.empty or share <= 0:
                return
            weights = pd.to_numeric(
                players["proj_minutes"], errors="coerce"
            ).fillna(0.0)
            total_w = float(weights.sum())
            if total_w <= 0:
                weights = pd.Series(1.0, index=players.index)
                total_w = float(len(players))

            for idx2 in players.index:
                orig_mins = float(
                    pd.to_numeric(
                        df.at[idx2, "proj_minutes"], errors="coerce"
                    )
                    or 0
                )
                already = cum_extra.get(idx2, 0.0)
                space = max(0.0, MAX_PLAYER_MINUTES - orig_mins - already)
                if space <= 0:
                    continue
                w = float(weights[idx2])
                extra = out_mins * share * w / total_w
                extra = min(extra, space)
                injury_bumps[idx2] = injury_bumps.get(idx2, 0.0) + extra

        _distribute(same_pos, _SAME_POS_SHARE)
        _distribute(adj, _ADJ_POS_SHARE)

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
            beneficiaries.append(
                {
                    "name": str(df.at[idx2, "player_name"]),
                    "original_proj": round(orig_proj, 2),
                    "adjusted_proj": adj_proj,
                    "bump": bump_fp,
                    "salary": int(sal),
                    "new_value_multiple": new_val,
                }
            )

        beneficiaries.sort(key=lambda x: x["bump"], reverse=True)

        if beneficiaries:
            cascade_report.append(
                {
                    "out_player": out_name,
                    "team": out_team,
                    "out_proj_mins": round(out_mins, 1),
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
