"""yak_core.board -- The Board: Stack Targets, Sniper Spots, and The Fade.

Computes three board sections from the player pool and edge analysis:
  1. Stack Targets (max 2): game stacks with highest combined ceiling,
     gated by vegas_total >= 225.
  2. Sniper Spots (max 3): top-20 by Ricky proj AND <10% ownership,
     excluding Core/Leverage/Value players.
  3. The Fade (max 2): >15% ownership players who rank low in Ricky's
     model relative to salary, excluding Core/Leverage/Value players.

Used by app/edge_tab.py to render The Board section.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

import pandas as pd


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

VEGAS_TOTAL_THRESHOLD = 225.0
SNIPER_OWN_MAX = 10.0          # projected ownership must be below this
SNIPER_TOP_N = 20              # must rank in top N by ricky_proj
SNIPER_MAX = 3                 # max spots to show
FADE_OWN_MIN = 15.0            # must be above this ownership %
FADE_MAX = 2                   # max fades to show
STACK_MAX = 2                  # max stack targets


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _get_own(df: pd.DataFrame) -> pd.Series:
    """Return ownership as 0-100 scale Series."""
    own_col = "ownership" if "ownership" in df.columns and df["ownership"].notna().any() else "own_pct"
    own = _safe_numeric(df.get(own_col, pd.Series(5.0, index=df.index)))
    if own.max() > 0 and own.max() <= 1.0:
        own = own * 100
    return own


def _get_tier_names(edge_analysis: Dict[str, Any]) -> Set[str]:
    """Collect all player names from Core/Leverage/Value tiers."""
    names: Set[str] = set()
    for tier in ("core_plays", "leverage_plays", "value_plays"):
        for p in edge_analysis.get(tier, []):
            name = p.get("player_name", "")
            if name:
                names.add(name)
    return names


def _get_ricky_proj(df: pd.DataFrame) -> pd.Series:
    """Return ricky_proj if available, else fall back to proj."""
    if "ricky_proj" in df.columns and df["ricky_proj"].notna().any():
        return _safe_numeric(df["ricky_proj"])
    return _safe_numeric(df.get("proj", pd.Series(0.0, index=df.index)))


def _get_ricky_ceil(df: pd.DataFrame) -> pd.Series:
    """Return ricky_ceil if available, else fall back to ceil."""
    if "ricky_ceil" in df.columns and df["ricky_ceil"].notna().any():
        return _safe_numeric(df["ricky_ceil"])
    return _safe_numeric(df.get("ceil", pd.Series(0.0, index=df.index)))


def compute_stack_targets(
    pool: pd.DataFrame,
    edge_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Find game stacks with highest combined ceiling, gated by vegas_total.

    Returns up to STACK_MAX entries, each with:
      game_id, team1, team2, vegas_total, top_player1, top_player2, combined_ceil
    """
    if pool is None or pool.empty:
        return []

    df = pool.copy()
    ceil = _get_ricky_ceil(df)
    df["_ceil"] = ceil

    # Determine vegas total per game
    vegas_col = None
    for col in ("over_under", "vegas_total", "total"):
        if col in df.columns and df[col].notna().any():
            vegas_col = col
            break

    if vegas_col is None:
        return []

    df["_vegas"] = _safe_numeric(df[vegas_col])

    # Need game_id or team to group by game
    game_col = None
    for col in ("game_id", "game"):
        if col in df.columns:
            game_col = col
            break

    if game_col is None:
        return []

    # Gate: only games above vegas threshold
    games = df.groupby(game_col).agg(
        vegas_total=("_vegas", "first"),
    ).reset_index()
    eligible_games = games[games["vegas_total"] >= VEGAS_TOTAL_THRESHOLD][game_col].tolist()

    if not eligible_games:
        return []

    results = []
    for game in eligible_games:
        game_players = df[df[game_col] == game].copy()
        if game_players.empty:
            continue

        vegas_total = float(game_players["_vegas"].iloc[0])

        # Get teams in the game
        teams = []
        if "team" in game_players.columns:
            teams = sorted(game_players["team"].dropna().unique().tolist())

        team1 = teams[0] if len(teams) > 0 else "?"
        team2 = teams[1] if len(teams) > 1 else "?"

        # Top 2 players by ceiling in this game
        top = game_players.nlargest(2, "_ceil")
        if len(top) < 2:
            continue

        p1 = top.iloc[0]
        p2 = top.iloc[1]
        combined_ceil = float(p1["_ceil"]) + float(p2["_ceil"])

        results.append({
            "game_id": game,
            "team1": team1,
            "team2": team2,
            "vegas_total": vegas_total,
            "top_player1": p1.get("player_name", "?"),
            "top_player2": p2.get("player_name", "?"),
            "combined_ceil": combined_ceil,
        })

    # Sort by combined ceiling descending, take top STACK_MAX
    results.sort(key=lambda x: x["combined_ceil"], reverse=True)
    return results[:STACK_MAX]


def compute_sniper_spots(
    pool: pd.DataFrame,
    edge_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Find high-ceiling + low-ownership players the field is sleeping on.

    Must rank in top 20 by Ricky projection AND be below 10% projected ownership.
    Excludes anyone in Core/Leverage/Value plays.

    Returns up to SNIPER_MAX entries with:
      player_name, team, salary, proj, ceil, own_pct
    """
    if pool is None or pool.empty:
        return []

    tier_names = _get_tier_names(edge_analysis)
    df = pool.copy()
    ricky_proj = _get_ricky_proj(df)
    ceil = _get_ricky_ceil(df)
    own = _get_own(df)

    df["_ricky_proj"] = ricky_proj
    df["_ceil"] = ceil
    df["_own"] = own

    # Exclude Core/Leverage/Value players
    if tier_names:
        df = df[~df["player_name"].isin(tier_names)].copy()

    if df.empty:
        return []

    # Exclude top-15 by salary — those are the obvious plays everyone sees.
    # Ricky's Plays should surface non-obvious edges.
    if "salary" in df.columns and len(df) > 20:
        sal = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
        top_sal_idx = sal.nlargest(15).index
        df = df.drop(index=top_sal_idx).copy()

    if df.empty:
        return []

    # Prioritize players with a situational edge (injury cascade, pace-up, value)
    # by adding a sniper_priority score
    df["_sniper_priority"] = df["_ricky_proj"].copy()
    # Boost players with injury cascade bump
    if "injury_bump_fp" in df.columns:
        bump = pd.to_numeric(df["injury_bump_fp"], errors="coerce").fillna(0)
        df["_sniper_priority"] += bump * 2.0  # strong boost for cascade beneficiaries
    # Boost players in high-total games
    for vc in ("vegas_total", "over_under", "total"):
        if vc in df.columns:
            vt = pd.to_numeric(df[vc], errors="coerce").fillna(0)
            df["_sniper_priority"] += (vt - 220).clip(lower=0) * 0.1  # small boost for pace-up
            break
    # Boost value plays (high pts/$1K)
    if "salary" in df.columns:
        sal = pd.to_numeric(df["salary"], errors="coerce").fillna(1)
        pts_per_k = df["_ricky_proj"] / (sal / 1000)
        df["_sniper_priority"] += (pts_per_k - 5.0).clip(lower=0) * 1.5

    # Must be in top 20 by sniper priority (not raw projection)
    top_n = df.nlargest(SNIPER_TOP_N, "_sniper_priority")

    # Must be below ownership threshold
    snipers = top_n[top_n["_own"] < SNIPER_OWN_MAX].copy()

    if snipers.empty:
        return []

    # Sort by ricky_proj descending
    snipers = snipers.sort_values("_ricky_proj", ascending=False)

    results = []
    for _, row in snipers.head(SNIPER_MAX + 2).iterrows():  # extra candidates for signal dedup
        floor_val = float(row.get("floor", row.get("sim15th", 0)) or 0)
        results.append({
            "player_name": row.get("player_name", "?"),
            "team": row.get("team", "?"),
            "salary": int(row.get("salary", 0)),
            "proj": float(row["_ricky_proj"]),
            "ceil": float(row["_ceil"]),
            "floor": floor_val,
            "own_pct": float(row["_own"]),
            "injury_bump_fp": float(row.get("injury_bump_fp", 0) or 0),
            "proj_minutes": float(row.get("proj_minutes", 0) or 0),
            "rolling_fp_5": float(row.get("rolling_fp_5", 0) or 0),
            "breakout_score": float(row.get("breakout_score", 0) or 0),
        })

    return results


def compute_fades(
    pool: pd.DataFrame,
    edge_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Find popular players who rank LOW in Ricky's model relative to salary.

    Must have >15% projected ownership and rank poorly vs salary expectations.
    Excludes anyone in Core/Leverage/Value plays.

    Returns up to FADE_MAX entries with:
      player_name, team, salary, proj, own_pct, reasoning
    """
    if pool is None or pool.empty:
        return []

    tier_names = _get_tier_names(edge_analysis)
    df = pool.copy()
    ricky_proj = _get_ricky_proj(df)
    own = _get_own(df)
    salary = _safe_numeric(df.get("salary", pd.Series(0.0, index=df.index)))

    df["_ricky_proj"] = ricky_proj
    df["_own"] = own
    df["_salary"] = salary

    # Exclude Core/Leverage/Value players
    if tier_names:
        df = df[~df["player_name"].isin(tier_names)].copy()

    if df.empty:
        return []

    # Must be above ownership threshold (popular)
    popular = df[df["_own"] > FADE_OWN_MIN].copy()

    if popular.empty:
        return []

    # Rank all players by Ricky projection and by salary
    all_df = df.copy()
    all_df["_ricky_rank"] = all_df["_ricky_proj"].rank(ascending=False, method="min")
    all_df["_salary_rank"] = all_df["_salary"].rank(ascending=False, method="min")

    # Merge ranks back into popular
    popular = popular.merge(
        all_df[["player_name", "_ricky_rank", "_salary_rank"]].drop_duplicates("player_name"),
        on="player_name",
        how="left",
        suffixes=("", "_dup"),
    )

    # Fade score: how much worse Ricky ranks them vs their salary rank
    # A player ranked #30 by Ricky but #5 by salary is a big fade
    popular["_fade_gap"] = popular["_ricky_rank"] - popular["_salary_rank"]

    # Sort by fade gap descending (biggest gap = biggest fade)
    popular = popular.sort_values("_fade_gap", ascending=False)

    # Fade voice lines -- rotate based on player index for variety
    _FADE_VOICE = [
        "The crowd will see a game log. We see a balance sheet of minutes, usage, and blowout risk.",
        "Good players, bad bets. That\u2019s who we\u2019re fading.",
        "Popular and overpriced. The field\u2019s favorite combination.",
    ]

    results = []
    for i, (_, row) in enumerate(popular.head(FADE_MAX).iterrows()):
        ricky_rank = int(row.get("_ricky_rank", 0))
        salary_rank = int(row.get("_salary_rank", 0))
        reasoning = (
            f"Ranked #{ricky_rank} by Ricky vs #{salary_rank} by salary. "
            f"{_FADE_VOICE[i % len(_FADE_VOICE)]}"
        )

        results.append({
            "player_name": row.get("player_name", "?"),
            "team": row.get("team", "?"),
            "salary": int(row.get("_salary", 0)),
            "proj": float(row["_ricky_proj"]),
            "own_pct": float(row["_own"]),
            "reasoning": reasoning,
        })

    return results
