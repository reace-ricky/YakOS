"""yak_core.scoring -- lineup scoring, backtest, and KPI functions."""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


def score_lineups(
    lineups: List[Dict],
    pool_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score a list of lineups against pool data.
    Each lineup is a dict with a 'players' key (list of player names).
    Returns a DataFrame with per-lineup totals.
    """
    rows = []
    for i, lu in enumerate(lineups):
        players = lu["players"]
        mask = pool_df["name"].isin(players)
        sub = pool_df.loc[mask]
        total_proj = sub["proj"].sum() if "proj" in sub.columns else 0.0
        total_actual = sub["actual_fp"].sum() if "actual_fp" in sub.columns else 0.0
        total_salary = sub["salary"].sum() if "salary" in sub.columns else 0
        rows.append({
            "lineup_id": i,
            "total_proj": round(total_proj, 2),
            "total_actual": round(total_actual, 2),
            "total_salary": int(total_salary),
            "proj_vs_actual": round(total_actual - total_proj, 2),
            "salary_remaining": 50000 - int(total_salary),
        })
    return pd.DataFrame(rows)


def backtest_summary(ldf: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate backtest KPIs from scored-lineup DataFrame."""
    result = {
        "slate_date": "",
        "num_lineups": len(ldf),
        "avg_proj": round(ldf["total_proj"].mean(), 2),
        "avg_actual": round(ldf["total_actual"].mean(), 2),
        "best_lineup_actual": round(ldf["total_actual"].max(), 2),
        "worst_lineup_actual": round(ldf["total_actual"].min(), 2),
        "avg_salary_used": int(ldf["total_salary"].mean()),
        "avg_salary_remaining": int(ldf["salary_remaining"].mean()),
        "pct_lineups_beat_proj": round(
            (ldf["total_actual"] >= ldf["total_proj"]).mean() * 100, 1
        ),
    }
    return result


def projection_pct(
    lineups: List[Dict],
    pool_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Projection %% for each lineup.
    projection_pct = lineup_total_proj / max_possible_proj * 100
    where max_possible_proj is the sum of the top-8 projections in the pool.
    Returns DataFrame with lineup_id, total_proj, max_proj, proj_pct.
    """
    # Max possible = top 8 proj values in pool (DK NBA = 8 roster spots)
    top_n = 8
    if "proj" not in pool_df.columns:
        raise ValueError("pool_df must have a 'proj' column")
    max_proj = pool_df.nlargest(top_n, "proj")["proj"].sum()
    rows = []
    for i, lu in enumerate(lineups):
        players = lu["players"]
        mask = pool_df["name"].isin(players)
        total_proj = pool_df.loc[mask, "proj"].sum()
        pct = round(total_proj / max_proj * 100, 1) if max_proj > 0 else 0.0
        rows.append({
            "lineup_id": i,
            "total_proj": round(total_proj, 2),
            "max_proj": round(max_proj, 2),
            "proj_pct": pct,
        })
    return pd.DataFrame(rows)


def ownership_kpis(
    lineups: List[Dict],
    pool_df: pd.DataFrame,
    ownership_col: str = "ownership",
) -> pd.DataFrame:
    """Compute ownership-based KPIs for each lineup.
    Requires an ownership column (0-100 scale) in pool_df.
    Returns DataFrame with lineup_id, own_sum, own_avg, own_max, leverage.
    """
    if ownership_col not in pool_df.columns:
        raise ValueError(f"pool_df must have '{ownership_col}' column")
    rows = []
    for i, lu in enumerate(lineups):
        players = lu["players"]
        mask = pool_df["name"].isin(players)
        owns = pool_df.loc[mask, ownership_col].fillna(0)
        own_sum = round(owns.sum(), 2)
        own_avg = round(owns.mean(), 2) if len(owns) > 0 else 0.0
        own_max = round(owns.max(), 2) if len(owns) > 0 else 0.0
        leverage = round(100 - own_avg, 2)
        rows.append({
            "lineup_id": i,
            "own_sum": own_sum,
            "own_avg": own_avg,
            "own_max": own_max,
            "leverage": leverage,
        })
    return pd.DataFrame(rows)


def print_dashboard(
    lineups: List[Dict],
    pool_df: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Print a text dashboard summarising the optimiser run."""
    ldf = score_lineups(lineups, pool_df)
    result = ldf.copy()

    print("=" * 70)
    print("  YakOS Dashboard")
    print("=" * 70)

    meta = [
        ("Date", config.get("SLATE_DATE", "")),
        ("Site", config.get("SITE", "")),
        ("Sport", config.get("SPORT", "")),
        ("Contest", config.get("CONTEST_TYPE", "")),
        ("Lineups", config.get("NUM_LINEUPS", "")),
        ("Pool size", len(pool_df)),
        ("Profile", config.get("LOGIC_PROFILE", "")),
        ("Band", config.get("BAND", "")),
        ("Salary cap", config.get("SALARY_CAP", 50000)),
        ("Max exposure", config.get("MAX_EXPOSURE", "")),
    ]
    for label, v in meta:
        print(f"  {label:<20s} {v}")
    print()

    # Top 5 lineups by projected
    cols = ["lineup_id", "total_proj", "total_actual", "total_salary"]
    ed = result.sort_values("total_proj", ascending=False)
    print("--- Top 5 Lineups (by proj) ---")
    print(ed[cols].head(10).to_string(index=False))

    # Projection %
    try:
        ppct = projection_pct(lineups, pool_df)
        avg_pct = ppct["proj_pct"].mean()
        max_pct = ppct["proj_pct"].max()
        min_pct = ppct["proj_pct"].min()
        print()
        print("--- Projection %% ---")
        print(f"  Avg proj %%:  {avg_pct:.1f}%%")
        print(f"  Best:        {max_pct:.1f}%%")
        print(f"  Worst:       {min_pct:.1f}%%")
    except ValueError:
        pass

    # Ownership KPIs (if ownership column exists)
    own_col = None
    for c in ["ownership", "OWNERSHIP", "proj_own", "POWN"]:
        if c in pool_df.columns:
            own_col = c
            break
    if own_col:
        try:
            okpi = ownership_kpis(lineups, pool_df, ownership_col=own_col)
            print()
            print("--- Ownership KPIs ---")
            print(f"  Avg lineup own sum:  {okpi['own_sum'].mean():.1f}%%")
            print(f"  Avg lineup own avg:  {okpi['own_avg'].mean():.1f}%%")
            print(f"  Avg leverage score:  {okpi['leverage'].mean():.1f}")
        except ValueError:
            pass

    # Top exposures
    from collections import Counter
    all_players = []
    for lu in lineups:
        all_players.extend(lu["players"])
    ctr = Counter(all_players)
    n = len(lineups)
    print()
    print("--- Top 10 Exposures ---")
    for name, cnt in ctr.most_common(10):
        print(f"  {name:<28s} {cnt:>3d} / {n}  ({cnt/n*100:.0f}%%)")

    # Backtest summary
    if "actual_fp" in ldf.columns:
        bt = backtest_summary(result)
        print()
        print("--- Backtest Summary ---")
        for k, v in bt.items():
            label = str(k).ljust(30, ".")
            print(f"  {label} {v}")
    else:
        print()
        print("[dashboard] No actual_fp -- skipping backtest")
    print()
    print("=" * 70)
    print("  Dashboard complete.")
    print("=" * 70)
