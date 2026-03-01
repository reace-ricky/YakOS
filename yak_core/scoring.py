"""yak_core.scoring -- lineup scoring, backtest, and KPI functions."""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


def _r_squared(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute R² between two series."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


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


def calibration_kpi_summary(hist_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive calibration KPIs from historical lineup data.

    Expects hist_df with columns: lineup_id, proj, actual, salary, pos,
    and optionally proj_own, own, proj_minutes, actual_minutes.

    Returns a dict with:
      - strategy: num_lineups, hit_rate, avg_actual, avg_proj
      - points_lineup: mean_error, std_error, mae, rmse, r_squared, avg_proj, avg_actual
      - points_player: mean_error, mae, r_squared, df (sorted by abs_error)
      - points_salary: df (error by salary bracket)
      - ownership: present only when both proj_own and own are available
          - mean_error, mae, bucket_df (binned proj vs actual own)
      - minutes: present only when both proj_minutes and actual_minutes are available
          - mean_error, mae, impact_df (avg pts error when minutes miss > 5 vs ≤ 5)
    """
    result: Dict[str, Any] = {}

    if hist_df.empty:
        return result

    # ── LINEUP-LEVEL POINTS ──────────────────────────────────────────────────
    agg_cols = {"proj": "sum", "actual": "sum", "salary": "sum"}
    lu_sum = hist_df.groupby("lineup_id").agg(agg_cols).reset_index()
    lu_sum["error"] = lu_sum["actual"] - lu_sum["proj"]

    result["strategy"] = {
        "num_lineups": int(lu_sum["lineup_id"].nunique()),
        "hit_rate": float((lu_sum["actual"] >= lu_sum["proj"]).mean()),
        "avg_actual": float(lu_sum["actual"].mean()),
        "avg_proj": float(lu_sum["proj"].mean()),
        "best_actual": float(lu_sum["actual"].max()),
    }

    result["points_lineup"] = {
        "mean_error": float(lu_sum["error"].mean()),
        "std_error": float(lu_sum["error"].std()),
        "mae": float(lu_sum["error"].abs().mean()),
        "rmse": float(np.sqrt((lu_sum["error"] ** 2).mean())),
        "r_squared": _r_squared(lu_sum["actual"], lu_sum["proj"]),
        "avg_proj": float(lu_sum["proj"].mean()),
        "avg_actual": float(lu_sum["actual"].mean()),
        "df": lu_sum,
    }

    # ── PLAYER-LEVEL POINTS ──────────────────────────────────────────────────
    pl_agg: Dict[str, Any] = {"proj": "mean", "actual": "mean", "salary": "first"}
    if "pos" in hist_df.columns:
        pl_agg["pos"] = "first"
    pl_sum = hist_df.groupby("name").agg(pl_agg).reset_index()
    pl_sum["error"] = pl_sum["actual"] - pl_sum["proj"]
    pl_sum["abs_error"] = pl_sum["error"].abs()

    result["points_player"] = {
        "mean_error": float(pl_sum["error"].mean()),
        "mae": float(pl_sum["abs_error"].mean()),
        "r_squared": _r_squared(pl_sum["actual"], pl_sum["proj"]),
        "df": pl_sum.sort_values("abs_error", ascending=False),
    }

    # ── PLAYER ERROR BY SALARY BRACKET ──────────────────────────────────────
    if "salary" in hist_df.columns:
        pl_sum["salary_bracket"] = pd.cut(
            pl_sum["salary"],
            bins=[0, 5000, 6500, 8000, 20000],
            labels=["<5K", "5-6.5K", "6.5-8K", ">8K"],
        )
        sal_br = (
            pl_sum.groupby("salary_bracket", observed=False)
            .agg(
                avg_proj=("proj", "mean"),
                avg_actual=("actual", "mean"),
                mae=("abs_error", "mean"),
                count=("name", "count"),
            )
            .reset_index()
        )
        sal_br["mean_error"] = sal_br["avg_actual"] - sal_br["avg_proj"]
        result["points_salary"] = {"df": sal_br}

    # ── OWNERSHIP KPIs ───────────────────────────────────────────────────────
    if "proj_own" in hist_df.columns and "own" in hist_df.columns:
        own_df = hist_df[["name", "proj_own", "own"]].copy()
        own_df["proj_own"] = pd.to_numeric(own_df["proj_own"], errors="coerce").fillna(0)
        own_df["own"] = pd.to_numeric(own_df["own"], errors="coerce").fillna(0)
        own_df["error"] = own_df["own"] - own_df["proj_own"]
        own_df["abs_error"] = own_df["error"].abs()

        own_df["bucket"] = pd.cut(
            own_df["proj_own"],
            bins=[0, 5, 10, 20, 100],
            labels=["0–5%", "5–10%", "10–20%", ">20%"],
            include_lowest=True,
        )
        bucket_df = (
            own_df.groupby("bucket", observed=False)
            .agg(
                avg_proj_own=("proj_own", "mean"),
                avg_actual_own=("own", "mean"),
                mae=("abs_error", "mean"),
                count=("name", "count"),
            )
            .reset_index()
        )
        bucket_df["mean_error"] = bucket_df["avg_actual_own"] - bucket_df["avg_proj_own"]

        result["ownership"] = {
            "mean_error": float(own_df["error"].mean()),
            "mae": float(own_df["abs_error"].mean()),
            "bucket_df": bucket_df,
        }

    # ── MINUTES KPIs ─────────────────────────────────────────────────────────
    if "proj_minutes" in hist_df.columns and "actual_minutes" in hist_df.columns:
        min_df = hist_df[["name", "proj_minutes", "actual_minutes", "proj", "actual"]].copy()
        min_df["proj_minutes"] = pd.to_numeric(min_df["proj_minutes"], errors="coerce").fillna(0)
        min_df["actual_minutes"] = pd.to_numeric(min_df["actual_minutes"], errors="coerce").fillna(0)
        min_df["min_error"] = min_df["actual_minutes"] - min_df["proj_minutes"]
        min_df["abs_min_error"] = min_df["min_error"].abs()
        min_df["pts_error"] = min_df["actual"] - min_df["proj"]

        large_miss = min_df[min_df["abs_min_error"] > 5]["pts_error"].mean()
        small_miss = min_df[min_df["abs_min_error"] <= 5]["pts_error"].mean()

        result["minutes"] = {
            "mean_error": float(min_df["min_error"].mean()),
            "mae": float(min_df["abs_min_error"].mean()),
            "avg_pts_err_large_min_miss": float(large_miss) if not np.isnan(large_miss) else 0.0,
            "avg_pts_err_small_min_miss": float(small_miss) if not np.isnan(small_miss) else 0.0,
        }

    return result


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
