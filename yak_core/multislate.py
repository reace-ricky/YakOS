"""YakOS Core - Multi-slate comparison and DK contest CSV ingest."""
import os
import re
import glob
import io
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

from .config import YAKOS_ROOT, merge_config
from .lineups import run_lineups_from_config



# ============================================================
# Slate Discovery
# ============================================================
def discover_slates(root: str = YAKOS_ROOT) -> pd.DataFrame:
    """Find all available parquet pool files and extract dates.

    Returns DataFrame with columns: slate_date, filename, path, size_kb.
    Deduplicates by date (prefers _reproj over base).
    """
    pattern = os.path.join(root, "tank_opt_pool_*.parquet")
    files = sorted(glob.glob(pattern))
    rows = []
    for fp in files:
        base = os.path.basename(fp)
        # Extract date from filename
        # Formats: tank_opt_pool_2024-10-22.parquet or tank_opt_pool_20241022.parquet
        m = re.search(r"tank_opt_pool_(\d{4}-\d{2}-\d{2})", base)
        if m:
            date_str = m.group(1)
        else:
            m2 = re.search(r"tank_opt_pool_(\d{8})", base)
            if m2:
                d = m2.group(1)
                date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            else:
                continue
        is_reproj = "_reproj" in base
        size_kb = round(os.path.getsize(fp) / 1024, 1)
        rows.append({
            "slate_date": date_str,
            "filename": base,
            "path": fp,
            "size_kb": size_kb,
            "is_reproj": is_reproj,
        })
    if not rows:
        return pd.DataFrame(columns=["slate_date", "filename", "path", "size_kb"])
    df = pd.DataFrame(rows)
    # Deduplicate: prefer _reproj files over base
    df["priority"] = df["is_reproj"].astype(int)
    df = df.sort_values(["slate_date", "priority"], ascending=[True, False])
    df = df.drop_duplicates(subset="slate_date", keep="first")
    df = df.drop(columns=["is_reproj", "priority"]).reset_index(drop=True)
    return df


# ============================================================
# Multi-Slate Batch Run
# ============================================================
def run_multi_slate(
    slate_dates: List[str],
    base_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the optimizer across multiple historical slates.

    Returns dict with:
      - "per_slate": list of {slate_date, result, error} dicts
      - "summary_df": DataFrame with per-slate KPIs
    """
    if base_cfg is None:
        base_cfg = {}
    per_slate = []
    for sd in slate_dates:
        cfg = dict(base_cfg)
        cfg["SLATE_DATE"] = sd
        cfg["DATA_MODE"] = "historical"
        try:
            result = run_lineups_from_config(cfg)
            per_slate.append({"slate_date": sd, "result": result, "error": None})
        except Exception as e:
            per_slate.append({"slate_date": sd, "result": None, "error": str(e)})

    # Build summary
    summary_rows = []
    for entry in per_slate:
        sd = entry["slate_date"]
        if entry["error"]:
            summary_rows.append({"slate_date": sd, "status": "FAIL", "error": entry["error"]})
            continue
        r = entry["result"]
        ldf = r["lineups_df"]
        pdf = r["pool_df"]
        meta = r["meta"]
        n_lineups = len(ldf["lineup_index"].unique())
        avg_lu_sal = ldf.groupby("lineup_index")["salary"].sum().mean()
        avg_lu_proj = ldf.groupby("lineup_index")["proj"].sum().mean()
        avg_proj = pdf["proj"].mean() if "proj" in pdf.columns else 0
        avg_salary = pdf["salary"].mean() if "salary" in pdf.columns else 0
        has_actual = "actual_fp" in pdf.columns
        avg_actual = pdf["actual_fp"].mean() if has_actual else None
        proj_diff = (avg_actual - avg_proj) if has_actual and avg_actual else None
        has_own = "ownership" in pdf.columns
        avg_own = pdf["ownership"].mean() if has_own else None
        summary_rows.append({
            "slate_date": sd,
            "status": "OK",
            "pool_size": len(pdf),
            "n_lineups": n_lineups,
            "avg_proj": round(avg_proj, 1),
            "avg_salary": round(avg_salary, 0),
            "avg_actual": round(avg_actual, 1) if avg_actual else None,
            "proj_diff": round(proj_diff, 1) if proj_diff is not None else None,
            "avg_lu_salary": round(avg_lu_sal, 0),
            "avg_lu_proj": round(avg_lu_proj, 1),
            "avg_ownership": round(avg_own, 1) if avg_own else None,
        })
    summary_df = pd.DataFrame(summary_rows)
    return {"per_slate": per_slate, "summary_df": summary_df}


# ============================================================
# Cross-Slate Comparison
# ============================================================
def compare_slates(multi_result: Dict[str, Any]) -> Dict[str, Any]:
    """Compute cross-slate comparison KPIs from run_multi_slate output."""
    sdf = multi_result["summary_df"]
    ok = sdf[sdf["status"] == "OK"]
    if ok.empty:
        return {"n_slates": 0, "trends": pd.DataFrame()}

    trends = {}
    for col in ["avg_proj", "avg_salary", "avg_actual", "avg_lu_proj", "avg_lu_salary", "avg_ownership"]:
        if col in ok.columns and ok[col].notna().any():
            vals = ok[col].dropna()
            trends[col] = {
                "mean": round(vals.mean(), 1),
                "std": round(vals.std(), 1) if len(vals) > 1 else 0,
                "min": round(vals.min(), 1),
                "max": round(vals.max(), 1),
            }

    # Consistency: how stable are optimizer outputs across slates?
    lu_proj_vals = ok["avg_lu_proj"].dropna()
    consistency = {
        "lu_proj_cv": round(lu_proj_vals.std() / lu_proj_vals.mean() * 100, 1) if len(lu_proj_vals) > 1 and lu_proj_vals.mean() > 0 else 0,
        "n_slates_ok": len(ok),
        "n_slates_fail": len(sdf[sdf["status"] == "FAIL"]),
    }

    # Proj accuracy (if actual_fp available)
    if "proj_diff" in ok.columns and ok["proj_diff"].notna().any():
        pd_vals = ok["proj_diff"].dropna()
        consistency["avg_proj_error"] = round(pd_vals.mean(), 1)
        consistency["proj_accuracy_std"] = round(pd_vals.std(), 1) if len(pd_vals) > 1 else 0

    return {
        "n_slates": len(ok),
        "trends": trends,
        "consistency": consistency,
        "summary_df": sdf,
    }


# ============================================================
# DK Contest CSV Parser
# ============================================================
def parse_dk_contest_csv(source) -> pd.DataFrame:
    """Parse a DraftKings contest results CSV.

    Accepts a file path (str) or file-like object (BytesIO from Streamlit uploader).
    Returns normalized DataFrame with columns:
      player_name, pos, salary, actual_fp, ownership, team
    """
    if isinstance(source, (str, os.PathLike)):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    # DK contest CSV column mapping
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("name", "player", "nickname"):
            col_map[c] = "player_name"
        elif cl in ("position", "pos", "roster position"):
            col_map[c] = "pos"
        elif cl in ("salary", "sal"):
            col_map[c] = "salary"
        elif cl in ("fpts", "fantasy points", "fp", "fantasypoints", "points"):
            col_map[c] = "actual_fp"
        elif cl in ("own", "ownership", "%owned", "pown", "%drafted"):
            col_map[c] = "ownership"
        elif cl in ("teamabbrev", "team", "tm"):
            col_map[c] = "team"
        elif cl in ("opp", "opponent", "game"):
            col_map[c] = "opponent"

    df = df.rename(columns=col_map)

    # Ensure numeric
    for nc in ["salary", "actual_fp", "ownership"]:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors="coerce")

    # Clean ownership (remove % sign if present)
    if "ownership" in df.columns:
        if df["ownership"].dtype == object:
            df["ownership"] = df["ownership"].astype(str).str.replace("%", "").astype(float)

    # Filter rows with valid player names
    if "player_name" in df.columns:
        df = df[df["player_name"].notna() & (df["player_name"] != "")]

    keep_cols = [c for c in ["player_name", "pos", "salary", "actual_fp", "ownership", "team", "opponent"] if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)
    return df


# ============================================================
# Quick Slate Preview
# ============================================================
def preview_slate(slate_date: str, root: str = YAKOS_ROOT) -> Dict[str, Any]:
    """Quick preview of a slate without running optimizer."""
    try:
        pool = run_lineups_from_config(slate_date, yakos_root=root)
        info = {
            "slate_date": slate_date,
            "n_players": len(pool),
            "cols": list(pool.columns),
        }
        if "salary" in pool.columns:
            info["avg_salary"] = round(pool["salary"].mean(), 0)
            sal_lo = pool["salary"].min()
            sal_hi = pool["salary"].max()
            info["salary_range"] = f"${sal_lo:,.0f} - ${sal_hi:,.0f}"
        if "proj" in pool.columns:
            info["avg_proj"] = round(pool["proj"].mean(), 1)
        if "actual_fp" in pool.columns:
            info["avg_actual"] = round(pool["actual_fp"].mean(), 1)
        if "team" in pool.columns:
            info["n_teams"] = pool["team"].nunique()
        return info
    except Exception as e:
        return {"slate_date": slate_date, "error": str(e)}
