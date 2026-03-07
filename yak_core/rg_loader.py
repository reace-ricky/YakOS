"""yak_core.rg_loader -- load and normalize RotoGrinders CSV exports."""
import os
from datetime import datetime
import pandas as pd
from typing import Optional


# ---------- column mapping ----------
# RG projections CSV -> our internal schema
_RG_PROJ_MAP = {
    "player_id": "rg_player_id",
    "name": "player_name",
    "team": "team",
    "opp": "opp",
    "pos": "pos",
    "fpts": "proj",
    "proj_own": "proj_own",
    "salary": "salary",
    "ceil": "ceil",
    "floor": "floor",
    "smash": "smash",
    "opto_pct": "opto_pct",
    "minutes": "minutes",
    "rg_value": "rg_value",
}

# RG contest results (53-col) -> our internal schema
_RG_CONTEST_MAP = {
    "PLAYERID": "dk_player_id",
    "PLAYER": "name",
    "SALARY": "salary",
    "POS": "pos",
    "TEAM": "team",
    "OPP": "opp",
    "FPTS": "actual_fp",
    "OWNERSHIP": "ownership",
    "POWN": "proj_own",
    "CEIL": "ceil",
    "FLOOR": "floor",
    "OPTO": "opto",
    "PERFECT": "perfect",
    "SMASH": "smash",
    "MINUTES": "minutes",
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "3PM": "fg3m",
    "TO": "tov",
    "FPTS/$": "fpts_per_dollar",
}

# RG contest results (24-col probability view)
_RG_PROB_MAP = {
    "PLAYERID": "dk_player_id",
    "PLAYER": "name",
    "SALARY": "salary",
    "POS": "pos",
    "TEAM": "team",
    "OPP": "opp",
    "FPTS": "actual_fp",
    "OWNERSHIP": "ownership",
    "POWN": "proj_own",
    "CEIL": "ceil",
    "FLOOR": "floor",
    "OPTO": "opto",
    "WIN_PROB": "win_prob",
    "TOP_5_PROB": "top_5_prob",
    "TOP_10_PROB": "top_10_prob",
    "TOP_20_PROB": "top_20_prob",
    "MAKE_CUT_PROB": "make_cut_prob",
    "NOTO_RATING": "noto_rating",
}


def _apply_map(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename columns using a mapping; keep only mapped columns."""
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    out = df.rename(columns=rename)
    keep = [v for v in rename.values() if v in out.columns]
    return out[keep].copy()


def load_rg_projections(path: str) -> pd.DataFrame:
    """Load a RotoGrinders projections CSV and normalize columns.
    Returns DataFrame with columns: player_name, team, opp, pos, proj, salary,
    proj_own, ceil, floor, smash, opto_pct, minutes, rg_value.
    """
    df = pd.read_csv(path)
    out = _apply_map(df, _RG_PROJ_MAP)
    # Ensure numeric
    for c in ["proj", "salary", "proj_own", "ceil", "floor", "smash",
              "opto_pct", "minutes", "rg_value"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_rg_contest(path: str) -> pd.DataFrame:
    """Load a RotoGrinders contest-results CSV (53-col format).
    Auto-detects whether it is the full-stats format or the probability view.
    Returns normalized DataFrame.
    """
    df = pd.read_csv(path)
    # Detect format by column count
    if "WIN_PROB" in df.columns:
        out = _apply_map(df, _RG_PROB_MAP)
    else:
        out = _apply_map(df, _RG_CONTEST_MAP)
    # Ensure numeric
    for c in ["salary", "actual_fp", "ownership", "proj_own", "ceil",
              "floor", "smash", "minutes", "pts", "reb", "ast", "stl",
              "blk", "fg3m", "tov", "fpts_per_dollar",
              "win_prob", "top_5_prob", "top_10_prob", "top_20_prob",
              "make_cut_prob", "noto_rating"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def merge_rg_with_pool(
    pool_df: pd.DataFrame,
    rg_df: pd.DataFrame,
    merge_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Merge RG data into an existing player pool by name+team.

    Adds any new columns from rg_df that pool_df doesn't have yet.
    **Critically**: if RG provides ownership data (``proj_own`` or
    ``ownership``), it OVERWRITES the pool's ``ownership`` column
    and sets ``own_proj`` so that both edge.py and sims.py consume
    real ownership instead of the salary-rank fallback.
    """
    if merge_cols is None:
        merge_cols = ["player_name", "team"]

    # Determine which RG column carries ownership
    _rg_own_col = None
    if "proj_own" in rg_df.columns:
        _rg_own_col = "proj_own"
    elif "ownership" in rg_df.columns:
        _rg_own_col = "ownership"

    # Find new columns in rg_df that pool_df lacks
    new_cols = [c for c in rg_df.columns if c not in pool_df.columns and c not in merge_cols]

    # Always include ownership column even if pool already has it — we want to overwrite
    if _rg_own_col and _rg_own_col not in new_cols:
        new_cols.append(_rg_own_col)

    if not new_cols:
        return pool_df

    rg_sub = rg_df[merge_cols + new_cols].drop_duplicates(subset=merge_cols)
    # If RG ownership would collide with existing column, drop pool's version first
    if _rg_own_col and _rg_own_col in pool_df.columns:
        merged = pool_df.drop(columns=[_rg_own_col]).merge(rg_sub, on=merge_cols, how="left")
    else:
        merged = pool_df.merge(rg_sub, on=merge_cols, how="left")

    # Overwrite the canonical ownership column with RG data
    if _rg_own_col and _rg_own_col in merged.columns:
        rg_own = pd.to_numeric(merged[_rg_own_col], errors="coerce")
        # Only overwrite where RG provided a value; keep existing for unmatched
        has_rg = rg_own.notna()
        if "ownership" in merged.columns:
            merged.loc[has_rg, "ownership"] = rg_own[has_rg]
        else:
            merged["ownership"] = rg_own
        # Also set own_proj (canonical for sims.py)
        if "own_proj" in merged.columns:
            merged.loc[has_rg, "own_proj"] = rg_own[has_rg]
        else:
            merged["own_proj"] = merged["ownership"]

        _matched = has_rg.sum()
        _total = len(merged)
        print(f"[rg_loader] Ownership updated: {_matched}/{_total} players now use RG ownership")

    # Archive the merged data for ownership model training
    _archive_for_training(merged)

    return merged


def _archive_for_training(merged_df: pd.DataFrame) -> None:
    """Save merged pool+RG data to the training archive for the ownership model.

    Each upload is timestamped and appended to ``data/ownership_training/``.
    The ownership model trainer reads all CSVs in that directory.
    """
    try:
        from yak_core.config import YAKOS_ROOT
        archive_dir = os.path.join(YAKOS_ROOT, "data", "ownership_training")
        os.makedirs(archive_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Keep only the columns useful for training
        _train_cols = [
            "player_name", "team", "pos", "salary", "proj", "floor", "ceil",
            "ownership", "own_proj", "proj_own",
        ]
        keep = [c for c in _train_cols if c in merged_df.columns]
        out = merged_df[keep].copy()
        path = os.path.join(archive_dir, f"own_train_{ts}.csv")
        out.to_csv(path, index=False)
        print(f"[rg_loader] Archived ownership training data: {path} ({len(out)} rows)")
    except Exception as exc:
        print(f"[rg_loader] Warning: could not archive training data: {exc}")


def hit_rate(
    contest_df: pd.DataFrame,
    cash_line: float = 0.0,
    col: str = "actual_fp",
) -> dict:
    """Compute hit-rate KPIs from a contest results DataFrame.
    cash_line: minimum FPTS to cash (if 0, skips cash calc).
    Returns dict with pct_above_cash, avg_fpts, median_fpts, etc.
    """
    if col not in contest_df.columns:
        return {"error": f"missing {col} column"}
    fpts = contest_df[col].dropna()
    result = {
        "n_players": len(fpts),
        "avg_fpts": round(fpts.mean(), 2),
        "median_fpts": round(fpts.median(), 2),
        "std_fpts": round(fpts.std(), 2),
    }
    if cash_line > 0:
        result["cash_line"] = cash_line
        result["pct_above_cash"] = round((fpts >= cash_line).mean() * 100, 1)
    return result
