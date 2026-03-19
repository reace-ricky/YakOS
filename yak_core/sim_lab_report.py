"""yak_core.sim_lab_report — Read-only Sim Lab analysis helpers.

Reads the CSV exports that the Sim Lab download button produces and
builds the same summary tables as the Sim_Lab_Explorer notebook.

This module has NO side-effects — it only reads files and returns
DataFrames.  Safe to call on every page render.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def summarize_sim_lab(
    data_dir: Path,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Summarize Sim Lab CSV exports.

    Parameters
    ----------
    data_dir : Path
        Folder containing ``ricky_ranked_lineups*.csv`` and ``*export.csv``.

    Returns
    -------
    rank_summary : DataFrame or None
        Lineup performance by Ricky rank bucket (1–5, 6–10, 11–20).
    tag_summary : DataFrame or None
        SE tag performance by date (SE Core / SE Spicy / SE Alt).
    lineups_df : DataFrame or None
        Combined lineups with diff and avg_own_pct for scatter plot.

    All three are ``None`` when no CSV files are found.
    """
    # ── Load ranked lineups ──────────────────────────────────────────────
    lineup_files = sorted(data_dir.glob("ricky_ranked_lineups*.csv"))
    if not lineup_files:
        return None, None, None

    lineups = pd.concat(
        [pd.read_csv(f) for f in lineup_files], ignore_index=True,
    )
    if lineups.empty:
        return None, None, None

    lineups["date"] = pd.to_datetime(lineups["date"], errors="coerce")

    if "diff" not in lineups.columns and {"total_actual", "total_proj"}.issubset(
        lineups.columns
    ):
        lineups["diff"] = lineups["total_actual"] - lineups["total_proj"]

    # ── Rank bucket summary ──────────────────────────────────────────────
    def _bucket(rank: int) -> str:
        if rank <= 5:
            return "1–5"
        if rank <= 10:
            return "6–10"
        return "11–20"

    lineups["rank_bucket"] = lineups["ricky_rank"].apply(_bucket)

    _agg = {}
    for col in ["diff", "total_proj", "total_actual", "total_gpp_score"]:
        if col in lineups.columns:
            _agg[col] = ["count", "mean", "median"] if col == "diff" else ["mean"]

    rank_summary = lineups.groupby("rank_bucket", sort=False).agg(_agg)
    rank_summary.columns = [
        f"{stat}_{col}" if stat != "count" else "Lineups"
        for col, stat in rank_summary.columns
    ]
    rank_summary = rank_summary.reindex(["1–5", "6–10", "11–20"]).dropna(how="all")
    rank_summary = rank_summary.rename(
        columns={
            "mean_diff": "Avg Diff",
            "median_diff": "Med Diff",
            "mean_total_proj": "Avg Proj",
            "mean_total_actual": "Avg Actual",
            "mean_total_gpp_score": "Avg GPP",
        }
    )
    rank_summary.index.name = "Rank Bucket"
    # Ensure Lineups is int
    if "Lineups" in rank_summary.columns:
        rank_summary["Lineups"] = rank_summary["Lineups"].astype(int)

    # ── Tag summary (from lineups, not a separate export file) ───────────
    tag_col = "ricky_tag" if "ricky_tag" in lineups.columns else None
    tag_summary: Optional[pd.DataFrame] = None

    if tag_col:
        tagged = lineups[lineups[tag_col].fillna("").str.strip().ne("")].copy()
        if not tagged.empty:
            _tag_agg = {}
            for col, ops in [
                ("diff", ["count", "mean", "median"]),
                ("total_proj", ["mean"]),
                ("total_actual", ["mean"]),
                ("avg_own_pct", ["mean"]),
            ]:
                if col in tagged.columns:
                    _tag_agg[col] = ops

            if _tag_agg:
                tag_summary = (
                    tagged.groupby(
                        [tagged["date"].dt.strftime("%Y-%m-%d"), tag_col],
                        sort=False,
                    )
                    .agg(_tag_agg)
                )
                tag_summary.columns = [
                    f"{stat}_{col}" if stat != "count" else "Lineups"
                    for col, stat in tag_summary.columns
                ]
                tag_summary = tag_summary.rename(
                    columns={
                        "mean_diff": "Avg Diff",
                        "median_diff": "Med Diff",
                        "mean_total_proj": "Avg Proj",
                        "mean_total_actual": "Avg Actual",
                        "mean_avg_own_pct": "Avg Own%",
                    }
                )
                tag_summary.index.names = ["Date", "Tag"]
                if "Lineups" in tag_summary.columns:
                    tag_summary["Lineups"] = tag_summary["Lineups"].astype(int)

    return rank_summary, tag_summary, lineups
