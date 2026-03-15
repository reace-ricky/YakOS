#!/usr/bin/env python3
"""GPP Calibration Analysis: Compare winning lineup patterns to optimizer output."""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from yak_core.config import DEFAULT_CONFIG, merge_config, SALARY_CAP
from yak_core.lineups import prepare_pool, build_multiple_lineups_with_exposure

# ============================================================
# 1. WINNING LINEUP ANALYSIS
# ============================================================
with open("/home/user/workspace/rg-resultsdb-winning-lineups.json") as f:
    data = json.load(f)

results = data["results"]
print("=" * 80)
print("PART 1: WINNING LINEUP PATTERN ANALYSIS")
print("=" * 80)
print(f"\nTotal contests analyzed: {len(results)}")
print(f"Date range: {results[-1]['date']} to {results[0]['date']}")

all_scores = []
all_players = []
salary_usages = []
studs_counts = []  # salary > $8K
value_counts = []  # salary < $5K
punt_counts = []   # salary < $4K
mid_counts = []    # salary $4K-$7K

for r in results:
    all_scores.append(r["winning_score"])
    lineup = r["lineup"]
    total_sal = sum(p["salary"] for p in lineup)
    salary_usages.append(total_sal)

    studs = [p for p in lineup if p["salary"] >= 8000]
    values = [p for p in lineup if p["salary"] < 5000]
    punts = [p for p in lineup if p["salary"] < 4000]
    mids = [p for p in lineup if 4000 <= p["salary"] <= 7000]

    studs_counts.append(len(studs))
    value_counts.append(len(values))
    punt_counts.append(len(punts))
    mid_counts.append(len(mids))

    for p in lineup:
        all_players.append({
            "date": r["date"],
            "contest": r["contest"],
            "winning_score": r["winning_score"],
            **p,
        })

pdf = pd.DataFrame(all_players)

print(f"\n--- Winning Scores ---")
print(f"  Min: {min(all_scores):.2f}")
print(f"  Max: {max(all_scores):.2f}")
print(f"  Avg: {np.mean(all_scores):.2f}")
print(f"  Median: {np.median(all_scores):.2f}")
print(f"  Scores: {all_scores}")

print(f"\n--- Salary Usage ---")
print(f"  Avg total salary: ${np.mean(salary_usages):,.0f} / $50,000")
print(f"  Avg % cap used: {np.mean(salary_usages)/50000*100:.1f}%")
print(f"  Min: ${min(salary_usages):,}")
print(f"  Max: ${max(salary_usages):,}")

print(f"\n--- Lineup Composition ---")
print(f"  Avg studs ($8K+): {np.mean(studs_counts):.1f}")
print(f"  Avg value ($4K-): {np.mean(value_counts):.1f}")
print(f"  Avg punts ($3.9K-): {np.mean(punt_counts):.1f}")
print(f"  Avg mid ($4K-$7K): {np.mean(mid_counts):.1f}")
print(f"  Stud counts per lineup: {studs_counts}")
print(f"  Value counts per lineup: {value_counts}")
print(f"  Punt counts per lineup: {punt_counts}")
print(f"  Mid counts per lineup: {mid_counts}")

print(f"\n--- Value Efficiency ---")
pdf["fp_per_1k"] = pdf["fpts"] / (pdf["salary"] / 1000)
print(f"  Avg FP per $1K salary: {pdf['fp_per_1k'].mean():.2f}")
print(f"  By salary tier:")

tiers = [
    ("$10K+", pdf["salary"] >= 10000),
    ("$8K-$10K", (pdf["salary"] >= 8000) & (pdf["salary"] < 10000)),
    ("$5K-$8K", (pdf["salary"] >= 5000) & (pdf["salary"] < 8000)),
    ("$4K-$5K", (pdf["salary"] >= 4000) & (pdf["salary"] < 5000)),
    ("Under $4K", pdf["salary"] < 4000),
]
for label, mask in tiers:
    subset = pdf[mask]
    if len(subset) > 0:
        print(f"    {label}: {len(subset)} players, avg salary ${subset['salary'].mean():,.0f}, "
              f"avg FP {subset['fpts'].mean():.1f}, avg FP/$1K {subset['fp_per_1k'].mean():.2f}")

print(f"\n--- Individual Ceiling Games ---")
top_scorers = pdf.nlargest(10, "fpts")
for _, p in top_scorers.iterrows():
    print(f"  {p['player']:25s} ${p['salary']:,}  → {p['fpts']:.2f} FP ({p['fp_per_1k']:.1f}x)")

print(f"\n--- Stud Performance (Top salary in each lineup) ---")
for r in results:
    lineup = sorted(r["lineup"], key=lambda x: x["salary"], reverse=True)
    top = lineup[0]
    print(f"  {r['date']} | {top['player']:25s} ${top['salary']:,} → {top['fpts']:.2f} FP")

# ============================================================
# 2. RUN OPTIMIZER ON MARCH 14 SLATE
# ============================================================
print("\n" + "=" * 80)
print("PART 2: OPTIMIZER OUTPUT (CURRENT V8 FORMULA)")
print("=" * 80)

pool_df = pd.read_parquet("data/published/nba/slate_pool.parquet")
print(f"\nSlate pool: {len(pool_df)} players")

cfg = merge_config({
    "CONTEST_TYPE": "gpp",
    "NUM_LINEUPS": 20,
    "MAX_EXPOSURE": 0.6,
    "PROJ_SOURCE": "parquet",
    "MIN_SALARY_USED": 49000,
})

prepared = prepare_pool(pool_df, cfg)
print(f"Prepared pool: {len(prepared)} players after filtering")

# Show top GPP-scored players
print(f"\n--- Top 20 Players by GPP Score ---")
top20 = prepared.nlargest(20, "gpp_score")[["player_name", "team", "position", "salary", "proj", "own_pct", "gpp_score"]].copy()
top20["own_pct"] = top20["own_pct"].map(lambda x: f"{x*100:.1f}%" if x < 1 else f"{x:.1f}%")
print(top20.to_string(index=False))

# Build lineups
lineups_df, exposures_df = build_multiple_lineups_with_exposure(prepared, cfg)
num_built = lineups_df["lineup_index"].nunique() if not lineups_df.empty else 0
print(f"\nGenerated {num_built} lineups")

# Analyze optimizer lineups
opt_scores = []
opt_salaries = []
opt_studs = []
opt_values = []
opt_punts = []
opt_mids = []

# Group lineups from the DataFrame
lineups_grouped = []
for lu_idx in lineups_df["lineup_index"].unique():
    lu_players = lineups_df[lineups_df["lineup_index"] == lu_idx]
    lineups_grouped.append(lu_players)

    total_proj = lu_players["proj"].sum()
    total_sal = lu_players["salary"].sum()

    opt_scores.append(total_proj)
    opt_salaries.append(total_sal)

    studs = len(lu_players[lu_players["salary"] >= 8000])
    values = len(lu_players[lu_players["salary"] < 5000])
    punts = len(lu_players[lu_players["salary"] < 4000])
    mids = len(lu_players[(lu_players["salary"] >= 4000) & (lu_players["salary"] <= 7000)])

    opt_studs.append(studs)
    opt_values.append(values)
    opt_punts.append(punts)
    opt_mids.append(mids)

print(f"\n--- Optimizer Lineup Projections ---")
print(f"  Min projected: {min(opt_scores):.1f}")
print(f"  Max projected: {max(opt_scores):.1f}")
print(f"  Avg projected: {np.mean(opt_scores):.1f}")
print(f"  Median projected: {np.median(opt_scores):.1f}")

print(f"\n--- Optimizer Salary Usage ---")
print(f"  Avg salary: ${np.mean(opt_salaries):,.0f}")
print(f"  Min salary: ${min(opt_salaries):,}")
print(f"  Max salary: ${max(opt_salaries):,}")

print(f"\n--- Optimizer Lineup Composition ---")
print(f"  Avg studs ($8K+): {np.mean(opt_studs):.1f}")
print(f"  Avg value ($4K-): {np.mean(opt_values):.1f}")
print(f"  Avg punts ($3.9K-): {np.mean(opt_punts):.1f}")
print(f"  Avg mid ($4K-$7K): {np.mean(opt_mids):.1f}")

# Show a few sample lineups
print(f"\n--- Sample Lineups (first 3) ---")
for i in range(min(3, len(lineups_grouped))):
    lu = lineups_grouped[i]
    total_proj = lu["proj"].sum()
    total_sal = lu["salary"].sum()
    total_gpp = lu["gpp_score"].sum()
    print(f"\n  Lineup {i+1}: Proj={total_proj:.1f}, Salary=${total_sal:,}, GPP_Score={total_gpp:.1f}")
    for _, p in lu.sort_values("salary", ascending=False).iterrows():
        own_val = p.get("own_pct", 0)
        own_str = f"{own_val*100:.1f}%" if own_val < 1 else f"{own_val:.1f}%"
        print(f"    {p.get('slot','??'):4s} {p['player_name']:25s} {p['position']:5s} ${p['salary']:>6,} proj={p['proj']:5.1f} own={own_str}")

# ============================================================
# 3. COMPARISON: OPTIMIZER vs WINNING PATTERNS
# ============================================================
print("\n" + "=" * 80)
print("PART 3: COMPARISON — OPTIMIZER vs WINNING PATTERNS")
print("=" * 80)

win_avg_score = np.mean(all_scores)
opt_avg_proj = np.mean(opt_scores)

print(f"\n  Metric                    | Winning Lineups  | Optimizer Output  | Gap")
print(f"  {'—'*25}|{'—'*18}|{'—'*19}|{'—'*15}")
print(f"  Avg Score/Proj            | {win_avg_score:>10.1f}       | {opt_avg_proj:>10.1f}        | {opt_avg_proj - win_avg_score:>+.1f}")
print(f"  Avg Salary                | ${np.mean(salary_usages):>10,.0f}    | ${np.mean(opt_salaries):>10,.0f}     |")
print(f"  Avg % Cap Used            | {np.mean(salary_usages)/50000*100:>10.1f}%      | {np.mean(opt_salaries)/50000*100:>10.1f}%       |")
print(f"  Avg Studs ($8K+)          | {np.mean(studs_counts):>10.1f}       | {np.mean(opt_studs):>10.1f}        |")
print(f"  Avg Mid ($4K-$7K)         | {np.mean(mid_counts):>10.1f}       | {np.mean(opt_mids):>10.1f}        |")
print(f"  Avg Value ($4K-)          | {np.mean(value_counts):>10.1f}       | {np.mean(opt_values):>10.1f}        |")
print(f"  Avg Punts ($3.9K-)        | {np.mean(punt_counts):>10.1f}       | {np.mean(opt_punts):>10.1f}        |")

# Check if optimizer proj is in winning range
in_range = sum(1 for s in opt_scores if 339 <= s <= 433)
print(f"\n  Optimizer lineups in winning range (339-433): {in_range}/{len(opt_scores)}")
print(f"  Optimizer lineups above minimum winning score (339): {sum(1 for s in opt_scores if s >= 339)}/{len(opt_scores)}")

# Sim-based upside analysis: what's the 90th percentile projection for optimizer lineups?
print(f"\n--- Upside Potential (Sim-Based) ---")
for i in range(min(3, len(lineups_grouped))):
    lu = lineups_grouped[i]
    total_sim90 = 0
    total_sim99 = 0
    for _, p in lu.iterrows():
        mask = prepared["player_name"] == p["player_name"]
        if mask.any():
            row = prepared[mask].iloc[0]
            s90 = row.get("sim90th", 0) or 0
            s99 = row.get("sim99th", 0) or 0
            total_sim90 += float(s90) if pd.notna(s90) else 0
            total_sim99 += float(s99) if pd.notna(s99) else 0
    total_proj = lu["proj"].sum()
    print(f"  Lineup {i+1}: Proj={total_proj:.1f}, Sim90={total_sim90:.1f}, Sim99={total_sim99:.1f}")

# Player salary distribution in optimizer
print(f"\n--- Salary Distribution in Optimizer Lineups ---")
opt_pdf = lineups_df.copy()
for label, lo, hi in [("$10K+", 10000, 99999), ("$8K-$10K", 8000, 9999),
                       ("$5K-$8K", 5000, 7999), ("$4K-$5K", 4000, 4999),
                       ("$3K-$4K", 3000, 3999)]:
    mask = (opt_pdf["salary"] >= lo) & (opt_pdf["salary"] <= hi)
    subset = opt_pdf[mask]
    pct = len(subset) / len(opt_pdf) * 100
    print(f"  {label}: {len(subset)} player-slots ({pct:.1f}%)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
