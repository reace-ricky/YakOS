#!/usr/bin/env python3
"""scripts/calc_field_ownership.py -- Compute heuristic own_proj from player pool.

Usage:
    python scripts/calc_field_ownership.py \
        --sport nba \
        --site dk \
        --slate-id 2026-03-24 \
        --contest-bucket gpp_main \
        [--n-field 5000] \
        [--random-seed 42] \
        [--alpha 0.3]

Outputs:
    data/field_ownership/{sport}_{site}_{slate_id}_{contest_bucket}_own.parquet

The own_proj column can then be joined back onto the player pool via
yak_core.ownership_store.attach_own_proj_to_pool().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Path bootstrap (same as other scripts) ───────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute heuristic field ownership from player pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sport", default="nba", help="Sport (nba / pga)")
    parser.add_argument("--site", default="dk", help="DFS site (dk)")
    parser.add_argument("--slate-id", required=True, help="Slate date or ID (e.g. 2026-03-24)")
    parser.add_argument("--contest-bucket", default="gpp_main", help="Contest bucket")
    parser.add_argument("--n-field", type=int, default=5000, help="Number of field lineups")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.3, help="Value-boost exponent")
    parser.add_argument("--salary-cap", type=int, default=0, help="Override salary cap (0 = auto)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    sport = args.sport.upper()
    site = args.site.lower()
    slate_id = args.slate_id
    contest_bucket = args.contest_bucket.lower()

    print(f"[calc_field_ownership] sport={sport} site={site} slate_id={slate_id} "
          f"contest_bucket={contest_bucket} n_field={args.n_field}")

    # ── 1. Load pool ──────────────────────────────────────────────────────────
    from yak_core.lineups import load_player_pool

    try:
        pool = load_player_pool(sport=sport, slate_date=slate_id)
    except FileNotFoundError:
        # Try loading from published dir
        from pathlib import Path as _P
        from yak_core.config import YAKOS_ROOT
        pool_path = _P(YAKOS_ROOT) / "data" / "published" / sport.lower() / "slate_pool.parquet"
        if pool_path.exists():
            import pandas as pd
            pool = pd.read_parquet(pool_path)
            print(f"[calc_field_ownership] Loaded pool from {pool_path}")
        else:
            print(
                f"[calc_field_ownership] ERROR: Could not load pool for {sport}/{slate_id}. "
                f"Run the Slate Hub to publish the pool first.",
                file=sys.stderr,
            )
            sys.exit(1)

    if pool.empty:
        print(
            f"[calc_field_ownership] ERROR: Pool is empty for {sport}/{slate_id}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[calc_field_ownership] Loaded pool: {len(pool)} players")

    # ── 2. Build field ────────────────────────────────────────────────────────
    from yak_core.field_ownership import build_field_lineups

    cfg: dict = {
        "n_field_lineups": args.n_field,
        "random_seed": args.random_seed,
        "contest_type": contest_bucket,
        "alpha": args.alpha,
        "sport": sport,
    }
    if args.salary_cap > 0:
        cfg["salary_cap"] = args.salary_cap

    field_lineups = build_field_lineups(pool, config=cfg)

    if not field_lineups:
        print(
            "[calc_field_ownership] ERROR: No valid lineups generated.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[calc_field_ownership] Generated {len(field_lineups)} field lineups")

    # ── 3. Estimate ownership ─────────────────────────────────────────────────
    from yak_core.field_ownership import estimate_ownership_from_field

    own_df = estimate_ownership_from_field(field_lineups)
    print(f"[calc_field_ownership] Estimated own_proj for {len(own_df)} players")
    if not own_df.empty:
        print(f"  Top 5:\n{own_df.head(5).to_string(index=False)}")

    # ── 4. Persist ────────────────────────────────────────────────────────────
    from yak_core.ownership_store import write_own_proj_to_archive

    write_own_proj_to_archive(sport, site, slate_id, contest_bucket, own_df)

    print("[calc_field_ownership] Done.")


if __name__ == "__main__":
    main()
