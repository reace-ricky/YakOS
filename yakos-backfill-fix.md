# YakOS Slate Archive: Backfill Fix & Snapshot-on-Publish

## Problem Summary

Three categories of broken archives:

| Date Range | Issue | Root Cause |
|-----------|-------|------------|
| 03-14, 03-15 | 15-47% actuals coverage | Multi-day slate: games spanned two dates, but only one day's actuals were fetched |
| 03-16 through 03-21 | Some have actuals, some need re-check | May have incomplete actuals; backfill should fix |
| 03-22 through 03-26 | **No archives exist** | Chicken-and-egg: published pool was stale (still showing 03-14/03-15), so nightly cron skipped these dates |

## What Was Done

### Task 1: Snapshot-on-Publish (prevents future gaps)

**File changed:** `app/lab_tab.py`

Added `_snapshot_on_publish()` function that runs inside `_publish_to_github()`. When the user clicks "Publish to GitHub" in the Lab tab:

1. Reads the current `slate_pool.parquet` and `slate_meta.json`
2. Creates an archive at `data/slate_archive/{date}_gpp_main.parquet`
3. Only creates if the archive doesn't exist yet, or if the existing archive has no actuals (safe to overwrite)
4. Non-fatal — if archiving fails, the publish still proceeds

This ensures the nightly cron always has a pool to work with, breaking the chicken-and-egg cycle.

### Task 2: Backfill Script (repairs half-baked slates)

**File changed:** `scripts/nightly_calibration.py`

Added `_fetch_nba_actuals_multi_day_backfill()` helper that handles archives **without** a `game_id` column:

- If the archive has `game_id`, delegates to the standard `fetch_actuals_multi_day()`
- If `game_id` is missing (older archives like 03-14, 03-15), fetches actuals for **both** the slate date AND the previous day
- Multi-day DK slates typically span two consecutive days, so fetching date-1 + date captures all players

The existing `--backfill` mode now calls this enhanced function instead of `fetch_actuals_multi_day()` directly.

**Command to run after merging:**

```bash
# Set your API key first
export RAPIDAPI_KEY="your-tank01-key"

# Run backfill — will repair all archives with <80% coverage
python scripts/nightly_calibration.py --backfill
```

Expected output: 03-14 should go from 15% → ~95%+, 03-15 from 47% → ~95%+.

### Task 3: Recovery of 03-22 through 03-26

**File created:** `scripts/recover_missing_archives.py`

Investigation results from git history:

| Date | Published Pool in Git? | Recoverable? |
|------|----------------------|-------------|
| 2026-03-22 | No | **Unrecoverable** |
| 2026-03-23 | No | **Unrecoverable** |
| 2026-03-24 | No | **Unrecoverable** |
| 2026-03-25 | No | **Unrecoverable** |
| 2026-03-26 | No (published pool was stale 03-14/03-15) | **Unrecoverable** |

The DK API does not serve historical draftables/player pools. The published pool was never updated for these dates (it was stuck on 03-14/03-15 data), so there is no pool data to recover from any source.

The recovery script (`scripts/recover_missing_archives.py --scan`) can verify this:

```bash
python scripts/recover_missing_archives.py --scan
```

## Architecture of the Fix

```
User publishes slate → _publish_to_github()
                        ├── _snapshot_on_publish()  ← NEW: archives pool immediately
                        └── sync to GitHub

Nightly cron runs     → nightly_calibration.py
                        ├── Loads pool from archive (not published/)
                        ├── fetch_actuals_multi_day() ← handles multi-day slates
                        └── Archives with actuals

Backfill mode         → nightly_calibration.py --backfill
                        ├── Scans all archives for <80% coverage
                        ├── _fetch_nba_actuals_multi_day_backfill() ← NEW: fetches date-1 too
                        ├── merge_actuals_three_pass()
                        └── Re-archives with improved coverage
```

## Preventing This in the Future

The snapshot-on-publish fix ensures:
1. Every published slate gets archived **before** games start
2. Nightly cron can always find the archive, even if published pool is overwritten
3. Archives with existing actuals are never overwritten by actuals-free snapshots
