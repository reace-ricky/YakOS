# YakOS Roadmap & Status

> **Purpose:** This file is the single source of truth for the project's end goal, current status, and remaining work.
> Future agent sessions should read this file first to understand context before making changes.

---

## End Goal

Build a production-quality **NBA DraftKings DFS lineup optimizer** called *YakOS / Right Angle Ricky* with:

1. A polished **Streamlit web UI** with three tabs:
   - 🏀 **Ricky's Slate Room** — pool loader, KPI dashboard, edge analysis, promoted lineups
   - ⚡ **Optimizer** — build lineups for any DK contest type with full override controls
   - 🔬 **Calibration Lab** — backtest, queue, ownership ingest, archetype knobs, sim module

2. A clean **`yak_core` Python library** that can be used headlessly (no Streamlit required)

3. Reliable **tests** covering core optimizer logic and data transforms

4. A working **DK upload workflow**: generate → download → paste into DraftKings bulk uploader

---

## What's Been Built ✅

| # | Feature | Module / File | PR |
|---|---------|---------------|----|
| 1 | LP-based lineup optimizer (PuLP) | `yak_core/lineups.py` | early |
| 2 | DK NBA salary cap + roster slots (8-man) | `yak_core/config.py` | early |
| 3 | Player pool loading — RotoGrinders CSV | `yak_core/rg_loader.py` | early |
| 4 | Player pool loading — Tank01 live API | `yak_core/live.py` | early |
| 5 | Projection modes: salary_implied, regression, blend | `yak_core/projections.py` | early |
| 6 | Projection noise + exposure controls | `yak_core/lineups.py` | early |
| 7 | Contest types: GPP, 50/50, Single Entry, MME, Showdown Captain | `yak_core/calibration.py` | early |
| 8 | Archetypes: Ceiling Hunter, Floor Lock, Balanced, Contrarian, Stacker | `yak_core/calibration.py` | early |
| 9 | Lock / Exclude / Bump player overrides | `yak_core/lineups.py`, `streamlit_app.py` | early |
| 10 | Right Angle Ricky edge analysis (stacks, pace, value plays) | `yak_core/right_angle.py` | early |
| 11 | Calibration Queue (review/action prior-day lineups) | `yak_core/calibration.py` | early |
| 12 | Ad Hoc Historical Lineup Builder (backtest vs. actuals) | `yak_core/calibration.py` | early |
| 13 | DK Contest CSV Ingest (import real ownership) | `yak_core/contest_ingest.py`, `yak_core/multislate.py` | early |
| 14 | Archetype Config Knobs (tune per-archetype params) | `yak_core/calibration.py` | early |
| 15 | Monte Carlo Sim Module + lineup promotion | `yak_core/sims.py` | early |
| 16 | Live injury/news updates (Tank01) | `yak_core/live.py` | early |
| 17 | Ownership model (salary-rank, leverage scoring) | `yak_core/ownership.py` | early |
| 18 | Scoring/KPI dashboard (hit rate, avg pts, projection %) | `yak_core/scoring.py` | early |
| 19 | Streamlit UI — 3-tab layout | `streamlit_app.py` | early |
| 20 | Optimizer cancellation / min-team-stack guards | `yak_core/lineups.py` | early |
| 21 | **DK bulk-upload CSV export** (`to_dk_upload_format`) | `yak_core/lineups.py` | #12 |
| 22 | Download DK upload CSV button in Optimizer + Sim tabs | `streamlit_app.py` | #12 |
| 23 | Unit tests: optimizer cancellations (10 tests) | `tests/test_optimizer_cancellations.py` | early |
| 24 | Unit tests: DK upload format (10 tests) | `tests/test_dk_upload_format.py` | #12 |

---

## What's Remaining 🔲

### High Priority

| # | Feature | Notes |
|---|---------|-------|
| R1 | **Fix `YAKOS_ROOT` hardcoded path** | `yak_core/config.py` line 6 has `/Users/franklynch/Library/…`. Should fall back to a relative or env-var path so the app runs on any machine. |
| R2 | **Multi-slate UI** | `yak_core/multislate.py` has `discover_slates()`, `run_multi_slate()`, `compare_slates()` but none are surfaced in the Streamlit UI yet. |
| R3 | **Persistent calibration config** | `save_calibration_config` writes to a local path. Need to decide on a committed default (e.g., `data/calibration_config.json`) so settings survive a fresh clone. |

### Medium Priority

| # | Feature | Notes |
|---|---------|-------|
| R4 | **CI/CD pipeline** | No GitHub Actions workflow. Add a simple `pytest` run on push/PR so tests stay green. |
| R5 | **Expanded test coverage** | Only optimizer-cancellations and DK-upload-format are tested. Add tests for projections, calibration metrics, ownership model, right-angle edge analysis. |
| R6 | **Lineup correlation / diversity controls** | Current exposure cap is the only uniqueness mechanism. Could add player-pair fade (same player not allowed in N consecutive lineups) or explicit game-stack enforcement. |
| R7 | **Showdown Captain mode full optimizer** | Captain slot logic exists in `apply_slate_filters` but `_eligible_slots` in lineups.py treats "CPT" as a regular slot. Verify Captain 1.5× salary / scoring multiplier is applied correctly end-to-end. |

### Low Priority / Nice-to-Have

| # | Feature | Notes |
|---|---------|-------|
| R8 | **Docker / one-click deploy** | `Dockerfile` + `docker-compose.yml` so anyone can spin up the Streamlit app without a local Python environment. |
| R9 | **Historical projection model training** | `proj_model()` exists in `yak_core/projections.py` but relies on parquet files at `YAKOS_ROOT`. Could export training data from `data/historical_lineups.csv` and train inline. |
| R10 | **Export / share lineups via URL** | Streamlit `st.query_params` could encode a shareable lineup state. |
| R11 | **Dark mode / UI polish** | Minor Streamlit theme tweaks. |

---

## Architecture Notes

```
YakOS/
├── streamlit_app.py          # Streamlit UI — imports from yak_core
├── yak_core/
│   ├── config.py             # DEFAULT_CONFIG, merge_config, DK constants
│   ├── lineups.py            # LP optimizer, exposure control, to_dk_upload_format
│   ├── projections.py        # salary_implied, regression, blend, proj_model
│   ├── calibration.py        # archetypes, queue, backtest, config knobs
│   ├── right_angle.py        # stack/pace/value edge analysis + lineup tagging
│   ├── sims.py               # Monte Carlo, live update, promote logic
│   ├── live.py               # Tank01 API (live pool + injury updates)
│   ├── ownership.py          # salary-rank ownership, leverage
│   ├── scoring.py            # KPIs, hit rate, projection %, backtest summary
│   ├── rg_loader.py          # RotoGrinders CSV parser
│   ├── multislate.py         # multi-slate discovery, run, compare, DK CSV ingest
│   ├── contest_ingest.py     # DK contest results CSV → ownership
│   └── validation.py         # lineup validity checks
├── data/
│   ├── NBADK20260227.csv     # sample RG pool file
│   ├── historical_lineups.csv
│   └── yakos_projections_2026-02-27.csv
├── tests/
│   ├── test_optimizer_cancellations.py  (10 tests)
│   └── test_dk_upload_format.py         (10 tests)
└── requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Tests:
```bash
python -m pytest tests/ -v
```

---

## Agent Memory Note

This `ROADMAP.md` file is the intended place to record context between agent sessions.
After completing work, update the ✅ table and move items from 🔲 Remaining to ✅ Built.
