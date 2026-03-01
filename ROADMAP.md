# YakOS Roadmap & Status

> **Purpose:** This file is the single source of truth for the project's end goal, current status, and remaining work.
> Future agent sessions should read this file first to understand context before making changes.

---

## End Goal

Build a production-quality **NBA DraftKings DFS lineup optimizer** called *YakOS / Right Angle Ricky* with:

1. A polished **Streamlit web UI** with four tabs:
   - 🏀 **Ricky's Slate Room** — pool loader, KPI dashboard, edge analysis, promoted lineups
   - ⚡ **Optimizer** — build lineups for any DK contest type with full override controls
   - 🔬 **Calibration Lab** — backtest, queue, ownership ingest, archetype knobs, sim module
   - 📡 **Ricky's Calibration Lab** — BacktestIQ-style backtesting: contest archetype ROI/cash-rate/percentile KPIs

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
| 25 | **Fix `YAKOS_ROOT` hardcoded path** — env var + relative fallback | `yak_core/config.py` | latest |
| 26 | **Persistent calibration config** — default `data/calibration_config.json` | `yak_core/calibration.py`, `data/calibration_config.json` | latest |
| 27 | **Multi-slate UI** — discover, batch-run, compare slates in Calibration Lab | `streamlit_app.py` | latest |
| 28 | **CI/CD pipeline** — GitHub Actions `pytest` on push/PR | `.github/workflows/ci.yml` | latest |
| 30 | **Lineup diversity controls** (`MAX_PAIR_APPEARANCES`) | `yak_core/lineups.py`, `yak_core/config.py`, `streamlit_app.py` | latest |
| 31 | **Showdown Captain full optimizer** — CPT × 1.5 salary/scoring, 6-man roster | `yak_core/lineups.py`, `yak_core/config.py`, `streamlit_app.py` | latest |
| 32 | **Docker / one-click deploy** | `Dockerfile`, `docker-compose.yml` | latest |
| 33 | **Dark mode / UI polish** | `.streamlit/config.toml` | latest |
| 34 | **Unit tests: diversity + Showdown** (25 tests) | `tests/test_diversity_and_showdown.py` | latest |
| 35 | **Dashboard API fetch + projection fallback** — fetch pool from API directly on Ricky's Slate Room; salary-implied fallback when API returns 0 proj | `streamlit_app.py` | latest |
| 36 | **Player Projections table on dashboard** — expanded by default, sorted by proj desc, clean columns, auto-sized | `streamlit_app.py` | latest |
| 37 | **Remove contest_entry / contest_entries from Calibration Queue display** — also removed fixed height for auto-sizing | `streamlit_app.py` | latest |
| 38 | **Auto-load sample pool on startup** — `NBADK20260227.csv` is loaded at first run so the dashboard immediately shows projections without manual upload; fixed NaN ownership KPI display | `streamlit_app.py` | latest |
| 39 | **Player Projections table in Calibration Lab Review & Action** — same sorted projections expander (expanded by default) shown under Review & Action in Section A | `streamlit_app.py` | latest |
| 40 | **Merge player projections into Review & Action table** — `floor`, `ceil` (and `proj_own` if missing) from pool joined into the Review & Action data editor; proj pts before act pts, proj own % before act own %; standalone expander removed | `streamlit_app.py` | latest |
| 41 | **Calibration KPI Dashboard** — `📊 Calibration KPI Dashboard` section at top of Calibration Lab: strategy KPIs (total lineups, hit rate, avg score), points accuracy (mean error, std, MAE, RMSE, R²) at lineup and player level, proj vs actual scatter chart, salary-bracket error table, ownership bucket calibration, and conditional minutes accuracy metrics | `yak_core/scoring.py`, `streamlit_app.py` | latest |
| 43 | **Calibration KPI Dashboard cleanup** — removed RAG status badges, `st.metric` circles, bold headers, and caption text from the 4 top-level KPI cards; replaced with clean bordered HTML boxes (label + value); removed `calibration_rag` import; expander titles cleaned of RAG emojis | `streamlit_app.py` | latest |
| 45 | **Calibration page redesign** — thin 4-KPI strip (Pts MAE player, Min MAE player, Own MAE player, Hit rate) with color-coded cards via `quality_color()`; advanced stats collapsed under "Advanced breakdown"; queue table shows focused columns (Player, Salary, Proj/Act FP, Error, Proj/Act Mins, Min Error, Proj/Act Own%, Own Error, Flag) | `yak_core/scoring.py`, `streamlit_app.py` | latest |
| 46 | **Sim module backtest** — `backtest_sim()` in `yak_core/sims.py` runs Monte Carlo on historical lineup data and compares sim predictions to actuals, returning `sim_mae`, `sim_rmse`, `sim_bias`, `within_range_pct`, and a per-lineup DataFrame | `yak_core/sims.py` | latest |
| 47 | **Slate Room 3-layer redesign** — Layer 1: color-coded KPI strip (Slate EV, Approved count by archetype, Exposure risk, Simmed hit rate, Last updated); Layer 2: data-driven Edge Analysis via `compute_stack_scores()` + `compute_value_scores()` (stack score, leverage tag, value index, ownership tag); Layer 3: Approved Lineups with archetype tabs, expandable compact lineup cards, late-swap badge, calibration note. Added `ApprovedLineup` dataclass + `build_approved_lineups()` / `get_approved_lineups_by_archetype()` / `compute_slate_kpis()` to `calibration.py`. Optimizer now injects stack/value scores into LP objective via `STACK_WEIGHT` / `VALUE_WEIGHT` config. 40 new unit tests. | `yak_core/right_angle.py`, `yak_core/calibration.py`, `yak_core/lineups.py`, `yak_core/config.py`, `streamlit_app.py`, `tests/test_slate_room_features.py` | latest |
| 46 | **Ricky's Calibration Lab (BacktestIQ-style)** — new 4th tab with backtest controls (sport, date range, site, contest-archetype multi-select, build config override, # lineups, Run Backtest), global KPI strip (ROI / cash rate / avg finish %ile / best finish with green/yellow/red coloring), archetype summary table (sorted worst ROI first, row coloring), slate-level drilldown, and Player Calibration Queue integration; `BACKTEST_ARCHETYPES` config + `run_archetype_backtest()` engine + `_reconstruct_pool_from_slate()` helper added to `yak_core/calibration.py`; 19 new unit tests in `tests/test_backtest_engine.py` | `yak_core/calibration.py`, `streamlit_app.py`, `tests/test_backtest_engine.py` | latest |
| 48 | **Sim Module table column glossary** — "Lineup-level sim metrics" expander now shows friendly column names (Avg Score, Std Dev, Smash %, Bust %, Median Score, P85 (Upside), P15 (Floor)) plus an inline markdown table explaining each metric in plain English; threshold values (`SMASH_THRESHOLD`, `BUST_THRESHOLD`) exported from `sims.py` and referenced dynamically in the UI | `streamlit_app.py`, `yak_core/sims.py` | latest |
| 49 | **Sim Module player accuracy table** — `build_sim_player_accuracy_table()` in `yak_core/sims.py` compares per-player sim projections to real actuals (MAE, RMSE, bias, hit rate ±10 FP, R²); actuals CSV uploader added to Sim Module section C in the Calibration Lab; player table sorted by abs error, with download button; 23 new unit tests in `tests/test_sim_player_accuracy.py` | `yak_core/sims.py`, `streamlit_app.py`, `tests/test_sim_player_accuracy.py` | latest |
| 50 | **Sim Module redesign** — Section C in Calibration Lab rebuilt from scratch: mode toggle at top (🔴 Live / 📅 Historical Date); historical mode pre-fills actuals date picker with selected date; Custom Lineup Builder (DK Classic slot selectors from sim pool with salary+proj summary); Sim vs Custom Lineup Comparison (side-by-side best-sim lineup vs custom, actual scores, "What the Sim Missed" miss-analysis table with download) | `streamlit_app.py` | latest |
| 51 | **Fix Custom Lineup Builder multi-position filtering** — `_players_for_slot` now splits position strings on "/" before matching eligible slots, so dual-eligibility players (e.g. "SG/SF", "PF/C", "PG/SG", "SF/PF") appear in all correct slot dropdowns; also added `.fillna("")` guard for null position values | `streamlit_app.py` | latest |
| 53 | **Calibration Lab page cleanup** — removed caption text under KPI cards; added tiny target indicator inside each bubble (≤ 6 pts, ≤ 3 min, ≤ 3%, ≥ 70%); removed Advanced breakdown expander, Stack Hit Log section, Build Best Lineup for a Slate expander, and Compare vs Contest type expander | `streamlit_app.py` | latest |
| 54 | **Tank01 API dict-body fix** — `fetch_live_dfs` now unwraps `body` dict (e.g., `{"DraftKings": [...]}`) via `_TANK01_DFS_PLAYER_KEYS` constant + longest-list fallback; 5 new tests added | `yak_core/live.py`, `tests/test_live_actuals.py` | latest |

---

## What's Remaining 🔲

### High Priority

| # | Feature | Notes |
|---|---------|-------|
| ~~R1~~ | ~~**Fix `YAKOS_ROOT` hardcoded path**~~ | ✅ Done — uses `YAKOS_ROOT` env var, falls back to repo root via `Path(__file__).parent.parent`. |
| ~~R2~~ | ~~**Multi-slate UI**~~ | ✅ Done — Section F in Calibration Lab surfaces `discover_slates`, `run_multi_slate`, `compare_slates`. |
| ~~R3~~ | ~~**Persistent calibration config**~~ | ✅ Done — defaults to `data/calibration_config.json`; committed default ships with the repo. |

### Medium Priority

| # | Feature | Notes |
|---|---------|-------|
| ~~R4~~ | ~~**CI/CD pipeline**~~ | ✅ Done — `.github/workflows/ci.yml` runs `pytest` on every push/PR. |
| ~~R5~~ | ~~**Expanded test coverage**~~ | ✅ Done — 75 tests added across `test_projections.py` (26 tests), `test_ownership.py` (21 tests), `test_right_angle.py` (28 tests). |
| ~~R6~~ | ~~**Lineup correlation / diversity controls**~~ | ✅ Done — `MAX_PAIR_APPEARANCES` config key prevents any player pair from appearing together more than N times across all lineups. 0 = disabled (default). Exposed as "Max pair appearances" number input in the Optimizer tab. |
| ~~R7~~ | ~~**Showdown Captain mode full optimizer**~~ | ✅ Done — `build_showdown_lineups()` implements CPT + 5 FLEX roster (6 players); Captain costs 1.5× salary and scores 1.5× fantasy points; `to_dk_showdown_upload_format()` exports the correct DK Showdown CSV; Optimizer tab dispatches to the correct function based on Slate Type. |

### Low Priority / Nice-to-Have

| # | Feature | Notes |
|---|---------|-------|
| ~~R8~~ | ~~**Docker / one-click deploy**~~ | ✅ Done — `Dockerfile` + `docker-compose.yml` added. Run `docker compose up` to spin up the Streamlit app at http://localhost:8501. |
| ~~R11~~ | ~~**Dark mode / UI polish**~~ | ✅ Done — `.streamlit/config.toml` updated with dark base theme, orange primary colour, and clean dark background palette. |
| R9 | **Historical projection model training** | `proj_model()` exists in `yak_core/projections.py` but relies on parquet files at `YAKOS_ROOT`. Could export training data from `data/historical_lineups.csv` and train inline. |
| R10 | **Export / share lineups via URL** | Streamlit `st.query_params` could encode a shareable lineup state. |

---

## Architecture Notes

```
YakOS/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions — pytest on push/PR
├── Dockerfile                    # Docker image for one-click deploy
├── docker-compose.yml            # docker compose up → app at :8501
├── streamlit_app.py          # Streamlit UI — imports from yak_core
├── yak_core/
│   ├── config.py             # DEFAULT_CONFIG, merge_config, DK constants (Classic + Showdown), YAKOS_ROOT
│   ├── lineups.py            # LP optimizer, exposure control, pair-fade diversity, Showdown optimizer, DK upload formats
│   ├── projections.py        # salary_implied, regression, blend, proj_model
│   ├── calibration.py        # archetypes, queue, backtest, config knobs, persistent calibration_config.json
│   ├── right_angle.py        # stack/pace/value edge analysis + lineup tagging
│   ├── sims.py               # Monte Carlo, live update, promote logic, player accuracy table
│   ├── live.py               # Tank01 API (live pool + injury updates)
│   ├── ownership.py          # salary-rank ownership, leverage
│   ├── scoring.py            # KPIs, hit rate, projection %, backtest summary
│   ├── rg_loader.py          # RotoGrinders CSV parser
│   ├── multislate.py         # multi-slate discovery, run, compare, DK CSV ingest
│   ├── contest_ingest.py     # DK contest results CSV → ownership
│   └── validation.py         # lineup validity checks
├── data/
│   ├── calibration_config.json  # committed default calibration config
│   ├── NBADK20260227.csv     # sample RG pool file
│   ├── historical_lineups.csv
│   └── yakos_projections_2026-02-27.csv
├── tests/
│   ├── test_optimizer_cancellations.py  (10 tests)
│   ├── test_dk_upload_format.py         (10 tests)
│   ├── test_projections.py              (26 tests)
│   ├── test_ownership.py                (21 tests)
│   ├── test_right_angle.py              (28 tests)
│   ├── test_diversity_and_showdown.py   (25 tests)
│   ├── test_calibration_queue.py        (12 tests)
│   ├── test_backtest_engine.py          (19 tests)
│   ├── test_slate_room_features.py      (40 tests)
│   ├── test_sim_backtest.py             (19 tests)
│   ├── test_sim_player_accuracy.py      (23 tests)
│   └── test_live_actuals.py             (23 tests)
└── requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Docker (one-click):
```bash
docker compose up
# App available at http://localhost:8501
```

Tests:
```bash
python -m pytest tests/ -v
```

---

## Agent Memory Note

This `ROADMAP.md` file is the intended place to record context between agent sessions.
After completing work, update the ✅ table and move items from 🔲 Remaining to ✅ Built.
