# YakOS
Right Angle Ricky DFS Optimizer — an NBA DraftKings lineup optimizer with a Streamlit web UI.

## Features

- **🏀 Ricky's Slate Room** — load today's player pool (RotoGrinders CSV or live Tank01 API), view Pool KPIs, run edge analysis (stack alerts, pace environment, value plays), and surface high-confidence lineups promoted from the Calibration Lab.
- **⚡ Optimizer** — build DraftKings lineups for any contest type (GPP, 50/50, Single Entry, MME, Showdown) with configurable archetype, projection style, and salary controls.
  - **Lock players** into every lineup
  - **Exclude players** from the pool
  - **Bump projections** up or down by a custom multiplier
- **🔬 Calibration Lab**
  - Calibration Queue — review prior-day lineups and action them (reviewed / apply_config / dismissed)
  - Ad Hoc Historical Lineup Builder — backtest against real actuals and inspect projection accuracy by player, position, and salary bracket
  - DK Contest CSV Ingest — upload a DraftKings contest results CSV to import real ownership data into the live pool
  - Archetype Config Knobs — tune Ceiling Hunter, Floor Lock, Balanced, Contrarian, and Stacker parameters
  - Sim Module — Monte Carlo simulations on the live player pool with live injury / news updates; promote high-confidence lineups to Ricky's Slate Room

## Setup

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

A Tank01 RapidAPI key is optional but required for live pool fetch and injury updates.  
Set it via the sidebar or the `RAPIDAPI_KEY` environment variable.

## Data

Place RotoGrinders NBA projection CSVs in `data/` and register them in the `RG_POOL_FILES` dict in `streamlit_app.py`.  
Historical contest entries go in `data/historical_lineups.csv` (columns: `slate_date, contest_name, lineup_id, pos, team, name, salary, own, actual`).
