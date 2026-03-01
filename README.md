# YakOS
Right Angle Ricky DFS Optimizer — an NBA DraftKings lineup optimizer with a Streamlit web UI.

## Features

- **🏀 Ricky's Slate Room** — load today's player pool (RotoGrinders CSV or live Tank01 API), view Pool KPIs, run edge analysis (stack alerts, pace environment, value plays), and surface high-confidence lineups promoted from the Calibration Lab.
- **⚡ Optimizer** — build DraftKings lineups for any contest type (GPP, 50/50, Single Entry, MME, Showdown) with configurable archetype, projection style, and salary controls.
  - **Lock players** into every lineup
  - **Exclude players** from the pool
  - **Bump projections** up or down by a custom multiplier
  - **Lineup diversity** — set *Max pair appearances* to prevent any two players from teaming up more than N times across all lineups
  - **Showdown Captain mode** — full CPT + 5 FLEX optimizer with correct 1.5× captain salary and scoring; exports DK Showdown bulk-upload CSV
- **🔬 Calibration Lab**
  - Calibration Queue — review prior-day lineups and action them (reviewed / apply_config / dismissed)
  - Ad Hoc Historical Lineup Builder — backtest against real actuals and inspect projection accuracy by player, position, and salary bracket
  - DK Contest CSV Ingest — upload a DraftKings contest results CSV to import real ownership data into the live pool
  - Archetype Config Knobs — tune Ceiling Hunter, Floor Lock, Balanced, Contrarian, and Stacker parameters
  - Sim Module — Monte Carlo simulations on the live player pool with live injury / news updates; promote high-confidence lineups to Ricky's Slate Room

## Setup

### Local development

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app will open at **http://localhost:8501**.

### Docker (one-click)

```bash
docker compose up
```

The app will be available at **http://localhost:8501**.  
Pass your Tank01 key via environment variable: `RAPIDAPI_KEY=your_key docker compose up`.

### Streamlit Community Cloud (free web deployment)

1. Fork or push this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repository → set the main file to `streamlit_app.py`.
4. Click **Deploy**. The app will be live at `https://<your-app>.streamlit.app` in ~2 minutes.

No extra configuration is required — `requirements.txt` and `.streamlit/config.toml` are already committed to the repo.

### Optional: Tank01 RapidAPI key

Required for **Fetch Pool from API** and live injury updates.  
Set it via the sidebar text box, the `RAPIDAPI_KEY` environment variable, or — on Streamlit Cloud — add it as a secret named `RAPIDAPI_KEY` under **Settings → Secrets**.

## Data

Place RotoGrinders NBA projection CSVs in `data/` and register them in the `RG_POOL_FILES` dict in `streamlit_app.py`.  
Historical contest entries go in `data/historical_lineups.csv` (columns: `slate_date, contest_name, lineup_id, pos, team, name, salary, own, actual`).
