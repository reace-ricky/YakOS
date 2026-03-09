# Right Angle Ricky — Blueprint

> This is the single source of truth. Every build session starts here.
> If the code doesn't match this doc, the code is wrong.

---

## Why This Exists

On 2026-03-08, we ran 150 GPP lineups through the optimizer.
- **Zero** hit 280 points (typical GPP cash line)
- Best actual score: 265.5
- Average actual score (top 50): 233.0
- Hindsight optimal lineup: **376.0 points**
- Gap: **120 points**

The optimizer stacked the highest-projected chalk players every time, producing 7-8 groups of near-identical lineups. It had no awareness of ownership, contest targets, or value. It did the dumbest possible thing — max projection, fill salary — and called it done.

### The Evidence (2026-03-08 GPP Slate)

**Our #1 Lineup (actual: 256.0):**

| Player | Pos | Salary | Proj | Actual | Miss | Own% |
|--------|-----|--------|------|--------|------|------|
| Deni Avdija | SF/PF | $8,600 | 46.4 | 44.2 | -2.1 | 0.97 |
| Dejounte Murray | PG | $6,900 | 39.5 | 33.5 | -6.0 | 1.06 |
| Saddiq Bey | SF/PF | $6,800 | 39.5 | 44.5 | +5.0 | 0.94 |
| Trey Murphy III | PG/SG | $7,700 | 38.6 | 39.2 | +0.6 | 0.92 |
| Russell Westbrook | PG | $7,000 | 34.6 | 57.8 | +23.1 | 0.83 |
| Myles Turner | C | $5,500 | 30.8 | 5.5 | **-25.3** | 0.95 |
| Julian Reese | C | $4,400 | 27.0 | 27.8 | +0.7 | 0.85 |
| Matisse Thybulle | SF/PF | $3,100 | 19.2 | 3.5 | **-15.7** | 0.60 |

- Avg ownership: ~0.89 (extreme chalk)
- Myles Turner and Matisse Thybulle alone lost us 41 points

**Hindsight Winning Lineup (actual: 376.0):**

| Player | Pos | Salary | Proj | Actual | Beat By | Own% |
|--------|-----|--------|------|--------|---------|------|
| Russell Westbrook | PG | $7,000 | 34.6 | 57.8 | +23.1 | 0.83 |
| Maxime Raynaud | C | $6,300 | 23.4 | 53.8 | +30.3 | 0.46 |
| Desmond Bane | SG/SF | $8,000 | 35.0 | 46.8 | +11.7 | 0.68 |
| Scoot Henderson | PG | $5,700 | 20.8 | 46.8 | +25.9 | 0.34 |
| Malik Monk | PG/SG | $4,500 | 21.4 | 45.5 | +24.1 | 0.47 |
| Saddiq Bey | SF/PF | $6,800 | 39.5 | 44.5 | +5.0 | 0.94 |
| Jay Huff | C | $5,200 | 20.1 | 41.0 | +20.9 | 0.39 |
| Bobby Portis | PF/C | $5,300 | 28.1 | 40.0 | +11.9 | 0.83 |

- Avg ownership: **0.62** (significantly lower than our chalk-heavy lineup)
- Avg salary: $6,100 (didn't need to max salary)
- 5 of 8 players were LOW projected, LOW owned, LOW salary — and they exploded

**Pattern in the winners:** Low projection + low ownership + upside potential = GPP gold. Our system never even considered these players.

---

## The System (4 Layers)

### Layer 1: Projection Model

Produces a projected fantasy point total for every player on the slate.

**Inputs (with approximate weights from industry research):**

| Input | Weight | Source |
|-------|--------|--------|
| Fantasy Points Per Minute (FPPM) | 50-55% | Historical game logs (season avg 65%, last 5 games 35%) |
| Minutes Projection | 20-25% | Season avg + recent avg + injury/rotation adjustments |
| Turnover-Adjusted Usage Rate | 15-20% | NBA.com stats |
| Vegas Team Totals | 10-14% | Betting lines (compare to season avg) |
| Defense vs Position (DvP) | 5-10% | Matchup-based adjustment |
| Projected Pace | 5-10% | Vegas total → pace proxy, diff from team avg pace |
| Vegas Spread (Blowout Risk) | Adjustment | Large spreads → reduce minutes projection for favorites' starters |

**Output:** Projected FP, projected floor, projected ceiling for each player.

**"Done" means:** Projection correlation with actuals > 0.80 across 20+ slates. MAE < 6.0.

---

### Layer 2: Ownership Model

Estimates what % of the field will roster each player.

**Inputs:**
- Salary rank within position
- Projection rank (higher proj = higher owned)
- Recent performance / narrative (hot streaks get owned)
- Injury news impact on teammates (backup gets owned when starter sits)
- Third-party ownership projections (RotoGrinders, FantasyPros) as baseline

**Output:** Estimated ownership % for each player.

**Key metric — Leverage:**
```
leverage = (player_edge / ownership%)
```
High projection + low ownership = high leverage (GPP gold).
High projection + high ownership = chalk (fine for cash, bad for GPP).

**"Done" means:** Ownership estimates within 5% MAE of actual DK contest ownership across 10+ slates.

---

### Layer 3: Optimizer (Contest-Aware)

Builds lineups that TARGET A SCORE BAND for the specific contest type. NOT "maximize projection."

**Contest Bands (calibrate over time):**

| Contest Type | Target Band | Strategy |
|-------------|-------------|----------|
| Cash / 50-50 | > median score (~240-260) | High floor, high ownership, consistency. Minimize bust risk. |
| GPP Top 15% | > ~280 | Balanced — some chalk anchors + 2-3 leverage plays |
| GPP Top 1% | > ~320 | Max leverage. Low-owned upside players. Game stacks. Accept bust risk. |
| Showdown | Varies by game | Captain leverage is key. Low-owned captain = massive differentiation. |

**How the optimizer works (by contest type):**

**Cash mode:**
- Objective: maximize FLOOR (not projection)
- Constraint: all players must have ownership > X% (they're chalk for a reason in cash)
- Constraint: projected minutes > 28 (no bench guys)
- No duplicate lineups needed — build 1-3 rock-solid lineups

**GPP mode:**
- Objective: maximize (projection × leverage) + upside bonus
- Must include 2-3 players with ownership < 10% and ceiling > 40
- Must include at least 1 game stack (2+ players from same game)
- Generate 20 unique lineups with max 60% exposure per player
- Rank by "SaberSim-style" score: projection + upside + negative weight on ownership

**Showdown mode:**
- Captain selection weighted by leverage (low-owned captain = massive edge)
- Flex spots balance floor + upside

**"Done" means:** On historical slates, at least 20% of GPP lineups would have finished in the top 15%. At least 1 cash lineup cashes on 70%+ of slates.

---

### Layer 4: Calibration (Outcome-Based)

**We calibrate against the CASH LINE, not against projection accuracy.**

Old (wrong): "Was our projection for Player X close to their actual?"
New (right): "Did our lineup hit the contest's cash threshold?"

**After each slate:**

1. Score all generated lineups using actuals
2. Compare each lineup's actual total to the contest band
3. Record: hit rate by contest type (% of lineups that cashed)
4. If we missed, diagnose WHY:
   - Projection error? (player busted or smashed vs projection)
   - Ownership error? (we were chalk-heavy in GPP)
   - Construction error? (right players identified but wrong lineup combos)
   - Minutes error? (player got fewer/more minutes than projected)
5. Feed the diagnosis back into the appropriate layer

**Calibration targets:**

| Metric | Target | Current |
|--------|--------|---------|
| Cash lineup cash rate | > 70% of slates | ~0% |
| GPP top-15% hit rate | > 20% of lineups | 0% |
| GPP top-1% hit rate | > 3% of lineups | 0% |
| Projection MAE | < 6.0 | 6.94 |
| Projection correlation | > 0.80 | 0.736 |
| Ownership MAE | < 5.0% | Unknown (ownership data was broken) |

---

## What We Do NOT Build

- Elaborate diagnostic UI (gauges, charts, status bars)
- Tests that check if code runs instead of if output is good
- Features that aren't directly wired to producing better lineups
- Anything scoped but not wired end-to-end

---

## Build Order

1. **Projection model** — get the inputs right (FPPM, minutes, pace, DvP, Vegas)
2. **Ownership model** — estimate field ownership, calculate leverage
3. **Optimizer** — contest-aware lineup builder using leverage + bands
4. **Calibration** — score against cash lines, diagnose misses, feed back

Each layer is **proven on real historical data** before moving to the next. "Proven" means visible results you can see — not test counts.

---

## The Bar

If Yak can beat the system by hand, the system has negative value.

Every output must answer: **would this lineup have cashed?**
