# ⚽ FIFA World Cup 2026 Prediction

An end-to-end machine learning pipeline for predicting FIFA World Cup 2026 match outcomes. The project scrapes real player performance data from [WhoScored](https://www.whoscored.com), engineers 120+ features from historical match data, and simulates the entire tournament 50,000 times using a four-model ensemble to produce win probabilities for all 48 qualified nations.

![Winning Probabilities](wc2026_probs_stacked_bar_chart.png)

## Pipeline Overview

### 1. Data Collection

Five Playwright-based scrapers collect player data from WhoScored for all 48 World Cup teams. The scrapers use stealth patches to bypass Cloudflare protection and employ a multi-strategy extraction approach — trying XHR interception, embedded `<script>` parsing, and DOM selector scraping in sequence, falling back gracefully when one method is blocked.

- `scrape_players.py` : Scrapes rosters for all 48 teams (~1,500 players), outputting player names, IDs, and profile links
- `scrape_player_stats.py` : Summary stats: rating, goals, assists, shots, aerial duels, cards
- `scrape_player_stats_defensive.py` : Defensive stats: tackles, interceptions, clearances, blocks
- `scrape_player_stats_offensive.py` : Offensive stats: shots, key passes, dribbles, dispossessions
- `scrape_player_stats_passing.py` : Passing stats: pass accuracy, crosses, long balls, through balls

`results.csv` provides 49,000+ historical international match results from 1872 to June 2026, used for Elo computation and training data

### 2. Data Processing

Six Jupyter notebooks (run in Google Colab) clean and aggregate the raw scraped data:

- `stats_summary.ipynb`, `stats_defensive.ipynb`, `stats_offensive.ipynb`, `stats_passing.ipynb` : Drop metadata columns, aggregate per-tournament rows into a single row per player (summing counting stats, averaging per-game rates)
- `merged_stats.ipynb` : Merges all four stat categories into one file (`merged_player_stats.csv`)
- `results.ipynb` : Filters historical results to 2023–2026 and drops location columns (`filtered_results.csv`)

### 3. Feature Engineering

`wc2026_phase2.py` transforms the cleaned data into a model-ready training matrix (~1,500 matches × 125 features):

- **Elo ratings** : Custom implementation with tournament-weighted K-factor, goal-difference multiplier, and no home advantage (all WC matches are on neutral ground)
- **Rolling form** : Goals scored/conceded, win/draw rates, and clean sheet rates over the last 10 and 20 matches
- **Head-to-head** : Historical win percentage, average goal difference, and recent form between each pair of opponents
- **Squad quality** : Aggregated from player data: average/max rating, total goals and assists, offensive/defensive ratios, experience depth, star concentration
- **Momentum** : Qualifying campaign points per game, form trajectory, current streaks, days since last competitive win
- **Context** : Tournament type flags (World Cup, qualifier, continental, friendly)
- **Differentials** : Pairwise differences between teams for key features (Elo, form, squad strength)

Outputs: `training_features.csv`, `wc2026_fixture_features.csv` (72 group-stage fixtures), and `team_snapshot.csv` (48-team profile)

### 4. Simulation

`wc2026_phase3_simulation.py` trains four models on historical data, then simulates the entire tournament 50,000 times

**Models:**
- **Dixon-Coles** : A bivariate Poisson model that estimates team-level attack/defense strengths with a correlation correction for low-scoring draws, using exponential time-decay weighting
- **XGBoost** : Gradient-boosted trees trained on core match features (Elo, form, H2H, context)
- **LightGBM** : A second gradient-boosted model for ensemble diversity
- **MLP** : A neural network trained on the full feature set including squad quality metrics

Models are weighted by inverse Brier score on a validation set, and all 1,128 pairwise match probabilities are pre-computed before simulation. Training data is augmented by swapping home/away perspectives to double the sample size.

**Adjustments:** Host-nation Elo boost (+50 for USA, Canada, Mexico), pedigree bonuses for recent tournament performance (e.g., Argentina as defending champion), and confederation strength discounts.

**Simulation:** Each run plays out the full group stage (generating scorelines from Dixon-Coles goal rates), ranks teams by points/GD/GF, advances the top 2 + best 8 third-place teams, then simulates the knockout bracket through to the final using the official FIFA R32 bracket structure.

### 5. Visualization

- `predict_32_teams.py` : Generates a group-by-group advancement report with status labels (Lock / Likely / Toss-up / Tough / Unlikely), prints predicted R32 qualifiers, and exports a formatted PDF (`wc2026_group_stage_predictions.pdf`)
- `create_chart.py` : Produces the stacked bar chart showing cumulative stage-reaching probabilities for the top 20 teams

## Results

Top 10 predicted winners (50,000 simulations):

| # | Team | Group | Win | Final | SF | QF |
|---|---|---|---|---|---|---|
| 1 | Argentina | J | 24.9% | 34.3% | 47.0% | 60.0% |
| 2 | Spain | H | 14.3% | 22.2% | 37.4% | 49.5% |
| 3 | England | L | 10.9% | 20.4% | 32.5% | 49.7% |
| 4 | Colombia | K | 5.7% | 10.2% | 20.3% | 35.4% |
| 5 | Morocco | C | 5.6% | 13.0% | 23.6% | 40.9% |
| 6 | France | I | 5.6% | 13.1% | 23.8% | 40.6% |
| 7 | Brazil | C | 5.6% | 13.2% | 24.1% | 40.2% |
| 8 | Portugal | K | 4.3% | 8.4% | 18.9% | 37.5% |
| 9 | Ecuador | E | 3.9% | 8.7% | 15.5% | 27.0% |
| 10 | Uruguay | H | 2.9% | 6.3% | 13.9% | 25.0% |

Full probabilities for all 48 teams (Champion through Group Exit) in `wc2026_probs.csv`. Group-by-group breakdown with advancement status in `wc2026_group_stage_predictions.pdf`.

## Setup & Usage

```bash
pip install pandas numpy scipy scikit-learn xgboost lightgbm matplotlib reportlab playwright playwright-stealth

# 1. Scrape rosters and stats
python scrape_players.py
python scrape_player_stats.py
python scrape_player_stats_defensive.py
python scrape_player_stats_offensive.py
python scrape_player_stats_passing.py

# 2. Run processing notebooks in Jupyter/Colab

# 3. Build features
python wc2026_phase2.py

# 4. Run simulation
python wc2026_phase3_simulation.py

# 5. Generate reports and charts
python predict_32_teams.py wc2026_probs.csv
python create_chart.py wc2026_probs.csv
```

All scrapers support `--headless`, `--headed`, `--debug`, `-i`, `-o` flags. Stats scrapers also accept `--delay-min` / `--delay-max` to control request pacing. A residential IP is recommended to avoid Cloudflare blocks.

## License

For personal research and educational use. Please respect the terms of service of all data sources.
