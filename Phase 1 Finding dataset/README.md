# ⚽ FIFA World Cup 2026 — Player Data Scraper

Python toolkit for scraping player rosters and performance statistics from [WhoScored](https://www.whoscored.com) for all 48 teams in the 2026 FIFA World Cup. 
Also includes a historical international match results dataset (1872–2026).

## Repository Structure

```
team_urls.json                         # WhoScored URLs for all 48 national teams
scrape_players.py                      # Scrape player rosters → whoscored_wc_players.csv
scrape_player_stats.py                 # Scrape summary stats → whoscored_player_stats.csv
scrape_player_stats_defensive.py       # Scrape defensive stats
scrape_player_stats_offensive.py       # Scrape offensive stats
scrape_player_stats_passing.py         # Scrape passing stats
results.csv                            # 49,000+ historical international match results
```

## Setup

```bash
pip install pandas playwright playwright-stealth
playwright install chromium
```

A residential IP is recommended — datacenter/VPN IPs are more likely to trigger Cloudflare challenges.

## Usage

**1. Scrape player rosters:**

```bash
python scrape_players.py
```

**2. Scrape player statistics (run each separately):**

```bash
python scrape_player_stats.py
python scrape_player_stats_defensive.py
python scrape_player_stats_offensive.py
python scrape_player_stats_passing.py
```

## How It Works

The player roster scraper uses Playwright with stealth patches and tries three extraction strategies per team page: XHR interception, embedded `<script>` parsing, and DOM selector scraping — falling back gracefully through each. The stats scrapers query WhoScored's internal statistics API per player, flattening the nested JSON into tabular CSV rows (one row per player per tournament).

Anti-detection includes randomized delays, Cloudflare challenge detection, automatic cookie banner dismissal, and checkpoint saving for resumable runs.

## Output Data

**Player roster** — team, player name, WhoScored profile link, player ID (~1,500 players across 48 teams).

**Statistics CSVs** — per-player, per-tournament rows covering: rating, apps, minutes played, goals, assists, shots, key passes, dribbles, tackles, interceptions, clearances, pass accuracy, crosses, through balls, and more.

**Historical results** — date, home/away teams, score, tournament, city, country, neutral venue flag.

## License

Intended for personal research and educational use. Please respect the terms of service of all data sources.
