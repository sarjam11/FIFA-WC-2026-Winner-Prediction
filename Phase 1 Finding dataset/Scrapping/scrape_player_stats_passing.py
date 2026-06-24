
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
from playwright_stealth import Stealth


# Configuration


BASE_URL = "https://www.whoscored.com"

STATS_URL_TEMPLATE = (
    "https://www.whoscored.com/statisticsfeed/1/getplayerstatistics"
    "?category=summary"
    "&subcategory=passing"
    "&statsAccumulationType=0"
    "&isCurrent=true"
    "&playerId={player_id}"
    "&teamIds="
    "&matchId="
    "&stageId="
    "&tournamentOptions="
    "&sortBy=Rating"
    "&sortAscending="
    "&age="
    "&ageComparisonType="
    "&appearances="
    "&appearancesComparisonType="
    "&field=Overall"
    "&nationality="
    "&positionOptions="
    "&timeOfTheGameEnd="
    "&timeOfTheGameStart="
    "&isMinApp=false"
    "&page="
    "&includeZeroValues=true"
    "&numberOfPlayersToPick="
    "&incPens="
)

DEFAULT_INPUT  = "whoscored_wc_players.csv"
DEFAULT_OUTPUT = "whoscored_player_stats_passing]" \
".csv"
CHECKPOINT_SUFFIX = "_checkpoint.csv"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



# JSON Flattening


def flatten_json(obj: Any, parent_key: str = "", sep: str = "_") -> dict:
 
    items: dict = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(val, dict):
                items.update(flatten_json(val, new_key, sep))
            elif isinstance(val, list):
                if not val:
                    items[new_key] = ""
                elif isinstance(val[0], dict):
                    # List of dicts → store count, skip deep expansion
                    items[f"{new_key}_count"] = len(val)
                else:
                    # List of primitives → join as semicolon-separated string
                    items[new_key] = "; ".join(str(v) for v in val)
            else:
                items[new_key] = val
    else:
        items[parent_key] = obj
    return items


def extract_player_rows(raw_json: dict, player_id: int) -> list[dict]:

    rows: list[dict] = []

    # Case A: object with playerTableStats array 
    if isinstance(raw_json, dict) and "playerTableStats" in raw_json:
        stats_list = raw_json["playerTableStats"]
        if isinstance(stats_list, list):
            for entry in stats_list:
                flat = flatten_json(entry)
                flat["Player_ID"] = player_id
                rows.append(flat)
            return rows if rows else []

    # Case B: top-level array of stat objects 
    if isinstance(raw_json, list):
        for entry in raw_json:
            if isinstance(entry, dict):
                flat = flatten_json(entry)
                flat["Player_ID"] = player_id
                rows.append(flat)
        return rows if rows else []

    # Case C: single object (or unknown wrapper) 
    if isinstance(raw_json, dict):
        # Try common wrapper keys
        for key in ["data", "stats", "result", "response", "players"]:
            if key in raw_json and isinstance(raw_json[key], (list, dict)):
                return extract_player_rows(raw_json[key], player_id)

        # Flatten the whole thing as a single row
        flat = flatten_json(raw_json)
        flat["Player_ID"] = player_id
        rows.append(flat)

    return rows



# Helpers


def human_delay(lo: float, hi: float):
    """Sleep a random duration to mimic organic browsing"""
    d = random.uniform(lo, hi)
    time.sleep(d)
    return d


def handle_cloudflare(page, max_rounds: int = 12) -> bool:
    """Wait for Cloudflare/Imperva to clear.  Returns True if cleared"""
    for i in range(max_rounds):
        title = page.title().lower()
        try:
            snippet = page.inner_text("body")[:400].lower()
        except Exception:
            snippet = ""

        challenged = (
            any(kw in title for kw in
                ["just a moment", "attention required", "please wait"]) or
            any(kw in snippet for kw in
                ["checking your browser", "enable javascript", "ray id"])
        )
        if challenged:
            log.info("  ⏳ Cloudflare challenge (%d/%d)…", i + 1, max_rounds)
            human_delay(4, 7)
        else:
            return True
    return False


def dismiss_cookies(page):
    """Attempt to close cookie-consent banners."""
    for sel in [
        "#qc-cmp2-ui button[mode='primary']",
        ".qc-cmp2-summary-buttons button:first-child",
        "button:has-text('Accept')",
        "button:has-text('I Accept')",
        "button:has-text('Agree')",
        "button:has-text('AGREE')",
        "button:has-text('OK')",
        "[id*='accept']",
        "[class*='cookie'] button",
    ]:
        try:
            btn = page.query_selector(sel)
            if btn and btn.is_visible():
                btn.click()
                log.info("  🍪 Cookie banner dismissed (%s)", sel)
                human_delay(1, 2)
                return
        except Exception:
            continue



# Core: fetch stats for a single player


def fetch_player_stats(
    page,
    player_id: int,
    retries: int = 2,
) -> list[dict]:

    url = STATS_URL_TEMPLATE.format(player_id=player_id)

    for attempt in range(1, retries + 2):
        try:
            log.info("    Fetching stats (attempt %d)…", attempt)
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)

            # Wait a moment for the page to settle
            human_delay(1.5, 3.0)

            # Handle Cloudflare if it appears on this request
            if not handle_cloudflare(page, max_rounds=6):
                log.warning("    Cloudflare did not clear for player %d", player_id)
                if attempt <= retries:
                    human_delay(5, 10)
                    continue
                return []

         
            raw_text = ""

           
            pre = page.query_selector("pre")
            if pre:
                raw_text = pre.inner_text()
            else:
                raw_text = page.inner_text("body")

            raw_text = raw_text.strip()

            if not raw_text:
                log.warning("    Empty body for player %d", player_id)
                if attempt <= retries:
                    human_delay(3, 6)
                    continue
                return []

            if raw_text.startswith("<!") or raw_text.startswith("<html"):
                log.warning("    Got HTML instead of JSON (likely blocked)")
                if attempt <= retries:
                    human_delay(8, 15)
                    continue
                return []

            if "Access Denied" in raw_text or "Forbidden" in raw_text:
                log.warning("    Access denied for player %d", player_id)
                if attempt <= retries:
                    human_delay(10, 20)
                    continue
                return []

            # Parse JSON 
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as e:
                log.warning("    JSON parse error for player %d: %s", player_id, e)
                log.debug("    First 300 chars: %s", raw_text[:300])
                if attempt <= retries:
                    human_delay(3, 6)
                    continue
                return []

            # Extract & flatten
            rows = extract_player_rows(data, player_id)

            if rows:
                log.info("    ✅ Got %d stat row(s) for player %d", len(rows), player_id)
            else:
                log.info("    ⚠  JSON parsed but no stat rows found for %d", player_id)

            return rows

        except PwTimeout:
            log.warning("    Timeout on attempt %d for player %d", attempt, player_id)
            if attempt <= retries:
                human_delay(5, 10)
        except Exception as exc:
            log.error("    Error on attempt %d for player %d: %s", attempt, player_id, exc)
            if attempt <= retries:
                human_delay(5, 10)

    log.error("    ❌ All %d attempts failed for player %d", retries + 1, player_id)
    return []



# Orchestrator


def run(
    input_file: str,
    output_file: str,
    headless: bool,
    delay_min: float,
    delay_max: float,
):
    # Load player IDs 
    input_path = Path(input_file)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    df_input = pd.read_csv(input_path)

    if "Player_ID" not in df_input.columns:
        log.error("Column 'Player_ID' not found. Available columns: %s",
                  list(df_input.columns))
        sys.exit(1)

    # De-duplicate IDs and drop NaN
    player_ids = (
        df_input["Player_ID"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )
    log.info("Loaded %d unique Player IDs from %s", len(player_ids), input_path)

    # Also keep any extra columns (Team_Name, Player_Name) for merging later
    id_metadata = {}
    for col in ["Team_Name", "Player_Name"]:
        if col in df_input.columns:
            mapping = df_input.drop_duplicates(subset="Player_ID").set_index("Player_ID")[col].to_dict()
            id_metadata[col] = mapping

    # Launch Playwright 
    all_rows: list[dict] = []
    failed_ids: list[int] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
            java_script_enabled=True,
        )
        page = context.new_page()

        # Apply stealth anti-detection
        Stealth().apply_stealth_sync(page)
        log.info("✓ Stealth patches applied")

      
        log.info("=" * 60)
        log.info("WARM-UP: Visiting WhoScored homepage…")
        log.info("=" * 60)

        try:
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60_000)
            human_delay(4, 7)
        except PwTimeout:
            log.warning("Homepage timed out — continuing anyway")

        # Clear Cloudflare challenge
        cf_ok = handle_cloudflare(page)
        if not cf_ok:
            log.error("❌ Cloudflare challenge did not clear on homepage.")
            if not headless:
                log.info("   Waiting 90s for you to manually solve the CAPTCHA…")
                for _ in range(18):
                    human_delay(5, 5)
                    if "just a moment" not in page.title().lower():
                        log.info("   ✓ Cloudflare cleared!")
                        break
            else:
                log.error("   Try running with --headed so you can solve CAPTCHAs.")

        dismiss_cookies(page)
        human_delay(2, 4)

        homepage_title = page.title()
        log.info("Homepage title: %s", homepage_title)

        # Iterate through every Player ID 
        total = len(player_ids)
        log.info("=" * 60)
        log.info("Starting stats scrape for %d players", total)
        log.info("Delay between requests: %.1f–%.1f s", delay_min, delay_max)
        log.info("=" * 60)

        for idx, pid in enumerate(player_ids, 1):
            # Build metadata string for logging
            pname = id_metadata.get("Player_Name", {}).get(pid, "")
            tname = id_metadata.get("Team_Name", {}).get(pid, "")
            label = f"{pname} ({tname})" if pname else str(pid)

            log.info("━" * 60)
            log.info("[%d/%d]  Player ID %d — %s", idx, total, pid, label)
            log.info("━" * 60)

            rows = fetch_player_stats(page, pid)

            if rows:
                # Attach metadata from the input CSV
                for row in rows:
                    for col, mapping in id_metadata.items():
                        if col not in row:
                            row[col] = mapping.get(pid, "")
                all_rows.extend(rows)
            else:
                failed_ids.append(pid)

            # Polite delay 
            if idx < total:
                d = human_delay(delay_min, delay_max)
                log.info("  💤 Sleeping %.1f s…", d)

            # Checkpoint save every 10 players 
            if idx % 10 == 0 and all_rows:
                cp_path = Path(output_file).stem + CHECKPOINT_SUFFIX
                pd.DataFrame(all_rows).to_csv(cp_path, index=False)
                log.info("  💾 Checkpoint: %d rows → %s", len(all_rows), cp_path)

            #  Periodic warm-up refresh every 50 players 
            # Re-visit the homepage to refresh cookies and avoid session-based rate limits
            if idx % 50 == 0 and idx < total:
                log.info("  🔄 Refreshing session (revisiting homepage)…")
                try:
                    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
                    human_delay(4, 7)
                    handle_cloudflare(page, max_rounds=6)
                except Exception:
                    pass

        browser.close()

    # Build final DataFrame
    if not all_rows:
        log.error("❌ No data collected.  Output CSV will be empty.")
        log.error("   Run with --headed to debug.  Common causes:")
        log.error("     • Cloudflare blocking (check if homepage loads)")
        log.error("     • JSON endpoint returns HTML (403 page)")
        log.error("     • IP is rate-limited (increase --delay-min)")
        pd.DataFrame(columns=["Player_ID"]).to_csv(output_file, index=False)
        return

    df = pd.DataFrame(all_rows)

    # Move Player_ID, Team_Name, Player_Name to the front
    priority_cols = ["Player_ID", "Team_Name", "Player_Name"]
    front = [c for c in priority_cols if c in df.columns]
    rest  = [c for c in df.columns if c not in front]
    df = df[front + rest]

    # Ensure Player_ID is integer
    df["Player_ID"] = pd.to_numeric(df["Player_ID"], errors="coerce").astype("Int64")

    # Summary 
    log.info("=" * 60)
    log.info("SCRAPING COMPLETE")
    log.info("  Players attempted : %d", total)
    log.info("  Players succeeded : %d", total - len(failed_ids))
    log.info("  Players failed    : %d", len(failed_ids))
    log.info("  Total stat rows   : %d", len(df))
    log.info("  Columns           : %d", len(df.columns))
    log.info("=" * 60)

    if failed_ids:
        log.warning("  Failed IDs: %s", failed_ids[:50])
        # Save failed IDs for easy re-run
        failed_path = Path(output_file).stem + "_failed_ids.json"
        Path(failed_path).write_text(json.dumps(failed_ids), encoding="utf-8")
        log.info("  Failed IDs saved → %s", failed_path)

    # Export
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    log.info("Saved → %s", Path(output_file).resolve())

    # Preview
    print("\n" + "─" * 80)
    print("COLUMN LIST:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col}")
    print("─" * 80)
    print(df.head(5).to_string(index=False))
    print("─" * 80 + "\n")



# CLI


def main():
    parser = argparse.ArgumentParser(
        description="Scrape WhoScored player statistics via their JSON API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape_player_stats.py
  python scrape_player_stats.py --input my_players.csv --headed
  python scrape_player_stats.py --delay-min 5 --delay-max 12
  python scrape_player_stats.py --headless -o stats.csv
        """,
    )
    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_INPUT,
        help=f"CSV with a Player_ID column (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium without a visible window",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Chromium WITH a visible window (recommended for first run)",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=3.0,
        help="Minimum seconds between requests (default: 3)",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=7.0,
        help="Maximum seconds between requests (default: 7)",
    )
    args = parser.parse_args()

    # headed takes priority over headless
    headless = not args.headed if args.headed else args.headless

    run(
        input_file=args.input,
        output_file=args.output,
        headless=headless,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
    )


if __name__ == "__main__":
    main()
