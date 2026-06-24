
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
from playwright_stealth import Stealth


# Configuration


BASE_URL       = "https://www.whoscored.com"
DEFAULT_INPUT  = "team_urls.json"
DEFAULT_OUTPUT = "whoscored_wc_players.csv"
DEBUG_DIR      = Path("debug")

# CSS selectors (tried in order for DOM strategy)
PLAYER_SELECTORS = [
    "a.player-link",
    "a[class*='player-link']",
    "a[class*='player'][href*='/Players/']",
    "a[href*='/Players/'][href*='/Show/']",
    "td a[href*='/Players/']",
    "#player-table-statistics-body a[href*='/Players/']",
    "#top-player-stats a[href*='/Players/']",
    "table a[href*='/Players/']",
    "[class*='squad'] a[href*='/Players/']",
]

# Regex patterns
PLAYER_ID_RE   = re.compile(r"/Players/(\d+)/")
PLAYER_HREF_RE = re.compile(r"/Players/(\d+)/Show/([\w.-]+)")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



# Helpers


def human_delay(lo: float = 2.0, hi: float = 5.0):
    time.sleep(random.uniform(lo, hi))


def parse_player_id(href: str) -> Optional[int]:
    m = PLAYER_ID_RE.search(href)
    return int(m.group(1)) if m else None


def name_from_slug(slug: str) -> str:
    """Convert URL slug like 'Lionel-Messi' to 'Lionel Messi'."""
    return slug.replace("-", " ").strip()


def handle_cloudflare(page, max_wait: int = 10):
    """Wait for Cloudflare/Imperva challenge to clear."""
    for i in range(max_wait):
        title = page.title().lower()
        try:
            body_snip = page.inner_text("body")[:500].lower()
        except Exception:
            body_snip = ""

        is_challenge = (
            any(kw in title for kw in ["just a moment", "attention required",
                                        "checking", "please wait"]) or
            any(kw in body_snip for kw in ["checking your browser",
                                            "ray id", "enable javascript"])
        )
        if is_challenge:
            log.info("     Cloudflare challenge (%d/%d)…", i + 1, max_wait)
            human_delay(4, 7)
        else:
            return True
    log.warning("      Cloudflare did not clear after %d waits", max_wait)
    return False


def dismiss_cookies(page):
    """Try to dismiss cookie/consent banners."""
    selectors = [
        "#qc-cmp2-ui button[mode='primary']",
        ".qc-cmp2-summary-buttons button:first-child",
        "button:has-text('Accept')",
        "button:has-text('I Accept')",
        "button:has-text('Agree')",
        "button:has-text('AGREE')",
        "button:has-text('OK')",
        "button:has-text('Got it')",
        "[id*='accept']",
        "[class*='consent'] button",
        "[class*='cookie'] button:has-text('Accept')",
    ]
    for sel in selectors:
        try:
            btn = page.query_selector(sel)
            if btn and btn.is_visible():
                btn.click()
                log.info("     Dismissed cookie banner via %s", sel)
                human_delay(1, 2)
                return True
        except Exception:
            continue
    return False



# Strategy 1: Network interception


def extract_players_from_xhr(captured_responses: list[dict],
                              team_name: str) -> list[dict]:
 
    players = []
    seen_ids = set()

    for resp in captured_responses:
        body = resp.get("body", "")
        if not body:
            continue

        # Try parsing as JSON
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            # Not JSON — try extracting JSON-like structures
            continue

        # Recursively search for player data in the JSON structure
        _extract_from_json(data, team_name, players, seen_ids)

    return players


def _extract_from_json(obj, team_name, players, seen_ids, depth=0):
    
    if depth > 10:
        return

    if isinstance(obj, dict):
        # Check if this dict looks like a player entry
        player_id = (
            obj.get("playerId") or obj.get("PlayerId") or
            obj.get("player_id") or obj.get("id")
        )
        player_name = (
            obj.get("name") or obj.get("Name") or
            obj.get("playerName") or obj.get("PlayerName") or
            obj.get("fullName") or obj.get("knownName")
        )

        if player_id and player_name and isinstance(player_id, (int, str)):
            pid = int(player_id) if str(player_id).isdigit() else None
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                players.append({
                    "Team_Name": team_name,
                    "Player_Name": str(player_name),
                    "Profile_Link": f"{BASE_URL}/Players/{pid}/Show/{player_name.replace(' ', '-')}",
                    "Player_ID": pid,
                })

        # Recurse into values
        for v in obj.values():
            _extract_from_json(v, team_name, players, seen_ids, depth + 1)

    elif isinstance(obj, list):
        for item in obj:
            _extract_from_json(item, team_name, players, seen_ids, depth + 1)



# Strategy 2: Embedded <script> parsing


def extract_players_from_scripts(page, team_name: str) -> list[dict]:
    """
    WhoScored sometimes injects player data directly into <script> tags
    as JavaScript variables. We extract and parse that data.
    """
    players = []
    seen_ids = set()
    html = page.content()

    # Pattern 1: Find all /Players/{id}/Show/{name} patterns in the source
    matches = PLAYER_HREF_RE.findall(html)
    for pid_str, slug in matches:
        pid = int(pid_str)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        players.append({
            "Team_Name": team_name,
            "Player_Name": name_from_slug(slug),
            "Profile_Link": f"{BASE_URL}/Players/{pid}/Show/{slug}",
            "Player_ID": pid,
        })

    if players:
        log.info("     Script parsing found %d players via URL patterns", len(players))
        return players

    # Pattern 2: Look for JSON data structures with playerId
    # Try to find and parse JSON blobs in script tags
    script_contents = page.evaluate("""
        () => {
            const scripts = document.querySelectorAll('script:not([src])');
            const contents = [];
            for (const s of scripts) {
                const text = s.textContent || '';
                if (text.length > 100 && (
                    text.includes('player') || text.includes('Player') ||
                    text.includes('squad') || text.includes('Squad')
                )) {
                    contents.push(text.substring(0, 50000));
                }
            }
            return contents;
        }
    """)

    for content in script_contents:
        # Try to extract JSON objects/arrays from the script
        # Look for patterns like: var data = {...} or [{...}, ...]
        json_patterns = [
            r'(\[{".*?"playerId".*?}\])',
            r'(\{".*?"players".*?\})',
            r'var\s+\w+\s*=\s*(\[.*?\]);',
            r'var\s+\w+\s*=\s*(\{.*?\});',
        ]
        for pat in json_patterns:
            for m in re.finditer(pat, content, re.DOTALL):
                try:
                    data = json.loads(m.group(1))
                    _extract_from_json(data, team_name, players, seen_ids)
                except (json.JSONDecodeError, IndexError):
                    continue

    if players:
        log.info("     Script parsing found %d players via JSON extraction", len(players))

    return players



# Strategy 3: DOM selector scraping (with tab navigation)


def extract_players_from_dom(page, team_name: str) -> list[dict]:
   
    players = []
    seen_ids = set()

    def _try_extract():
        """Inner helper: try all selectors on the current DOM state."""
        found = []
        for sel in PLAYER_SELECTORS:
            try:
                elements = page.query_selector_all(sel)
                if not elements:
                    continue

                log.info("     Selector '%s' → %d elements", sel, len(elements))
                for el in elements:
                    href = el.get_attribute("href") or ""
                    text = (el.inner_text() or "").strip()

                    if "/Players/" not in href:
                        continue

                    pid = parse_player_id(href)
                    if pid is None or pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    # Build name: prefer text content, fall back to slug
                    name = text
                    if not name:
                        slug_match = re.search(r"/Show/([\w.-]+)", href)
                        name = name_from_slug(slug_match.group(1)) if slug_match else f"Unknown ({pid})"

                    full_link = href if href.startswith("http") else BASE_URL + href
                    found.append({
                        "Team_Name": team_name,
                        "Player_Name": name,
                        "Profile_Link": full_link,
                        "Player_ID": pid,
                    })

                if found:
                    return found
            except Exception:
                continue
        return found

    # First try: immediate DOM state
    players = _try_extract()
    if players:
        return players

    # Second try: click through sub-navigation tabs
    tab_texts = ["Squad", "Players", "Statistics", "Summary"]
    for tab_text in tab_texts:
        try:
            # Try multiple ways to find the tab
            tab = None
            for tab_sel in [
                f"a:has-text('{tab_text}')",
                f"li:has-text('{tab_text}') a",
                f"[class*='sub'] a:has-text('{tab_text}')",
                f"nav a:has-text('{tab_text}')",
            ]:
                tab = page.query_selector(tab_sel)
                if tab and tab.is_visible():
                    break
                tab = None

            if tab:
                log.info("      Clicking '%s' tab…", tab_text)
                tab.click()
                human_delay(3, 6)

                try:
                    page.wait_for_load_state("networkidle", timeout=15_000)
                except PwTimeout:
                    pass

                # Scroll to trigger lazy loading
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                human_delay(1, 2)
                page.evaluate("window.scrollTo(0, 0)")
                human_delay(1, 2)

                players = _try_extract()
                if players:
                    log.info("     Found players after clicking '%s'", tab_text)
                    return players
        except Exception:
            continue

    return players



# Per-team orchestrator


def scrape_team_players(
    page,
    team_name: str,
    team_url: str,
    retries: int = 2,
    save_debug: bool = False,
) -> list[dict]:
 
    players = []

    # Prepare to capture XHR responses during page load
    xhr_captures: list[dict] = []

    def on_response(response):
        if response.request.resource_type in ("xhr", "fetch"):
            try:
                body = response.text()
                if any(kw in body.lower() for kw in
                       ["player", "squad", "playerid", "playername"]):
                    xhr_captures.append({
                        "url": response.url,
                        "status": response.status,
                        "body": body,
                    })
            except Exception:
                pass

    for attempt in range(1, retries + 2):
        try:
            xhr_captures.clear()
            page.on("response", on_response)

            log.info("  Loading page (attempt %d)…  %s", attempt, team_url)

            page.goto(team_url, wait_until="domcontentloaded", timeout=60_000)
            human_delay(3, 5)

            # Handle Cloudflare
            handle_cloudflare(page)

            # Dismiss cookies (on first load)
            if attempt == 1:
                dismiss_cookies(page)

            # Wait for network to settle
            try:
                page.wait_for_load_state("networkidle", timeout=20_000)
            except PwTimeout:
                log.info("    Network didn't reach idle — continuing")

            human_delay(2, 4)

            # Scroll to trigger any lazy-loaded content
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            human_delay(1.5, 2.5)
            page.evaluate("window.scrollTo(0, 0)")
            human_delay(1, 2)

            # Save debug artifacts if requested
            if save_debug:
                safe = re.sub(r'[^\w]', '_', team_name)[:30]
                DEBUG_DIR.mkdir(exist_ok=True)
                page.screenshot(path=str(DEBUG_DIR / f"team_{safe}.png"), full_page=True)

            # Try all strategies in order 

            # Strategy 1: XHR data
            if xhr_captures:
                log.info("     Trying Strategy 1: XHR interception (%d captures)…",
                         len(xhr_captures))
                players = extract_players_from_xhr(xhr_captures, team_name)
                if players:
                    log.info("     Strategy 1 SUCCESS: %d players from XHR", len(players))
                    page.remove_listener("response", on_response)
                    return players

            # Strategy 2: Embedded scripts
            log.info("     Trying Strategy 2: embedded <script> parsing…")
            players = extract_players_from_scripts(page, team_name)
            if players:
                log.info("     Strategy 2 SUCCESS: %d players from scripts", len(players))
                page.remove_listener("response", on_response)
                return players

            # Strategy 3: DOM selectors
            log.info("    🔍 Trying Strategy 3: DOM selectors…")
            players = extract_players_from_dom(page, team_name)
            if players:
                log.info("     Strategy 3 SUCCESS: %d players from DOM", len(players))
                page.remove_listener("response", on_response)
                return players

            # No strategy worked
            log.warning("      No players found (attempt %d/%d)", attempt, retries + 1)
            page.remove_listener("response", on_response)

            if attempt <= retries:
                # On retry, save debug info
                DEBUG_DIR.mkdir(exist_ok=True)
                safe = re.sub(r'[^\w]', '_', team_name)[:30]
                page.screenshot(
                    path=str(DEBUG_DIR / f"failed_{safe}_attempt{attempt}.png"),
                    full_page=True,
                )
                human_delay(8, 15)

        except Exception as exc:
            log.error("     Error on attempt %d: %s", attempt, exc)
            try:
                page.remove_listener("response", on_response)
            except Exception:
                pass
            if attempt <= retries:
                human_delay(8, 15)
            else:
                log.error("    Giving up on %s", team_name)

    return players



# Main orchestrator


def run(
    input_file: str  = DEFAULT_INPUT,
    output_file: str = DEFAULT_OUTPUT,
    headless: bool   = False,
    debug: bool      = False,
):
    # Load team URLs 
    input_path = Path(input_file)
    if not input_path.exists():
        log.error("File not found: %s  — run scrape_team_urls.py first.", input_path)
        sys.exit(1)

    teams: dict[str, str] = json.loads(input_path.read_text(encoding="utf-8"))
    log.info("Loaded %d teams from %s", len(teams), input_path)

    # Launch browser 
    all_rows: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
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

        # Apply stealth
        Stealth().apply_stealth_sync(page)
        log.info(" Stealth patches applied")

        # Warm-up visit (establish cookies, pass Cloudflare) 
        log.info("Warm-up: visiting WhoScored homepage…")
        try:
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60_000)
            human_delay(4, 7)
        except PwTimeout:
            log.warning("Homepage timed out — continuing")

        handle_cloudflare(page)
        dismiss_cookies(page)
        human_delay(2, 3)

        # Verify we got past Cloudflare
        title = page.title()
        log.info("Homepage title: %s", title)
        if "just a moment" in title.lower():
            log.error(" Cannot bypass Cloudflare. Try:")
            log.error("   1. Run with --headed to see the challenge")
            log.error("   2. Use a residential IP (not VPN/datacenter)")
            log.error("   3. Manually solve the CAPTCHA in the browser window")
            log.error("   (The script will detect when you clear it)")
            if not headless:
                log.info("Waiting 60s for manual CAPTCHA solve…")
                for _ in range(12):
                    human_delay(5, 5)
                    if "just a moment" not in page.title().lower():
                        log.info("✓ Cloudflare cleared!")
                        break

        # Iterate teams
        total = len(teams)
        failed_teams = []

        for idx, (team_name, team_url) in enumerate(teams.items(), 1):
            log.info("━" * 60)
            log.info("[%d/%d]  %s", idx, total, team_name)
            log.info("━" * 60)

            players = scrape_team_players(
                page, team_name, team_url,
                save_debug=debug,
            )

            if players:
                all_rows.extend(players)
            else:
                failed_teams.append(team_name)

            # Polite delay between teams
            if idx < total:
                delay = random.uniform(6, 14)
                log.info("   Waiting %.1f s before next team…", delay)
                time.sleep(delay)

            # Checkpoint save every 5 teams
            if idx % 5 == 0 and all_rows:
                checkpoint = Path(output_file).stem + "_checkpoint.csv"
                pd.DataFrame(all_rows).to_csv(checkpoint, index=False)
                log.info("   Checkpoint: %d rows → %s", len(all_rows), checkpoint)

        browser.close()

    # Build final DataFrame 
    df = pd.DataFrame(
        all_rows,
        columns=["Team_Name", "Player_Name", "Profile_Link", "Player_ID"],
    )
    df["Player_ID"] = pd.to_numeric(df["Player_ID"], errors="coerce").astype("Int64")

    # Summary
    log.info("=" * 60)
    log.info("SCRAPING COMPLETE")
    log.info("  Total players : %d", len(df))
    log.info("  Unique teams  : %d", df["Team_Name"].nunique())
    if failed_teams:
        log.warning("  Failed teams  : %d → %s", len(failed_teams), failed_teams)
    log.info("=" * 60)

    # Export 
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    log.info("Saved → %s", Path(output_file).resolve())

    if not df.empty:
        print("\n", df.head(20).to_string(index=False), "\n")
    else:
        log.warning("DataFrame is empty. Run  python diagnose_whoscored.py --headed")
        log.warning("to see what the browser actually encounters.")

    return df



# CLI


def main():
    parser = argparse.ArgumentParser(
        description="Scrape WhoScored player data for World Cup teams.",
    )
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--headless", action="store_true",
                        help="Headless mode (default: headed)")
    parser.add_argument("--debug", action="store_true",
                        help="Save screenshots for each team to debug/")
    args = parser.parse_args()

    run(
        input_file=args.input,
        output_file=args.output,
        headless=args.headless,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
