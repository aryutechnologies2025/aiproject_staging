"""
utils.py — Stealth browser setup, date parsing, text normalisation.

Key goals:
  • Bypass LinkedIn's bot detection (undetected-chromedriver + stealth patches)
  • Human-like delays so automation is invisible
  • Robust date parsing for any locale / format
  • Text normalisation helpers used across parser
"""

from __future__ import annotations

import logging
import platform
import random
import re
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────── Browser Fingerprint Patches ──────────────────────

# These JS snippets are injected before any page load.
# They remove Selenium artefacts that LinkedIn checks.
_STEALTH_SCRIPTS = [
    # 1. Remove webdriver flag
    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})",

    # 2. Fake plugin list (empty in headless Chrome → bot signal)
    """
    Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5],
    });
    """,

    # 3. Fake language list
    """
    Object.defineProperty(navigator, 'languages', {
        get: () => ['en-US', 'en'],
    });
    """,

    # 4. Fix Chrome runtime object (missing in automation)
    """
    window.chrome = { runtime: {} };
    """,

    # 5. Permissions API patch
    """
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
        Promise.resolve({ state: Notification.permission }) :
        originalQuery(parameters)
    );
    """,
]


def apply_stealth(driver) -> None:
    """Inject all stealth JS patches into an active WebDriver session."""
    for script in _STEALTH_SCRIPTS:
        try:
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": script},
            )
        except Exception as exc:
            logger.warning(f"[Stealth] Failed to inject script: {exc}")


def build_chrome_options(headless: bool = False):
    """
    Build ChromeOptions with stealth settings.

    headless=False  → Shows a real visible window (for user login popup)
    headless=True   → Silent background scraping after cookies are saved
    """
    try:
        import undetected_chromedriver as uc
        options = uc.ChromeOptions()
    except ImportError:
        from selenium.webdriver.chrome.options import Options
        options = Options()

    # ── Core stealth args ────────────────────────────────────────────────────
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")          # Hides "Chrome is being controlled" bar
    options.add_argument("--disable-extensions")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,800")

    # ── Realistic user-agent ─────────────────────────────────────────────────
    system = platform.system()
    if system == "Windows":
        ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    elif system == "Darwin":
        ua = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    else:
        ua = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    options.add_argument(f"--user-agent={ua}")

    # ── Locale & timezone for global users ──────────────────────────────────
    options.add_argument("--lang=en-US")

    if headless:
        options.add_argument("--headless=new")   # Chrome 112+ new headless

    return options


def create_driver(headless: bool = False, profile_dir: Optional[str] = None):
    import undetected_chromedriver as uc

    options = uc.ChromeOptions()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,800")
    options.add_argument("--lang=en-US")

    if profile_dir:
        options.add_argument(f"--user-data-dir={profile_dir}")

    driver = uc.Chrome(options=options)

    # Apply stealth patches
    apply_stealth(driver)

    driver.set_page_load_timeout(30)
    driver.implicitly_wait(5)

    logger.info("[Browser] Using undetected_chromedriver ✓")

    return driver


# ─────────────────────────── Human-like Delays ────────────────────────────────

def human_sleep(min_s: float = 1.0, max_s: float = 3.0) -> None:
    """Random sleep to mimic human reading/scrolling pace."""
    duration = random.uniform(min_s, max_s)
    time.sleep(duration)


def scroll_slowly(driver, pause: float = 0.4, steps: int = 8) -> None:
    """Scroll page gradually — mimics a human reading through the profile."""
    scroll_height = driver.execute_script("return document.body.scrollHeight")
    step_px = scroll_height // steps
    for i in range(steps):
        driver.execute_script(f"window.scrollTo(0, {step_px * (i + 1)});")
        time.sleep(pause + random.uniform(0, 0.3))
    # Scroll back to top
    driver.execute_script("window.scrollTo(0, 0);")


# ─────────────────────────── Date Parsing ─────────────────────────────────────

# Month name → int (English + common international aliases)
_MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    # Spanish
    "ene": 1, "abr": 4, "ago": 8, "dic": 12,
    # French
    "janv": 1, "févr": 2, "avr": 4, "mai": 5, "juin": 6,
    "juil": 7, "août": 8, "sept": 9, "oct.": 10, "nov.": 11, "déc": 12,
    # Portuguese
    "jan.": 1, "fev": 2, "abr.": 4, "jun.": 6, "ago.": 8, "set": 9,
    "out": 10, "dez": 12,
    # German
    "mär": 3, "mai": 5, "okt": 10,
    # Full English names
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}


def parse_month(token: str) -> Optional[int]:
    """Convert a month string token to int (1–12)."""
    clean = token.lower().strip(" .,")
    # Try direct map
    if clean in _MONTH_MAP:
        return _MONTH_MAP[clean]
    # Try first 3 chars
    if clean[:3] in _MONTH_MAP:
        return _MONTH_MAP[clean[:3]]
    # Numeric
    if clean.isdigit():
        val = int(clean)
        if 1 <= val <= 12:
            return val
    return None


def parse_year(token: str) -> Optional[int]:
    """Extract a 4-digit year from a token."""
    match = re.search(r"(\d{4})", token)
    if match:
        year = int(match.group(1))
        if 1950 <= year <= 2100:
            return year
    return None


def parse_date_range(text: str) -> Tuple[
    Optional[int], Optional[int], Optional[int], Optional[int], bool
]:
    """
    Parse a LinkedIn date range string into (start_month, start_year, end_month, end_year, is_current).

    Handles formats like:
      "Jan 2020 – Present"
      "March 2018 - Dec 2021"
      "2015 – 2019"
      "Jun 2023 – Present"
      "2020 - current"
    """
    if not text:
        return None, None, None, None, False

    text = text.strip()
    is_current = bool(re.search(r"\b(present|current|now|hoje|maintenant|jetzt|ahora)\b", text, re.I))

    # Split on dash/en-dash/em-dash
    parts = re.split(r"\s*[–—-]\s*", text, maxsplit=1)
    start_part = parts[0].strip()
    end_part   = parts[1].strip() if len(parts) > 1 else ""

    def _extract(part: str) -> Tuple[Optional[int], Optional[int]]:
        tokens = part.split()
        month, year = None, None
        for t in tokens:
            if year is None:
                year = parse_year(t)
            if month is None:
                month = parse_month(t)
        return month, year

    sm, sy = _extract(start_part)
    em, ey = _extract(end_part) if end_part and not re.search(r"present|current|now", end_part, re.I) else (None, None)

    return sm, sy, em, ey, is_current


# ─────────────────────────── Text Cleaning ────────────────────────────────────

def clean_text(text: Optional[str]) -> Optional[str]:
    """Strip whitespace, normalise unicode, remove zero-width chars."""
    if not text:
        return None
    # Remove zero-width chars
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def extract_linkedin_username(url: str) -> Optional[str]:
    """Extract the username slug from a LinkedIn profile URL."""
    match = re.search(r"linkedin\.com/in/([^/?#&]+)", url, re.I)
    return match.group(1) if match else None


def is_valid_linkedin_url(url: str) -> bool:
    """Basic validation of a LinkedIn profile URL."""
    return bool(re.match(r"https?://(www\.)?linkedin\.com/in/[^/?#&]+", url.strip(), re.I))


def sanitize_url(url: str) -> str:
    """Ensure the LinkedIn URL is properly formed."""
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    # Remove locale query params
    url = re.sub(r"\?.*$", "", url)
    url = url.rstrip("/")
    return url