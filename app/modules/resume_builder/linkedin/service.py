"""
service.py — Core LinkedIn scraping service.

Extraction flow:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  1. Check cache → return immediately if fresh                        │
  │  2. Check saved cookies → try silent background scrape               │
  │  3. If no cookies / session expired:                                 │
  │     a. Show clear consent popup (frontend calls /linkedin/consent)   │
  │     b. Open visible Chrome window at linkedin.com/login              │
  │     c. Poll for successful login (li_at cookie appears)              │
  │     d. Save cookies → close login window                             │
  │     e. Open headless window → scrape profile silently                │
  │  4. Parse HTML → structured JSON → cache → return                    │
  └──────────────────────────────────────────────────────────────────────┘

Anti-detection measures:
  • undetected_chromedriver (no webdriver flag)
  • Human-like scroll + random delays
  • "Chrome is being controlled" banner suppressed
  • Real user-agent string
  • All stealth JS patches applied before first page load
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cache import ProfileCache, profile_cache
from .parser import parse_profile_html
from .save_cookies import (
    cookies_exist, delete_cookies, is_logged_in,
    load_cookies, save_cookies,
)
from .schemas import (
    ExtractionMeta, ExtractionRequest, ExtractionStatus,
    LinkedInProfile, LinkedInResponse,
)
from .utils import (
    create_driver, human_sleep, is_valid_linkedin_url,
    sanitize_url, scroll_slowly,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
LOGIN_URL          = "https://www.linkedin.com/login"
LINKEDIN_HOME      = "https://www.linkedin.com/feed/"
LOGIN_TIMEOUT_S    = 120   # How long to wait for user to log in (2 min)
LOGIN_POLL_S       = 2     # Check for login every N seconds
SCRAPE_TIMEOUT_S   = 30    # Page load + scroll time


# ─────────────────────────── Login Window Manager ─────────────────────────────

class LoginWindowManager:
    """
    Opens a visible Chrome window for the user to log in.
    Polls for the LinkedIn session cookie then saves it and closes the window.
    """

    def __init__(
        self,
        cookie_path: Path,
        on_success_callback=None,
    ) -> None:
        self.cookie_path   = cookie_path
        self.on_success    = on_success_callback
        self._driver       = None
        self._logged_in    = False

    def open_login_window(self) -> bool:
        """
        Open a clean Chrome window at linkedin.com/login.
        Returns True when login detected, False on timeout.
        """
        logger.info("[Login] Opening LinkedIn login window for user")
        driver = create_driver(headless=False)  # VISIBLE window
        self._driver = driver

        try:
            driver.get(LOGIN_URL)
            logger.info("[Login] Waiting for user to log in…")

            deadline = time.time() + LOGIN_TIMEOUT_S
            while time.time() < deadline:
                time.sleep(LOGIN_POLL_S)
                if is_logged_in(driver):
                    logger.info("[Login] ✅ Login detected!")
                    # Navigate to feed to ensure all session cookies are set
                    driver.get(LINKEDIN_HOME)
                    time.sleep(2)
                    save_cookies(driver, self.cookie_path)
                    self._logged_in = True
                    if self.on_success:
                        self.on_success()
                    return True

            logger.warning("[Login] ⏱ Login timed out")
            return False

        except Exception as exc:
            logger.error(f"[Login] Window error: {exc}")
            return False

        finally:
            # Always close the login window — never leave it open
            try:
                driver.quit()
                logger.info("[Login] Login window closed")
            except Exception:
                pass
            self._driver = None

    @property
    def logged_in(self) -> bool:
        return self._logged_in


# ─────────────────────────── Profile Scraper ──────────────────────────────────

class LinkedInScraper:
    """
    Scrapes a LinkedIn profile page and returns the raw HTML.
    Requires valid session cookies.
    """

    def __init__(self, cookie_path: Path) -> None:
        self.cookie_path = cookie_path

    def scrape(self, profile_url: str) -> Tuple[Optional[str], List[str]]:
        """
        Returns (page_html, warnings).
        Runs in a background thread so FastAPI stays async.
        """
        warnings: List[str] = []
        driver = None

        try:
            # Start headless driver
            driver = create_driver(headless=True)

            # Load cookies (must visit domain first)
            driver.get("https://www.linkedin.com")
            human_sleep(1.5, 2.5)

            loaded = load_cookies(driver, self.cookie_path)
            if not loaded:
                warnings.append("No valid cookies — login required")
                return None, warnings

            # Navigate to the profile
            driver.get(profile_url)
            human_sleep(2.5, 4.0)

            # Check if we're still logged in
            if "authwall" in driver.current_url or "login" in driver.current_url:
                warnings.append("Session expired — login required")
                delete_cookies(self.cookie_path)
                return None, warnings

            # Scroll to trigger lazy-loaded sections
            scroll_slowly(driver, pause=0.5, steps=10)
            human_sleep(1.0, 2.0)

            # Expand "Show all" sections for complete data
            self._expand_sections(driver)
            human_sleep(0.5, 1.0)

            html = driver.page_source
            logger.info(f"[Scraper] ✅ Scraped {profile_url} ({len(html):,} bytes)")
            return html, warnings

        except Exception as exc:
            logger.error(f"[Scraper] Error scraping {profile_url}: {exc}")
            warnings.append(f"Scrape error: {exc}")
            return None, warnings

        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def _expand_sections(self, driver) -> None:
        """Click 'Show all X experiences/education' buttons to get full data."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        expand_selectors = [
            "button[aria-label*='show all']",
            "button.pvs-list__footer-actioned",
            "a.optional-action-btn-wrapper",
        ]
        for selector in expand_selectors:
            try:
                buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                for btn in buttons[:5]:  # Max 5 expands per selector
                    try:
                        driver.execute_script("arguments[0].click();", btn)
                        human_sleep(0.3, 0.8)
                    except Exception:
                        pass
            except Exception:
                pass


# ─────────────────────────── Main Service ─────────────────────────────────────

class LinkedInService:
    """
    Orchestrates the full extraction pipeline.
    FastAPI endpoints delegate to this class.
    """

    def __init__(
        self,
        cache: ProfileCache = profile_cache,
        cookie_path: Path = Path("/tmp/linkedin_cookies.pkl"),
    ) -> None:
        self.cache       = cache
        self.cookie_path = cookie_path
        self._executor   = ThreadPoolExecutor(max_workers=2)

    # ── Public API ────────────────────────────────────────────────────────────

    async def extract_profile(self, request: ExtractionRequest) -> LinkedInResponse:
        """
        Main entry point. Returns a LinkedInResponse.
        """
        warnings: List[str] = []

        # Validate URL
        if not request.linkedin_url:
            return self._error_response("LinkedIn URL is required", warnings)

        profile_url = sanitize_url(request.linkedin_url)
        if not is_valid_linkedin_url(profile_url):
            return self._error_response("Invalid LinkedIn profile URL", warnings)

        # ── Step 1: Cache check ───────────────────────────────────────────────
        if request.use_cache:
            cached_data, age_hours = self.cache.get(profile_url, request.cache_ttl_hours)
            if cached_data:
                try:
                    profile = LinkedInProfile(**cached_data)
                    return LinkedInResponse(
                        meta=ExtractionMeta(
                            status=ExtractionStatus.CACHED,
                            extracted_at=datetime.now(timezone.utc).isoformat(),
                            cache_hit=True,
                            cache_age_hours=round(age_hours, 2),
                            sections_found=list(cached_data.keys()),
                        ),
                        profile=profile,
                    )
                except Exception as exc:
                    logger.warning(f"[Service] Cache deserialize error: {exc}")
                    self.cache.invalidate(profile_url)

        # ── Step 2: Check if login is needed ─────────────────────────────────
        if not cookies_exist(self.cookie_path):
            return LinkedInResponse(
                meta=ExtractionMeta(
                    status=ExtractionStatus.LOGIN_NEEDED,
                    extracted_at=datetime.now(timezone.utc).isoformat(),
                    warnings=["Please log in to LinkedIn to extract your profile"],
                ),
            )

        # ── Step 3: Scrape in thread (blocking I/O) ───────────────────────────
        scraper = LinkedInScraper(self.cookie_path)
        html, scrape_warnings = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            scraper.scrape,
            profile_url,
        )
        warnings.extend(scrape_warnings)

        if html is None:
            status = (
                ExtractionStatus.LOGIN_NEEDED
                if any("login" in w.lower() or "session" in w.lower() for w in warnings)
                else ExtractionStatus.FAILED
            )
            return LinkedInResponse(
                meta=ExtractionMeta(
                    status=status,
                    extracted_at=datetime.now(timezone.utc).isoformat(),
                    warnings=warnings,
                ),
            )

        # ── Step 4: Parse HTML → structured JSON ─────────────────────────────
        try:
            profile, sections_found = parse_profile_html(html, profile_url=profile_url)
        except Exception as exc:
            logger.error(f"[Service] Parse error: {exc}")
            return self._error_response(f"Profile parsing failed: {exc}", warnings)

        # ── Step 5: Cache the result ──────────────────────────────────────────
        profile_dict = profile.dict()
        content_hash = self.cache.put(profile_url, profile_dict, request.cache_ttl_hours)

        status = ExtractionStatus.SUCCESS if sections_found else ExtractionStatus.PARTIAL
        if not sections_found:
            warnings.append("Profile parsed but no sections were detected — profile may be private or layout changed")

        return LinkedInResponse(
            meta=ExtractionMeta(
                status=status,
                extracted_at=datetime.now(timezone.utc).isoformat(),
                cache_hit=False,
                cache_age_hours=0.0,
                profile_hash=content_hash,
                sections_found=sections_found,
                warnings=warnings,
            ),
            profile=profile,
        )

    async def start_login_flow(self) -> Dict[str, Any]:
        """
        Trigger the visible login popup.
        Run in background thread since Selenium is synchronous.
        Returns status dict.
        """
        def _run_login():
            manager = LoginWindowManager(cookie_path=self.cookie_path)
            success = manager.open_login_window()
            return {"success": success, "logged_in": manager.logged_in}

        result = await asyncio.get_event_loop().run_in_executor(
            self._executor, _run_login
        )
        return result

    def has_active_session(self) -> bool:
        return cookies_exist(self.cookie_path)

    def logout(self) -> None:
        delete_cookies(self.cookie_path)
        logger.info("[Service] User logged out — cookies cleared")

    def invalidate_cache(self, url: str) -> bool:
        return self.cache.invalidate(sanitize_url(url))

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _error_response(message: str, warnings: List[str]) -> LinkedInResponse:
        return LinkedInResponse(
            meta=ExtractionMeta(
                status=ExtractionStatus.FAILED,
                extracted_at=datetime.now(timezone.utc).isoformat(),
                warnings=[message] + warnings,
            ),
        )


# Singleton
linkedin_service = LinkedInService()

