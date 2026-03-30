"""
scraper.py — Playwright async LinkedIn profile scraper.

Why Playwright instead of Selenium / undetected_chromedriver:
  • Full async support — no ThreadPoolExecutor hacks needed
  • Ships its own Chromium — no Chrome version mismatch
  • Built-in stealth patches that work reliably
  • Page.wait_for_selector / route intercepts are first-class

Setup (one-time):
  pip install playwright
  playwright install chromium --with-deps

Auth strategy:
  We inject the user's `li_at` session cookie. The user obtains this ONCE
  after logging into LinkedIn (via OAuth redirect or manually), and we store
  it server-side. See service.py for the full session flow.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Stealth init script injected before every page load
_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins',   {get: () => [1,2,3,4,5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.chrome = { runtime: {} };
const origQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (p) =>
  p.name === 'notifications'
    ? Promise.resolve({ state: Notification.permission })
    : origQuery(p);
"""


class PlaywrightLinkedInScraper:
    """
    Headless Playwright scraper for full LinkedIn profile extraction.

    Usage:
        scraper = PlaywrightLinkedInScraper()
        html, warnings = await scraper.scrape_profile(
            profile_url="https://www.linkedin.com/in/someone",
            li_at="AQE...",
        )
    """

    # ── Public API ────────────────────────────────────────────────────────────

    async def scrape_profile(
        self,
        profile_url: str,
        li_at: str,
        extra_cookies: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[str], List[str]]:
        """
        Scrape a LinkedIn profile page and return raw HTML.

        Args:
            profile_url:   Full LinkedIn profile URL
            li_at:         LinkedIn session cookie value (required for auth)
            extra_cookies: Optional additional cookies for better session fidelity

        Returns:
            (html | None, list_of_warnings)
        """
        warnings: List[str] = []

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            warnings.append(
                "playwright is not installed. Run: pip install playwright && playwright install chromium --with-deps"
            )
            return None, warnings

        if not li_at or len(li_at) < 10:
            warnings.append("li_at cookie is empty or too short — auth will fail")
            return None, warnings

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=self._chrome_args(),
            )
            context = await browser.new_context(
                user_agent=self._user_agent(),
                viewport={"width": 1280, "height": 900},
                locale="en-US",
                timezone_id="America/New_York",
                java_script_enabled=True,
                bypass_csp=False,
            )

            # Inject stealth patches before first page load
            await context.add_init_script(_STEALTH_JS)

            # Build cookie jar
            cookies = [
                {
                    "name":     "li_at",
                    "value":    li_at,
                    "domain":   ".linkedin.com",
                    "path":     "/",
                    "secure":   True,
                    "httpOnly": True,
                    "sameSite": "None",
                }
            ]
            if extra_cookies:
                cookies.extend(extra_cookies)

            await context.add_cookies(cookies)

            page = await context.new_page()

            try:
                # Navigate to profile
                response = await page.goto(
                    profile_url,
                    wait_until="domcontentloaded",
                    timeout=35_000,
                )

                # Human-like pause
                await asyncio.sleep(random.uniform(2.0, 3.5))

                # Check for auth wall
                current_url = page.url
                if any(kw in current_url for kw in ("authwall", "login", "checkpoint", "uas/authenticate")):
                    warnings.append(
                        "Session expired or li_at is invalid. Please reconnect your LinkedIn account."
                    )
                    return None, warnings

                if response and response.status >= 400:
                    warnings.append(f"LinkedIn returned HTTP {response.status}")
                    return None, warnings

                # Scroll to trigger lazy-loaded sections
                await self._scroll_slowly(page)

                # Click "Show all" / "See more" buttons
                await self._expand_sections(page)

                # Final settle
                await asyncio.sleep(1.0)

                html = await page.content()
                logger.info(f"[Scraper] ✅ Scraped {profile_url} ({len(html):,} bytes)")
                return html, warnings

            except asyncio.TimeoutError:
                warnings.append("Page timed out after 35 s — LinkedIn may be slow or blocking")
                return None, warnings
            except Exception as exc:
                logger.error(f"[Scraper] Error: {exc}", exc_info=True)
                warnings.append(f"Scrape error: {exc}")
                return None, warnings
            finally:
                await page.close()
                await context.close()
                await browser.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _scroll_slowly(self, page) -> None:
        """Gradually scroll to reveal lazy-loaded sections."""
        total = await page.evaluate("document.body.scrollHeight")
        steps = 12
        step_px = total // steps

        for i in range(steps):
            await page.evaluate(f"window.scrollTo(0, {step_px * (i + 1)})")
            await asyncio.sleep(random.uniform(0.25, 0.55))

        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.8)

    async def _expand_sections(self, page) -> None:
        """Click expand buttons so we get full content."""
        selectors = [
            "button[aria-label*='show all']",
            "button[aria-label*='Show all']",
            "button.pvs-list__footer-actioned",
            "a.optional-action-btn-wrapper",
            "button[aria-label*='See more']",
        ]
        for sel in selectors:
            try:
                buttons = await page.query_selector_all(sel)
                for btn in buttons[:6]:
                    try:
                        await btn.scroll_into_view_if_needed()
                        await btn.click(timeout=2_000)
                        await asyncio.sleep(random.uniform(0.3, 0.6))
                    except Exception:
                        pass
            except Exception:
                pass

    @staticmethod
    def _chrome_args() -> List[str]:
        return [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--disable-accelerated-2d-canvas",
            "--no-first-run",
            "--disable-gpu",
            "--disable-infobars",
            "--window-size=1280,900",
            "--lang=en-US",
        ]

    @staticmethod
    def _user_agent() -> str:
        agents = [
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        ]
        return random.choice(agents)


# Singleton
playwright_scraper = PlaywrightLinkedInScraper()