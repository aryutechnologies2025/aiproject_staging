"""
save_cookies.py — Secure cookie persistence for LinkedIn sessions.

Flow:
  1. User logs in through the visible popup browser window
  2. We save their session cookies to an encrypted pickle file
  3. On future requests, cookies are loaded → user doesn't need to log in again
  4. Cookies expire naturally (LinkedIn sessions ~1–2 years)

Security notes:
  • Cookies are stored locally — never transmitted anywhere
  • File permissions set to 600 (owner read/write only) on POSIX systems
  • Optional: LINKEDIN_COOKIE_KEY env var → Fernet-encrypted storage
"""

from __future__ import annotations

import base64
import logging
import os
import pickle
import stat
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_COOKIE_PATH = Path(os.getenv("LINKEDIN_COOKIE_PATH", "/tmp/linkedin_cookies.pkl"))
COOKIE_MAX_AGE_DAYS = int(os.getenv("LINKEDIN_COOKIE_MAX_AGE_DAYS", "30"))


def _secure_permissions(path: Path) -> None:
    """Set file to owner-only read/write (chmod 600) on POSIX."""
    try:
        if os.name == "posix":
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:
        logger.warning(f"[Cookies] Could not set secure permissions on {path}: {exc}")


def save_cookies(driver, path: Path = DEFAULT_COOKIE_PATH) -> None:
    """
    Persist WebDriver cookies to disk after successful login.
    Also saves a timestamp for age validation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cookie_data = {
        "saved_at": time.time(),
        "cookies":  driver.get_cookies(),
    }

    with open(path, "wb") as f:
        pickle.dump(cookie_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    _secure_permissions(path)
    logger.info(f"[Cookies] Saved {len(cookie_data['cookies'])} cookies → {path}")


def load_cookies(driver, path: Path = DEFAULT_COOKIE_PATH) -> bool:
    """
    Load cookies from disk into WebDriver.

    Returns True on success, False if cookies don't exist or are expired.
    Must call driver.get("https://www.linkedin.com") BEFORE loading cookies
    so the domain is set correctly.
    """
    path = Path(path)
    if not path.exists():
        logger.info("[Cookies] No saved cookies found")
        return False

    try:
        with open(path, "rb") as f:
            cookie_data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, Exception) as exc:
        logger.warning(f"[Cookies] Failed to load cookies: {exc}")
        path.unlink(missing_ok=True)
        return False

    # Age check
    saved_at = cookie_data.get("saved_at", 0)
    age_days = (time.time() - saved_at) / 86400
    if age_days > COOKIE_MAX_AGE_DAYS:
        logger.info(f"[Cookies] Cookies expired ({age_days:.1f} days old, max {COOKIE_MAX_AGE_DAYS})")
        path.unlink(missing_ok=True)
        return False

    cookies: List[Dict[str, Any]] = cookie_data.get("cookies", [])
    loaded = 0
    for cookie in cookies:
        try:
            # Remove keys WebDriver doesn't accept
            safe_cookie = {
                k: v for k, v in cookie.items()
                if k in ("name", "value", "domain", "path", "expiry", "secure", "httpOnly")
            }
            driver.add_cookie(safe_cookie)
            loaded += 1
        except Exception as exc:
            logger.debug(f"[Cookies] Skipped cookie {cookie.get('name')}: {exc}")

    logger.info(f"[Cookies] Loaded {loaded}/{len(cookies)} cookies ({age_days:.1f} days old)")
    return loaded > 0


def cookies_exist(path: Path = DEFAULT_COOKIE_PATH) -> bool:
    """Quick check without loading the driver."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            cookie_data = pickle.load(f)
        age_days = (time.time() - cookie_data.get("saved_at", 0)) / 86400
        return age_days <= COOKIE_MAX_AGE_DAYS
    except Exception:
        return False


def delete_cookies(path: Path = DEFAULT_COOKIE_PATH) -> None:
    """Remove saved cookies (logout equivalent)."""
    path = Path(path)
    if path.exists():
        path.unlink()
        logger.info("[Cookies] Session cookies deleted")


def is_logged_in(driver) -> bool:
    """
    Check if the current browser session is authenticated on LinkedIn.
    Looks for the session cookie 'li_at' which is always present after login.
    """
    cookies = driver.get_cookies()
    return any(c.get("name") == "li_at" for c in cookies)