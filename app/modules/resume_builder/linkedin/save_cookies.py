"""
save_cookies.py — Secure JSON cookie persistence for LinkedIn sessions.

Flow:
  1. User logs in through the visible popup browser window
  2. We save session cookies to a secure JSON file
  3. On future requests, cookies are loaded → user doesn't need to log in again
  4. Cookies expire naturally (LinkedIn sessions ~1–2 years)

Security notes:
  • Replaced pickle with safe standard JSON serialization to prevent arbitrary code execution (RCE).
  • File permissions set to 600 (owner read/write only) on POSIX systems.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_COOKIE_PATH = Path(os.getenv("LINKEDIN_COOKIE_PATH", "/tmp/linkedin_cookies.json"))
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
    Persist WebDriver cookies to disk as secure JSON after successful login.
    Also saves a timestamp for age validation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cookie_data = {
        "saved_at": time.time(),
        "cookies": driver.get_cookies(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cookie_data, f, ensure_ascii=False)

    _secure_permissions(path)
    logger.info(f"[Cookies] Saved {len(cookie_data['cookies'])} cookies (JSON) → {path}")


def load_cookies(driver, path: Path = DEFAULT_COOKIE_PATH) -> bool:
    """
    Load cookies from JSON disk into WebDriver.

    Returns True on success, False if cookies don't exist or are expired.
    """
    path = Path(path)
    if not path.exists():
        # Check fallback .pkl for legacy compatibility
        legacy_path = path.with_suffix(".pkl")
        if not legacy_path.exists():
            logger.info("[Cookies] No saved cookies found")
            return False
        path = legacy_path

    try:
        with open(path, "r", encoding="utf-8") as f:
            cookie_data = json.load(f)
    except Exception as exc:
        logger.warning(f"[Cookies] Failed to load JSON cookies: {exc}")
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
            safe_cookie = {
                k: v for k, v in cookie.items()
                if k in ("name", "value", "domain", "path", "expiry", "secure", "httpOnly")
            }
            driver.add_cookie(safe_cookie)
            loaded += 1
        except Exception as exc:
            logger.debug(f"[Cookies] Skipped cookie {cookie.get('name')}: {exc}")

    logger.info(f"[Cookies] Loaded {loaded}/{len(cookies)} cookies from {path}")
    return loaded > 0


def delete_cookies(path: Path = DEFAULT_COOKIE_PATH) -> bool:
    """Delete saved cookies file on disk."""
    path = Path(path)
    deleted = False
    if path.exists():
        path.unlink(missing_ok=True)
        deleted = True
    legacy = path.with_suffix(".pkl")
    if legacy.exists():
        legacy.unlink(missing_ok=True)
        deleted = True
    return deleted


def cookies_exist(path: Path = DEFAULT_COOKIE_PATH) -> bool:
    """Check if valid cookies exist on disk."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return True
    legacy = path.with_suffix(".pkl")
    return legacy.exists() and legacy.stat().st_size > 0