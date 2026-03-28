"""
cache.py — Smart profile cache with content-hash deduplication and TTL freshness.

Strategy:
  • File-based JSON cache (zero external dependencies for simple deploys)
  • Keyed by normalized LinkedIn profile URL
  • Freshness: configurable TTL (default 24 h)
  • Content-hash: SHA-256 of serialized profile → detects if profile actually changed
  • Thread-safe writes via temp-file + atomic rename
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Default cache directory — can be overridden via LINKEDIN_CACHE_DIR env var
DEFAULT_CACHE_DIR = Path(os.getenv("LINKEDIN_CACHE_DIR", "/tmp/linkedin_cache"))
DEFAULT_TTL_HOURS = int(os.getenv("LINKEDIN_CACHE_TTL_HOURS", "24"))


def _normalize_url(url: str) -> str:
    """Strip trailing slashes, query params, and lowercase the path."""
    url = url.strip().lower().rstrip("/")
    # Remove locale prefix e.g. /in/username?locale=en_US
    if "?" in url:
        url = url[: url.index("?")]
    return url


def _url_to_key(url: str) -> str:
    """Convert a LinkedIn URL to a safe filesystem key."""
    normalized = _normalize_url(url)
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def _compute_hash(data: Dict[str, Any]) -> str:
    """SHA-256 hash of serialized profile data (sorted keys for stability)."""
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


class ProfileCache:
    """
    Thread-safe, file-based LinkedIn profile cache.

    Cache entry format (JSON):
    {
        "url":          "https://linkedin.com/in/username",
        "profile_hash": "<sha256>",
        "cached_at":    <unix_timestamp_float>,
        "ttl_hours":    24,
        "data":         { ...LinkedInProfile dict... }
    }
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        default_ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Cache] Initialized at {self.cache_dir}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        url: str,
        ttl_hours: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Return (cached_profile_dict, age_hours) if fresh, else (None, age_hours).
        age_hours = 0.0 when no cache exists.
        """
        ttl = ttl_hours if ttl_hours is not None else self.default_ttl
        key = _url_to_key(url)
        path = self._path(key)

        if not path.exists():
            return None, 0.0

        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"[Cache] Corrupt entry for {url}: {exc}")
            path.unlink(missing_ok=True)
            return None, 0.0

        cached_at = entry.get("cached_at", 0)
        age_hours = (time.time() - cached_at) / 3600

        if age_hours > ttl:
            logger.debug(f"[Cache] STALE  {url}  ({age_hours:.1f}h > {ttl}h TTL)")
            return None, age_hours

        logger.info(f"[Cache] HIT    {url}  ({age_hours:.1f}h old)")
        return entry.get("data"), age_hours

    def put(self, url: str, data: Dict[str, Any], ttl_hours: Optional[int] = None) -> str:
        """
        Store profile data. Returns the content hash.
        Uses atomic write to avoid partial reads under concurrency.
        """
        ttl = ttl_hours if ttl_hours is not None else self.default_ttl
        key = _url_to_key(url)
        path = self._path(key)
        content_hash = _compute_hash(data)

        # Check if content actually changed
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("profile_hash") == content_hash:
                    logger.info(f"[Cache] UNCHANGED {url} — refreshing timestamp only")
                    old["cached_at"] = time.time()
                    old["ttl_hours"] = ttl
                    self._write_atomic(path, old)
                    return content_hash
            except Exception:
                pass

        entry = {
            "url":          url,
            "profile_hash": content_hash,
            "cached_at":    time.time(),
            "ttl_hours":    ttl,
            "data":         data,
        }
        self._write_atomic(path, entry)
        logger.info(f"[Cache] STORED {url}  hash={content_hash[:8]}…")
        return content_hash

    def invalidate(self, url: str) -> bool:
        """Delete the cache entry for a URL. Returns True if it existed."""
        key = _url_to_key(url)
        path = self._path(key)
        if path.exists():
            path.unlink()
            logger.info(f"[Cache] INVALIDATED {url}")
            return True
        return False

    def is_fresh(self, url: str, ttl_hours: Optional[int] = None) -> bool:
        """Quick check without returning data."""
        data, _ = self.get(url, ttl_hours)
        return data is not None

    def stats(self) -> Dict[str, Any]:
        """Return basic cache stats."""
        entries = list(self.cache_dir.glob("*.json"))
        total_size = sum(p.stat().st_size for p in entries)
        return {
            "entry_count": len(entries),
            "total_size_kb": round(total_size / 1024, 1),
            "cache_dir": str(self.cache_dir),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def _write_atomic(path: Path, data: Dict[str, Any]) -> None:
        """Write JSON atomically via temp file + rename."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp_", suffix=".json"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# Singleton instance — import and reuse across the app
profile_cache = ProfileCache()

