"""
session_manager.py — OAuth session + li_at cookie store.

For a single-worker dev deployment, an in-memory dict is fine.
For multi-worker production (Gunicorn / k8s), swap out the backend
for Redis (see RedisSessionManager below — just change the import).

Session lifecycle:
  PENDING   → session created, OAuth URL returned to frontend
  AUTHORIZED → OAuth callback received, access_token stored
  SCRAPING  → Playwright job running
  COMPLETED → full profile ready
  FAILED    → something went wrong
"""

from __future__ import annotations

import logging
import secrets
import time
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Status enum ────────────────────────────────────────────────────────────────
class SessionStatus:
    PENDING    = "pending"
    AUTHORIZED = "authorized"
    SCRAPING   = "scraping"
    COMPLETED  = "completed"
    FAILED     = "failed"


# ── In-memory backend ──────────────────────────────────────────────────────────

class InMemorySessionManager:
    """
    Thread-safe in-memory session store.

    Replace with RedisSessionManager for multi-process production.
    TTL default: 10 minutes (enough to complete OAuth + scrape).
    """

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock  = Lock()
        self.ttl    = ttl_seconds

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(self, state: Optional[str] = None) -> str:
        """Create a new session, returns session_id."""
        session_id = secrets.token_urlsafe(32)
        state      = state or secrets.token_urlsafe(24)

        with self._lock:
            self._store[session_id] = {
                "session_id":    session_id,
                "state":         state,
                "status":        SessionStatus.PENDING,
                "created_at":    time.time(),
                "updated_at":    time.time(),
                # OAuth
                "access_token":  None,
                "token_expiry":  None,
                # LinkedIn identity
                "linkedin_id":   None,
                "linkedin_url":  None,
                "basic_profile": None,
                # Scraping
                "li_at":         None,
                "full_profile":  None,
                # Errors
                "error":         None,
                "warnings":      [],
            }

        logger.info(f"[Session] Created {session_id[:12]}… state={state[:8]}…")
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        self._cleanup()
        with self._lock:
            return dict(self._store[session_id]) if session_id in self._store else None

    def get_by_state(self, state: str) -> Optional[Dict[str, Any]]:
        self._cleanup()
        with self._lock:
            for s in self._store.values():
                if s["state"] == state:
                    return dict(s)
        return None

    def update(self, session_id: str, **kwargs) -> bool:
        with self._lock:
            if session_id not in self._store:
                return False
            kwargs["updated_at"] = time.time()
            self._store[session_id].update(kwargs)
            return True

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    # ── Lifecycle helpers ──────────────────────────────────────────────────────

    def mark_authorized(
        self,
        session_id: str,
        access_token: str,
        basic_profile: Dict[str, Any],
        linkedin_url: str = "",
        token_expires_in: int = 3600,
    ) -> bool:
        return self.update(
            session_id,
            status        = SessionStatus.AUTHORIZED,
            access_token  = access_token,
            token_expiry  = time.time() + token_expires_in,
            basic_profile = basic_profile,
            linkedin_url  = linkedin_url,
            linkedin_id   = basic_profile.get("id", ""),
        )

    def mark_scraping(self, session_id: str) -> bool:
        return self.update(session_id, status=SessionStatus.SCRAPING)

    def mark_completed(
        self,
        session_id: str,
        full_profile: Dict[str, Any],
        warnings: Optional[List[str]] = None,
    ) -> bool:
        return self.update(
            session_id,
            status       = SessionStatus.COMPLETED,
            full_profile = full_profile,
            warnings     = warnings or [],
        )

    def mark_failed(self, session_id: str, error: str) -> bool:
        return self.update(
            session_id,
            status = SessionStatus.FAILED,
            error  = error,
        )

    def set_li_at(self, session_id: str, li_at: str) -> bool:
        return self.update(session_id, li_at=li_at)

    def add_warning(self, session_id: str, warning: str) -> None:
        with self._lock:
            if session_id in self._store:
                self._store[session_id]["warnings"].append(warning)

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Remove sessions older than TTL."""
        now = time.time()
        with self._lock:
            expired = [
                k for k, v in self._store.items()
                if now - v.get("created_at", 0) > self.ttl
            ]
            for k in expired:
                del self._store[k]
        if expired:
            logger.debug(f"[Session] Expired {len(expired)} sessions")

    def active_count(self) -> int:
        self._cleanup()
        return len(self._store)


# ── Optional Redis backend ─────────────────────────────────────────────────────

class RedisSessionManager:
    """
    Redis-backed session store for multi-worker production.

    Swap InMemorySessionManager → RedisSessionManager in the
    singleton assignment below.

    Requires: pip install redis
    Env:      REDIS_URL=redis://localhost:6379/0
    """

    def __init__(self, ttl_seconds: int = 600) -> None:
        import os
        import json
        import redis
        self.ttl = ttl_seconds
        self.r   = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self._json = json

    def _key(self, session_id: str) -> str:
        return f"linkedin_session:{session_id}"

    def _state_key(self, state: str) -> str:
        return f"linkedin_state:{state}"

    def create(self, state: Optional[str] = None) -> str:
        session_id = secrets.token_urlsafe(32)
        state      = state or secrets.token_urlsafe(24)
        data = {
            "session_id":    session_id,
            "state":         state,
            "status":        SessionStatus.PENDING,
            "created_at":    time.time(),
            "updated_at":    time.time(),
            "access_token":  None,
            "token_expiry":  None,
            "linkedin_id":   None,
            "linkedin_url":  None,
            "basic_profile": None,
            "li_at":         None,
            "full_profile":  None,
            "error":         None,
            "warnings":      [],
        }
        pipe = self.r.pipeline()
        pipe.setex(self._key(session_id), self.ttl, self._json.dumps(data))
        pipe.setex(self._state_key(state), self.ttl, session_id)
        pipe.execute()
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        raw = self.r.get(self._key(session_id))
        return self._json.loads(raw) if raw else None

    def get_by_state(self, state: str) -> Optional[Dict[str, Any]]:
        sid = self.r.get(self._state_key(state))
        if not sid:
            return None
        return self.get(sid.decode())

    def update(self, session_id: str, **kwargs) -> bool:
        raw = self.r.get(self._key(session_id))
        if not raw:
            return False
        data = self._json.loads(raw)
        kwargs["updated_at"] = time.time()
        data.update(kwargs)
        self.r.setex(self._key(session_id), self.ttl, self._json.dumps(data))
        return True

    def delete(self, session_id: str) -> None:
        self.r.delete(self._key(session_id))

    # Delegate lifecycle methods to parent-style calls
    def mark_authorized(self, session_id, access_token, basic_profile, linkedin_url="", token_expires_in=3600):
        return self.update(session_id, status=SessionStatus.AUTHORIZED, access_token=access_token,
                           token_expiry=time.time() + token_expires_in, basic_profile=basic_profile,
                           linkedin_url=linkedin_url, linkedin_id=basic_profile.get("id", ""))

    def mark_scraping(self, session_id):
        return self.update(session_id, status=SessionStatus.SCRAPING)

    def mark_completed(self, session_id, full_profile, warnings=None):
        return self.update(session_id, status=SessionStatus.COMPLETED, full_profile=full_profile, warnings=warnings or [])

    def mark_failed(self, session_id, error):
        return self.update(session_id, status=SessionStatus.FAILED, error=error)

    def set_li_at(self, session_id, li_at):
        return self.update(session_id, li_at=li_at)

    def add_warning(self, session_id, warning):
        raw = self.r.get(self._key(session_id))
        if raw:
            data = self._json.loads(raw)
            data.setdefault("warnings", []).append(warning)
            self.r.setex(self._key(session_id), self.ttl, self._json.dumps(data))


# ── Singleton ──────────────────────────────────────────────────────────────────
# Swap to RedisSessionManager for production multi-worker deployment
session_manager: InMemorySessionManager = InMemorySessionManager(ttl_seconds=600)

