"""
linkedin_router.py — All LinkedIn extraction endpoints.

Mount this router at: /api/v1/resume-builder

Endpoints:
  GET  /linkedin/auth-url              → Start OAuth flow
  GET  /linkedin/callback              → OAuth callback (LinkedIn redirects here)
  POST /linkedin/connect-session       → Provide li_at + trigger full scrape
  GET  /linkedin/status                → Poll scrape status
  GET  /linkedin/profile               → Get basic API profile
  GET  /linkedin/resume-data           → Get full resume-ready data
  POST /linkedin/extract-direct        → Direct extraction (li_at + URL, no OAuth)
  DELETE /linkedin/session             → Clean up session
  GET  /linkedin/health                → Health check
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field, validator

from app.modules.resume_builder.linkedin.service import linkedin_service
from app.modules.resume_builder.linkedin.session_manager import SessionStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/linkedin", tags=["LinkedIn"])


# ── Request / Response Models ──────────────────────────────────────────────────

class AuthUrlResponse(BaseModel):
    session_id: str
    auth_url:   str
    message:    str


class ConnectSessionRequest(BaseModel):
    session_id:  str  = Field(..., description="From /linkedin/auth-url")
    li_at:       str  = Field(..., description="LinkedIn li_at session cookie")
    profile_url: Optional[str] = Field(None, description="LinkedIn profile URL (optional if OAuth done)")

    @validator("li_at")
    def li_at_not_empty(cls, v):
        if not v or len(v.strip()) < 20:
            raise ValueError("li_at must be at least 20 characters")
        return v.strip()


class DirectExtractRequest(BaseModel):
    li_at:       str = Field(..., description="LinkedIn li_at session cookie")
    profile_url: str = Field(..., description="Full LinkedIn profile URL")


class StatusResponse(BaseModel):
    session_id:    str
    status:        str
    has_full_data: bool = False
    name:          Optional[str] = None
    email:         Optional[str] = None
    error:         Optional[str] = None
    warnings:      list          = []


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "/auth-url",
    response_model=AuthUrlResponse,
    summary="Step 1 — Get LinkedIn OAuth URL",
    description=(
        "Creates a new session and returns the LinkedIn OAuth redirect URL. "
        "Frontend should redirect the user (or open a popup) to `auth_url`. "
        "After the user authorizes, LinkedIn redirects to /linkedin/callback automatically."
    ),
)
async def get_auth_url() -> AuthUrlResponse:
    result = linkedin_service.get_auth_url()
    return AuthUrlResponse(**result)


@router.get(
    "/callback",
    summary="Step 2 — OAuth callback (LinkedIn redirects here)",
    description=(
        "LinkedIn calls this after the user authorizes. "
        "Exchanges the code for an access token and fetches basic profile. "
        "Redirects to FRONTEND_REDIRECT_URL with ?session_id=xxx&status=authorized"
    ),
    include_in_schema=True,
)
async def oauth_callback(
    code:  str = Query(..., description="Authorization code from LinkedIn"),
    state: str = Query(..., description="State token for CSRF protection"),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
) -> Any:
    import os

    frontend_redirect = os.getenv("LINKEDIN_FRONTEND_REDIRECT", "http://localhost:3000/linkedin-callback")

    # Handle user denial
    if error:
        logger.warning(f"[Callback] LinkedIn auth denied: {error} — {error_description}")
        return RedirectResponse(
            url=f"{frontend_redirect}?error={error}&error_description={error_description or ''}",
            status_code=302,
        )

    result = await linkedin_service.handle_oauth_callback(code=code, state=state)

    if not result.get("success"):
        error_msg = result.get("error", "unknown")
        return RedirectResponse(
            url=f"{frontend_redirect}?error=callback_failed&detail={error_msg}",
            status_code=302,
        )

    session_id = result["session_id"]
    basic      = result.get("basic_profile", {})
    name       = basic.get("full_name", "")

    logger.info(f"[Callback] OAuth success — session={session_id[:12]}… user={name}")

    # Redirect frontend with session_id — frontend stores it and proceeds
    return RedirectResponse(
        url=f"{frontend_redirect}?session_id={session_id}&status=authorized&name={name}",
        status_code=302,
    )


@router.post(
    "/connect-session",
    summary="Step 3 — Provide li_at cookie to start full profile scrape",
    description=(
        "After OAuth, the user provides their li_at session cookie. "
        "This triggers a background Playwright scrape of their full profile. "
        "Returns immediately — poll /linkedin/status to track progress.\n\n"
        "**How to get li_at:** See the /linkedin/health endpoint for instructions, "
        "or refer to FRONTEND_INTEGRATION.md."
    ),
)
async def connect_session(body: ConnectSessionRequest) -> Dict[str, Any]:
    return await linkedin_service.connect_session(
        session_id  = body.session_id,
        li_at       = body.li_at,
        profile_url = body.profile_url,
    )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Poll session / scrape status",
    description=(
        "Poll this endpoint every 2–3 seconds while status == 'scraping'. "
        "Transitions: pending → authorized → scraping → completed | failed"
    ),
)
async def get_status(
    session_id: str = Query(..., description="Session ID from /linkedin/auth-url"),
) -> StatusResponse:
    result = linkedin_service.get_status(session_id)
    return StatusResponse(**result)


@router.get(
    "/profile",
    summary="Get basic API profile (no scraping required)",
    description=(
        "Returns the basic LinkedIn profile from the OAuth API. "
        "Available immediately after OAuth (status=authorized). "
        "Contains: name, email, headline, picture, profile_url."
    ),
)
async def get_api_profile(
    session_id: str = Query(...),
) -> Dict[str, Any]:
    return await linkedin_service.get_api_profile(session_id)


@router.get(
    "/resume-data",
    summary="Step 4 — Get full resume-ready profile data",
    description=(
        "Returns the complete structured LinkedIn profile ready for resume generation. "
        "Only available after status == 'completed'. "
        "Response includes: name, email, phone, location, summary, experience[], "
        "education[], skills[], certifications[], languages[], awards[], projects[]."
    ),
)
async def get_resume_data(
    session_id: str = Query(...),
) -> Dict[str, Any]:
    result = linkedin_service.get_resume_data(session_id)
    if not result.get("success"):
        status = result.get("status", "")
        if status == SessionStatus.SCRAPING:
            raise HTTPException(
                status_code=202,
                detail={"message": "Still scraping, please wait", "status": status},
            )
        raise HTTPException(status_code=400, detail=result.get("error", "Not ready"))
    return result


@router.post(
    "/extract-direct",
    summary="Direct extraction — li_at + profile URL, no OAuth",
    description=(
        "Developer / internal endpoint. "
        "Provide li_at cookie + profile URL directly to get full data immediately. "
        "This is a synchronous wait — response comes back when scrape completes (~10–30 s)."
    ),
)
async def extract_direct(body: DirectExtractRequest) -> Dict[str, Any]:
    result = await linkedin_service.extract_with_cookie(
        li_at       = body.li_at,
        profile_url = body.profile_url,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Extraction failed"))
    return result


@router.delete(
    "/session",
    summary="Delete a session",
    description="Clean up a session. Call this after you've saved the resume data.",
)
async def delete_session(
    session_id: str = Query(...),
) -> Dict[str, str]:
    from app.modules.resume_builder.linkedin.session_manager import session_manager
    session_manager.delete(session_id)
    return {"message": "Session deleted", "session_id": session_id}


@router.get(
    "/health",
    summary="Health check + li_at instructions",
    include_in_schema=True,
)
async def health() -> Dict[str, Any]:
    from app.modules.resume_builder.linkedin.session_manager import session_manager
    return {
        "status":          "ok",
        "active_sessions": session_manager.active_count(),
        "how_to_get_li_at": {
            "step_1": "Log into LinkedIn in Chrome (https://www.linkedin.com)",
            "step_2": "Open DevTools: press F12 (Windows/Linux) or Cmd+Option+I (Mac)",
            "step_3": "Go to: Application → Cookies → https://www.linkedin.com",
            "step_4": "Find the cookie named  li_at",
            "step_5": "Copy its Value and send it to POST /linkedin/connect-session",
            "note":   (
                "The li_at cookie is valid for ~1 year. "
                "Your app should store it server-side per user (encrypted) for re-use."
            ),
        },
        "oauth_env_vars": {
            "LINKEDIN_CLIENT_ID":          "Required — LinkedIn app client ID",
            "LINKEDIN_CLIENT_SECRET":      "Required — LinkedIn app client secret",
            "LINKEDIN_REDIRECT_URI":       "Required — must match LinkedIn app settings",
            "LINKEDIN_FRONTEND_REDIRECT":  "Where to redirect after OAuth (your frontend URL)",
        },
    }