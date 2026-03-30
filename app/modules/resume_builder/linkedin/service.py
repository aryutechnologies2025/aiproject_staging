"""
service.py — LinkedIn extraction service (production rewrite).

Flow summary
────────────
1.  Frontend calls  GET  /linkedin/auth-url
      → session created, OAuth URL returned

2.  User redirected to LinkedIn in THEIR browser (no server window needed)

3.  LinkedIn redirects back to  GET  /linkedin/callback?code=xxx&state=xxx
      → backend exchanges code for access_token
      → fetches basic profile from LinkedIn API
      → session status: AUTHORIZED

4.  Frontend receives session_id, now has two options:

    a) FULL SCRAPE (recommended):
         Frontend calls  POST /linkedin/connect-session
         with { session_id, li_at } — see FRONTEND_INTEGRATION.md
         for how the user obtains their li_at cookie
         → session status: SCRAPING → COMPLETED

    b) API-ONLY:
         Frontend calls  GET /linkedin/profile?session_id=xxx
         → returns basic profile (name, email, headline, picture)
         → good enough to pre-fill the resume builder form

5.  Frontend calls  GET /linkedin/resume-data?session_id=xxx
      → returns the full structured profile ready for resume generation
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .auth           import linkedin_oauth
from .parser         import parse_profile_html
from .scraper        import playwright_scraper
from .session_manager import SessionStatus, session_manager

logger = logging.getLogger(__name__)


class LinkedInService:
    """
    Orchestrates the full LinkedIn extraction pipeline.
    All public methods are async and safe to call from FastAPI routes.
    """

    # ── Step 1: Auth URL ───────────────────────────────────────────────────────

    def get_auth_url(self) -> Dict[str, str]:
        """
        Create a new session and return the LinkedIn OAuth URL.
        Frontend should redirect the user (or open a popup) to `auth_url`.
        """
        import secrets
        state      = secrets.token_urlsafe(24)
        session_id = session_manager.create(state=state)
        auth_url   = linkedin_oauth.generate_auth_url(state=state)

        logger.info(f"[Service] Auth URL generated — session={session_id[:12]}…")

        return {
            "session_id": session_id,
            "auth_url":   auth_url,
            "message":    "Redirect the user to auth_url. After authorization, poll /linkedin/status?session_id=...",
        }

    # ── Step 2: OAuth Callback ─────────────────────────────────────────────────

    async def handle_oauth_callback(
        self,
        code:  str,
        state: str,
    ) -> Dict[str, Any]:
        """
        Called by the OAuth callback endpoint.
        Exchanges auth code, fetches basic profile, updates session.
        Returns a dict that the callback endpoint can redirect with.
        """
        # Find session by state
        session = session_manager.get_by_state(state)
        if not session:
            logger.error(f"[Service] No session found for state={state[:8]}…")
            return {"success": False, "error": "Invalid or expired state"}

        session_id = session["session_id"]

        try:
            # Exchange code for token
            token_data    = await linkedin_oauth.exchange_code(code)
            access_token  = token_data.get("access_token", "")
            expires_in    = token_data.get("expires_in", 3600)

            if not access_token:
                session_manager.mark_failed(session_id, "Token exchange returned no access_token")
                return {"success": False, "error": "Token exchange failed"}

            # Fetch basic profile from LinkedIn API
            basic_profile = await linkedin_oauth.get_full_api_profile(access_token)
            linkedin_url  = basic_profile.get("profile_url", "")

            # Update session
            session_manager.mark_authorized(
                session_id,
                access_token   = access_token,
                basic_profile  = basic_profile,
                linkedin_url   = linkedin_url,
                token_expires_in = expires_in,
            )

            logger.info(
                f"[Service] OAuth success — session={session_id[:12]}… "
                f"user={basic_profile.get('full_name', '?')}"
            )

            return {
                "success":    True,
                "session_id": session_id,
                "basic_profile": basic_profile,
                "message":    (
                    "OAuth complete. To get full profile data (experience, education, skills), "
                    "call POST /linkedin/connect-session with your li_at cookie."
                ),
            }

        except Exception as exc:
            logger.error(f"[Service] OAuth callback error: {exc}", exc_info=True)
            session_manager.mark_failed(session_id, str(exc))
            return {"success": False, "error": str(exc)}

    # ── Step 3a: Connect session with li_at cookie ─────────────────────────────

    async def connect_session(
        self,
        session_id:    str,
        li_at:         str,
        profile_url:   Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Accept the user's li_at cookie and trigger full profile scraping.

        The li_at cookie is the LinkedIn session cookie that enables
        authenticated scraping. Frontend obtains it after user logs in.
        See FRONTEND_INTEGRATION.md → "Getting li_at".

        Args:
            session_id:  From /linkedin/auth-url response
            li_at:       LinkedIn session cookie
            profile_url: LinkedIn profile URL (optional — auto-detected if omitted)
        """
        session = session_manager.get(session_id)
        if not session:
            return {"success": False, "error": "Session not found or expired"}

        if session["status"] not in (SessionStatus.AUTHORIZED, SessionStatus.PENDING):
            return {
                "success": False,
                "error":   f"Session is {session['status']} — cannot accept cookies in this state",
            }

        if not li_at or len(li_at) < 20:
            return {"success": False, "error": "li_at cookie is too short or empty"}

        # Store li_at
        session_manager.set_li_at(session_id, li_at)

        # Determine profile URL
        if not profile_url:
            profile_url = session.get("linkedin_url", "")
        if not profile_url:
            basic = session.get("basic_profile") or {}
            profile_url = basic.get("profile_url", "")

        if not profile_url:
            return {
                "success": False,
                "error":   "Could not determine LinkedIn profile URL. Pass profile_url in the request body.",
            }

        # Launch background scrape
        logger.info(f"[Service] Starting scrape — session={session_id[:12]}… url={profile_url}")
        session_manager.mark_scraping(session_id)

        # Fire and forget — client polls /linkedin/status
        asyncio.create_task(
            self._scrape_and_parse(session_id, profile_url, li_at)
        )

        return {
            "success":    True,
            "session_id": session_id,
            "status":     SessionStatus.SCRAPING,
            "message":    "Scraping started. Poll GET /linkedin/status?session_id=... for updates.",
        }

    # ── Step 3b: API-only profile (no scraping) ────────────────────────────────

    async def get_api_profile(self, session_id: str) -> Dict[str, Any]:
        """
        Return the basic LinkedIn API profile without scraping.
        Available immediately after OAuth callback.
        """
        session = session_manager.get(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        if session["status"] == SessionStatus.PENDING:
            return {"success": False, "error": "OAuth not yet completed for this session"}

        return {
            "success":       True,
            "session_id":    session_id,
            "status":        session["status"],
            "basic_profile": session.get("basic_profile"),
            "full_profile":  session.get("full_profile"),
            "warnings":      session.get("warnings", []),
        }

    # ── Step 4: Poll status ────────────────────────────────────────────────────

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """
        Returns current session status.
        Frontend should poll this every 2 s while status == 'scraping'.
        """
        session = session_manager.get(session_id)
        if not session:
            return {"status": "not_found", "error": "Session not found or expired"}

        resp: Dict[str, Any] = {
            "session_id":    session_id,
            "status":        session["status"],
            "has_full_data": session.get("full_profile") is not None,
            "warnings":      session.get("warnings", []),
        }

        if session.get("error"):
            resp["error"] = session["error"]

        if session.get("basic_profile"):
            bp = session["basic_profile"]
            resp["name"]  = bp.get("full_name", "")
            resp["email"] = bp.get("email", "")

        return resp

    # ── Step 5: Get resume-ready profile ──────────────────────────────────────

    def get_resume_data(self, session_id: str) -> Dict[str, Any]:
        """
        Returns the full structured profile formatted for resume generation.
        Only available after status == 'completed'.
        """
        session = session_manager.get(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        if session["status"] != SessionStatus.COMPLETED:
            return {
                "success": False,
                "status":  session["status"],
                "error":   f"Profile not ready yet (status: {session['status']})",
            }

        full   = session.get("full_profile") or {}
        basic  = session.get("basic_profile") or {}

        # Merge API data with scraped data
        resume_data = self._build_resume_payload(basic, full)

        return {
            "success":     True,
            "session_id":  session_id,
            "resume_data": resume_data,
            "warnings":    session.get("warnings", []),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Manual li_at endpoint (no OAuth needed) ────────────────────────────────

    async def extract_with_cookie(
        self,
        li_at:        str,
        profile_url:  str,
    ) -> Dict[str, Any]:
        """
        Direct extraction using only a li_at cookie + profile URL.
        No OAuth required — useful for internal/developer use.
        """
        if not li_at or len(li_at) < 20:
            return {"success": False, "error": "li_at cookie required"}
        if not profile_url:
            return {"success": False, "error": "profile_url required"}

        # Create a temporary session
        session_id = session_manager.create()
        session_manager.set_li_at(session_id, li_at)
        session_manager.update(session_id, linkedin_url=profile_url)
        session_manager.mark_scraping(session_id)

        try:
            html, warnings = await playwright_scraper.scrape_profile(
                profile_url=profile_url,
                li_at=li_at,
            )

            if html is None:
                error = warnings[0] if warnings else "Scrape returned no HTML"
                session_manager.mark_failed(session_id, error)
                return {"success": False, "error": error, "warnings": warnings}

            profile, sections_found = parse_profile_html(html, profile_url=profile_url)
            profile_dict = profile.dict()

            session_manager.mark_completed(session_id, profile_dict, warnings)

            resume_data = self._build_resume_payload({}, profile_dict)
            return {
                "success":       True,
                "session_id":    session_id,
                "resume_data":   resume_data,
                "sections_found": sections_found,
                "warnings":      warnings,
            }

        except Exception as exc:
            logger.error(f"[Service] Direct extraction error: {exc}", exc_info=True)
            session_manager.mark_failed(session_id, str(exc))
            return {"success": False, "error": str(exc)}

    # ── Internal: Background scrape ────────────────────────────────────────────

    async def _scrape_and_parse(
        self,
        session_id:  str,
        profile_url: str,
        li_at:       str,
    ) -> None:
        """Background task — scrapes and parses, then updates session."""
        try:
            html, warnings = await playwright_scraper.scrape_profile(
                profile_url=profile_url,
                li_at=li_at,
            )

            for w in warnings:
                session_manager.add_warning(session_id, w)

            if html is None:
                err = warnings[0] if warnings else "Scrape returned empty response"
                session_manager.mark_failed(session_id, err)
                return

            profile, sections_found = parse_profile_html(html, profile_url=profile_url)
            profile_dict = profile.dict()

            if not sections_found:
                session_manager.add_warning(
                    session_id,
                    "Profile parsed but no sections detected — profile may be private",
                )

            session_manager.mark_completed(session_id, profile_dict, warnings)
            logger.info(
                f"[Service] Scrape complete — session={session_id[:12]}… "
                f"sections={sections_found}"
            )

        except Exception as exc:
            logger.error(f"[Service] Background scrape error: {exc}", exc_info=True)
            session_manager.mark_failed(session_id, str(exc))

    # ── Internal: Build resume payload ────────────────────────────────────────

    @staticmethod
    def _build_resume_payload(
        basic:  Dict[str, Any],
        scraped: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge LinkedIn API (basic) and scraped (full) data into a single
        dict that matches the existing CV generation endpoints' format.
        """
        # Identity — prefer scraped, fall back to basic API
        full_name = scraped.get("full_name") or basic.get("full_name", "")
        first = scraped.get("first_name") or basic.get("first_name", "")
        last  = scraped.get("last_name")  or basic.get("last_name",  "")
        if not full_name:
            full_name = f"{first} {last}".strip()

        # Contact
        contact = scraped.get("contact") or {}
        location_obj = scraped.get("location") or {}
        location_str = ""
        if isinstance(location_obj, dict):
            parts = [
                location_obj.get("city"),
                location_obj.get("state"),
                location_obj.get("country"),
            ]
            location_str = ", ".join(p for p in parts if p)
        elif isinstance(location_obj, str):
            location_str = location_obj

        email = (
            contact.get("email")
            if isinstance(contact, dict) else ""
        ) or basic.get("email", "")

        phone = contact.get("phone", "") if isinstance(contact, dict) else ""

        # Summary
        summary = scraped.get("about") or scraped.get("summary") or scraped.get("headline", "")

        # Experience — convert scraped Experience objects to plain dicts
        raw_exp = scraped.get("experiences") or scraped.get("experience") or []
        experience = []
        for exp in raw_exp:
            if isinstance(exp, dict):
                company  = exp.get("company", "")
                title    = exp.get("title", "")
                dr       = exp.get("date_range") or {}
                duration = ""
                if isinstance(dr, dict):
                    sy = dr.get("start_year")
                    ey = dr.get("end_year")
                    is_current = dr.get("is_current", False)
                    if sy:
                        end_label = "Present" if is_current else str(ey or "")
                        duration = f"{sy} – {end_label}".strip(" –")
                loc_obj = exp.get("location") or {}
                loc_str = ""
                if isinstance(loc_obj, dict):
                    loc_parts = [loc_obj.get("city"), loc_obj.get("country")]
                    loc_str   = ", ".join(p for p in loc_parts if p)
                experience.append({
                    "title":    title,
                    "company":  company,
                    "duration": duration,
                    "location": loc_str,
                    "bullets":  exp.get("skills_used") or [],
                    "description": exp.get("description") or "",
                })

        # Education
        raw_edu = scraped.get("educations") or scraped.get("education") or []
        education = []
        for edu in raw_edu:
            if isinstance(edu, dict):
                dr   = edu.get("date_range") or {}
                year = str(dr.get("end_year") or "") if isinstance(dr, dict) else ""
                education.append({
                    "degree":  edu.get("degree")         or edu.get("field_of_study", ""),
                    "college": edu.get("institution", ""),
                    "year":    year,
                    "grade":   edu.get("grade", ""),
                })

        # Skills
        raw_skills = scraped.get("skills") or []
        skills = []
        for sk in raw_skills:
            if isinstance(sk, dict):
                skills.append(sk.get("name", ""))
            elif isinstance(sk, str):
                skills.append(sk)

        # Certifications
        raw_certs = scraped.get("certifications") or []
        certifications = []
        for cert in raw_certs:
            if isinstance(cert, dict):
                certifications.append(cert.get("name", ""))

        # Languages
        raw_langs = scraped.get("languages") or []
        languages = []
        for lang in raw_langs:
            if isinstance(lang, dict):
                name = lang.get("name", "")
                prof = lang.get("proficiency") or ""
                languages.append(f"{name} ({prof})" if prof else name)
            elif isinstance(lang, str):
                languages.append(lang)

        # Awards
        raw_awards = scraped.get("awards") or []
        awards = []
        for aw in raw_awards:
            if isinstance(aw, dict):
                awards.append(aw.get("title", ""))

        # Projects
        raw_projects = scraped.get("projects") or []
        projects = []
        for pr in raw_projects:
            if isinstance(pr, dict):
                projects.append({
                    "name":        pr.get("name", ""),
                    "description": pr.get("description", ""),
                })

        # Publications
        raw_pubs = scraped.get("publications") or []
        publications = []
        for pub in raw_pubs:
            if isinstance(pub, dict):
                publications.append(pub.get("title", ""))

        return {
            # Identity
            "name":           full_name,
            "first_name":     first,
            "last_name":      last,
            "email":          email,
            "phone":          phone,
            "location":       location_str,
            "profile_url":    scraped.get("profile_url") or basic.get("profile_url", ""),
            "picture":        scraped.get("profile_picture") or basic.get("picture", ""),
            # Content
            "summary":        summary,
            "headline":       scraped.get("headline") or basic.get("headline", ""),
            "experience":     experience,
            "education":      education,
            "skills":         [s for s in skills if s],
            "certifications": [c for c in certifications if c],
            "languages":      [l for l in languages if l],
            "awards":         [a for a in awards if a],
            "projects":       projects,
            "publications":   [p for p in publications if p],
            # Meta
            "connections":    scraped.get("connections"),
            "followers":      scraped.get("followers"),
            "open_to_work":   scraped.get("open_to_work", False),
        }


# Singleton
linkedin_service = LinkedInService()