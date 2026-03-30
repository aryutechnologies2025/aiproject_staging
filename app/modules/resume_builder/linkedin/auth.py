"""
auth.py — LinkedIn OAuth 2.0 client.

Handles the full redirect-based OAuth flow:
  1. generate_auth_url()  → send user to LinkedIn
  2. exchange_code()      → trade code for access token
  3. get_user_info()      → fetch basic profile from API
  4. get_profile_v2()     → fetch expanded profile fields

Why OAuth instead of Selenium window:
  A visible Chrome window on a server has no display — users can never see it.
  OAuth redirect runs entirely in the USER's browser, so no server display is needed.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
LINKEDIN_CLIENT_ID     = os.getenv("LINKEDIN_CLIENT_ID", "")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET", "")
LINKEDIN_REDIRECT_URI  = os.getenv(
    "LINKEDIN_REDIRECT_URI",
    "http://localhost:8000/api/v1/resume-builder/linkedin/callback",
)

# OpenID Connect scopes (LinkedIn sign-in)
# For partner API add: "r_basicprofile", "r_liteprofile"
SCOPES = ["openid", "profile", "email"]

AUTH_BASE  = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL  = "https://www.linkedin.com/oauth/v2/accessToken"
USERINFO   = "https://api.linkedin.com/v2/userinfo"
PROFILE_V2 = "https://api.linkedin.com/v2/me"
EMAIL_V2   = "https://api.linkedin.com/v2/emailAddress"


# ── Client ─────────────────────────────────────────────────────────────────

class LinkedInOAuthClient:
    """LinkedIn OAuth 2.0 + basic API client."""

    # ─ Auth URL ──────────────────────────────────────────────────────────────

    def generate_auth_url(self, state: str, scopes: Optional[list] = None) -> str:
        """
        Build the LinkedIn authorization URL.
        Frontend should redirect the user (or open a popup) to this URL.
        """
        params = {
            "response_type": "code",
            "client_id":     LINKEDIN_CLIENT_ID,
            "redirect_uri":  LINKEDIN_REDIRECT_URI,
            "state":         state,
            "scope":         " ".join(scopes or SCOPES),
        }
        url = f"{AUTH_BASE}?{urlencode(params)}"
        logger.info(f"[OAuth] Generated auth URL (state={state[:8]}…)")
        return url

    # ─ Token exchange ─────────────────────────────────────────────────────────

    async def exchange_code(self, code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for an access token.
        Returns the full token response dict.
        """
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type":   "authorization_code",
                    "code":         code,
                    "redirect_uri": LINKEDIN_REDIRECT_URI,
                    "client_id":    LINKEDIN_CLIENT_ID,
                    "client_secret": LINKEDIN_CLIENT_SECRET,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info("[OAuth] Token exchange successful")
            return data

    # ─ User info (OpenID Connect) ─────────────────────────────────────────────

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Standard OpenID Connect userinfo.
        Returns: sub, name, given_name, family_name, picture, email, locale
        """
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                USERINFO,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"[OAuth] userinfo failed: {resp.status_code} {resp.text[:200]}")
            return {}

    # ─ Profile API v2 (r_liteprofile) ────────────────────────────────────────

    async def get_profile_v2(self, access_token: str) -> Dict[str, Any]:
        """
        LinkedIn API v2 basic profile + email.
        NOTE: Only available if app has r_liteprofile / r_emailaddress permissions.
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        result: Dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=10) as client:
            # Profile fields
            p_resp = await client.get(
                PROFILE_V2,
                headers=headers,
                params={
                    "projection": (
                        "(id,firstName,lastName,headline,"
                        "vanityName,profilePicture(displayImage~:playableStreams))"
                    )
                },
            )
            if p_resp.status_code == 200:
                result["v2_profile"] = p_resp.json()

            # Email
            e_resp = await client.get(
                EMAIL_V2,
                headers=headers,
                params={"q": "members", "projection": "(elements*(handle~))"},
            )
            if e_resp.status_code == 200:
                elements = e_resp.json().get("elements", [])
                if elements:
                    result["email"] = (
                        elements[0].get("handle~", {}).get("emailAddress", "")
                    )

        return result

    # ─ Combined profile fetch ─────────────────────────────────────────────────

    async def get_full_api_profile(self, access_token: str) -> Dict[str, Any]:
        """
        Merge data from userinfo + v2 profile into a clean dict.
        This is what we get WITHOUT scraping (name, email, headline, picture only).
        """
        userinfo  = await self.get_user_info(access_token)
        v2        = await self.get_profile_v2(access_token)

        v2_profile = v2.get("v2_profile", {})

        # Best-effort name
        first = userinfo.get("given_name", "")
        last  = userinfo.get("family_name", "")
        if not first and "firstName" in v2_profile:
            loc_map = v2_profile["firstName"].get("localized", {})
            first = next(iter(loc_map.values()), "")
        if not last and "lastName" in v2_profile:
            loc_map = v2_profile["lastName"].get("localized", {})
            last = next(iter(loc_map.values()), "")

        headline_map = v2_profile.get("headline", {}).get("localized", {})
        headline = next(iter(headline_map.values()), userinfo.get("headline", ""))

        vanity = v2_profile.get("vanityName", "")
        profile_url = (
            f"https://www.linkedin.com/in/{vanity}" if vanity else ""
        )

        return {
            "id":          userinfo.get("sub") or v2_profile.get("id", ""),
            "first_name":  first,
            "last_name":   last,
            "full_name":   userinfo.get("name") or f"{first} {last}".strip(),
            "email":       userinfo.get("email") or v2.get("email", ""),
            "headline":    headline,
            "picture":     userinfo.get("picture", ""),
            "profile_url": profile_url,
            "vanity_name": vanity,
        }


# Singleton
linkedin_oauth = LinkedInOAuthClient()