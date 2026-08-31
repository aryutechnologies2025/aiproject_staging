"""
request_validator.py — Ingress HTTP Request Validation & Cloud Proxy Security.

Handles:
- Real client IP extraction across multi-tier reverse proxies (Cloudflare, AWS ALB, Nginx).
- Header validation and CRLF injection defense.
- Blacklist and whitelist IP filtering.
"""

import logging
import re
from typing import Optional, Set, Tuple
from fastapi import Request

logger = logging.getLogger("resume_builder.request_validator")


def _is_safe_header_value(value: str) -> bool:
    """Validate header string against CRLF injection and control characters."""
    if not isinstance(value, str) or len(value) > 2000:
        return False
    # Check for CRLF injection or null bytes
    if any(c in value for c in ('\n', '\r', '\0', '%00', '%0a', '%0d')):
        return False
    return True


class RequestValidator:
    """
    Validates ingress HTTP requests for security compliance without breaking cloud reverse proxies.
    """

    BLACKLISTED_IPS: Set[str] = set()
    WHITELISTED_IPS: Set[str] = set()

    @classmethod
    def validate_request(cls, request: Request) -> Tuple[bool, Optional[str]]:
        client_ip = cls.get_client_ip(request)

        if client_ip in cls.BLACKLISTED_IPS:
            logger.warning(f"[RequestValidator] Blocked blacklisted IP: {client_ip}")
            return False, "Your IP address has been blocked due to suspicious activity."

        if not cls._validate_headers(request):
            logger.warning(f"[RequestValidator] Invalid or malicious headers from {client_ip}")
            return False, "Invalid request headers."

        if request.method not in ["POST", "GET", "OPTIONS", "HEAD", "PUT", "PATCH", "DELETE"]:
            logger.warning(f"[RequestValidator] Disallowed HTTP method '{request.method}' from {client_ip}")
            return False, f"HTTP method '{request.method}' not allowed."

        return True, None

    @classmethod
    def get_client_ip(cls, request: Request) -> str:
        """
        Extracts real client IP resolving proxies in order: CF-Connecting-IP -> X-Forwarded-For -> X-Real-IP -> client.host.
        """
        # Cloudflare
        cf_ip = request.headers.get("CF-Connecting-IP")
        if cf_ip and _is_safe_header_value(cf_ip):
            return cf_ip.strip()

        # Standard Forwarded proxy list
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded and _is_safe_header_value(forwarded):
            return forwarded.split(",")[0].strip()

        # Nginx / ALB
        real_ip = request.headers.get("X-Real-IP")
        if real_ip and _is_safe_header_value(real_ip):
            return real_ip.strip()

        if request.client and request.client.host:
            return request.client.host

        return "127.0.0.1"

    @classmethod
    def _validate_headers(cls, request: Request) -> bool:
        # Check all present headers for CRLF injection
        for header_name, header_value in request.headers.items():
            if not _is_safe_header_value(header_name) or not _is_safe_header_value(header_value):
                return False

        # User-Agent length limit
        ua = request.headers.get("User-Agent", "")
        if len(ua) > 1000:
            return False

        return True

    @classmethod
    def blacklist_ip(cls, ip_address: str) -> None:
        cls.BLACKLISTED_IPS.add(ip_address.strip())
        logger.warning(f"[RequestValidator] Blacklisted IP: {ip_address}")

    @classmethod
    def whitelist_ip(cls, ip_address: str) -> None:
        cls.WHITELISTED_IPS.add(ip_address.strip())
        logger.info(f"[RequestValidator] Whitelisted IP: {ip_address}")

    @classmethod
    def clear_rules(cls) -> None:
        cls.BLACKLISTED_IPS.clear()
        cls.WHITELISTED_IPS.clear()
