"""
security_middleware.py — Production Security & Latency Middlewares for resume_builder.

Includes:
- SecurityHeadersMiddleware: Sets CSP, HSTS, X-Content-Type-Options, Frame protection.
- RequestValidationMiddleware: Ingress validation, real client IP extraction, request latency tracking.
- RateLimitMiddleware: Sliding-window rate limit enforcement with standard Retry-After headers.
"""

import logging
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.modules.resume_builder.request_validator import RequestValidator
from app.modules.resume_builder.security_manager import SecurityManager

logger = logging.getLogger("resume_builder.security_middleware")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Applies production security headers to all HTTP responses.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validates ingress HTTP requests and attaches processing latency headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = RequestValidator.get_client_ip(request)
        start_time = time.time()

        is_valid, error_message = RequestValidator.validate_request(request)
        if not is_valid:
            logger.warning(f"[SecurityMiddleware] Blocked request from {client_ip}: {error_message}")
            return JSONResponse(
                status_code=400,
                content={"error": error_message, "client_ip": client_ip},
            )

        response = await call_next(request)

        process_time_ms = round((time.time() - start_time) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(process_time_ms)

        logger.debug(f"{client_ip} {request.method} {request.url.path} -> {response.status_code} ({process_time_ms}ms)")
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Enforces timestamp sliding-window rate limiting with RFC Retry-After response headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        user_id = request.headers.get("X-User-ID", "anonymous")
        client_ip = RequestValidator.get_client_ip(request)

        is_allowed, error_message = SecurityManager.check_rate_limit(user_id, client_ip)
        if not is_allowed:
            logger.warning(f"[RateLimitMiddleware] Rate limit exceeded for user '{user_id}' from {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"error": error_message, "retry_after_seconds": 60},
                headers={"Retry-After": "60"},
            )

        return await call_next(request)