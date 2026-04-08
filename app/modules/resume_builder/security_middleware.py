import logging
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .request_validator import RequestValidator
from .security_manager import SecurityManager

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"  # Prevent MIME sniffing
        response.headers["X-Frame-Options"] = "DENY"  # Prevent clickjacking
        response.headers["X-XSS-Protection"] = "1; mode=block"  # XSS protection
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"  # HTTPS only
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate all incoming requests
    """
    
    async def dispatch(self, request: Request, call_next):
        
        # Get client IP
        client_ip = RequestValidator.get_client_ip(request)
        start_time = time.time()
        
        # Validate request
        is_valid, error_message = RequestValidator.validate_request(request)
        
        if not is_valid:
            logger.warning(f"Invalid request from {client_ip}: {error_message}")
            return JSONResponse(
                status_code=400,
                content={"error": error_message}
            )
        
        # Process request
        response = await call_next(request)
        
        # Log request
        process_time = time.time() - start_time
        logger.info(
            f"{client_ip} {request.method} {request.url.path} "
            f"{response.status_code} {process_time:.2f}s"
        )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Enforce rate limiting
    """
    
    async def dispatch(self, request: Request, call_next):
        
        # Get user ID from auth header (if available)
        user_id = request.headers.get("X-User-ID", "anonymous")
        
        # Get client IP
        client_ip = RequestValidator.get_client_ip(request)
        
        # Check rate limit
        is_allowed, error_message = SecurityManager.check_rate_limit(user_id, client_ip)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {user_id} from {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"error": error_message}
            )
        
        response = await call_next(request)
        return response
    