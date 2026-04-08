import logging
from typing import Optional, Tuple
from functools import wraps
from fastapi import Request, HTTPException, status
import time

logger = logging.getLogger(__name__)


class RequestValidator:
    """
    Validate and sanitize incoming HTTP requests
    """
    
    # Blacklisted IPs (banned malicious actors)
    BLACKLISTED_IPS = set()
    
    # Whitelist for testing (if needed)
    WHITELISTED_IPS = set()
    
    @staticmethod
    def validate_request(request: Request) -> Tuple[bool, Optional[str]]:
        """
        Validate incoming request
        Returns: (is_valid, error_message)
        """
        
        # Get client IP
        client_ip = RequestValidator.get_client_ip(request)
        
        # Check if IP is blacklisted
        if client_ip in RequestValidator.BLACKLISTED_IPS:
            logger.warning(f"Request from blacklisted IP: {client_ip}")
            return False, "Your IP address has been blocked due to suspicious activity"
        
        # Check request headers
        if not RequestValidator._validate_headers(request):
            logger.warning(f"Invalid headers from {client_ip}")
            return False, "Invalid request headers"
        
        # Check request method
        if request.method not in ["POST", "GET", "OPTIONS"]:
            logger.warning(f"Disallowed HTTP method: {request.method} from {client_ip}")
            return False, "HTTP method not allowed"
        
        logger.info(f"✓ Request validation passed for {client_ip}")
        return True, None
    
    @staticmethod
    def get_client_ip(request: Request) -> str:
        """
        Get real client IP (handles proxies)
        """
        # Check X-Forwarded-For header (for proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Use client connection IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    @staticmethod
    def _validate_headers(request: Request) -> bool:
        """
        Validate HTTP headers
        """
        
        # Check for suspicious headers
        dangerous_headers = [
            "X-Forwarded-Host",
            "X-Forwarded-Proto",
            "X-Original-URL",
            "X-Rewrite-URL",
        ]
        
        for header in dangerous_headers:
            if header in request.headers:
                # These can be used for header injection
                value = request.headers[header]
                if not _is_safe_header_value(value):
                    return False
        
        # Check User-Agent
        user_agent = request.headers.get("User-Agent", "")
        if len(user_agent) > 500:  # Suspiciously long user agent
            return False
        
        # Check Content-Type
        if request.method == "POST":
            content_type = request.headers.get("Content-Type", "")
            allowed_types = [
                "multipart/form-data",
                "application/json",
                "application/x-www-form-urlencoded"
            ]
            
            if not any(allowed in content_type for allowed in allowed_types):
                return False
        
        return True
    
    @staticmethod
    def blacklist_ip(ip_address: str):
        """
        Blacklist an IP address
        """
        RequestValidator.BLACKLISTED_IPS.add(ip_address)
        logger.warning(f"IP blacklisted: {ip_address}")
    
    @staticmethod
    def whitelist_ip(ip_address: str):
        """
        Whitelist an IP address
        """
        RequestValidator.WHITELISTED_IPS.add(ip_address)
        logger.info(f"IP whitelisted: {ip_address}")


def _is_safe_header_value(value: str) -> bool:
    """
    Check if header value is safe
    """
    # Check length
    if len(value) > 1000:
        return False
    
    # Check for injection patterns
    dangerous_chars = ['\n', '\r', '\0', '%00']
    for char in dangerous_chars:
        if char in value:
            return False
    
    return True
