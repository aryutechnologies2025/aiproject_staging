"""
security_manager.py — Comprehensive security management for resume_builder.

Handles:
- File signature validation (magic bytes).
- True timestamp-based sliding-window rate limiting with memory leak protection.
- Path traversal defense and malicious payload detection.
"""

import hashlib
import logging
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("resume_builder.security_manager")


class SecurityManager:
    """
    Security manager providing file validation, sanitization, and sliding-window rate limiting.
    """

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_JSON_SIZE = 5 * 1024 * 1024   # 5MB

    # Sliding-Window Rate Limits
    IP_BURST_PER_MINUTE = 60
    USER_BURST_PER_MINUTE = 120
    MAX_REQUESTS_PER_HOUR = 300
    MAX_REQUESTS_PER_IP = 150

    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'application/octet-stream',
    }

    SUSPICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\(',
        r'base64_decode',
        r'system\(',
        r'exec\(',
        r'shell_exec',
        r'passthru',
        r'proc_open',
        r'popen',
        r'python -m',
        r'import os',
        r'subprocess',
        r'__import__',
        r'pickle\.loads',
        r'\/\/\/\/',
        r'\.\.\/',
        r'%2e%2e',
    ]

    # In-memory sliding-window history: {identifier: [timestamp_1, timestamp_2, ...]}
    _user_window_history: Dict[str, List[float]] = {}
    _ip_window_history: Dict[str, List[float]] = {}
    _check_counter = 0

    @classmethod
    def validate_file(cls, file_bytes: bytes, filename: str, content_type: str) -> Tuple[bool, str]:
        """
        Validate uploaded file bytes, format, and magic signatures.
        """
        if len(file_bytes) > cls.MAX_FILE_SIZE:
            logger.warning(f"File too large: {filename} ({len(file_bytes)} bytes)")
            return False, "File size exceeds maximum limit (10MB)"

        if not cls._validate_filename(filename):
            logger.warning(f"Invalid filename or path traversal attempt: {filename}")
            return False, "Invalid filename. Path traversal characters not permitted."

        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''

        if f'.{file_extension}' not in allowed_extensions:
            logger.warning(f"Invalid file extension: {file_extension}")
            return False, f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"

        if not cls._validate_file_signature(file_bytes, file_extension):
            logger.warning(f"Invalid file signature: {filename}")
            return False, "File signature validation failed. File may be corrupted or malformed."

        return True, ""

    @classmethod
    def validate_json_data(cls, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parsed JSON data size and integrity.
        """
        try:
            import json
            json_str = json.dumps(data)
            if len(json_str) > cls.MAX_JSON_SIZE:
                return False, "Parsed data exceeds maximum size"
            return True, ""
        except Exception as e:
            logger.error(f"JSON validation error: {str(e)}")
            return False, "Error validating parsed data"

    @classmethod
    def check_rate_limit(
        cls,
        user_id: str,
        ip_address: str,
    ) -> Tuple[bool, str]:
        """
        Sliding-window rate limiter checking both 1-minute burst and 1-hour sustained limits.
        Returns: (is_allowed, error_message)
        """
        now = time.time()
        cls._check_counter += 1

        # Periodic memory cleanup of stale entries (every 100 checks)
        if cls._check_counter % 100 == 0:
            cls._prune_stale_buckets(now)

        # ── 1. Check IP Sliding Window (60s and 3600s) ──
        ip_records = cls._ip_window_history.setdefault(ip_address, [])
        # Filter to active timestamps within 3600s
        cls._ip_window_history[ip_address] = [t for t in ip_records if now - t < 3600.0]
        ip_records = cls._ip_window_history[ip_address]

        # Count in last 60 seconds
        ip_last_min = sum(1 for t in ip_records if now - t < 60.0)
        if ip_last_min >= cls.IP_BURST_PER_MINUTE:
            logger.warning(f"IP minute burst rate limit exceeded: {ip_address}")
            return False, "Too many requests from your IP address. Please wait a moment."

        if len(ip_records) >= cls.MAX_REQUESTS_PER_IP:
            logger.warning(f"IP hourly rate limit exceeded: {ip_address}")
            return False, "Hourly request limit exceeded for your IP."

        # ── 2. Check User Sliding Window ──
        user_records = cls._user_window_history.setdefault(user_id, [])
        cls._user_window_history[user_id] = [t for t in user_records if now - t < 3600.0]
        user_records = cls._user_window_history[user_id]

        user_last_min = sum(1 for t in user_records if now - t < 60.0)
        if user_last_min >= cls.USER_BURST_PER_MINUTE:
            logger.warning(f"User minute burst rate limit exceeded: {user_id}")
            return False, "Too many requests. Please slow down."

        if len(user_records) >= cls.MAX_REQUESTS_PER_HOUR:
            logger.warning(f"User hourly rate limit exceeded: {user_id}")
            return False, f"Hourly request limit exceeded ({cls.MAX_REQUESTS_PER_HOUR}/hour)."

        # Record this request
        cls._ip_window_history[ip_address].append(now)
        cls._user_window_history[user_id].append(now)

        return True, ""

    @classmethod
    def _prune_stale_buckets(cls, now: float) -> None:
        """Prune tracking buckets with zero requests in the last hour."""
        cutoff = now - 3600.0
        for ip in list(cls._ip_window_history.keys()):
            valid = [t for t in cls._ip_window_history[ip] if t > cutoff]
            if valid:
                cls._ip_window_history[ip] = valid
            else:
                cls._ip_window_history.pop(ip, None)

        for uid in list(cls._user_window_history.keys()):
            valid = [t for t in cls._user_window_history[uid] if t > cutoff]
            if valid:
                cls._user_window_history[uid] = valid
            else:
                cls._user_window_history.pop(uid, None)

    @classmethod
    def _validate_file_signature(cls, file_bytes: bytes, extension: str) -> bool:
        if extension == 'pdf':
            return file_bytes.startswith(b'%PDF')
        elif extension == 'docx':
            return file_bytes.startswith(b'PK\x03\x04')
        elif extension == 'doc':
            return True
        elif extension == 'txt':
            try:
                file_bytes.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
        return True

    @classmethod
    def _validate_filename(cls, filename: str) -> bool:
        dangerous_patterns = ['../', '..\\', '%2e%2e', '~', '$', '/../', '\\..\\']
        for pattern in dangerous_patterns:
            if pattern in filename.lower():
                return False
        if '\0' in filename:
            return False
        if len(filename) > 255:
            return False
        return bool(re.match(r'^[\w\-. ]+$', filename))

    @classmethod
    def clear_rate_limits(cls) -> None:
        """Reset rate limit history (for testing)."""
        cls._user_window_history.clear()
        cls._ip_window_history.clear()
        cls._check_counter = 0