import logging
import hashlib
import secrets
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
import re

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Comprehensive security management for resume parsing
    Handles: validation, sanitization, rate limiting, malware detection
    """
    
    # Constants
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_JSON_SIZE = 5 * 1024 * 1024   # 5MB
    MAX_REQUESTS_PER_HOUR = 100
    MAX_REQUESTS_PER_IP = 50
    
    # Allowed file types (MIME types)
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain'
    }
    
    # Suspicious patterns (malware/injection attempts)
    SUSPICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # JavaScript injection
        r'javascript:',                 # JavaScript protocol
        r'on\w+\s*=',                   # Event handlers (onclick, etc)
        r'eval\(',                      # eval() function
        r'base64_decode',               # Base64 decoding
        r'system\(',                    # System commands
        r'exec\(',                      # exec() function
        r'shell_exec',                  # Shell execution
        r'passthru',                    # Passthru execution
        r'proc_open',                   # Process opening
        r'popen',                       # Process popen
        r'curl',                        # Curl commands
        r'wget',                        # Wget commands
        r'python -m',                   # Python module execution
        r'import os',                   # OS import
        r'subprocess',                  # Subprocess execution
        r'__import__',                  # Dynamic imports
        r'pickle\.loads',               # Pickle deserialization
        r'\/\/\/\/',                    # Path traversal patterns
        r'\.\.\/',                      # Directory traversal
        r'%2e%2e',                      # URL encoded traversal
    ]
    
    # Rate limiting storage (in production, use Redis)
    _request_history: Dict[str, list] = {}
    _ip_history: Dict[str, list] = {}
    
    @staticmethod
    def validate_file(file_bytes: bytes, filename: str, content_type: str) -> tuple[bool, str]:
        """
        Validate file before processing
        Returns: (is_valid, error_message)
        """
        
        # 1. Check file size
        if len(file_bytes) > SecurityManager.MAX_FILE_SIZE:
            logger.warning(f"File too large: {filename} ({len(file_bytes)} bytes)")
            return False, "File size exceeds maximum limit (10MB)"
        
        # 2. Check file extension
        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if f'.{file_extension}' not in allowed_extensions:
            logger.warning(f"Invalid file extension: {file_extension}")
            return False, f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"
        
        # 3. Check MIME type
        if content_type not in SecurityManager.ALLOWED_MIME_TYPES:
            logger.warning(f"Invalid MIME type: {content_type}")
            return False, f"Invalid file type: {content_type}"
        
        # 4. Validate file signature (magic bytes)
        if not SecurityManager._validate_file_signature(file_bytes, file_extension):
            logger.warning(f"Invalid file signature: {filename}")
            return False, "File signature validation failed. File may be corrupted or malicious."
        
        # 5. Check for suspicious content
        try:
            file_text = file_bytes.decode('utf-8', errors='ignore')
            if SecurityManager._contains_suspicious_patterns(file_text):
                logger.warning(f"Suspicious patterns detected in: {filename}")
                return False, "File contains suspicious patterns. Please upload a clean resume."
        except Exception as e:
            logger.warning(f"Error checking file content: {str(e)}")
            return False, "Error validating file content"
        
        # 6. Verify filename (no path traversal)
        if not SecurityManager._validate_filename(filename):
            logger.warning(f"Invalid filename pattern: {filename}")
            return False, "Invalid filename. Contains prohibited characters."
        
        logger.info(f"✓ File validation passed: {filename}")
        return True, ""
    
    @staticmethod
    def validate_json_data(data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate parsed JSON data for safety
        Returns: (is_valid, error_message)
        """
        
        try:
            json_str = json.dumps(data)
            
            # Check JSON size
            if len(json_str) > SecurityManager.MAX_JSON_SIZE:
                return False, "Parsed data exceeds maximum size"
            
            # Check for suspicious patterns in parsed data
            if SecurityManager._contains_suspicious_patterns(json_str):
                return False, "Suspicious patterns detected in parsed data"
            
            logger.info("✓ JSON data validation passed")
            return True, ""
        
        except Exception as e:
            logger.error(f"JSON validation error: {str(e)}")
            return False, "Error validating parsed data"
    
    @staticmethod
    def check_rate_limit(user_id: str, ip_address: str) -> tuple[bool, str]:
        """
        Check rate limiting for user and IP
        Returns: (is_allowed, error_message)
        """
        
        now = time.time()
        one_hour_ago = now - 3600
        
        # Check user rate limit
        if user_id not in SecurityManager._request_history:
            SecurityManager._request_history[user_id] = []
        
        # Clean old requests
        SecurityManager._request_history[user_id] = [
            t for t in SecurityManager._request_history[user_id] 
            if t > one_hour_ago
        ]
        
        # Check limit
        if len(SecurityManager._request_history[user_id]) >= SecurityManager.MAX_REQUESTS_PER_HOUR:
            logger.warning(f"Rate limit exceeded for user: {user_id}")
            return False, f"Rate limit exceeded. Maximum {SecurityManager.MAX_REQUESTS_PER_HOUR} requests per hour"
        
        # Check IP rate limit
        if ip_address not in SecurityManager._ip_history:
            SecurityManager._ip_history[ip_address] = []
        
        # Clean old requests
        SecurityManager._ip_history[ip_address] = [
            t for t in SecurityManager._ip_history[ip_address]
            if t > one_hour_ago
        ]
        
        # Check limit
        if len(SecurityManager._ip_history[ip_address]) >= SecurityManager.MAX_REQUESTS_PER_IP:
            logger.warning(f"IP rate limit exceeded: {ip_address}")
            return False, f"Too many requests from your IP. Please try again later."
        
        # Add current request
        SecurityManager._request_history[user_id].append(now)
        SecurityManager._ip_history[ip_address].append(now)
        
        logger.info(f"✓ Rate limit check passed for {user_id} from {ip_address}")
        return True, ""
    
    @staticmethod
    def _validate_file_signature(file_bytes: bytes, extension: str) -> bool:
        """
        Validate file by checking magic bytes (file signature)
        """
        # PDF signature
        if extension == 'pdf':
            return file_bytes.startswith(b'%PDF')
        
        # Word documents
        elif extension in ('doc', 'docx'):
            # DOCX is ZIP, should start with PK
            if extension == 'docx':
                return file_bytes.startswith(b'PK\x03\x04')
            # DOC is binary, harder to validate
            return True
        
        # Text files
        elif extension == 'txt':
            try:
                file_bytes.decode('utf-8')
                return True
            except:
                return False
        
        return True
    
    @staticmethod
    def _contains_suspicious_patterns(content: str) -> bool:
        """
        Check if content contains suspicious patterns
        """
        content_lower = content.lower()
        
        for pattern in SecurityManager.SUSPICIOUS_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected: {pattern}")
                return True
        
        return False
    
    @staticmethod
    def _validate_filename(filename: str) -> bool:
        """
        Validate filename to prevent path traversal
        """
        
        # Check for path traversal attempts
        dangerous_patterns = ['../', '..\\', '../', '..\\', '%2e%2e', '~', '$']
        
        for pattern in dangerous_patterns:
            if pattern in filename.lower():
                return False
        
        # Check for null bytes
        if '\0' in filename:
            return False
        
        # Allow only alphanumeric, dash, underscore, dot
        if not re.match(r'^[\w\-. ]+$', filename):
            return False
        
        # Max filename length
        if len(filename) > 255:
            return False
        
        return True
    
    @staticmethod
    def sanitize_text_input(text: str, max_length: int = 10000) -> str:
        """
        Sanitize text input from user
        """
        
        # Limit length
        text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\0', '')
        
        # Remove control characters (except newline, tab)
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove script tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """
        Generate secure random token for sessions
        """
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """
        Hash sensitive data (user IDs, emails for logging)
        """
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    @staticmethod
    def validate_json_structure(data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate parsed resume JSON structure
        """
        
        required_sections = ["header", "experience", "education", "skills"]
        
        # Check if dict
        if not isinstance(data, dict):
            return False, "Invalid data structure"
        
        # Check for required sections (don't need all, but structure should be valid)
        if "parsed" in data:
            data = data["parsed"]
        
        if not isinstance(data, dict):
            return False, "Invalid parsed data structure"
        
        # Validate header
        if "header" in data:
            if not isinstance(data["header"], dict):
                return False, "Invalid header structure"
            
            # Check for dangerous fields in header
            for key in data["header"]:
                if not isinstance(key, str):
                    return False, "Invalid header key type"
        
        # Validate experience
        if "experience" in data:
            if not isinstance(data["experience"], list):
                return False, "Invalid experience structure"
            
            for exp in data["experience"]:
                if not isinstance(exp, dict):
                    return False, "Invalid experience entry"
        
        # Validate education
        if "education" in data:
            if not isinstance(data["education"], list):
                return False, "Invalid education structure"
        
        # Validate skills
        if "skills" in data:
            if not isinstance(data["skills"], list):
                return False, "Invalid skills structure"
        
        logger.info("✓ JSON structure validation passed")
        return True, ""
    