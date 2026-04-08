import logging
import re
from typing import Dict, Any, List
import html
import json

logger = logging.getLogger(__name__)


class InputSanitizer:
    """
    Sanitize all input data to prevent injection attacks
    """
    
    @staticmethod
    def sanitize_resume_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep sanitize all resume data
        """
        
        if not isinstance(data, dict):
            return {}
        
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            safe_key = InputSanitizer._sanitize_string(key)
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[safe_key] = InputSanitizer._sanitize_string(value)
            
            elif isinstance(value, dict):
                sanitized[safe_key] = InputSanitizer.sanitize_resume_data(value)
            
            elif isinstance(value, list):
                sanitized[safe_key] = [
                    InputSanitizer._sanitize_value(item) for item in value
                ]
            
            elif isinstance(value, bool):
                sanitized[safe_key] = value
            
            elif isinstance(value, (int, float)):
                sanitized[safe_key] = value
            
            else:
                # Unknown type - skip it
                logger.warning(f"Skipping unknown type for key {safe_key}: {type(value)}")
        
        return sanitized
    
    @staticmethod
    def _sanitize_string(text: str, max_length: int = 5000) -> str:
        """
        Sanitize string input
        """
        
        if not isinstance(text, str):
            return ""
        
        # Limit length
        text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\0', '')
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove script tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove HTML tags (but keep text)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove event handlers
        text = re.sub(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*on\w+\s*=\s*[^\s>]*', '', text, flags=re.IGNORECASE)
        
        # Remove JavaScript protocol
        text = re.sub(r'javascript:\s*', '', text, flags=re.IGNORECASE)
        
        # Remove data: protocol (can be used for XSS)
        text = re.sub(r'data:[^,]*,', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        """
        Sanitize individual value
        """
        
        if isinstance(value, str):
            return InputSanitizer._sanitize_string(value)
        
        elif isinstance(value, dict):
            return InputSanitizer.sanitize_resume_data(value)
        
        elif isinstance(value, list):
            return [InputSanitizer._sanitize_value(item) for item in value]
        
        elif isinstance(value, (bool, int, float)):
            return value
        
        else:
            return None
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format
        """
        
        if not isinstance(email, str):
            return False
        
        # RFC 5322 simplified pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            return False
        
        # Check length
        if len(email) > 254:
            return False
        
        return True
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        Validate phone number format
        """
        
        if not isinstance(phone, str):
            return False
        
        # Remove common formatting characters
        cleaned = re.sub(r'[\s\-\(\)\+\.]', '', phone)
        
        # Should be 7-15 digits
        if not re.match(r'^\d{7,15}$', cleaned):
            return False
        
        return True
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format
        """
        
        if not isinstance(url, str):
            return False
        
        # Simple URL validation
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        
        if not re.match(pattern, url, re.IGNORECASE):
            return False
        
        # Check length
        if len(url) > 2048:
            return False
        
        return True
    
    