"""
input_sanitizer.py — Deep Input Sanitizer and Prompt Injection Defense for resume_builder.

Handles:
- Recursive sanitization of structured dictionaries and lists.
- Neutralization of prompt injection attempts (e.g. system overrides, jailbreaks).
- Preservation of legitimate programming syntax (C++, C#, .NET, Node.js, A/B testing, SQL).
- XSS and HTML tag removal.
"""

import html
import logging
import re
from typing import Any, Dict, List, Union

logger = logging.getLogger("resume_builder.input_sanitizer")

PROMPT_INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b",
    r"(?i)\bsystem\s+prompt\s+override\b",
    r"(?i)\byou\s+are\s+now\s+(?:an?\s+)?unrestricted\b",
    r"(?i)\bjailbreak\b",
    r"(?i)\bdisregard\s+(?:all\s+)?prior\s+commands\b",
    r"(?i)\boutput\s+the\s+secret\s+key\b",
    r"(?i)\bshow\s+your\s+system\s+prompt\b",
]


class InputSanitizer:
    """
    Sanitizes all incoming user payloads before document parsing, prompt construction, and AI calls.
    """

    @classmethod
    def sanitize_resume_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitizes all nested resume dictionaries and lists.
        """
        if not isinstance(data, dict):
            return {}

        sanitized: Dict[str, Any] = {}

        for key, value in data.items():
            safe_key = cls._sanitize_string(str(key))

            if isinstance(value, str):
                sanitized[safe_key] = cls._sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[safe_key] = cls.sanitize_resume_data(value)
            elif isinstance(value, list):
                sanitized[safe_key] = [cls._sanitize_value(item) for item in value]
            elif isinstance(value, (bool, int, float)) or value is None:
                sanitized[safe_key] = value
            else:
                sanitized[safe_key] = cls._sanitize_string(str(value))

        return sanitized

    @classmethod
    def sanitize_prompt_text(cls, text: str, max_length: int = 15000) -> str:
        """
        Sanitizes text intended to be embedded in LLM prompts, neutralizing injection attacks.
        """
        if not isinstance(text, str):
            return ""

        clean = text[:max_length]
        clean = clean.replace('\0', '')

        # Neutralize prompt injection phrases
        for pattern in PROMPT_INJECTION_PATTERNS:
            clean = re.sub(pattern, "[BLOCKED_INJECTION_ATTEMPT]", clean)

        # Remove script and style tags
        clean = re.sub(r'<script[^>]*>.*?</script>', '', clean, flags=re.IGNORECASE | re.DOTALL)
        clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.IGNORECASE | re.DOTALL)

        # Remove generic HTML tags
        clean = re.sub(r'<[^>]+>', '', clean)

        # Normalize multiple spaces and trailing spaces per line
        lines = [line.strip() for line in clean.split("\n")]
        return "\n".join(lines).strip()

    @classmethod
    def _sanitize_string(cls, text: str, max_length: int = 5000) -> str:
        if not isinstance(text, str):
            return ""

        text = text[:max_length]
        text = text.replace('\0', '')
        text = html.unescape(text)

        # Strip scripts and event handlers
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        text = re.sub(r'javascript:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'data:[^,]*,', '', text, flags=re.IGNORECASE)

        # Neutralize prompt injection
        for pattern in PROMPT_INJECTION_PATTERNS:
            text = re.sub(pattern, "[SANITIZED]", text)

        return ' '.join(text.split()).strip()

    @classmethod
    def _sanitize_value(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._sanitize_string(value)
        elif isinstance(value, dict):
            return cls.sanitize_resume_data(value)
        elif isinstance(value, list):
            return [cls._sanitize_value(i) for i in value]
        return value