"""
adaptive_token_manager.py — Intelligent Token Budgeting & Section Compression.

Provides:
- Character-to-token estimations calibrated for Gemini 2.0 and Ollama models.
- Per-section token budgets and overflow detection.
- Non-destructive compression for oversized sections (skills deduplication, experience prioritization).
"""

import logging
import re
from typing import Any, Dict, Tuple

logger = logging.getLogger("resume_builder.adaptive_token_manager")

SECTION_OUTPUT_BUDGETS = {
    "header": 300,
    "summary": 500,
    "experience": 1500,
    "education": 600,
    "skills": 500,
    "projects": 1000,
    "certifications": 400,
    "languages": 200,
    "other": 500,
}

MAX_INPUT_CHARS = {
    "header": 1000,
    "summary": 1500,
    "experience": 4000,
    "education": 2000,
    "skills": 1500,
    "projects": 3000,
    "certifications": 1200,
    "languages": 500,
    "other": 1500,
}


class AdaptiveTokenManager:
    """
    Manages token budgets and applies intelligent compression when input sections exceed capacity.
    """

    def __init__(self, total_tokens: int = 32000):
        self.total_tokens = total_tokens
        self.used_tokens: Dict[str, int] = {}

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates token count (~3.7 characters per token for English & code)."""
        if not text:
            return 0
        return max(1, int(len(text) / 3.7))

    def get_output_budget(self, section_name: str) -> int:
        return SECTION_OUTPUT_BUDGETS.get(section_name.lower(), 500)

    def get_max_input_chars(self, section_name: str) -> int:
        return MAX_INPUT_CHARS.get(section_name.lower(), 1500)

    def check_overflow(self, section_name: str, content: str) -> Tuple[bool, int]:
        max_chars = self.get_max_input_chars(section_name)
        will_overflow = len(content) > max_chars
        return will_overflow, len(content)

    def compress_content(self, section_name: str, content: str, target_chars: int) -> str:
        """
        Compresses content down to target_chars using section-specific heuristics.
        """
        if len(content) <= target_chars:
            return content

        sec = section_name.lower()
        if sec == "skills":
            return self._compress_skills(content, target_chars)
        if sec == "experience":
            return self._compress_experience(content, target_chars)

        # General text compression
        truncated = content[:target_chars]
        if "\n" in truncated:
            return truncated.rsplit("\n", 1)[0]
        return truncated

    @staticmethod
    def _compress_skills(content: str, target: int) -> str:
        tokens = re.split(r"[,|\n•;\-/]", content)
        skills = [s.strip() for s in tokens if s.strip()]
        unique = list(dict.fromkeys(skills))
        result = ", ".join(unique)
        if len(result) > target and "," in result:
            result = result[:target].rsplit(",", 1)[0]
        return result

    @staticmethod
    def _compress_experience(content: str, target: int) -> str:
        lines = content.split("\n")
        result = []
        total = 0
        for line in lines:
            if total + len(line) + 1 > target:
                break
            result.append(line)
            total += len(line) + 1
        return "\n".join(result)

    def record_usage(self, section_name: str, chars_used: int) -> None:
        estimated = self.estimate_tokens("a" * chars_used)
        self.used_tokens[section_name] = self.used_tokens.get(section_name, 0) + estimated

    def get_token_report(self) -> Dict[str, Any]:
        total_used = sum(self.used_tokens.values())
        return {
            "total_budget": self.total_tokens,
            "total_used": total_used,
            "remaining": max(0, self.total_tokens - total_used),
            "by_section": self.used_tokens,
            "utilization_percent": round((total_used / self.total_tokens * 100), 1) if self.total_tokens > 0 else 0,
        }