import logging
import re
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

SECTION_OUTPUT_BUDGETS = {
    "header": 300,
    "summary": 500,
    "experience": 1200,
    "education": 600,
    "skills": 400,
    "projects": 900,
    "certifications": 400,
    "languages": 200,
    "other": 500,
}

MAX_INPUT_CHARS = {
    "header": 800,
    "summary": 1200,
    "experience": 3000,
    "education": 1500,
    "skills": 1000,
    "projects": 2500,
    "certifications": 1000,
    "languages": 400,
    "other": 1200,
}


class AdaptiveTokenManager:
    def __init__(self, total_tokens: int = 20000):
        self.total_tokens = total_tokens
        self.used_tokens: Dict[str, int] = {}

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    def get_output_budget(self, section_name: str) -> int:
        return SECTION_OUTPUT_BUDGETS.get(section_name, 500)

    def get_max_input_chars(self, section_name: str) -> int:
        return MAX_INPUT_CHARS.get(section_name, 1200)

    def check_overflow(self, section_name: str, content: str) -> Tuple[bool, int]:
        max_chars = self.get_max_input_chars(section_name)
        will_overflow = len(content) > max_chars
        return will_overflow, len(content)

    def compress_content(self, section_name: str, content: str, target_chars: int) -> str:
        if len(content) <= target_chars:
            return content
        if section_name == "skills":
            return self._compress_skills(content, target_chars)
        if section_name == "experience":
            return self._compress_experience(content, target_chars)
        return content[:target_chars].rsplit("\n", 1)[0]

    @staticmethod
    def _compress_skills(content: str, target: int) -> str:
        items = [s.strip() for s in re.split(r"[,|\n•\-]", content) if s.strip()]
        unique = list(dict.fromkeys(items))
        result = ", ".join(unique)
        if len(result) > target:
            result = result[:target].rsplit(",", 1)[0]
        return result

    @staticmethod
    def _compress_experience(content: str, target: int) -> str:
        lines = content.split("\n")
        result = []
        total = 0
        for line in lines:
            total += len(line)
            result.append(line)
            if total >= target:
                break
        return "\n".join(result)

    def record_usage(self, section_name: str, chars_used: int):
        self.used_tokens[section_name] = self.used_tokens.get(section_name, 0) + (chars_used // 4)

    def get_token_report(self) -> Dict[str, Any]:
        total_used = sum(self.used_tokens.values())
        return {
            "total_budget": self.total_tokens,
            "total_used": total_used,
            "remaining": self.total_tokens - total_used,
            "by_section": self.used_tokens,
            "utilization_percent": round((total_used / self.total_tokens * 100), 1) if self.total_tokens > 0 else 0
        }
    