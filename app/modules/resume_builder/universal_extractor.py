"""
universal_extractor.py — Layout-aware and block extraction helpers for resume parsing.

Provides:
- Block item flattening with layout tags ([HEADING], [LIST], [TEXT], [LINK]).
- Aggregated content string extraction.
- Contact extraction from raw layout items.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger("resume_builder.universal_extractor")


class UniversalExtractor:
    """
    Universal content extractor preserving block layout hints and contact details.
    """

    @staticmethod
    def get_all_items_flat(raw_items: List[Dict[str, Any]]) -> List[str]:
        """
        Return text lines annotated with block types for layout-aware parsing.
        """
        items: List[str] = []

        for item in raw_items:
            text = item.get("text", "").strip()
            block_type = item.get("type", "text").lower()

            if not text:
                continue

            if block_type in ("heading", "h1", "h2", "h3", "title"):
                tag = "[HEADING]"
            elif block_type in ("list", "bullet", "li"):
                tag = "[LIST]"
            elif block_type in ("link", "url"):
                tag = "[LINK]"
            else:
                tag = "[TEXT]"

            items.append(f"{tag} {text}")

            for nested in item.get("items", []):
                val = nested.strip() if isinstance(nested, str) else ""
                if val:
                    items.append(f"[LIST] {val}")

        return items

    @staticmethod
    def extract_all_content(raw_items: List[Dict[str, Any]]) -> str:
        """
        Aggregates all raw layout items into a clean multiline document string.
        """
        lines = []
        for item in raw_items:
            text = item.get("text", "").strip()
            if text:
                lines.append(text)
            for nested in item.get("items", []):
                val = nested.strip() if isinstance(nested, str) else ""
                if val:
                    lines.append(f"• {val}")
        return "\n".join(lines).strip()

    @staticmethod
    def extract_contact_info_raw(raw_items: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract contact details directly from raw layout items.
        """
        contact = {"name": "", "email": "", "phone": "", "location": "", "link": ""}

        full_text = UniversalExtractor.extract_all_content(raw_items[:10])

        # Email
        email_match = re.search(r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b', full_text)
        if email_match:
            contact["email"] = email_match.group(0)

        # Phone
        phone_match = re.search(r'(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{3,5}', full_text)
        if phone_match and len(re.findall(r'\d', phone_match.group(0))) >= 7:
            contact["phone"] = phone_match.group(0).strip()

        # Name heuristic
        for item in raw_items[:5]:
            text = item.get("text", "").strip()
            if text and "@" not in text and not any(c.isdigit() for c in text) and len(text.split()) <= 4:
                contact["name"] = text
                break

        return contact
