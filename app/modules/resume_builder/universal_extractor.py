import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class UniversalExtractor:
    """
    Universal content extractor — layout-aware, domain-agnostic.
    Preserves block type hints so the section identifier LLM gets structured context.
    """

    # ------------------------------------------------------------------ #
    #  Public: full structured text for LLM section identification         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_all_items_flat(raw_items: List[Dict[str, Any]]) -> List[str]:
        """
        Return text lines annotated with block type so the LLM can detect
        headings, body text, and list items without guessing layout.

        Format injected:  [HEADING] …  /  [LIST] …  /  [TEXT] …
        The LLM strips these tags internally — they are structural hints only.
        """
        items: List[str] = []

        for item in raw_items:
            text = item.get("text", "").strip()
            block_type = item.get("type", "text").lower()

            if not text:
                continue

            # Map LlamaParse block types to readable tags
            if block_type in ("heading", "h1", "h2", "h3", "title"):
                tag = "[HEADING]"
            elif block_type in ("list", "bullet", "li"):
                tag = "[LIST]"
            elif block_type in ("link", "url"):
                tag = "[LINK]"
            else:
                tag = "[TEXT]"

            items.append(f"{tag} {text}")

            # Expand nested list items
            for nested in item.get("items", []):
                val = nested.strip() if isinstance(nested, str) else ""
                if val:
                    items.append(f"[LIST] {val}")

        return items

    # ------------------------------------------------------------------ #
    #  Public: plain content dump (used by other utilities)               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_all_content(raw_items: List[Dict[str, Any]]) -> str:
        """Plain concatenated text — no type tags."""
        parts = []
        for item in raw_items:
            text = item.get("text", "").strip()
            if text:
                parts.append(text)
            for nested in item.get("items", []):
                val = nested.strip() if isinstance(nested, str) else ""
                if val:
                    parts.append(val)
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Public: regex-based contact extraction (LLM-free fallback)         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_contact_info_raw(raw_items: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract all contact fields deterministically via regex.
        Works for any domain, any resume format.
        """
        all_text = "\n".join(item.get("text", "") for item in raw_items)

        email = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", all_text)

        # Broad international phone — captures +91, +1, bare 10-digit, etc.
        phone = re.search(
            r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,5}\)?[\s.\-]?)?\d{3,5}[\s.\-]?\d{3,5}[\s.\-]?\d{0,5}",
            all_text,
        )

        linkedin = re.search(r"linkedin\.com/in/[\w\-]+", all_text, re.IGNORECASE)
        github = re.search(r"github\.com/[\w\-]+", all_text, re.IGNORECASE)

        # Portfolio: anything that looks like a personal domain / hosted URL
        portfolio = re.search(
            r"(?:https?://)?(?:www\.)?[\w\-]+\.(?:netlify\.app|vercel\.app|github\.io|me|dev|io|app|co|net|com)"
            r"(?:/[\w\-./]*)?",
            all_text,
            re.IGNORECASE,
        )

        # Name heuristic: first [HEADING] or first [TEXT] line before any known contact field
        name = UniversalExtractor._extract_name_heuristic(raw_items)

        # Location heuristic: look for city/state/country patterns
        location = UniversalExtractor._extract_location_heuristic(all_text)

        return {
            "name": name,
            "email": email.group(0).strip() if email else "",
            "phone": UniversalExtractor._clean_phone(phone.group(0)) if phone else "",
            "linkedin": linkedin.group(0).strip() if linkedin else "",
            "github": github.group(0).strip() if github else "",
            "portfolio": portfolio.group(0).strip() if portfolio else "",
            "location": location,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_name_heuristic(raw_items: List[Dict[str, Any]]) -> str:
        """
        The resume name is almost always the very first heading or the first
        short (<= 6 words) text block on page 1 before any contact detail.
        """
        contact_signals = re.compile(
            r"@|linkedin|github|http|www\.|\.com|\.io|\.net|\+\d|\d{5,}|"
            r"resume|curriculum|vitae|objective|summary|profile|experience|"
            r"education|skill|project|certification|language",
            re.IGNORECASE,
        )

        for item in raw_items:
            if item.get("page", 1) > 1:
                break
            text = item.get("text", "").strip()
            block_type = item.get("type", "text").lower()
            if not text or contact_signals.search(text):
                continue
            words = text.split()
            # Headings or short text lines with 1–6 words → likely the name
            if block_type in ("heading", "h1", "h2", "title") or (1 <= len(words) <= 6):
                # Must look like a name: mostly alphabetic + spaces/hyphens
                if re.match(r"^[A-Za-z][A-Za-z\s\-'.]{1,50}$", text):
                    return text
        return ""

    @staticmethod
    def _extract_location_heuristic(text: str) -> str:
        """
        Match common location patterns:
        'City, State'  /  'City, Country'  /  'City, ST 12345'
        """
        match = re.search(
            r"\b([A-Z][a-zA-Z\s]+),\s*([A-Z][a-zA-Z\s]{2,}(?:\s+\d{5,6})?)\b",
            text,
        )
        return match.group(0).strip() if match else ""

    @staticmethod
    def _clean_phone(raw: str) -> str:
        """Strip leading/trailing noise; keep digits, +, spaces, dashes, parens."""
        cleaned = re.sub(r"[^\d\+\-\(\)\s]", "", raw).strip()
        # Reject if fewer than 7 digits remain
        digits = re.sub(r"\D", "", cleaned)
        return cleaned if len(digits) >= 7 else ""