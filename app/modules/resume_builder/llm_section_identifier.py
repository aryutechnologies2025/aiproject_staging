import json
import logging
import re
from typing import Dict, List, Any

from .ai_client import call_ai
from .universal_extractor import UniversalExtractor

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 6000

SECTION_KEYS = [
    "header", "summary", "experience", "education",
    "skills", "projects", "certifications", "languages", "other",
]


class LLMSectionIdentifier:

    # ------------------------------------------------------------------ #
    #  JSON cleaning helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_json_response(text: str) -> str:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
        text = text.strip()
        text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        return text

    @staticmethod
    def _extract_json_fallback(text: str) -> Dict[str, str]:
        result = {k: "" for k in SECTION_KEYS}
        for match in re.finditer(r'"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL):
            key, value = match.group(1), match.group(2)
            value = (
                value.replace('\\"', '"')
                     .replace('\\\\', '\\')
                     .replace('\\n', '\n')
                     .replace('\\t', '\t')
            )
            if key in result:
                result[key] = value
        return result

    # ------------------------------------------------------------------ #
    #  Text preparation                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_structured_text(raw_items: List[Dict[str, Any]], max_chars: int) -> str:
        """
        Produce a type-annotated text representation so the LLM can see
        which lines are headings vs body vs list items.
        Annotations: [HEADING], [LIST], [LINK], [TEXT]
        """
        lines = UniversalExtractor.get_all_items_flat(raw_items)
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars]
            last_nl = text.rfind("\n")
            if last_nl > max_chars * 0.8:
                text = text[:last_nl]
        return text

    # ------------------------------------------------------------------ #
    #  Pre-extract header via regex so it is never lost                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _regex_header_block(raw_items: List[Dict[str, Any]]) -> str:
        """
        Deterministically pull the first ~15 non-empty lines from page 1.
        This guarantees the LLM always sees the header region even if the
        full resume is truncated.
        """
        lines = []
        for item in raw_items:
            if item.get("page", 1) > 1 and len(lines) >= 5:
                break
            text = item.get("text", "").strip()
            if text:
                lines.append(text)
            if len(lines) >= 15:
                break
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def identify_and_extract_sections(
        raw_items: List[Dict[str, Any]],
    ) -> Dict[str, str]:

        resume_text = LLMSectionIdentifier._build_structured_text(raw_items, MAX_TEXT_CHARS)
        header_block = LLMSectionIdentifier._regex_header_block(raw_items)

        prompt = f"""You are a universal resume section extractor. The resume below uses type-annotated lines:
  [HEADING] = section heading or person's name/title
  [TEXT]    = regular paragraph / contact line
  [LIST]    = bullet point / list item
  [LINK]    = hyperlink

TASK: Split the resume into these EXACT JSON keys:
  "header", "summary", "experience", "education", "skills",
  "projects", "certifications", "languages", "other"

RULES (read carefully):
1. "header" MUST contain: full name, job title, email, phone, location, and all URLs/links.
   - The header is ALWAYS at the very top of the resume (first [HEADING] and first [TEXT] lines).
   - Do NOT put any header content into "other".
   - Works for ALL domains: IT, healthcare, finance, law, teaching, etc.
2. "summary" = Objective / Profile / About / Summary section text (any label variation).
3. "experience" = Work Experience / Employment History / Career / Professional Background.
4. "education" = Education / Academic Background / Qualifications.
5. "skills" = Skills / Competencies / Technical Skills / Core Competencies.
6. "projects" = Projects / Portfolio / Works / Case Studies.
7. "certifications" = Certifications / Licenses / Credentials / Courses.
8. "languages" = Languages / Language Proficiency.
9. "other" = Anything that does NOT fit the above (awards, publications, volunteer, hobbies).
   - NEVER put name/contact/title in "other".
10. Values must be the COMPLETE raw text for each section. Empty string "" if not present.
11. Return ONLY a compact JSON object. No markdown. No explanation.

HEADER REGION (first lines of resume — always belongs in "header"):
{header_block}

FULL RESUME (type-annotated):
{resume_text}"""

        try:
            response = await call_ai(
                prompt=prompt,
                system_prompt=(
                    "You are a resume section extractor. "
                    "Return only a valid compact JSON object with the exact keys provided. "
                    "Never put the person's name, title, email, phone, or links into 'other'."
                ),
                max_output_tokens=3500,
                use_gemini_first=False,
            )

            cleaned = LLMSectionIdentifier._clean_json_response(response)
            try:
                sections = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("JSON parse failed, using fallback extraction")
                sections = LLMSectionIdentifier._extract_json_fallback(cleaned)

            # Ensure all keys exist
            for key in SECTION_KEYS:
                if key not in sections:
                    sections[key] = ""

            # --- Safety net: if header is empty but other has contact-looking content,
            #     pull the header_block directly into "header" so it is never lost.
            header_val = str(sections.get("header", "")).strip()
            if not header_val:
                logger.warning("LLM returned empty header — injecting regex header block")
                sections["header"] = header_block

            # --- Safety net: remove header content that leaked into "other"
            contact_re = re.compile(
                r"@[a-zA-Z0-9.\-]+|linkedin\.com|github\.com|"
                r"\+\d[\d\s\-]{6,}|\d{10}",
                re.IGNORECASE,
            )
            other_val = str(sections.get("other", ""))
            if contact_re.search(other_val) and not str(sections.get("header", "")).strip():
                # Move entire other block into header as fallback
                sections["header"] = other_val
                sections["other"] = ""

            found = [k for k, v in sections.items() if str(v).strip()]
            logger.info(f"Sections identified: {found}")
            return sections

        except Exception as e:
            logger.error(f"Section identification failed: {str(e)}", exc_info=True)
            # Last-resort: return header_block in header so name/contact never vanish
            result = {k: "" for k in SECTION_KEYS}
            result["header"] = header_block
            return result
        