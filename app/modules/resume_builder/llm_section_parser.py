import json
import logging
import re
from typing import Dict, Any

from .ai_client import call_ai

logger = logging.getLogger(__name__)

SECTION_INPUT_LIMITS = {
    "header": 800,
    "summary": 800,
    "experience": 2500,
    "education": 1200,
    "skills": 800,
    "projects": 2000,
    "certifications": 800,
    "languages": 300,
    "other": 800,
}

SECTION_OUTPUT_TOKENS = {
    "experience": 1200,
    "projects": 1100,
    "skills": 400,
    "education": 600,
    "header": 550,
    "summary": 400,
    "certifications": 400,
    "languages": 200,
    "other": 400,
}

SECTION_PROMPTS = {
    "header": (
        'Extract from the text below and return ONLY this JSON object:\n'
        '{{"name":"","title":"","location":"","email":"","phone":"","link":""}}\n\n'
        'Rules:\n'
        '- "name": full person name (not company, not section heading like "Resume")\n'
        '- "title": professional title / designation\n'
        '- "location": city, state/country\n'
        '- "email": email address\n'
        '- "phone": full phone number including country code if present\n'
        '- "link": comma-separated list of all URLs (LinkedIn, GitHub, portfolio, website)\n'
        '- If a field is not found leave it as empty string ""\n'
        '- JSON only, no explanation\n\n'
        'TEXT:\n{content}'
    ),
    "summary": (
        'Extract the professional summary/objective/profile as a single paragraph string.\n'
        'Return ONLY: {{"summary":"<text>"}}\n'
        'JSON only.\n\n'
        'TEXT:\n{content}'
    ),
    "experience": (
        'Extract ALL jobs/positions from the text below.\n'
        'Return ONLY a JSON array:\n'
        '[{{"position":"","company":"","location":"","fromYear":"","toYear":"","isOngoing":false,"bullets":[]}}]\n\n'
        'Rules:\n'
        '- "isOngoing": true if current/present role\n'
        '- "bullets": list of achievement/responsibility strings\n'
        '- "fromYear"/"toYear": 4-digit year strings\n'
        '- Extract ALL jobs, do not skip any\n'
        '- JSON only\n\n'
        'TEXT:\n{content}'
    ),
    "education": (
        'Extract ALL education entries.\n'
        'Return ONLY a JSON array:\n'
        '[{{"degree":"","institution":"","location":"","fromYear":"","toYear":""}}]\n'
        'JSON only.\n\n'
        'TEXT:\n{content}'
    ),
    "skills": (
        'Extract ALL skills mentioned. Flatten all categories into one list.\n'
        'Return ONLY a JSON string array: ["skill1","skill2",...]\n'
        'No duplicates. JSON only.\n\n'
        'TEXT:\n{content}'
    ),
    "projects": (
        'Extract ALL projects.\n'
        'Return ONLY a JSON array:\n'
        '[{{"title":"","description":"","technologies":[],"fromYear":"","toYear":"","bullets":[]}}]\n'
        'JSON only.\n\n'
        'TEXT:\n{content}'
    ),
    "certifications": (
        'Extract ALL certifications/licenses/credentials.\n'
        'Return ONLY a JSON array:\n'
        '[{{"title":"","issuer":"","year":""}}]\n'
        'JSON only.\n\n'
        'TEXT:\n{content}'
    ),
    "languages": (
        'Extract ALL human languages (e.g. English, Tamil, French).\n'
        'Return ONLY a JSON string array: ["lang1","lang2"]\n'
        'JSON only.\n\n'
        'TEXT:\n{content}'
    ),
    "other": (
        'Extract miscellaneous items (awards, publications, volunteer work, hobbies).\n'
        'Return ONLY a JSON string array.\n'
        'JSON only.\n\n'
        'TEXT:\n{content}'
    ),
}

EMPTY_SECTIONS: Dict[str, Any] = {
    "header": {"name": "", "title": "", "email": "", "phone": "", "location": "", "link": ""},
    "summary": {"summary": ""},
    "experience": [],
    "education": [],
    "skills": [],
    "projects": [],
    "certifications": [],
    "languages": [],
    "other": [],
}


# ------------------------------------------------------------------ #
#  Regex-based header pre-extraction (deterministic, LLM-independent) #
# ------------------------------------------------------------------ #

def _regex_extract_header(text: str) -> Dict[str, str]:
    """
    Extract all header fields that can be found deterministically.
    Used to (a) pre-fill before LLM call and (b) patch LLM output.
    """
    email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)

    # International phone — broad pattern, cleaned afterward
    phone_match = re.search(
        r"(?:\+\d{1,3}[\s.\-]?)?(?:\(?\d{2,5}\)?[\s.\-]?)?\d{3,5}[\s.\-]?\d{3,5}(?:[\s.\-]?\d{2,5})?",
        text,
    )

    linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)
    github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+", text, re.IGNORECASE)
    portfolio_match = re.search(
        r"(?:https?://)?(?:www\.)?[\w\-]+\.(?:netlify\.app|vercel\.app|github\.io|me|dev|io|app)(?:/[\w\-./]*)?",
        text,
        re.IGNORECASE,
    )

    # Collect links
    links = []
    for m in [linkedin_match, github_match, portfolio_match]:
        if m:
            val = m.group(0).strip()
            if val not in links:
                links.append(val)

    # Location: "City, State" or "City, Country" pattern
    location_match = re.search(
        r"\b([A-Z][a-zA-Z\s\-]+),\s*([A-Z][a-zA-Z\s]{2,}(?:\s+\d{5,6})?)\b",
        text,
    )

    # Phone cleanup
    phone_val = ""
    if phone_match:
        raw = phone_match.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 7:
            phone_val = raw

    return {
        "email": email_match.group(0).strip() if email_match else "",
        "phone": phone_val,
        "link": ", ".join(links),
        "location": location_match.group(0).strip() if location_match else "",
    }


def _extract_name_from_text(text: str) -> str:
    """
    Heuristic: name is a short line (1–5 words) of mostly alphabetic chars
    that appears before contact details and section headings.
    """
    contact_re = re.compile(
        r"@|http|www\.|linkedin|github|\.com|\.io|\+\d|\d{5,}|"
        r"resume|curriculum|vitae|objective|summary|profile|"
        r"experience|education|skill|project|certification",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or contact_re.search(line):
            continue
        words = line.split()
        if 1 <= len(words) <= 6 and re.match(r"^[A-Za-z][A-Za-z\s\-'.]{1,50}$", line):
            return line
    return ""


class LLMSectionParser:

    @staticmethod
    def _clean_json_response(res: str) -> str:
        res = res.strip()
        if "```json" in res:
            res = res.split("```json")[1].split("```")[0]
        elif "```" in res:
            res = res.split("```")[1].split("```")[0]
        res = res.strip()
        res = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', res)
        res = re.sub(r',(\s*[}\]])', r'\1', res)
        return res

    @staticmethod
    def _trim_input(section_name: str, text: str) -> str:
        limit = SECTION_INPUT_LIMITS.get(section_name, 800)
        if len(text) <= limit:
            return text
        trimmed = text[:limit]
        last_nl = trimmed.rfind("\n")
        if last_nl > limit * 0.7:
            return trimmed[:last_nl]
        return trimmed

    @staticmethod
    def _deduplicate_skills(data: Any) -> Any:
        if isinstance(data, list):
            return sorted(list(dict.fromkeys(str(s).strip() for s in data if s)))
        return data

    @staticmethod
    def _patch_header(parsed: Dict[str, Any], regex_data: Dict[str, str], raw_text: str) -> Dict[str, Any]:
        """
        Merge LLM output with deterministic regex extractions.
        Regex values win for fields that are reliably detectable (email, phone, links).
        LLM values are used for name, title, location when regex has nothing.
        """
        if not isinstance(parsed, dict):
            parsed = {}

        # Fields where regex is authoritative
        for field in ("email", "phone", "link"):
            if regex_data.get(field):
                parsed[field] = regex_data[field]
            elif not parsed.get(field):
                parsed[field] = ""

        # Location: prefer regex result if LLM missed it
        if not parsed.get("location") and regex_data.get("location"):
            parsed["location"] = regex_data["location"]

        # Name: if LLM returned empty or a non-name string, try heuristic
        if not parsed.get("name") or len(parsed.get("name", "").split()) > 7:
            name_guess = _extract_name_from_text(raw_text)
            if name_guess:
                parsed["name"] = name_guess

        # Ensure all keys exist
        for key in ("name", "title", "email", "phone", "location", "link"):
            parsed.setdefault(key, "")

        return parsed

    @staticmethod
    async def parse_section(section_name: str, section_text: str) -> Any:
        if not section_text.strip():
            return EMPTY_SECTIONS.get(section_name, [])

        content = LLMSectionParser._trim_input(section_name, section_text)
        template = SECTION_PROMPTS.get(section_name, "Extract data as JSON array.\n{content}")
        prompt = template.format(content=content)
        max_tokens = SECTION_OUTPUT_TOKENS.get(section_name, 400)

        # For header: run regex extraction before LLM so we have a fallback ready
        regex_header: Dict[str, str] = {}
        if section_name == "header":
            regex_header = _regex_extract_header(section_text)

        try:
            response = await call_ai(
                prompt=prompt,
                system_prompt="Return only valid compact JSON. No explanation.",
                max_output_tokens=max_tokens,
                use_gemini_first=True,
            )
            json_text = LLMSectionParser._clean_json_response(response)
            try:
                parsed = json.loads(json_text)
            except json.JSONDecodeError:
                logger.warning(f"JSON parse failed for {section_name}, using empty default")
                parsed = EMPTY_SECTIONS.get(section_name, [])

            if section_name == "header":
                return LLMSectionParser._patch_header(parsed, regex_header, section_text)

            if section_name == "skills":
                return LLMSectionParser._deduplicate_skills(parsed)

            return parsed

        except Exception as e:
            logger.error(f"Parse error for {section_name}: {str(e)}")
            if section_name == "header":
                # Return whatever regex found — never return empty header
                fallback = dict(EMPTY_SECTIONS["header"])
                fallback.update({k: v for k, v in regex_header.items() if v})
                name_guess = _extract_name_from_text(section_text)
                if name_guess and not fallback.get("name"):
                    fallback["name"] = name_guess
                return fallback
            return EMPTY_SECTIONS.get(section_name, [])