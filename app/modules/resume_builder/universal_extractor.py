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

        # Strip the email out before link-hunting so its local-part/domain
        # fragments (e.g. "alex.lee", "mail.com") never get misread as a
        # standalone portfolio/personal-site link.
        link_search_text = all_text
        if email:
            link_search_text = all_text.replace(email.group(0), " ")

        linkedin = re.search(r"linkedin\.com/in/[\w\-]+", link_search_text, re.IGNORECASE)
        github = re.search(r"github\.com/[\w\-]+", link_search_text, re.IGNORECASE)

        # Portfolio: anything that looks like a personal domain / hosted URL
        portfolio = re.search(
            r"(?:https?://)?(?:www\.)?[\w\-]+\.(?:netlify\.app|vercel\.app|github\.io|me|dev|io|app|co|net|com)"
            r"(?:/[\w\-./]*)?",
            link_search_text,
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
        Restricted to a single line so it can never accidentally span
        across unrelated lines/sections that happen to follow it.
        """
        for line in text.split("\n"):
            match = re.search(
                r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*),\s*([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*(?:\s+\d{5,6})?)\b",
                line,
            )
            if match:
                return match.group(0).strip()
        return ""

    @staticmethod
    def _clean_phone(raw: str) -> str:
        """Strip leading/trailing noise; keep digits, +, spaces, dashes, parens."""
        cleaned = re.sub(r"[^\d\+\-\(\)\s]", "", raw).strip()
        # Reject if fewer than 7 digits remain
        digits = re.sub(r"\D", "", cleaned)
        return cleaned if len(digits) >= 7 else ""

    # ------------------------------------------------------------------ #
    #  Deterministic section parsers (regex-only, zero LLM cost)          #
    #                                                                      #
    #  Each method below produces the SAME output schema that             #
    #  llm_section_parser.SECTION_PROMPTS would produce for that section, #
    #  so callers can swap a Gemini call for a deterministic call with    #
    #  zero changes downstream. Each method also returns a confidence     #
    #  score in [0, 100] reflecting how much of the input text it was     #
    #  able to confidently structure — low confidence signals that the   #
    #  caller should still invoke the LLM for that section.               #
    # ------------------------------------------------------------------ #

    DATE_RANGE_RE = re.compile(
        r"(?P<from>(?:\d{4})|(?:[A-Za-z]{3,9}\.?\s+\d{4}))\s*"
        r"(?:-|–|—|to)\s*"
        r"(?P<to>(?:\d{4})|(?:[A-Za-z]{3,9}\.?\s+\d{4})|present|current|ongoing|now)",
        re.IGNORECASE,
    )
    YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
    BULLET_PREFIX_RE = re.compile(r"^[\s]*[•\-\*\u2022\u2023\u25E6\u2043\u2219►▶→]+[\s]*")
    DEGREE_RE = re.compile(
        r"\b(bachelor|master|b\.?\s?tech|m\.?\s?tech|b\.?\s?e\.?|m\.?\s?e\.?|b\.?\s?sc|m\.?\s?sc|"
        r"phd|ph\.?d\.?|mba|bba|b\.?\s?com|m\.?\s?com|associate degree|diploma|"
        r"high school|gpa|cgpa)\b",
        re.IGNORECASE,
    )
    JOB_TITLE_HINT_RE = re.compile(
        r"\b(engineer|developer|manager|analyst|intern|consultant|director|"
        r"designer|architect|specialist|coordinator|administrator|officer|"
        r"executive|lead|head|nurse|teacher|professor|attorney|lawyer|"
        r"physician|doctor|accountant|technician|supervisor|associate|"
        r"scientist|researcher|representative)\b",
        re.IGNORECASE,
    )
    CERT_HINT_RE = re.compile(
        r"\b(certified|certificate|certification|license|licensed|credential|"
        r"AWS|Azure|PMP|CISSP|CCNA|Scrum|ITIL)\b",
        re.IGNORECASE,
    )
    LANGUAGE_NAMES = {
        "english", "spanish", "french", "german", "tamil", "hindi", "mandarin",
        "chinese", "japanese", "korean", "arabic", "portuguese", "italian",
        "russian", "bengali", "telugu", "kannada", "malayalam", "marathi",
        "gujarati", "punjabi", "urdu", "dutch", "swedish", "turkish", "vietnamese",
        "thai", "polish", "greek", "hebrew", "indonesian",
    }
    URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[\w\-]+\.[a-z]{2,}(?:/[\w\-./?=&%#]*)?", re.IGNORECASE)

    @staticmethod
    def _strip_bullet(line: str) -> str:
        return UniversalExtractor.BULLET_PREFIX_RE.sub("", line).strip()

    @staticmethod
    def _nonempty_lines(text: str) -> List[str]:
        return [l.strip() for l in text.split("\n") if l.strip()]

    # -- header ---------------------------------------------------------- #

    @staticmethod
    def deterministic_parse_header(text: str) -> Dict[str, Any]:
        """
        Header is rarely fully deterministic (title/name require judgement),
        so confidence is capped — this exists mainly to pre-fill fields and
        let llm_section_parser._patch_header shortcut when regex already
        found everything (email/phone/link), still calling the LLM only
        for the remaining ambiguous fields (name/title) when needed.
        """
        email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
        phone_match = re.search(
            r"(?:\+\d{1,3}[\s.\-]?)?(?:\(?\d{2,5}\)?[\s.\-]?)?\d{3,5}[\s.\-]?\d{3,5}(?:[\s.\-]?\d{2,5})?",
            text,
        )
        # Strip the email out before URL-hunting so its local-part/domain
        # fragments ("john.smith", "email.com") never get misread as links.
        link_search_text = text
        if email_match:
            link_search_text = text.replace(email_match.group(0), " ")

        links = []
        for m in UniversalExtractor.URL_RE.finditer(link_search_text):
            val = m.group(0).strip().rstrip(".,;)")
            if "@" in val:
                continue
            if val not in links:
                links.append(val)

        location_match = re.search(
            r"\b([A-Z][a-zA-Z\s\-]+),\s*([A-Z][a-zA-Z\s]{2,}(?:\s+\d{5,6})?)\b", text
        )

        name = UniversalExtractor._extract_name_heuristic(
            [{"text": l, "type": "text", "page": 1} for l in UniversalExtractor._nonempty_lines(text)]
        )

        result = {
            "name": name,
            "title": "",
            "location": location_match.group(0).strip() if location_match else "",
            "email": email_match.group(0).strip() if email_match else "",
            "phone": UniversalExtractor._clean_phone(phone_match.group(0)) if phone_match else "",
            "link": ", ".join(links),
        }

        # Confidence: header is only "safe to skip the LLM" if we found
        # email/phone AND a plausible name AND a title-shaped second line.
        score = 0
        if result["email"]:
            score += 30
        if result["phone"]:
            score += 20
        if result["name"]:
            score += 20
        if result["link"]:
            score += 10
        # Title detection: look for a short line with a job-title hint
        # near the top that isn't the name line.
        for line in UniversalExtractor._nonempty_lines(text)[:6]:
            if line == name:
                continue
            if UniversalExtractor.JOB_TITLE_HINT_RE.search(line) and len(line.split()) <= 8:
                result["title"] = line
                score += 20
                break

        return {"data": result, "confidence": min(score, 100)}

    # -- summary ----------------------------------------------------------#

    @staticmethod
    def deterministic_parse_summary(text: str) -> Dict[str, Any]:
        """
        Summary is free-form prose — true paraphrase/condensation needs an
        LLM. Deterministic pass only cleans bullet markers and joins lines;
        confidence is intentionally capped well below the skip threshold
        unless the section is trivially short (e.g. already one paragraph).
        """
        lines = [UniversalExtractor._strip_bullet(l) for l in UniversalExtractor._nonempty_lines(text)]
        joined = " ".join(lines).strip()
        data = {"summary": joined}

        # Only confident when content is already a single clean paragraph
        # (one line, or lines that look like simple wrapped prose).
        score = 0
        if joined:
            score = 40
            if len(lines) <= 2:
                score = 75
        return {"data": data, "confidence": score}

    # -- education --------------------------------------------------------#

    @staticmethod
    def deterministic_parse_education(text: str) -> Dict[str, Any]:
        lines = UniversalExtractor._nonempty_lines(text)
        entries: List[Dict[str, str]] = []
        matched_lines = 0

        def _years_from(line: str):
            date_m = UniversalExtractor.DATE_RANGE_RE.search(line)
            if date_m:
                fy = re.sub(r"\D", "", date_m.group("from"))[:4] or date_m.group("from")
                to_val = date_m.group("to")
                ty = "" if to_val.lower() in ("present", "current", "ongoing", "now") else (
                    re.sub(r"\D", "", to_val)[:4] or to_val
                )
                return fy, ty
            all_years = re.findall(r"\b(?:19|20)\d{2}\b", line)
            if all_years:
                return all_years[0], (all_years[1] if len(all_years) > 1 else "")
            return "", ""

        i = 0
        while i < len(lines):
            line = lines[i]
            if UniversalExtractor.DEGREE_RE.search(line):
                matched_lines += 1
                from_year, to_year = _years_from(line)

                # institution: next non-degree line (consumed as lookahead),
                # but only if it doesn't itself look like a bare date line.
                institution = ""
                location = ""
                consumed = 0
                if "," in line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        location = parts[-1]
                elif i + 1 < len(lines) and not UniversalExtractor.DEGREE_RE.search(lines[i + 1]):
                    candidate = lines[i + 1]
                    cand_years = re.findall(r"\b(?:19|20)\d{2}\b", candidate)
                    # Treat as institution unless it's predominantly a date
                    # range (e.g. "2016 - 2020" on its own line).
                    if not (cand_years and len(candidate.split()) <= 5):
                        institution = candidate
                        matched_lines += 1
                        consumed = 1

                # If no year found yet, check the institution line and the
                # line immediately after it for a standalone date range.
                if not from_year:
                    for lookahead_idx in (i + 1, i + 1 + consumed):
                        if lookahead_idx < len(lines):
                            fy, ty = _years_from(lines[lookahead_idx])
                            if fy:
                                from_year, to_year = fy, ty
                                break

                i += consumed
                degree_clean = UniversalExtractor.DATE_RANGE_RE.sub("", line).strip(" ,-")
                entries.append({
                    "degree": degree_clean,
                    "institution": institution,
                    "location": location,
                    "fromYear": from_year,
                    "toYear": to_year,
                })
            i += 1

        score = 0
        if lines:
            score = int((matched_lines / len(lines)) * 100)
            if entries:
                score = max(score, 70)
        return {"data": entries, "confidence": score}

    # -- experience ---------------------------------------------------------#

    @staticmethod
    def deterministic_parse_experience(text: str) -> Dict[str, Any]:
        lines = UniversalExtractor._nonempty_lines(text)
        entries: List[Dict[str, Any]] = []
        current = None
        title_lines = 0

        for raw_line in lines:
            is_bullet = bool(UniversalExtractor.BULLET_PREFIX_RE.match(raw_line))
            line = UniversalExtractor._strip_bullet(raw_line)

            if not is_bullet and UniversalExtractor.JOB_TITLE_HINT_RE.search(line) and len(line.split()) <= 10:
                if current:
                    entries.append(current)
                title_lines += 1

                date_m = UniversalExtractor.DATE_RANGE_RE.search(line)
                from_year, to_year, is_ongoing = "", "", False
                if date_m:
                    is_ongoing = date_m.group("to").lower() in ("present", "current", "ongoing", "now")
                    all_years = re.findall(r"\b(?:19|20)\d{2}\b", line)
                    if all_years:
                        from_year = all_years[0]
                        to_year = "" if is_ongoing else (all_years[1] if len(all_years) > 1 else "")

                position = UniversalExtractor.DATE_RANGE_RE.sub("", line).strip(" ,-|")
                company = ""
                # Common single-line format: "Title | Company" or "Title - Company"
                if "|" in position:
                    parts = [p.strip() for p in position.split("|") if p.strip()]
                    if len(parts) >= 2:
                        position, company = parts[0], parts[1]

                current = {
                    "position": position,
                    "company": company,
                    "location": "",
                    "fromYear": from_year,
                    "toYear": to_year,
                    "isOngoing": is_ongoing,
                    "description": [],
                    "bullets": [],
                }
                continue

            if current and not is_bullet and not current["company"] and len(line.split()) <= 8:
                current["company"] = line
                continue

            if current and is_bullet:
                current["bullets"].append(line)

        if current:
            entries.append(current)

        score = 0
        if entries:
            # Confident only if every entry has a company AND at least one bullet
            complete = sum(1 for e in entries if e["company"] and e["bullets"])
            score = int((complete / len(entries)) * 100)
        return {"data": entries, "confidence": score}

    # -- skills -------------------------------------------------------------#

    @staticmethod
    def deterministic_parse_skills(text: str) -> Dict[str, Any]:
        # Strip "Category: " prefixes (e.g. "Languages: Python, Java")
        cleaned_lines = []
        for line in UniversalExtractor._nonempty_lines(text):
            line = re.sub(r"^[A-Za-z /]{2,30}:\s*", "", line)
            cleaned_lines.append(line)
        joined = "\n".join(cleaned_lines)

        parts = re.split(r"[,|\u2022\n•\-]", joined)
        skills = [p.strip() for p in parts if p.strip() and len(p.strip()) <= 40]
        unique = sorted(list(dict.fromkeys(skills)))

        score = 0
        if unique:
            # High confidence when items are short tokens (skill-shaped),
            # low confidence if lines look like prose (long avg length).
            avg_len = sum(len(s) for s in unique) / len(unique)
            score = 85 if avg_len <= 25 else 40
        return {"data": unique, "confidence": score}

    # -- certifications ------------------------------------------------------#

    @staticmethod
    def deterministic_parse_certifications(text: str) -> Dict[str, Any]:
        lines = UniversalExtractor._nonempty_lines(text)
        entries = []
        matched = 0
        for line in lines:
            clean = UniversalExtractor._strip_bullet(line)
            if not clean:
                continue
            years = re.findall(r"\b(?:19|20)\d{2}\b", clean)
            year = years[-1] if years else ""
            title = clean
            issuer = ""
            if "-" in clean:
                left, _, right = clean.partition("-")
                title, issuer = left.strip(), right.strip()
            elif "," in clean:
                left, _, right = clean.partition(",")
                title, issuer = left.strip(), right.strip()
            entries.append({"title": title, "issuer": issuer, "year": year})
            if UniversalExtractor.CERT_HINT_RE.search(clean) or year:
                matched += 1

        score = 0
        if lines:
            score = int((matched / len(lines)) * 100) if lines else 0
            if entries:
                score = max(score, 60)
        return {"data": entries, "confidence": score}

    # -- languages ------------------------------------------------------------#

    @staticmethod
    def deterministic_parse_languages(text: str) -> Dict[str, Any]:
        parts = re.split(r"[,|\u2022\n•\-]", text)
        found = []
        matched = 0
        total = 0
        for p in parts:
            token = p.strip()
            if not token:
                continue
            total += 1
            # Take leading word(s) as the language name, ignore proficiency
            # qualifiers like "(Fluent)" or "- Native".
            name_token = re.split(r"[\(\-:]", token)[0].strip()
            if name_token.lower() in UniversalExtractor.LANGUAGE_NAMES:
                found.append(name_token)
                matched += 1

        unique = sorted(list(dict.fromkeys(found)))
        score = 0
        if total:
            score = int((matched / total) * 100)
        return {"data": unique, "confidence": score}

    # -- projects (low determinism, kept conservative) ------------------------#

    @staticmethod
    def deterministic_parse_projects(text: str) -> Dict[str, Any]:
        lines = UniversalExtractor._nonempty_lines(text)
        entries: List[Dict[str, Any]] = []
        current = None

        for raw_line in lines:
            is_bullet = bool(UniversalExtractor.BULLET_PREFIX_RE.match(raw_line))
            line = UniversalExtractor._strip_bullet(raw_line)

            if not is_bullet and len(line.split()) <= 8 and not current:
                current = {
                    "title": line, "description": "", "technologies": [],
                    "fromYear": "", "toYear": "", "bullets": [],
                }
                continue
            if not is_bullet and len(line.split()) <= 8 and current and current["bullets"]:
                entries.append(current)
                current = {
                    "title": line, "description": "", "technologies": [],
                    "fromYear": "", "toYear": "", "bullets": [],
                }
                continue
            if current and is_bullet:
                current["bullets"].append(line)
            elif current and not current["description"]:
                current["description"] = line

        if current:
            entries.append(current)

        # Projects free-text is structurally ambiguous; never claim high
        # confidence — always prefer the LLM unless trivially simple.
        score = 30 if entries else 0
        return {"data": entries, "confidence": score}

    # -- other ------------------------------------------------------------#

    @staticmethod
    def deterministic_parse_other(text: str) -> Dict[str, Any]:
        lines = [UniversalExtractor._strip_bullet(l) for l in UniversalExtractor._nonempty_lines(text)]
        # "Other" is a catch-all grab-bag; never confidently structured.
        return {"data": lines, "confidence": 30 if lines else 0}

    # -- dispatch table ---------------------------------------------------#

    DETERMINISTIC_PARSERS = {
        "header": "deterministic_parse_header",
        "summary": "deterministic_parse_summary",
        "education": "deterministic_parse_education",
        "experience": "deterministic_parse_experience",
        "skills": "deterministic_parse_skills",
        "certifications": "deterministic_parse_certifications",
        "languages": "deterministic_parse_languages",
        "projects": "deterministic_parse_projects",
        "other": "deterministic_parse_other",
    }

    @staticmethod
    def deterministic_parse_section(section_name: str, text: str) -> Dict[str, Any]:
        """
        Single entry point: dispatch to the right deterministic parser and
        return {"data": <schema-matching value>, "confidence": int 0-100}.
        Unknown section names fall back to zero confidence (always LLM).
        """
        if not text or not text.strip():
            return {"data": None, "confidence": 0}
        method_name = UniversalExtractor.DETERMINISTIC_PARSERS.get(section_name)
        if not method_name:
            return {"data": None, "confidence": 0}
        method = getattr(UniversalExtractor, method_name)
        try:
            return method(text)
        except Exception as e:
            logger.warning(f"Deterministic parse failed for {section_name}: {e}")
            return {"data": None, "confidence": 0}

