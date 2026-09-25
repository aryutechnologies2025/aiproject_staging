"""
deterministic_parser.py — Pure Algorithmic Abstract Syntax Tree (AST) Resume Parser.

Provides:
- 100% offline, zero-token, sub-50ms deterministic resume parsing.
- Robust entity association (Position + Company + Dates + Responsibilities).
- Multi-project description preservation and education credentials extraction.
- Serves as the high-availability safety fallback when Gemini API is unavailable or rate-limited.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.modules.resume_builder.schemas import (
    CanonicalResume,
    PersonalInformation,
    ExperienceItem,
    EducationItem,
    ProjectItem,
    CertificationItem,
)

logger = logging.getLogger("resume_builder.deterministic_parser")

# Regex patterns
EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
PHONE_REGEX = r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{3,5}[\s.-]?\d{0,5}"
URL_REGEX = r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in/[^\s,]+|github\.com/[^\s,]+|[a-zA-Z0-9-]+\.(?:com|io|org|dev|me)/[^\s,]*)"
DATE_RANGE_REGEX = r"(?i)(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)?\.?\s*\d{4})\s*(?:-|–|—|to)\s*(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)?\.?\s*\d{4}|present|current|ongoing|now)\b"

SECTION_PATTERNS = {
    "experience": r"(?i)^(?:work\s+)?experience|employment(?:\s+history)?|work\s+history|professional\s+experience|internships$",
    "education": r"(?i)^education|academic(?:\s+background)?|qualifications|academic\s+qualifications$",
    "skills": r"(?i)^technical\s+skills|skills(?:\s+&\s+tools)?|tech\s+stack|core\s+competencies|technologies(?:\s+used)?$",
    "summary": r"(?i)^professional\s+summary|summary|profile|about\s+me|executive\s+summary|objective$",
    "projects": r"(?i)^projects|key\s+projects|academic\s+projects|personal\s+projects$",
    "certifications": r"(?i)^certifications|certificates|licenses(?:\s+&\s+certifications)?$",
    "languages": r"(?i)^languages|language\s+proficiency$",
}

DEGREE_KEYWORDS = [
    "bachelor", "master", "phd", "b.tech", "m.tech", "b.e", "m.e", "b.sc", "m.sc",
    "b.s.", "b.s", "m.s.", "m.s", "b.a.", "b.a", "m.a.", "m.a", "bba", "mba", "bca", "mca",
    "diploma", "associate", "doctorate", "high school", "hr sec school", "secondary school",
    "matriculation", "engineering", "science", "arts", "degree",
]

JOB_TITLE_KEYWORDS = [
    "engineer", "developer", "architect", "manager", "lead", "analyst", "consultant",
    "specialist", "administrator", "officer", "intern", "internship", "associate", "director", "designer",
    "scientist", "programmer", "coordinator", "executive", "trainee", "founder", "lead",
]

COMPANY_INDICATORS = [
    "pvt ltd", "ltd", "inc", "corp", "llc", "enterprises", "infotech", "technologies",
    "solutions", "studios", "systems", "consulting", "group", "services", "labs", "agency",
]


class DeterministicResumeParser:
    """
    Algorithmic AST parser converting plain text or block structures into CanonicalResume.
    """

    @classmethod
    def parse_text(cls, raw_text: str) -> CanonicalResume:
        """
        Parse raw text into CanonicalResume structure deterministically.
        """
        if not raw_text or len(raw_text.strip()) < 10:
            return CanonicalResume()

        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        # 1. Extract Contact Information
        personal_info = cls._extract_personal_info(raw_text, lines)

        # 2. Segment into Sections
        section_map = cls._segment_sections(lines)

        # Also check for inline skill definitions
        skills = cls._parse_skills(section_map.get("skills", []))
        if not skills:
            for line in lines:
                if re.match(r"(?i)^(?:tech(?:nical)?\s+)?skills?\s*[:\-]", line):
                    inline_skills = re.sub(r"(?i)^(?:tech(?:nical)?\s+)?skills?\s*[:\-]\s*", "", line)
                    skills.extend([s.strip() for s in re.split(r"[,|•;/]", inline_skills) if s.strip()])
            skills = list(dict.fromkeys(skills))

        # 3. Parse Sections
        summary = cls._parse_summary(section_map.get("summary", []))
        experience = cls._parse_experience(section_map.get("experience", []))
        education = cls._parse_education(section_map.get("education", []))
        projects = cls._parse_projects(section_map.get("projects", []))
        certifications = cls._parse_certifications(section_map.get("certifications", []))
        languages = cls._parse_languages(section_map.get("languages", []))

        return CanonicalResume(
            personal_information=personal_info,
            summary=summary,
            skills=skills,
            experience=experience,
            education=education,
            projects=projects,
            certifications=certifications,
            languages=languages,
        )

    @classmethod
    def _extract_personal_info(cls, text: str, lines: List[str]) -> PersonalInformation:
        email_match = re.search(EMAIL_REGEX, text)
        email = email_match.group(0) if email_match else ""

        # Multi-candidate phone extraction: reject short 5-6 digit zip codes
        phone = ""
        phone_matches = re.finditer(PHONE_REGEX, text[:1500])
        candidates = []
        for m in phone_matches:
            cand = m.group(0).strip()
            digits = re.findall(r"\d", cand)
            if len(digits) >= 10 or (len(digits) >= 7 and cand.startswith("+")):
                candidates.append(cand)
        if candidates:
            candidates.sort(key=lambda c: (c.startswith("+"), len(re.findall(r"\d", c))), reverse=True)
            phone = candidates[0]

        urls = re.findall(URL_REGEX, text, re.IGNORECASE)
        unique_urls = list(dict.fromkeys(urls))
        link = ", ".join(unique_urls)

        name = ""
        title = ""
        candidate_lines = lines[:6]

        for line in candidate_lines:
            if "@" in line or "http" in line or any(re.match(p, line) for p in SECTION_PATTERNS.values()):
                continue
            if not name and len(line.split()) in (1, 2, 3, 4) and not any(c.isdigit() for c in line):
                name = line.title()
                continue
            if name and not title and any(kw in line.lower() for kw in JOB_TITLE_KEYWORDS):
                title = line
                continue

        location = ""
        loc_match = re.search(r"\b([A-Z][a-zA-Z\s]+,\s*[A-Z]{2}|[A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+(?:\s+\d{5,6})?)\b", text[:800])
        if loc_match:
            location = loc_match.group(0).strip()

        return PersonalInformation(
            name=name,
            title=title,
            email=email,
            phone=phone,
            location=location,
            link=link,
        )

    @classmethod
    def _segment_sections(cls, lines: List[str]) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = {}
        current_sec = "header"
        sections[current_sec] = []

        for line in lines:
            matched_sec = None
            clean_head = line.strip(" :-\t").lower()

            for sec_name, pattern in SECTION_PATTERNS.items():
                if re.match(pattern, clean_head):
                    matched_sec = sec_name
                    break

            if matched_sec:
                current_sec = matched_sec
                if current_sec not in sections:
                    sections[current_sec] = []
            else:
                sections[current_sec].append(line)

        return sections

    @classmethod
    def _parse_summary(cls, lines: List[str]) -> str:
        return " ".join(lines).strip()

    @classmethod
    def _parse_skills(cls, lines: List[str]) -> List[str]:
        raw_text = " ".join(lines)
        tokens = re.split(r"[,|•;/\n?]", raw_text)
        skills = []
        for t in tokens:
            cleaned = t.strip(" -:*•[]()?\t").strip()
            if ":" in cleaned:
                cleaned = cleaned.split(":")[-1].strip()
            if cleaned and len(cleaned) < 40 and not any(kw in cleaned.lower() for kw in ("skills", "tools", "competencies", "technologies")):
                skills.append(cleaned)
        return list(dict.fromkeys(skills))

    @classmethod
    def _parse_experience(cls, lines: List[str]) -> List[ExperienceItem]:
        items: List[ExperienceItem] = []
        cur: Optional[Dict[str, Any]] = None

        for line in lines:
            clean = line.strip()
            if not clean:
                continue

            clean_lower = clean.lower()
            date_match = re.search(DATE_RANGE_REGEX, clean)
            is_bullet = clean.startswith(("•", "-", "*", "–", "—", "·", "+"))
            is_title = any(kw in clean_lower for kw in JOB_TITLE_KEYWORDS) and not is_bullet
            is_comp = any(ci in clean_lower for ci in COMPANY_INDICATORS) and not is_bullet

            # A new experience item starts when a new job title is encountered
            if (is_title or (date_match and not cur)) and not is_bullet:
                if cur and (cur.get("position") or cur.get("bullets") or cur.get("description")):
                    items.append(ExperienceItem(**cur))

                clean_title = re.sub(DATE_RANGE_REGEX, "", clean).strip(" -|•,\t")
                from_yr, to_yr, ongoing = "", "", False
                if date_match:
                    from_yr = date_match.group(1).strip()
                    to_yr = date_match.group(2).strip()
                    ongoing = to_yr.lower() in ("present", "current", "ongoing", "now")

                cur = {
                    "position": clean_title if clean_title else "Professional",
                    "company": "",
                    "location": "",
                    "fromYear": from_yr,
                    "toYear": to_yr,
                    "isOngoing": ongoing,
                    "description": "",
                    "bullets": [],
                }
            elif cur and date_match and not cur["fromYear"]:
                cur["fromYear"] = date_match.group(1).strip()
                cur["toYear"] = date_match.group(2).strip()
                cur["isOngoing"] = cur["toYear"].lower() in ("present", "current", "ongoing", "now")
                rem = re.sub(DATE_RANGE_REGEX, "", clean).strip(" -|•,\t")
                if rem and not cur["company"]:
                    cur["company"] = rem
            elif cur and is_comp and not cur["company"]:
                parts = [p.strip() for p in re.split(r"[-•|–—]", clean) if p.strip()]
                cur["company"] = parts[0]
                if len(parts) > 1:
                    cur["location"] = parts[1]
            elif cur:
                cleaned_line = clean.strip(" •-*–—·\t")
                if is_bullet:
                    cur["bullets"].append(cleaned_line)
                else:
                    if not cur["company"] and len(cleaned_line.split()) <= 6 and not any(kw in cleaned_line.lower() for kw in ("with", "experienced", "skilled", "working", "responsible")):
                        cur["company"] = cleaned_line
                    elif cur["description"]:
                        cur["description"] += " " + cleaned_line
                    else:
                        cur["description"] = cleaned_line

        if cur and (cur.get("position") or cur.get("bullets") or cur.get("description")):
            items.append(ExperienceItem(**cur))

        return items

    @classmethod
    def _parse_education(cls, lines: List[str]) -> List[EducationItem]:
        items: List[EducationItem] = []
        cur: Optional[Dict[str, Any]] = None

        for line in lines:
            clean = line.strip()
            if not clean:
                continue

            clean_lower = clean.lower()
            is_deg = any(kw in clean_lower for kw in DEGREE_KEYWORDS) or "university" in clean_lower or "college" in clean_lower or "school" in clean_lower
            date_match = re.search(r"\b(19\d\d|20\d\d)\b", clean)

            if is_deg:
                if cur and (cur.get("degree") or cur.get("institution")):
                    items.append(EducationItem(**cur))

                yr = date_match.group(1) if date_match else ""
                clean_line = re.sub(r"\b(19\d\d|20\d\d)\b", "", clean).strip(" •-*–—|,/\t")
                degree_part = clean_line
                institution_part = ""

                for sep in [" | ", " - ", " – ", " — ", " at ", " from ", ", "]:
                    if sep in clean_line:
                        parts = clean_line.split(sep, 1)
                        p0, p1 = parts[0].strip(), parts[1].strip()
                        if any(kw in p0.lower() for kw in DEGREE_KEYWORDS):
                            degree_part, institution_part = p0, p1
                        elif any(kw in p1.lower() for kw in DEGREE_KEYWORDS):
                            degree_part, institution_part = p1, p0
                        break

                cur = {
                    "degree": degree_part,
                    "institution": institution_part or degree_part,
                    "location": "",
                    "fromYear": "",
                    "toYear": yr,
                }
            elif cur and date_match and not cur["toYear"]:
                cur["toYear"] = date_match.group(1)
            elif cur and not cur["institution"]:
                cur["institution"] = clean

        if cur and (cur.get("degree") or cur.get("institution")):
            items.append(EducationItem(**cur))

        return items

    @classmethod
    def _parse_projects(cls, lines: List[str]) -> List[ProjectItem]:
        projects: List[ProjectItem] = []
        cur: Optional[Dict[str, Any]] = None

        for line in lines:
            clean = line.strip()
            if not clean:
                continue

            is_bullet = clean.startswith(("•", "-", "*", "–", "—", "·", "+"))
            clean_text = clean.strip(" •-*–—·:\t")

            # Project title line heuristic: short line without trailing period
            if not is_bullet and len(clean.split()) <= 8 and len(clean) < 70 and not clean.endswith("."):
                if cur and (cur.get("title") and (cur.get("description") or cur.get("bullets"))):
                    projects.append(ProjectItem(**cur))
                cur = {
                    "title": clean_text,
                    "description": "",
                    "technologies": [],
                    "fromYear": "",
                    "toYear": "",
                    "bullets": [],
                }
            elif cur:
                if is_bullet:
                    cur["bullets"].append(clean_text)
                else:
                    if cur["description"]:
                        cur["description"] += " " + clean_text
                    else:
                        cur["description"] = clean_text

        if cur and (cur.get("title") and (cur.get("description") or cur.get("bullets"))):
            projects.append(ProjectItem(**cur))

        return projects

    @classmethod
    def _parse_certifications(cls, lines: List[str]) -> List[CertificationItem]:
        certs: List[CertificationItem] = []
        for line in lines:
            clean = line.strip(" •-*–—:\t")
            if clean:
                certs.append(CertificationItem(title=clean, issuer="", year=""))
        return certs

    @classmethod
    def _parse_languages(cls, lines: List[str]) -> List[str]:
        raw_text = " ".join(lines)
        tokens = re.split(r"[,|•;\n]", raw_text)
        return [t.strip(" -:*•[]()") for t in tokens if t.strip() and len(t.strip()) < 25]
