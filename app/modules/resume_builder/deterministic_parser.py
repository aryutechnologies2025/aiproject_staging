"""
deterministic_parser.py — Pure Algorithmic Abstract Syntax Tree (AST) Resume Parser.

Provides:
- 100% offline, zero-token, sub-50ms deterministic resume parsing.
- Section segmentation, contact extraction, date parsing, and entity classification.
- Serves as the high-availability safety fallback when Gemini API is unavailable or rate-limited.
"""

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
    "experience": r"(?i)^(?:work\s+)?experience|employment(?:\s+history)?|work\s+history|professional\s+experience$",
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
    "diploma", "associate", "doctorate", "high school", "engineering", "science", "arts", "degree",
]

JOB_TITLE_KEYWORDS = [
    "engineer", "developer", "architect", "manager", "lead", "analyst", "consultant",
    "specialist", "administrator", "officer", "intern", "associate", "director", "designer",
    "scientist", "programmer", "coordinator", "executive",
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

        # Also check for inline skill definitions (e.g. "Skills: Python, FastAPI...")
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

        phone_match = re.search(PHONE_REGEX, text)
        phone = phone_match.group(0).strip() if phone_match else ""
        if len(re.findall(r"\d", phone)) < 7:
            phone = ""

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
        loc_match = re.search(r"\b([A-Z][a-zA-Z\s]+,\s*[A-Z]{2}|[A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)\b", text[:800])
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
        tokens = re.split(r"[,|•;/\n]", raw_text)
        skills = []
        for t in tokens:
            cleaned = t.strip(" -:*•[]()").strip()
            if ":" in cleaned:
                cleaned = cleaned.split(":")[-1].strip()
            if cleaned and len(cleaned) < 40 and not any(kw in cleaned.lower() for kw in ("skills", "tools", "competencies", "technologies")):
                skills.append(cleaned)
        return list(dict.fromkeys(skills))

    @classmethod
    def _parse_experience(cls, lines: List[str]) -> List[ExperienceItem]:
        items: List[ExperienceItem] = []
        current_exp: Optional[Dict[str, Any]] = None

        for line in lines:
            date_match = re.search(DATE_RANGE_REGEX, line)
            is_job_title = any(kw in line.lower() for kw in JOB_TITLE_KEYWORDS)
            is_bullet = line.startswith(("•", "-", "*", "–", "—", "+")) or (current_exp and len(line.split()) > 7)

            if date_match or (is_job_title and not is_bullet):
                if current_exp and (current_exp.get("position") or current_exp.get("bullets")):
                    items.append(ExperienceItem(**current_exp))

                from_yr, to_yr = "", ""
                is_ongoing = False
                if date_match:
                    from_yr = date_match.group(1).strip()
                    to_yr = date_match.group(2).strip()
                    is_ongoing = to_yr.lower() in ("present", "current", "ongoing", "now")

                clean_title = re.sub(DATE_RANGE_REGEX, "", line).strip(" -|•,\t")
                current_exp = {
                    "position": clean_title if clean_title else "Professional",
                    "company": "",
                    "location": "",
                    "fromYear": from_yr,
                    "toYear": to_yr,
                    "isOngoing": is_ongoing,
                    "bullets": [],
                }
            elif current_exp:
                cleaned_line = line.strip(" •-*–—\t")
                if cleaned_line:
                    current_exp["bullets"].append(cleaned_line)

        if current_exp and (current_exp.get("position") or current_exp.get("bullets")):
            items.append(ExperienceItem(**current_exp))

        return items

    @classmethod
    def _parse_education(cls, lines: List[str]) -> List[EducationItem]:
        items: List[EducationItem] = []
        for line in lines:
            line_lower = line.lower()
            is_degree = any(kw in line_lower for kw in DEGREE_KEYWORDS) or "university" in line_lower or "college" in line_lower
            date_match = re.search(r"\b(19\d\d|20\d\d)\b", line)
            if is_degree:
                yr = date_match.group(1) if date_match else ""
                clean_degree = line.strip(" •-*–—\t")
                items.append(
                    EducationItem(
                        degree=clean_degree,
                        institution="",
                        location="",
                        fromYear="",
                        toYear=yr,
                    )
                )
        return items

    @classmethod
    def _parse_projects(cls, lines: List[str]) -> List[ProjectItem]:
        projects: List[ProjectItem] = []
        current_proj: Optional[Dict[str, Any]] = None

        for line in lines:
            if line.startswith(("•", "-", "*")):
                if current_proj:
                    current_proj["bullets"].append(line.strip(" •-*–—\t"))
            else:
                if current_proj:
                    projects.append(ProjectItem(**current_proj))
                current_proj = {
                    "title": line.strip(" •-*–—:\t"),
                    "description": "",
                    "technologies": [],
                    "fromYear": "",
                    "toYear": "",
                    "bullets": [],
                }

        if current_proj:
            projects.append(ProjectItem(**current_proj))

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
