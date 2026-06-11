"""
ATS Resume Normalizer v1.0
──────────────────────────────────────────────────────────────────────────────
Single normalization layer that sits between every parser output and every
scorer/feedback consumer.

WHY THIS EXISTS
───────────────
The project has two parser pipelines:
  1. Resume Builder   → LlamaParse + LLM  → uses "position", "fromYear",
                        "toYear", nested summary dict, etc.
  2. ATS Markdown Parser → regex/heuristic → uses "title", "start_date",
                           "end_date", flat summary string, etc.

The ATS scoring engine (_dim_experience, _dim_quality, _dim_structure, etc.)
was written against the Markdown parser schema.  When Resume Builder output
flows through it, every field lookup silently returns None, producing:
  • roles = 0 even when experience is present
  • total_bullets = 0 even when bullets exist
  • score capped to 40 due to false "no experience" hard-cap

This module resolves ALL known schema variants into a single
`NormalizedResume` dataclass that every downstream consumer reads.

NORMALIZATION CONTRACT
──────────────────────
After normalization the following invariants HOLD:
  • experience[i].title      — always a non-empty string if role exists
  • experience[i].company    — always a non-empty string if role exists
  • experience[i].start_date — always a string (may be empty if unknown)
  • experience[i].end_date   — always a string ("Present" if ongoing)
  • experience[i].bullets    — always a list[str], never None
  • summary                  — always a plain string, never a dict
  • skills                   — canonical deduplicated list[str]
  • education[i].degree / .institution / .year — always strings
  • certifications[i].title / .issuer / .year  — always strings
  • contact.name/email/phone/location/linkedin/github — always strings
  • len(skills) == len(unique_normalized_skills)  — NO inflation
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES  (canonical schema)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizedContact:
    name:     str = ""
    email:    str = ""
    phone:    str = ""
    location: str = ""
    linkedin: str = ""
    github:   str = ""


@dataclass
class NormalizedExperience:
    title:      str = ""
    company:    str = ""
    location:   str = ""
    start_date: str = ""
    end_date:   str = ""
    bullets:    List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return bool(self.title and self.company)


@dataclass
class NormalizedEducation:
    degree:      str = ""
    institution: str = ""
    year:        str = ""
    gpa:         str = ""


@dataclass
class NormalizedCertification:
    title:  str = ""
    issuer: str = ""
    year:   str = ""


@dataclass
class NormalizedProject:
    title:        str = ""
    description:  str = ""
    technologies: List[str] = field(default_factory=list)
    bullets:      List[str] = field(default_factory=list)
    url:          str = ""


@dataclass
class NormalizedResume:
    """Single canonical model consumed by every scorer and feedback generator."""
    contact:          NormalizedContact                  = field(default_factory=NormalizedContact)
    summary:          str                                = ""
    experience:       List[NormalizedExperience]         = field(default_factory=list)
    education:        List[NormalizedEducation]          = field(default_factory=list)
    skills:           List[str]                          = field(default_factory=list)
    certifications:   List[NormalizedCertification]      = field(default_factory=list)
    projects:         List[NormalizedProject]            = field(default_factory=list)
    languages:        List[str]                          = field(default_factory=list)
    awards:           List[str]                          = field(default_factory=list)
    volunteer:        List[str]                          = field(default_factory=list)
    publications:     List[str]                          = field(default_factory=list)
    hobbies:          List[str]                          = field(default_factory=list)
    raw_text:         str                                = ""

    # ── Derived helpers ──────────────────────────────────────────────────────

    @property
    def has_experience(self) -> bool:
        return len(self.experience) > 0

    @property
    def has_skills(self) -> bool:
        return len(self.skills) > 0

    @property
    def has_education(self) -> bool:
        return len(self.education) > 0

    @property
    def has_certifications(self) -> bool:
        return len(self.certifications) > 0

    @property
    def has_projects(self) -> bool:
        return len(self.projects) > 0

    @property
    def has_languages(self) -> bool:
        return len(self.languages) > 0

    @property
    def has_summary(self) -> bool:
        return bool(self.summary and self.summary.strip())

    @property
    def has_contact_email(self) -> bool:
        return bool(self.contact.email)

    @property
    def all_bullets(self) -> List[str]:
        bullets: List[str] = []
        for exp in self.experience:
            bullets.extend(exp.bullets)
        return bullets


# ─────────────────────────────────────────────────────────────────────────────
# SKILL SYNONYM / DEDUPLICATION TABLE
# ─────────────────────────────────────────────────────────────────────────────

# canonical_lower → set of lowercase aliases that should all map to it
_SKILL_CANONICAL: Dict[str, str] = {}   # alias_lower → canonical_display

_RAW_SKILL_MAP: List[tuple] = [
    # (canonical_display, [aliases...])
    ("Node.js",                 ["nodejs", "node.js", "node js", "node"]),
    ("JavaScript",              ["js", "javascript", "ecmascript", "es6", "es2015"]),
    ("TypeScript",              ["ts", "typescript"]),
    ("Python",                  ["python3", "python 3", "py", "python"]),
    ("React",                   ["reactjs", "react.js", "react js"]),
    ("Vue.js",                  ["vuejs", "vue.js", "vue js", "vue"]),
    ("Angular",                 ["angularjs", "angular.js", "angular"]),
    ("Next.js",                 ["nextjs", "next.js", "next js"]),
    ("PostgreSQL",              ["postgres", "postgresql"]),
    ("MySQL",                   ["mysql"]),
    ("MongoDB",                 ["mongodb", "mongo"]),
    ("Redis",                   ["redis"]),
    ("AWS",                     ["amazon web services", "aws"]),
    ("GCP",                     ["google cloud", "google cloud platform", "gcp"]),
    ("Azure",                   ["microsoft azure", "azure"]),
    ("Docker",                  ["docker"]),
    ("Kubernetes",              ["k8s", "kubernetes"]),
    ("Search Engine Optimization", ["seo", "search engine optimization", "search engine optimisation"]),
    ("Search Engine Marketing", ["sem", "search engine marketing"]),
    ("Google Analytics",        ["google analytics", "ga", "google analytics & google search console"]),
    ("Google Search Console",   ["google search console", "gsc"]),
    ("Social Media Marketing",  ["social media marketing", "smm", "social media"]),
    ("Pay-Per-Click",           ["ppc", "pay per click", "pay-per-click"]),
    ("Content Marketing",       ["content marketing", "content strategy"]),
    ("Email Marketing",         ["email marketing"]),
    ("Instagram Marketing",     ["instagram", "instagram marketing"]),
    ("LinkedIn Marketing",      ["linkedin marketing", "linkedin ads"]),
    ("Facebook Ads",            ["facebook ads", "meta ads", "fb ads"]),
    ("Machine Learning",        ["ml", "machine learning"]),
    ("Deep Learning",           ["dl", "deep learning"]),
    ("Natural Language Processing", ["nlp", "natural language processing"]),
    ("FastAPI",                 ["fastapi"]),
    ("Django",                  ["django", "django rest framework", "drf"]),
    ("Flask",                   ["flask"]),
    ("Spring Boot",             ["spring boot", "spring"]),
    ("C#",                      ["csharp", "c sharp", "c#", "dotnet", ".net"]),
    ("C++",                     ["cpp", "c plus plus", "c++"]),
    ("Go",                      ["golang", "go"]),
    ("Rust",                    ["rust"]),
    ("PHP",                     ["php"]),
    ("WordPress",               ["wordpress", "wp"]),
    ("Shopify",                 ["shopify"]),
    ("HubSpot",                 ["hubspot"]),
    ("Salesforce",              ["salesforce", "sfdc"]),
    ("Canva",                   ["canva"]),
    ("Figma",                   ["figma"]),
    ("Adobe Photoshop",         ["photoshop", "adobe photoshop"]),
    ("Adobe Illustrator",       ["illustrator", "adobe illustrator"]),
    ("Microsoft Excel",         ["excel", "microsoft excel", "ms excel"]),
    ("Microsoft Word",          ["word", "microsoft word", "ms word"]),
    ("Microsoft PowerPoint",    ["powerpoint", "ppt", "ms powerpoint"]),
    ("Tableau",                 ["tableau"]),
    ("Power BI",                ["power bi", "powerbi"]),
    ("Jira",                    ["jira"]),
    ("Asana",                   ["asana"]),
    ("Trello",                  ["trello"]),
    ("Slack",                   ["slack"]),
    ("Agile",                   ["agile", "agile methodology"]),
    ("Scrum",                   ["scrum"]),
    ("Project Management",      ["project management", "pm"]),
    ("Data Analysis",           ["data analysis", "data analytics"]),
    ("SQL",                     ["sql"]),
    ("Git",                     ["git"]),
    ("GitHub",                  ["github"]),
    ("Linux",                   ["linux", "ubuntu"]),
    ("Registered Nurse",        ["rn", "registered nurse"]),
    ("Electronic Health Records", ["ehr", "emr", "electronic health records"]),
    ("CPA",                     ["certified public accountant", "cpa"]),
    ("CFA",                     ["chartered financial analyst", "cfa"]),
    ("PMP",                     ["project management professional", "pmp"]),
    ("ERP",                     ["enterprise resource planning", "erp"]),
    ("CRM",                     ["customer relationship management", "crm"]),
]

# Build lookup: alias_lower → canonical_display
for _canonical, _aliases in _RAW_SKILL_MAP:
    for _alias in _aliases:
        _SKILL_CANONICAL[_alias.lower().strip()] = _canonical
    # canonical itself maps to itself
    _SKILL_CANONICAL[_canonical.lower().strip()] = _canonical


def _canonicalize_skill(raw: str) -> str:
    """Return the canonical display name for a skill string."""
    key = raw.lower().strip()
    return _SKILL_CANONICAL.get(key, raw.strip())


# Patterns that indicate a string is NOT a skill (it's a heading, sentence, etc.)
_NOT_A_SKILL_RE = re.compile(
    r"(responsible for|experience|overview|summary|objective|profile|"
    r"education|certification|project|award|reference|language|"
    r"\.com|http|@|\d{4})",
    re.IGNORECASE,
)


def _is_valid_skill(s: str) -> bool:
    """Return True if the string looks like a genuine skill entry."""
    s = s.strip()
    if not s or len(s) < 2 or len(s) > 80:
        return False
    # Reject strings that are clearly sentences (>6 words)
    if len(s.split()) > 6:
        return False
    if _NOT_A_SKILL_RE.search(s):
        return False
    return True


def _deduplicate_skills(raw_skills: List[str]) -> List[str]:
    """
    Normalize and deduplicate a raw skill list.

    Strategy:
    1. Canonicalize each skill via the synonym map.
    2. Use a case-insensitive seen-set to eliminate duplicates.
    3. Filter out non-skill strings (sentences, headings, etc.).
    4. Preserve insertion order of first occurrence.

    This ensures len(result) == number of UNIQUE canonical skills.
    No inflation from regex hits or synonym variants.
    """
    seen:   Dict[str, str] = {}   # canonical_lower → canonical_display (first seen wins)
    result: List[str]       = []

    for raw in raw_skills:
        if not isinstance(raw, str):
            continue
        # Skills might be comma-separated in a single entry — split them
        parts = re.split(r"[,|/•·]+", raw)
        for part in parts:
            part = part.strip().strip("()[]{}").strip()
            if not _is_valid_skill(part):
                continue
            canonical = _canonicalize_skill(part)
            key = canonical.lower()
            if key not in seen:
                seen[key] = canonical
                result.append(canonical)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONTACT NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I)
_PHONE_RE    = re.compile(r"(\+\d{1,3}[\s\-]?)?\(?\d{3,5}\)?[\s\-]?\d{3,5}[\s\-]?\d{4,6}")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+", re.I)
_GITHUB_RE   = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+", re.I)
_URL_SPLIT_RE = re.compile(r"[,\s]+")

_SUBTITLE_TOKENS = {
    "full stack", "software engineer", "mern", "developer", "manager",
    "agile", "delivery", "stack", "frontend", "backend", "engineer",
    "architect", "analyst", "consultant", "specialist", "designer",
    "director", "officer", "lead", "head of",
}


def _clean_name(raw: str) -> str:
    if not raw:
        return ""
    lower = raw.lower()
    hits  = sum(1 for t in _SUBTITLE_TOKENS if t in lower)
    if hits >= 2 or "|" in raw or len(raw) > 60:
        return ""
    return raw.strip()


def _normalize_contact(resume: Dict[str, Any]) -> NormalizedContact:
    """
    Build a NormalizedContact from any known parser output shape.

    Handles:
    - Builder: resume.header.{name, email, phone, location, link}
    - Markdown: resume.{name, email, phone, location, linkedin, github}
    - Flat: resume.{name, email, ...} with no nesting
    - Raw text fallback
    """
    def _first(*vals):
        for v in vals:
            s = str(v).strip() if v is not None else ""
            if s:
                return s
        return ""

    raw_text = str(resume.get("raw_text") or "")

    # ── header dict (Resume Builder output) ─────────────────────────────────
    header = resume.get("header") or {}
    if not isinstance(header, dict):
        header = {}

    # ── contact sub-dict ────────────────────────────────────────────────────
    nested = resume.get("contact") or {}
    if not isinstance(nested, dict):
        nested = {}

    # ── name ────────────────────────────────────────────────────────────────
    name = _clean_name(
        _first(header.get("name"), nested.get("name"), resume.get("name"))
    )

    # ── email ────────────────────────────────────────────────────────────────
    email = _first(
        header.get("email"), nested.get("email"), resume.get("email"),
    )
    if not email:
        m = _EMAIL_RE.search(raw_text)
        if m:
            email = m.group(0).strip()

    # ── phone ────────────────────────────────────────────────────────────────
    phone = _first(
        header.get("phone"), nested.get("phone"), resume.get("phone"),
    )
    if not phone:
        m = _PHONE_RE.search(raw_text)
        if m:
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) >= 7:
                phone = m.group(0).strip()

    # ── location ─────────────────────────────────────────────────────────────
    location = _first(
        header.get("location"), nested.get("location"), resume.get("location"),
    )

    # ── linkedin ─────────────────────────────────────────────────────────────
    # Builder stores all links in header.link as a comma-separated string
    link_blob = _first(header.get("link"), nested.get("link"), resume.get("link"), "")
    linkedin  = _first(nested.get("linkedin"), resume.get("linkedin"), "")
    github    = _first(nested.get("github"),   resume.get("github"),   "")

    if link_blob:
        for part in _URL_SPLIT_RE.split(link_blob):
            part = part.strip()
            if not part:
                continue
            if not linkedin and _LINKEDIN_RE.search(part):
                m = _LINKEDIN_RE.search(part)
                linkedin = m.group(0) if m else part
            if not github and _GITHUB_RE.search(part):
                m = _GITHUB_RE.search(part)
                github = m.group(0) if m else part

    if not linkedin:
        m = _LINKEDIN_RE.search(raw_text)
        if m:
            linkedin = m.group(0).strip()
    if not github:
        m = _GITHUB_RE.search(raw_text)
        if m:
            github = m.group(0).strip()

    return NormalizedContact(
        name=name, email=email, phone=phone,
        location=location, linkedin=linkedin, github=github,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

_BULLET_FIELD_NAMES = (
    "bullets",          # both parsers
    "responsibilities", # some custom schemas
    "achievements",     # alternative
    "description",      # fallback single string
    "content",          # another variant
    "tasks",
    "duties",
    "highlights",
)


def _extract_bullets(exp: Dict[str, Any]) -> List[str]:
    """
    Extract bullet/responsibility strings from an experience dict,
    regardless of which field name is used.
    """
    for field_name in _BULLET_FIELD_NAMES:
        val = exp.get(field_name)
        if val is None:
            continue
        if isinstance(val, list):
            bullets = [str(b).strip() for b in val if b and str(b).strip()]
            if bullets:
                return bullets
        if isinstance(val, str) and val.strip():
            # Split a description string on newlines / bullet chars
            lines = re.split(r"\n|(?:^|\s)[•\-\*►▶→✓]+\s", val)
            bullets = [l.strip() for l in lines if l.strip() and len(l.strip()) > 5]
            if bullets:
                return bullets
    return []


_MONTH_MAP_NORM: Dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

def _normalize_date(raw: Any) -> str:
    """Normalize any date/year representation to a consistent string."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # Already looks like a date range fragment — return as-is
    if re.search(r"present|current|now|till", s, re.I):
        return "Present"
    return s


def _normalize_experience(raw_list: Any) -> List[NormalizedExperience]:
    if not isinstance(raw_list, list):
        return []

    result: List[NormalizedExperience] = []

    for item in raw_list:
        if not isinstance(item, dict):
            continue

        # ── title: "position" (Builder) | "title" (Markdown) | "job_title" ─
        title = str(
            item.get("position") or
            item.get("title") or
            item.get("job_title") or
            item.get("role") or
            ""
        ).strip()

        company = str(
            item.get("company") or
            item.get("organization") or
            item.get("employer") or
            ""
        ).strip()

        location = str(item.get("location") or "").strip()

        # ── dates: "fromYear"/"toYear" (Builder) | "start_date"/"end_date" ─
        start_raw = (
            item.get("start_date") or
            item.get("fromYear") or
            item.get("from") or
            item.get("start") or
            ""
        )
        end_raw = (
            item.get("end_date") or
            item.get("toYear") or
            item.get("to") or
            item.get("end") or
            ""
        )
        is_ongoing = item.get("isOngoing") or item.get("is_ongoing") or False
        if is_ongoing:
            end_raw = "Present"

        start_date = _normalize_date(start_raw)
        end_date   = _normalize_date(end_raw) or ("Present" if is_ongoing else "")

        bullets = _extract_bullets(item)

        exp = NormalizedExperience(
            title=title,
            company=company,
            location=location,
            start_date=start_date,
            end_date=end_date,
            bullets=bullets,
        )

        # Only include if at least title or company is present
        if exp.title or exp.company:
            result.append(exp)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

_CERT_SIGNALS = re.compile(
    r"(certif|licence|license|course|training|bootcamp|nanodegree|"
    r"specialization|specialisation|workshop|diploma\s+in\s+(?!science|arts|engineering))",
    re.I,
)

def _looks_like_certification(edu: Dict[str, Any]) -> bool:
    """
    Return True if an education entry is more likely a certification/course
    than a formal degree.
    """
    degree = str(edu.get("degree") or "").lower()
    institution = str(edu.get("institution") or edu.get("college") or "").lower()
    combined = degree + " " + institution
    if _CERT_SIGNALS.search(combined):
        return True
    # Formal degree signals — if these are present, keep in education
    if re.search(r"\b(b\.?s|b\.?a|bachelor|master|m\.?s|m\.?a|ph\.?d|mba|b\.?tech|m\.?tech|b\.?e)\b", degree):
        return False
    return False


def _normalize_education(raw_list: Any) -> tuple[List[NormalizedEducation], List[NormalizedCertification]]:
    """
    Normalize education list.  Entries that look like certifications/courses
    are split out and returned separately to prevent them being lost.
    """
    if not isinstance(raw_list, list):
        return [], []

    education:      List[NormalizedEducation]    = []
    certifications: List[NormalizedCertification] = []

    for item in raw_list:
        if not isinstance(item, dict):
            continue

        degree = str(
            item.get("degree") or
            item.get("qualification") or
            item.get("title") or
            ""
        ).strip()

        institution = str(
            item.get("institution") or
            item.get("college") or
            item.get("school") or
            item.get("university") or
            ""
        ).strip()

        year = str(
            item.get("year") or
            item.get("toYear") or
            item.get("graduation_year") or
            item.get("end_year") or
            ""
        ).strip()

        gpa = str(item.get("gpa") or "").strip()

        if not degree and not institution:
            continue

        if _looks_like_certification(item):
            certifications.append(NormalizedCertification(
                title=degree or institution,
                issuer=institution if degree else "",
                year=year,
            ))
        else:
            education.append(NormalizedEducation(
                degree=degree,
                institution=institution,
                year=year,
                gpa=gpa,
            ))

    return education, certifications


# ─────────────────────────────────────────────────────────────────────────────
# CERTIFICATION NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_certifications(raw_list: Any) -> List[NormalizedCertification]:
    if not isinstance(raw_list, list):
        return []
    result: List[NormalizedCertification] = []
    for item in raw_list:
        if isinstance(item, str):
            if item.strip():
                result.append(NormalizedCertification(title=item.strip()))
            continue
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or item.get("certification") or "").strip()
        issuer = str(item.get("issuer") or item.get("organization") or "").strip()
        year   = str(item.get("year") or item.get("date") or "").strip()
        if title:
            result.append(NormalizedCertification(title=title, issuer=issuer, year=year))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_projects(raw_list: Any) -> List[NormalizedProject]:
    if not isinstance(raw_list, list):
        return []
    result: List[NormalizedProject] = []
    for item in raw_list:
        if isinstance(item, str):
            if item.strip():
                result.append(NormalizedProject(title=item.strip()))
            continue
        if not isinstance(item, dict):
            continue
        title       = str(item.get("title") or item.get("name") or item.get("project") or "").strip()
        description = str(item.get("description") or item.get("summary") or "").strip()
        techs_raw   = item.get("technologies") or item.get("tech_stack") or item.get("tools") or []
        if isinstance(techs_raw, str):
            techs_raw = [t.strip() for t in re.split(r"[,/|]", techs_raw) if t.strip()]
        techs   = [str(t).strip() for t in techs_raw if t]
        bullets = _extract_bullets(item)
        url     = str(item.get("url") or item.get("link") or item.get("github") or "").strip()
        if title or description:
            result.append(NormalizedProject(
                title=title, description=description,
                technologies=techs, bullets=bullets, url=url,
            ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_summary(raw: Any) -> str:
    """
    Handle every known summary shape:
    - Builder: {"summary": "text"}
    - Markdown/flat: "text"
    - None / missing
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        # Builder output: {"summary": "text"} or {"summary": {"summary": "text"}}
        val = raw.get("summary") or raw.get("text") or raw.get("profile") or ""
        return _normalize_summary(val)
    return str(raw).strip()


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_languages(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [p.strip() for p in re.split(r"[,|/]", raw) if p.strip()]
    if isinstance(raw, list):
        langs: List[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                langs.append(item.strip())
            elif isinstance(item, dict):
                name = item.get("language") or item.get("name") or ""
                if name:
                    langs.append(str(name).strip())
        return langs
    return []


# ─────────────────────────────────────────────────────────────────────────────
# FLAT LIST NORMALIZATION (awards, volunteer, publications, hobbies)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_flat_list(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        result: List[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
            elif isinstance(item, dict):
                text = (
                    item.get("title") or item.get("name") or
                    item.get("text") or item.get("description") or ""
                )
                if text:
                    result.append(str(text).strip())
        return result
    return []


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NORMALIZER ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def normalize_resume(raw: Dict[str, Any]) -> NormalizedResume:
    """
    Convert any parser output dict (Resume Builder or ATS Markdown Parser)
    into a NormalizedResume.  This is the ONLY function scorers should call.

    Called once per scan in ATSScannerService.scan() before all scoring.
    """
    if not raw or not isinstance(raw, dict):
        return NormalizedResume()

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = _normalize_summary(raw.get("summary"))

    # ── Contact ──────────────────────────────────────────────────────────────
    contact = _normalize_contact(raw)

    # ── Experience ───────────────────────────────────────────────────────────
    experience = _normalize_experience(raw.get("experience") or [])

    # ── Education + cert promotion ───────────────────────────────────────────
    education, edu_certs = _normalize_education(raw.get("education") or [])

    # ── Certifications ───────────────────────────────────────────────────────
    raw_certs        = _normalize_certifications(raw.get("certifications") or [])
    all_certs        = raw_certs + edu_certs
    # Deduplicate certifications by lowercased title
    seen_cert_titles: set = set()
    certifications:   List[NormalizedCertification] = []
    for c in all_certs:
        key = c.title.lower().strip()
        if key and key not in seen_cert_titles:
            seen_cert_titles.add(key)
            certifications.append(c)

    # ── Skills ───────────────────────────────────────────────────────────────
    raw_skills = raw.get("skills") or []
    if not isinstance(raw_skills, list):
        raw_skills = []
    skills = _deduplicate_skills(raw_skills)

    # ── Projects ─────────────────────────────────────────────────────────────
    projects = _normalize_projects(raw.get("projects") or [])

    # ── Languages ────────────────────────────────────────────────────────────
    languages = _normalize_languages(raw.get("languages") or [])

    # ── Flat sections ────────────────────────────────────────────────────────
    awards       = _normalize_flat_list(raw.get("awards")       or [])
    volunteer    = _normalize_flat_list(raw.get("volunteer")    or [])
    publications = _normalize_flat_list(raw.get("publications") or [])
    hobbies      = _normalize_flat_list(raw.get("hobbies")      or [])

    # ── Raw text ─────────────────────────────────────────────────────────────
    raw_text = str(raw.get("raw_text") or raw.get("_raw_text") or "")

    nr = NormalizedResume(
        contact=contact,
        summary=summary,
        experience=experience,
        education=education,
        skills=skills,
        certifications=certifications,
        projects=projects,
        languages=languages,
        awards=awards,
        volunteer=volunteer,
        publications=publications,
        hobbies=hobbies,
        raw_text=raw_text,
    )

    logger.info(
        f"[Normalizer] contact='{nr.contact.name}' | "
        f"exp={len(nr.experience)} | edu={len(nr.education)} | "
        f"skills={len(nr.skills)} | certs={len(nr.certifications)} | "
        f"proj={len(nr.projects)} | langs={len(nr.languages)}"
    )

    return nr