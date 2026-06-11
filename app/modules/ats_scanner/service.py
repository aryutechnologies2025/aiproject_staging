from __future__ import annotations

import json
import logging
import math
import re
import asyncio
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.ats_scanner.utils.ats_rules_advanced import ATSRulesEngine, SeverityLevel
from app.modules.ats_scanner.utils.ats_keyword_engine import KeywordEngine, KeywordAnalysis, UNIVERSAL_SKILLS
from app.modules.ats_scanner.utils.ats_feedback_generator import (
    DetailedFeedbackGenerator,
    ComprehensiveFeedback,
    GLOBAL_ATS_TACTICS,
    RECRUITER_TIPS,
)
from app.utils.llm_client import call_llm

logger = logging.getLogger(__name__)

EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I)
PHONE_RE    = re.compile(r"(\+\d{1,3}[\s\-]?)?\(?\d{3,5}\)?[\s\-]?\d{3,5}[\s\-]?\d{4,6}")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.I)
GITHUB_RE   = re.compile(r"github\.com/[\w\-]+", re.I)

_SUBTITLE_TOKENS = {
    "full stack", "software engineer", "mern", "developer", "manager",
    "agile", "delivery", "performance", "optimization", "stack", "frontend",
    "backend", "engineer", "architect", "analyst", "consultant", "specialist",
    "designer", "director", "officer", "lead", "head of",
}

_MONTH_MAP: Dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_STRONG_VERBS: Set[str] = {
    "led", "managed", "directed", "spearheaded", "orchestrated", "oversaw",
    "achieved", "delivered", "exceeded", "surpassed", "attained",
    "developed", "built", "created", "designed", "architected", "engineered",
    "launched", "initiated", "pioneered", "founded", "innovated",
    "optimized", "improved", "enhanced", "streamlined", "accelerated",
    "increased", "reduced", "minimized", "maximized", "scaled", "expanded",
    "analyzed", "evaluated", "assessed", "diagnosed", "identified",
    "implemented", "deployed", "integrated", "established", "released",
    "transformed", "modernized", "reformed", "restructured", "negotiated",
    "generated", "secured", "raised", "grew", "drove", "cut", "saved",
}

_WEAK_OPENERS: Set[str] = {
    "responsible for", "involved in", "helped with", "assisted",
    "worked on", "was part of", "participated in", "contributed to",
    "was responsible", "duties included", "tasked with",
}

_METRIC_PATS: List[re.Pattern] = [
    re.compile(r"\$[\d,]+\.?\d*\s*[KMBkmb]?"),
    re.compile(r"\b\d+\s*%"),
    re.compile(r"\b\d+[xX]\b"),
    re.compile(r"\b\d+\+\s*(?:users?|clients?|customers?|projects?|employees?)"),
    re.compile(r"(?:increased|reduced|grew|improved|decreased|expanded|saved|cut)\s+by\s+\d+"),
    re.compile(r"\b\d[\d,]*\s*(?:million|billion|thousand|M|B|K)\b", re.I),
]

_SYNONYMS: Dict[str, Set[str]] = {
    "javascript":       {"js", "ecmascript", "es6"},
    "typescript":       {"ts"},
    "python":           {"python3", "py"},
    "c#":               {"csharp", "dotnet", ".net"},
    "c++":              {"cpp"},
    "node.js":          {"node", "nodejs"},
    "react":            {"reactjs", "react.js"},
    "vue":              {"vuejs", "vue.js"},
    "angular":          {"angularjs"},
    "postgresql":       {"postgres"},
    "kubernetes":       {"k8s"},
    "machine learning": {"ml"},
    "artificial intelligence": {"ai"},
    "natural language processing": {"nlp"},
    "registered nurse": {"rn"},
    "electronic health records": {"ehr", "emr"},
    "certified public accountant": {"cpa"},
    "chartered financial analyst": {"cfa"},
    "search engine optimization": {"seo"},
    "project management professional": {"pmp"},
    "enterprise resource planning": {"erp"},
    "customer relationship management": {"crm"},
}

_REV_SYN: Dict[str, str] = {}
for _can, _vars in _SYNONYMS.items():
    for _v in _vars:
        _REV_SYN[_v.lower()] = _can.lower()

_HARD_CAPS: Dict[str, int] = {
    "no_experience": 40,
    "no_skills":     50,
    "no_email":      85,
}

AI_ANALYSIS_PROMPT = """You are a senior ATS expert. Analyse this resume and respond ONLY with valid JSON.

RESUME:
Name: {name}
Role: {target_role}
Industry: {industry}
Summary: {summary}
Skills: {skills}
Experience: {experience_bullets}
Education: {education}
Certifications: {certifications}
Projects: {projects}
Extra sections: {additional_sections}
Job Description: {job_description}
Current ATS Score: {ats_score}/100

Return ONLY this JSON structure with no markdown fences, no extra text:
{{
  "industry_detected": "string",
  "role_level": "Entry|Mid|Senior|Executive",
  "ats_compatibility_verdict": "string",
  "ai_section_scores": {{
    "summary": {{"score": 70, "verdict": "string"}},
    "experience": {{"score": 70, "verdict": "string"}},
    "skills": {{"score": 70, "verdict": "string"}},
    "education": {{"score": 70, "verdict": "string"}}
  }},
  "content_strengths": ["string", "string", "string"],
  "critical_improvements": [
    {{"section": "string", "issue": "string", "rewritten_example": "string", "estimated_score_gain": 5}}
  ],
  "keyword_gaps": [
    {{"missing_keyword": "string", "importance": "critical", "add_to_section": "string"}}
  ],
  "ats_passing_tactics": ["string", "string", "string"],
  "bullet_rewrites": [
    {{"original": "string", "rewritten": "string", "why_better": "string"}}
  ],
  "summary_rewrite": {{"current": "string", "suggested": "string", "why_better": "string"}},
  "missing_sections": [
    {{"section": "string", "why_important": "string"}}
  ],
  "overall_assessment": "string",
  "priority_action_plan": ["string", "string", "string", "string", "string"]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# PURE HELPERS  (stateless, no class dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _safe(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_for_prompt(value: Any, max_len: int = 200) -> str:
    text = _safe(value)[:max_len]
    text = text.replace("\n", " ").replace("\r", " ").replace("\\", "")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return bool(value)


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _grade(score: int) -> str:
    if score >= 90: return "A+"
    if score >= 85: return "A"
    if score >= 80: return "A-"
    if score >= 75: return "B+"
    if score >= 70: return "B"
    if score >= 65: return "B-"
    if score >= 60: return "C+"
    if score >= 55: return "C"
    if score >= 50: return "C-"
    if score >= 40: return "D"
    return "F"


def _status_from_score(score: int) -> str:
    if score >= 85: return "excellent"
    if score >= 70: return "good"
    if score >= 50: return "needs_improvement"
    if score >  0:  return "critical"
    return "missing"


def _compatibility_label(score: int) -> str:
    if score >= 85: return "Excellent"
    if score >= 72: return "Good"
    if score >= 60: return "Moderate"
    if score >= 45: return "Poor"
    return "Critical"


def _ats_verdict(score: int) -> str:
    if score >= 85:
        return "Strong ATS profile — resume will pass most automated filters."
    if score >= 72:
        return "Passes ATS screening — ready to apply with minor refinements."
    if score >= 60:
        return "Borderline ATS pass — address high-priority improvements before applying."
    if score >= 45:
        return "High ATS rejection risk — significant keyword and quality gaps need fixing."
    return "Very high ATS rejection risk — major revision required before applying."


def _clean_name(raw: str) -> str:
    if not raw:
        return ""
    lower = raw.lower()
    hits  = sum(1 for t in _SUBTITLE_TOKENS if t in lower)
    if hits >= 2 or "|" in raw or len(raw) > 60:
        return ""
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# CONTACT EXTRACTION  — reads from canonical JSON fields first
# ─────────────────────────────────────────────────────────────────────────────

def _build_contact_from_resume(resume: Dict) -> Dict:
    """
    Extract contact info from the canonical parsed JSON.
    The header dict (from Resume Builder) is the primary source.
    Regex over raw_text is used ONLY as a last-resort fallback for
    fields that couldn't be parsed (e.g. phone split across a table).
    """
    header   = resume.get("header") or {}
    raw_text = _safe(resume.get("raw_text", ""))

    def _first(*sources):
        for s in sources:
            v = _safe(s)
            if v:
                return v
        return ""

    # Prefer structured header; fall back to top-level flat keys; then regex
    name_raw = _first(
        header.get("name") if isinstance(header, dict) else None,
        resume.get("name"),
    )
    name = _clean_name(name_raw)

    email_val = _first(
        header.get("email") if isinstance(header, dict) else None,
        resume.get("email"),
    )
    if not email_val and raw_text:
        m = EMAIL_RE.search(raw_text)
        email_val = m.group(0).strip() if m else ""

    phone_val = _first(
        header.get("phone") if isinstance(header, dict) else None,
        resume.get("phone"),
    )
    if not phone_val and raw_text:
        m = PHONE_RE.search(raw_text)
        phone_val = m.group(0).strip() if m else ""

    location_val = _first(
        header.get("location") if isinstance(header, dict) else None,
        resume.get("location"),
    )

    # LinkedIn / GitHub from the header "link" field or flat keys
    link_blob = _first(
        header.get("link") if isinstance(header, dict) else None,
        resume.get("linkedin"),
    )
    linkedin_val = ""
    github_val   = ""
    for fragment in re.split(r"[,\s]+", link_blob):
        if LINKEDIN_RE.search(fragment):
            linkedin_val = fragment.strip()
        elif GITHUB_RE.search(fragment):
            github_val = fragment.strip()

    if not linkedin_val:
        m = LINKEDIN_RE.search(raw_text)
        linkedin_val = m.group(0).strip() if m else ""
    if not github_val:
        m = GITHUB_RE.search(raw_text)
        github_val = m.group(0).strip() if m else ""

    return {
        "name":     name,
        "email":    email_val,
        "phone":    phone_val,
        "location": location_val,
        "linkedin": linkedin_val,
        "github":   github_val,
    }


def _score_contact(contact: Dict) -> Tuple[int, List[str], List[str], List[str]]:
    score     = 100
    missing   = []
    quality   = []
    strengths = []

    if not contact.get("name"):
        missing.append("Full name")
        score -= 25
    else:
        strengths.append(f"Name present: {contact['name']}")

    if not contact.get("email"):
        missing.append("Email address")
        score -= 25
    else:
        if re.search(r"@(hotmail|yahoo|rediffmail|ymail)\.", contact["email"], re.I):
            quality.append(f"Unprofessional email: {contact['email']} — use Gmail or custom domain")
        else:
            strengths.append(f"Professional email: {contact['email']}")

    if not contact.get("phone"):
        missing.append("Phone number")
        score -= 15
    else:
        strengths.append("Phone number present")

    if not contact.get("location"):
        quality.append("Location missing — many ATS filter by city/country")
        score -= 10

    if not contact.get("linkedin"):
        quality.append("No LinkedIn URL — 90% of recruiters check LinkedIn before contacting")
        score -= 5

    if contact.get("github"):
        strengths.append("GitHub profile linked")

    return max(score, 0), missing, quality, strengths


class _DictProxy:
    def __init__(self, d: Dict) -> None:
        self._d = d

    def __getattr__(self, name: str):
        try:
            return self._d[name]
        except KeyError:
            return None

    def get(self, key, default=None):
        return self._d.get(key, default)


def _build_contact_section_proxy(contact: Dict) -> _DictProxy:
    score, missing, quality, strengths = _score_contact(contact)
    ats_tips = [
        "Keep contact info in the resume body — NOT a header/footer.",
        "Use City, State/Country only — not your full street address.",
        "Always include a country code in your phone number.",
        "Add your LinkedIn URL — recruiters verify before scheduling interviews.",
        "Professional email format: firstname.lastname@gmail.com",
    ]
    data = {
        "section_name":   "contact",
        "current_score":  score,
        "missing_fields": missing,
        "quality_issues": quality,
        "strengths":      strengths,
        "ats_tips":       ats_tips,
        "improvements":   [f"Add {m}" for m in missing] + quality,
        "rewrite_examples": [],
        "current_status": (
            ["good"] if score >= 70
            else (["needs_improvement"] if score >= 40 else ["critical"])
        ),
        "complete": not missing,
    }
    return _DictProxy(data)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION ACCESSORS  — all read from structured JSON, never from raw text
# ─────────────────────────────────────────────────────────────────────────────

def _get_summary(resume: Dict) -> str:
    """Return the summary string regardless of whether it is nested or flat."""
    raw = resume.get("summary")
    if isinstance(raw, dict):
        return _safe(raw.get("summary", ""))
    return _safe(raw)


def _get_education(resume: Dict) -> List[Dict]:
    edu = resume.get("education") or []
    if not isinstance(edu, list):
        return []
    result = []
    for e in edu:
        if isinstance(e, dict) and (e.get("degree") or e.get("institution") or e.get("college")):
            result.append(e)
        elif isinstance(e, str) and e.strip():
            result.append({"raw_text": e})
    return result


def _get_skills(resume: Dict) -> List[str]:
    skills = resume.get("skills") or []
    if not isinstance(skills, list):
        return []
    return [_safe(s) for s in skills if _safe(s)]


def _get_experience(resume: Dict) -> List[Dict]:
    exp = resume.get("experience") or []
    if not isinstance(exp, list):
        return []
    return [e for e in exp if isinstance(e, dict)]


def _get_projects(resume: Dict) -> List[Dict]:
    proj = resume.get("projects") or []
    if not isinstance(proj, list):
        return []
    return proj


def _get_certifications(resume: Dict) -> List:
    return resume.get("certifications") or []


def _get_languages(resume: Dict) -> List:
    return resume.get("languages") or []


# ─────────────────────────────────────────────────────────────────────────────
# SCORING DIMENSIONS  — operate on structured JSON only
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date_to_months(raw: str) -> Optional[int]:
    raw = _safe(raw).lower()
    if not raw or re.match(r"present|current|now|till", raw, re.I):
        return 12 * 2026 + 6

    for abbr, num in _MONTH_MAP.items():
        if abbr in raw:
            ym = re.search(r"(19|20)\d{2}", raw)
            if ym:
                return int(ym.group(0)) * 12 + num

    ym = re.search(r"(19|20)\d{2}", raw)
    if ym:
        return int(ym.group(0)) * 12 + 6

    return None


def _duration_months(start: str, end: str) -> int:
    s = _parse_date_to_months(start)
    e = _parse_date_to_months(end)
    if s is None or e is None:
        return 0
    return max(0, e - s)


def _has_metric(text: str) -> bool:
    return any(p.search(text) for p in _METRIC_PATS)


def _has_strong_verb(text: str) -> bool:
    words = re.findall(r"\b\w+\b", text.lower())
    return bool(words and words[0] in _STRONG_VERBS)


def _has_weak_opener(text: str) -> bool:
    lower = text.lower()
    return any(lower.startswith(w) for w in _WEAK_OPENERS)


def _resume_flat_text(resume: Dict) -> str:
    """Build a searchable text blob from structured fields only."""
    parts: List[str] = []

    summary = _get_summary(resume)
    if summary:
        parts.append(summary)

    for s in _get_skills(resume):
        parts.append(s)

    for exp in _get_experience(resume):
        parts.append(_safe(exp.get("title")))
        parts.append(_safe(exp.get("company")))
        for b in (exp.get("bullets") or []):
            parts.append(_safe(b))

    for proj in _get_projects(resume):
        parts.append(_safe(proj.get("title") or proj.get("name", "")))
        for b in (proj.get("bullets") or []):
            parts.append(_safe(b))
        for t in (proj.get("technologies") or []):
            parts.append(_safe(t))

    for cert in _get_certifications(resume):
        if isinstance(cert, dict):
            parts.append(_safe(cert.get("name") or cert.get("title", "")))
        else:
            parts.append(_safe(cert))

    # raw_text is only appended for density/length purposes, not section detection
    raw = resume.get("raw_text", "")
    if raw:
        parts.append(_safe(raw))

    return " ".join(p for p in parts if p).lower()


def _known_skills_set() -> Set[str]:
    known: Set[str] = set()
    for cats in UNIVERSAL_SKILLS.values():
        for skills in cats.values():
            known.update(s.lower() for s in skills)
    return known


_KNOWN_SKILLS: Set[str] = _known_skills_set()


def _dim_keyword(resume: Dict, job_description: Optional[str]) -> Dict:
    resume_text       = _resume_flat_text(resume)
    skills_raw        = _get_skills(resume)
    resume_skills_set = {_REV_SYN.get(s.lower(), s.lower()) for s in skills_raw} | {s.lower() for s in skills_raw}

    if not job_description or not job_description.strip():
        skill_count = len(skills_raw)
        raw = _clamp(100.0 / (1.0 + math.exp(-0.2 * (skill_count - 12))))
        return {
            "raw_score": raw, "weight": 0.30,
            "skill_count": skill_count, "jd_provided": False,
            "penalties": (["No job description provided"] if skill_count < 5 else []),
            "bonuses":   ([f"{skill_count} skills listed"] if skill_count >= 8 else []),
        }

    jd_lower   = job_description.lower()
    jd_words   = re.findall(r"\b[\w\+#\.]+\b", jd_lower)
    jd_bigrams = [f"{jd_words[i]} {jd_words[i+1]}" for i in range(len(jd_words) - 1)]
    jd_tokens  = list(set(jd_words + jd_bigrams))

    freq: Dict[str, int] = {}
    for tok in jd_tokens:
        canon = _REV_SYN.get(tok, tok)
        if canon in _KNOWN_SKILLS or tok in _KNOWN_SKILLS:
            freq[canon] = freq.get(canon, 0) + jd_lower.count(tok)

    if not freq:
        skill_count = len(skills_raw)
        raw = _clamp(100.0 / (1.0 + math.exp(-0.18 * (skill_count - 10))))
        return {
            "raw_score": raw, "weight": 0.30,
            "jd_skills_detected": 0, "skill_count": skill_count,
            "penalties": [], "bonuses": [],
        }

    total_weight   = 0.0
    matched_weight = 0.0
    missing_critical: List[str] = []
    matched_skills:   List[str] = []

    for skill, cnt in freq.items():
        w = 1.0 + math.log(max(cnt, 1))
        total_weight += w
        found = False
        if skill in resume_skills_set or skill in resume_text:
            found = True
        else:
            for var in _SYNONYMS.get(skill, set()):
                if var.lower() in resume_skills_set or var.lower() in resume_text:
                    found = True
                    break
        if found:
            matched_weight += w
            matched_skills.append(skill)
        elif cnt >= 3:
            missing_critical.append(skill)

    match_ratio = matched_weight / total_weight if total_weight > 0 else 0.0
    raw = _clamp(match_ratio * 100)

    words_in_resume = len(resume_text.split())
    density = len(resume_skills_set) / max(words_in_resume, 1)
    if density > 0.06:
        raw = min(raw + 5, 100)

    return {
        "raw_score":          raw,
        "weight":             0.30,
        "jd_skills_detected": len(freq),
        "matched":            len(matched_skills),
        "match_ratio":        round(match_ratio, 3),
        "missing_critical":   missing_critical,
        "keyword_density":    round(density, 4),
        "penalties":          [f'Missing critical keyword: "{s}"' for s in missing_critical[:5]],
        "bonuses":            [f'Matched: "{s}"' for s in matched_skills[:5]],
    }


def _dim_experience(resume: Dict) -> Dict:
    experience = _get_experience(resume)
    if not experience:
        return {
            "raw_score": 0.0, "weight": 0.25,
            "total_months": 0, "roles": 0,
            "penalties": ["No work experience entries found"], "bonuses": [],
        }

    total_months   = 0
    complete_roles = 0
    role_count     = len(experience)
    penalties: List[str] = []
    bonuses:   List[str] = []

    for idx, exp in enumerate(experience):
        # Support both "title"/"position" keys from different parser outputs
        title   = _safe(exp.get("title") or exp.get("position", ""))
        company = _safe(exp.get("company", ""))
        start   = _safe(exp.get("start_date") or exp.get("fromYear", ""))
        end     = _safe(exp.get("end_date") or exp.get("toYear", ""))

        # isOngoing flag from Resume Builder schema
        if exp.get("isOngoing"):
            end = "present"

        bullets = exp.get("bullets") or []

        dur = _duration_months(start, end)
        if dur == 0 and (start or exp.get("isOngoing")):
            dur = 12
        total_months += dur

        complete = bool(title and company and (start or dur > 0) and bullets)
        if complete:
            complete_roles += 1
        else:
            missing_parts = []
            if not title:   missing_parts.append("title")
            if not company: missing_parts.append("company")
            if not start:   missing_parts.append("dates")
            if not bullets: missing_parts.append("bullets")
            if missing_parts:
                penalties.append(f"Role {idx+1}: missing {', '.join(missing_parts)}")

        title_lower = title.lower()
        if any(t in title_lower for t in ("senior", "lead", "principal", "head", "director", "vp", "chief", "manager")):
            bonuses.append(f"Senior role: {title}")

    if total_months == 0:
        dur_score = 0.0
    elif total_months <= 6:
        dur_score = 15.0
    else:
        dur_score = _clamp(
            35.0 + 30.0 * math.log(total_months / 12.0)
            if total_months >= 12 else total_months / 12.0 * 35.0
        )

    role_bonus       = _clamp(min(role_count * 5.0, 15.0))
    completeness     = complete_roles / max(role_count, 1)
    completeness_pts = completeness * 20.0
    raw              = _clamp(dur_score + role_bonus * completeness + completeness_pts)

    return {
        "raw_score":        raw,
        "weight":           0.25,
        "total_months":     total_months,
        "total_years":      round(total_months / 12, 1),
        "roles":            role_count,
        "complete_roles":   complete_roles,
        "completeness_pct": round(completeness * 100),
        "penalties":        penalties,
        "bonuses":          bonuses,
    }


def _dim_quality(resume: Dict) -> Dict:
    experience = _get_experience(resume)
    summary    = _get_summary(resume)
    skills     = _get_skills(resume)

    all_bullets: List[str] = []
    for exp in experience:
        for b in (exp.get("bullets") or []):
            if _safe(b):
                all_bullets.append(_safe(b))

    total_bullets = len(all_bullets)
    strong_count  = 0
    metric_count  = 0
    weak_count    = 0
    penalties: List[str] = []
    bonuses:   List[str] = []

    for b in all_bullets:
        if _has_strong_verb(b):  strong_count += 1
        if _has_metric(b):       metric_count += 1
        if _has_weak_opener(b):  weak_count   += 1

    if total_bullets == 0:
        bullet_score = 0.0
        penalties.append("No achievement bullets in experience section")
    else:
        strong_ratio = strong_count / total_bullets
        metric_ratio = metric_count / total_bullets
        weak_ratio   = weak_count   / total_bullets
        bullet_score = _clamp((strong_ratio * 40.0) + (metric_ratio * 50.0) - (weak_ratio * 20.0))

        if metric_ratio > 0.5:
            bonuses.append(f"{int(metric_ratio * 100)}% of bullets contain measurable results")
        elif metric_ratio < 0.2:
            penalties.append(f"Only {int(metric_ratio * 100)}% of bullets have metrics — add numbers")

        if weak_ratio > 0.3:
            penalties.append(f"{int(weak_ratio * 100)}% of bullets start with weak openers")

    summary_score = 0.0
    if not summary:
        penalties.append("Professional summary missing")
    else:
        words = len(summary.split())
        if words < 20:
            summary_score = 8.0
            penalties.append(f"Summary too brief ({words} words)")
        elif words > 150:
            summary_score = 15.0
            penalties.append("Summary too long — trim to 50-80 words")
        else:
            summary_score = 20.0
        if _has_metric(summary):
            summary_score = min(summary_score + 5.0, 25.0)
            bonuses.append("Summary contains quantified achievement")

    generic_terms = {
        "ms office", "microsoft office", "computers", "communication",
        "teamwork", "adaptability", "hardworking", "detail oriented",
        "fast learner", "quick learner", "team player",
    }
    specific_count = sum(
        1 for s in skills
        if s.lower() not in generic_terms and len(s) > 2
    )
    specificity_score = _clamp(min(specific_count * 1.5, 15.0))

    raw = _clamp(bullet_score * 0.60 + summary_score + specificity_score)

    return {
        "raw_score":         raw,
        "weight":            0.20,
        "total_bullets":     total_bullets,
        "strong_verb_pct":   round(strong_count / max(total_bullets, 1) * 100),
        "metric_pct":        round(metric_count / max(total_bullets, 1) * 100),
        "weak_opener_pct":   round(weak_count   / max(total_bullets, 1) * 100),
        "summary_score":     round(summary_score),
        "specificity_score": round(specificity_score),
        "penalties":         penalties,
        "bonuses":           bonuses,
    }


def _dim_structure(resume: Dict) -> Dict:
    """
    Evaluate section completeness from canonical JSON.
    Sections are present if their structured lists/strings are non-empty.
    """
    earned   = 0.0
    max_pts  = 85.0
    penalties: List[str] = []
    bonuses:   List[str] = []

    summary = _get_summary(resume)
    if summary and len(summary.split()) >= 15:
        earned += 20
        bonuses.append("Professional summary present")
    elif summary:
        earned += 10
        penalties.append("Summary present but too brief")
    else:
        penalties.append("Professional summary missing")

    experience = _get_experience(resume)
    if experience:
        earned += 20
        bonuses.append(f"{len(experience)} experience entry/entries")
    else:
        penalties.append("Work experience section missing")

    skills = _get_skills(resume)
    if len(skills) >= 5:
        earned += 20
        bonuses.append(f"{len(skills)} skills listed")
    elif skills:
        earned += 10
        penalties.append(f"Too few skills ({len(skills)} — add at least 5)")
    else:
        penalties.append("Skills section missing or empty")

    education = _get_education(resume)
    if education:
        earned += 20
        bonuses.append(f"{len(education)} education entry/entries")
    else:
        penalties.append("Education section missing")

    contact = _build_contact_from_resume(resume)
    if contact.get("email"):
        earned += 5
        bonuses.append("Email present")
    else:
        penalties.append("Email address missing")

    projects = _get_projects(resume)
    if projects:
        bonuses.append(f"{len(projects)} project(s) listed")

    raw = _clamp((earned / max_pts) * 100)
    return {
        "raw_score":  raw,
        "weight":     0.15,
        "pts_earned": round(earned),
        "pts_max":    round(max_pts),
        "penalties":  penalties,
        "bonuses":    bonuses,
    }


def _dim_format(resume: Dict) -> Dict:
    """Format/ATS compliance uses raw extraction metadata only."""
    score    = 100.0
    penalties: List[str] = []
    bonuses:   List[str] = []

    if resume.get("uses_table") or resume.get("has_tables"):
        score -= 15
        penalties.append("Tables detected — ATS cannot parse table content")

    if resume.get("uses_columns") or resume.get("multi_column"):
        score -= 12
        penalties.append("Multi-column layout — ATS reads columns out of order")

    if resume.get("has_images") or resume.get("uses_graphics"):
        score -= 8
        penalties.append("Images/graphics detected — ATS cannot read visual elements")

    contact = _build_contact_from_resume(resume)
    if contact.get("linkedin"):
        bonuses.append("LinkedIn profile linked")
        score = min(score + 3, 100)
    if contact.get("github"):
        bonuses.append("GitHub profile linked")
        score = min(score + 2, 100)

    return {
        "raw_score": _clamp(score), "weight": 0.10,
        "penalties": penalties, "bonuses": bonuses,
    }


def _apply_hard_caps(score: float, resume: Dict) -> Tuple[float, List[str]]:
    applied: List[str] = []
    exp     = _get_experience(resume)
    skills  = _get_skills(resume)
    contact = _build_contact_from_resume(resume)
    email   = contact.get("email", "")

    if not exp:
        if score > _HARD_CAPS["no_experience"]:
            score = float(_HARD_CAPS["no_experience"])
            applied.append(f"No work experience — score capped at {_HARD_CAPS['no_experience']}")

    if not skills:
        if score > _HARD_CAPS["no_skills"]:
            score = float(_HARD_CAPS["no_skills"])
            applied.append(f"No skills section — score capped at {_HARD_CAPS['no_skills']}")

    if not email:
        if score > _HARD_CAPS["no_email"]:
            score = float(_HARD_CAPS["no_email"])
            applied.append(f"Missing email — score capped at {_HARD_CAPS['no_email']}")

    return score, applied


def _calculate_dynamic_score(
    resume:          Dict,
    job_description: Optional[str],
    ai_insights:     Dict,
) -> Tuple[int, Dict]:
    d_kw      = _dim_keyword(resume, job_description)
    d_exp     = _dim_experience(resume)
    d_quality = _dim_quality(resume)
    d_struct  = _dim_structure(resume)
    d_format  = _dim_format(resume)

    raw_weighted = (
        d_kw["raw_score"]      * d_kw["weight"]      +
        d_exp["raw_score"]     * d_exp["weight"]      +
        d_quality["raw_score"] * d_quality["weight"]  +
        d_struct["raw_score"]  * d_struct["weight"]   +
        d_format["raw_score"]  * d_format["weight"]
    )

    logger.info(
        f"  kw={d_kw['raw_score']:.1f}  exp={d_exp['raw_score']:.1f}  "
        f"qual={d_quality['raw_score']:.1f}  struct={d_struct['raw_score']:.1f}  "
        f"fmt={d_format['raw_score']:.1f}  → weighted={raw_weighted:.1f}"
    )

    ai_bonus = 0
    if ai_insights.get("success"):
        ai_scores = ai_insights.get("ai_section_scores") or {}
        if ai_scores:
            vals = [
                v.get("score", 70)
                for v in ai_scores.values()
                if isinstance(v, dict) and isinstance(v.get("score"), (int, float))
            ]
            if vals:
                ai_bonus = int((sum(vals) / len(vals) - 70) * 0.08)

    raw_with_ai = raw_weighted + ai_bonus
    capped, caps_applied = _apply_hard_caps(raw_with_ai, resume)
    final = max(0, min(100, round(capped)))

    breakdown = {
        "keyword_skill_density":      {"raw": round(d_kw["raw_score"],      1), "weight": d_kw["weight"],      "weighted": round(d_kw["raw_score"]      * d_kw["weight"],      2), **{k: v for k, v in d_kw.items()      if k not in ("raw_score", "weight")}},
        "experience_depth_duration":  {"raw": round(d_exp["raw_score"],     1), "weight": d_exp["weight"],     "weighted": round(d_exp["raw_score"]     * d_exp["weight"],     2), **{k: v for k, v in d_exp.items()     if k not in ("raw_score", "weight")}},
        "achievement_quality":        {"raw": round(d_quality["raw_score"], 1), "weight": d_quality["weight"], "weighted": round(d_quality["raw_score"] * d_quality["weight"], 2), **{k: v for k, v in d_quality.items() if k not in ("raw_score", "weight")}},
        "structure_completeness":     {"raw": round(d_struct["raw_score"],  1), "weight": d_struct["weight"],  "weighted": round(d_struct["raw_score"]  * d_struct["weight"],  2), **{k: v for k, v in d_struct.items()  if k not in ("raw_score", "weight")}},
        "format_ats_compliance":      {"raw": round(d_format["raw_score"],  1), "weight": d_format["weight"],  "weighted": round(d_format["raw_score"]  * d_format["weight"],  2), **{k: v for k, v in d_format.items()  if k not in ("raw_score", "weight")}},
        "hard_caps_applied":          caps_applied,
        "ai_bonus":                   ai_bonus,
    }

    return final, breakdown


def _score_explanation(
    rule_score:    int,
    keyword_score: int,
    final_score:   int,
    has_jd:        bool,
    dim_breakdown: Dict,
) -> Dict:
    return {
        "resume_quality_score": {
            "score":          rule_score,
            "grade":          _grade(rule_score),
            "what_it_means":  "How well-written the resume is: structure, formatting, action verbs, bullet quality.",
            "interpretation": (
                "Excellent resume writing quality."     if rule_score >= 85 else
                "Good quality with minor improvements." if rule_score >= 70 else
                "Quality issues that need addressing."
            ),
        },
        "keyword_match_score": {
            "score":         keyword_score if has_jd else None,
            "grade":         _grade(keyword_score) if has_jd else "N/A (no JD provided)",
            "what_it_means": "How many keywords from the job description appear in the resume.",
            "interpretation": (
                "Not calculated — provide a job description to measure keyword match." if not has_jd else
                "High keyword alignment."              if keyword_score >= 70 else
                "Low keyword match — resume will likely be filtered out." if keyword_score < 50 else
                "Moderate keyword match — add more JD keywords."
            ),
        },
        "final_ats_score": {
            "score":         final_score,
            "grade":         _grade(final_score),
            "what_it_means": "Multi-dimensional score: keyword density 30% + experience depth 25% + achievement quality 20% + structure 15% + format 10%.",
            "interpretation": _ats_verdict(final_score),
        },
        "dimension_breakdown": dim_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# JSON REPAIR  (for AI response parsing)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text
    depth        = 0
    in_string    = False
    escape_next  = False
    end          = start
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    return text[start:end + 1]


def _escape_newlines_in_strings(text: str) -> str:
    result:      List[str] = []
    in_string    = False
    escape_next  = False
    for ch in text:
        if escape_next:
            escape_next = False
            result.append(ch)
            continue
        if ch == "\\" and in_string:
            escape_next = True
            result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            if ch == "\n":   result.append("\\n")
            elif ch == "\r": result.append("\\r")
            elif ch == "\t": result.append("\\t")
            else:            result.append(ch)
        else:
            result.append(ch)
    return "".join(result)


def _repair_json(raw: str) -> str:
    text = re.sub(r",\s*([\]}])", r"\1", raw)
    text = _escape_newlines_in_strings(text)
    text = text.rstrip().rstrip(",")
    open_braces   = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    text += "]" * max(open_brackets, 0)
    text += "}" * max(open_braces,   0)
    return text


def _try_parse_json(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = _strip_markdown_fences(text)
    cleaned = _extract_json_object(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    repaired = _repair_json(cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    try:
        swapped = re.sub(r"(?<![\\])'", '"', repaired)
        return json.loads(swapped)
    except Exception:
        pass

    return None


def _extract_fields_by_regex(text: str) -> Dict:
    result: Dict = {}
    for field in ("industry_detected", "role_level", "ats_compatibility_verdict", "overall_assessment"):
        m = re.search(rf'"{field}"\s*:\s*"([^"{{}}[\]]*)"', text)
        if m:
            result[field] = m.group(1).strip()

    for field in ("content_strengths", "ats_passing_tactics", "priority_action_plan"):
        m = re.search(rf'"{field}"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
        if m:
            items = re.findall(r'"([^"]*)"', m.group(1))
            if items:
                result[field] = items

    section_scores = {}
    for sec in ("summary", "experience", "skills", "education"):
        m = re.search(rf'"{sec}"\s*:\s*\{{\s*"score"\s*:\s*(\d+)', text)
        if m:
            section_scores[sec] = {"score": int(m.group(1)), "verdict": ""}
    if section_scores:
        result["ai_section_scores"] = section_scores

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class ATSScannerService:

    def __init__(self) -> None:
        self.rules_engine       = ATSRulesEngine()
        self.keyword_engine     = KeywordEngine()
        self.feedback_generator = DetailedFeedbackGenerator()

    async def scan(
        self,
        resume:          Dict,
        job_description: Optional[str] = None,
        db:              Optional[AsyncSession] = None,
        include_ai:      bool = True,
    ) -> Dict:
        logger.info("=== ATS Scan v5 Starting ===")

        contact_built = _build_contact_from_resume(resume)
        logger.info(
            f"  name='{contact_built.get('name')}' "
            f"edu={len(_get_education(resume))} "
            f"exp={len(_get_experience(resume))} "
            f"skills={len(_get_skills(resume))}"
        )

        logger.info("[Stage 1] ATSRulesEngine")
        rules_score = self.rules_engine.analyze(resume)

        logger.info("[Stage 2] KeywordEngine")
        keyword_analysis: Optional[KeywordAnalysis] = None
        keyword_score = 0
        has_jd = bool(job_description and job_description.strip())

        if has_jd:
            try:
                keyword_analysis = self.keyword_engine.match_skills(resume, job_description)
                keyword_score    = self.keyword_engine.calculate_keyword_score(keyword_analysis)
                logger.info(f"  match={keyword_analysis.match_percentage}% score={keyword_score}")
            except Exception as e:
                logger.warning(f"  Keyword engine error: {e}")

        ai_insights: Dict = {}
        if include_ai and db:
            logger.info("[Stage 3] AI analysis")
            try:
                ai_insights = await self._run_ai_analysis(
                    resume, job_description, rules_score.total_score,
                    keyword_analysis, db,
                )
                logger.info(f"  AI success={ai_insights.get('success')} partial={ai_insights.get('partial', False)}")
            except Exception as e:
                logger.warning(f"  AI failed (graceful fallback): {e}")
                ai_insights = {"success": False, "error": str(e)}

        logger.info("[Stage 3.5] Dynamic multi-dimensional scoring")
        final_score, dim_breakdown = _calculate_dynamic_score(resume, job_description, ai_insights)
        logger.info(f"  final={final_score}")

        logger.info("[Stage 4] Feedback generation")
        rules_score.section_issues["contact"] = _build_contact_section_proxy(contact_built)

        section_scores: Dict[str, int] = {}
        for s in [
            "contact", "summary", "experience", "education", "skills",
            "projects", "certifications", "languages", "volunteer",
            "publications", "awards", "hobbies", "references",
        ]:
            sa = rules_score.section_issues.get(s)
            section_scores[s] = getattr(sa, "current_score", 0) if sa else 0

        if ai_insights.get("success") and ai_insights.get("ai_section_scores"):
            for sec, ai_data in ai_insights["ai_section_scores"].items():
                if isinstance(ai_data, dict) and "score" in ai_data:
                    r_s = section_scores.get(sec, 0)
                    a_s = int(ai_data.get("score", r_s))
                    section_scores[sec] = int(r_s * 0.6 + a_s * 0.4)

        detailed_feedback = self.feedback_generator.generate_detailed_feedback(
            ats_score         = final_score,
            section_scores    = section_scores,
            resume            = resume,
            ats_issues        = rules_score.all_issues,
            section_analyses  = rules_score.section_issues,
        )

        logger.info("[Stage 5] Assembling response")
        response = self._build_response(
            rules_score, keyword_analysis, keyword_score, final_score,
            detailed_feedback, ai_insights, resume,
            section_scores, contact_built, has_jd, dim_breakdown,
        )
        logger.info(f"=== Scan complete — final={final_score}/100 ===")
        return response

    async def _run_ai_analysis(
        self,
        resume, job_description, ats_score,
        keyword_analysis, db,
    ) -> Dict:
        contact  = _build_contact_from_resume(resume)
        name     = _safe_for_prompt(contact.get("name") or "", 60)
        industry = (
            keyword_analysis.detected_industry if keyword_analysis
            else self.keyword_engine.detect_industry(resume)
        )
        summary = _safe_for_prompt(_get_summary(resume), 250)
        skills  = _safe_for_prompt(
            ", ".join(_get_skills(resume)[:15]), 200
        )

        exp_lines: List[str] = []
        for exp in _get_experience(resume)[:3]:
            for b in (exp.get("bullets") or [])[:2]:
                clean = _safe_for_prompt(b, 120)
                if clean:
                    exp_lines.append(clean)
        experience_bullets = "; ".join(exp_lines[:6]) or "Not provided"

        edu_parts: List[str] = []
        for edu in _get_education(resume)[:2]:
            if isinstance(edu, dict):
                d = _safe_for_prompt(edu.get("degree", ""), 60)
                i = _safe_for_prompt(edu.get("institution") or edu.get("college", ""), 80)
                y = _safe_for_prompt(edu.get("year", ""), 10)
                if d or i:
                    edu_parts.append(f"{d} at {i} ({y})".strip())
        education = "; ".join(edu_parts) or "Not provided"

        certs = ", ".join([
            _safe_for_prompt(c.get("name") or c.get("title", "") if isinstance(c, dict) else c, 60)
            for c in _get_certifications(resume)[:5] if c
        ]) or "None"

        projects = ", ".join([
            _safe_for_prompt(p.get("name") or p.get("title", "") if isinstance(p, dict) else p, 60)
            for p in _get_projects(resume)[:3] if p
        ]) or "None"

        additional = ", ".join([
            s for s in ["languages", "volunteer", "publications", "awards", "hobbies"]
            if _is_present(resume.get(s))
        ]) or "None"

        jd_text = _safe_for_prompt(job_description or "Not provided", 600)

        prompt = AI_ANALYSIS_PROMPT.format(
            name                = name or "Candidate",
            target_role         = _safe_for_prompt(resume.get("target_role") or self._guess_target_role(resume) or "Not specified", 60),
            industry            = _safe_for_prompt(industry, 40),
            summary             = summary or "Not provided",
            skills              = skills  or "Not provided",
            experience_bullets  = experience_bullets,
            education           = education,
            certifications      = certs,
            projects            = projects,
            additional_sections = additional,
            job_description     = jd_text,
            ats_score           = ats_score,
        )

        raw = await call_llm(user_message=prompt, agent_name="ats_scanner", db=db)
        return self._parse_ai_response(raw)

    def _parse_ai_response(self, raw: str) -> Dict:
        if not raw or not raw.strip():
            return {"success": False, "error": "Empty AI response"}

        parsed = _try_parse_json(raw)
        if parsed and isinstance(parsed, dict):
            parsed["success"] = True
            logger.info("AI JSON parsed successfully")
            return parsed

        logger.warning("Full JSON parse failed — extracting partial fields via regex")
        partial = _extract_fields_by_regex(raw)
        if partial:
            partial["success"] = bool(
                partial.get("overall_assessment") or
                partial.get("ai_section_scores") or
                partial.get("content_strengths")
            )
            partial["partial"] = True
            return partial

        return {"success": False, "error": "AI response could not be parsed", "raw": raw[:200]}

    def _build_response(
        self,
        rules_score,
        keyword_analysis: Optional[KeywordAnalysis],
        keyword_score:    int,
        final_score:      int,
        detailed_feedback: ComprehensiveFeedback,
        ai_insights:       Dict,
        resume:            Dict,
        section_scores:    Dict[str, int],
        contact_built:     Dict,
        has_jd:            bool,
        dim_breakdown:     Dict,
    ) -> Dict:
        section_output: Dict = {}
        for sn, sf in (detailed_feedback.section_feedback or {}).items():
            section_output[sn] = {
                "score":                sf.current_score,
                "target_score":         sf.target_score,
                "status":               sf.status,
                "grade":                _grade(sf.current_score),
                "is_present":           sf.is_present,
                "is_complete":          sf.is_complete,
                "quality_level":        sf.quality_level,
                "impact_potential":     sf.impact_potential,
                "missing_elements":     sf.missing_elements,
                "elements_to_remove":   sf.elements_to_remove,
                "quality_issues":       sf.quality_issues,
                "top_priority_fixes":   sf.top_priority_fixes,
                "quick_wins":           sf.quick_wins,
                "detailed_suggestions": sf.detailed_suggestions,
                "ats_passing_tips":     sf.ats_passing_tips,
                "rewrite_examples":     sf.rewrite_examples,
                "strengths":            sf.strengths,
            }

        keyword_output = None
        if keyword_analysis:
            keyword_output = {
                "total_jd_keywords":   keyword_analysis.total_jd_keywords,
                "matched_keywords":    keyword_analysis.matched_keywords,
                "match_percentage":    keyword_analysis.match_percentage,
                "detected_industry":   keyword_analysis.detected_industry,
                "keyword_density":     keyword_analysis.keyword_density,
                "missing_critical":    keyword_analysis.missing_critical_skills,
                "missing_preferred":   keyword_analysis.missing_preferred_skills,
                "found_strengths":     keyword_analysis.found_strengths,
                "ats_keyword_gaps":    keyword_analysis.ats_keyword_gaps[:15],
                "keyword_suggestions": keyword_analysis.keyword_suggestions[:10],
            }
            if ai_insights.get("success") and ai_insights.get("keyword_gaps"):
                existing = {g["keyword"] for g in keyword_analysis.ats_keyword_gaps}
                for ai_gap in ai_insights["keyword_gaps"]:
                    kw = ai_gap.get("missing_keyword", "")
                    if kw and kw not in existing:
                        keyword_output["ats_keyword_gaps"].append({
                            "keyword":           kw,
                            "criticality":       2.0 if ai_gap.get("importance") == "critical" else 1.0,
                            "is_required":       ai_gap.get("importance") == "critical",
                            "suggested_section": ai_gap.get("add_to_section", "skills"),
                            "source":            "ai",
                        })

        if ai_insights.get("success"):
            ai_block = {
                "status":                    "success",
                "partial":                   ai_insights.get("partial", False),
                "industry_detected":         ai_insights.get("industry_detected"),
                "role_level":                ai_insights.get("role_level"),
                "ats_compatibility_verdict": ai_insights.get("ats_compatibility_verdict"),
                "content_strengths":         ai_insights.get("content_strengths", []),
                "critical_improvements":     ai_insights.get("critical_improvements", []),
                "bullet_rewrites":           ai_insights.get("bullet_rewrites", []),
                "summary_rewrite":           ai_insights.get("summary_rewrite"),
                "missing_sections":          ai_insights.get("missing_sections", []),
                "ats_passing_tactics":       ai_insights.get("ats_passing_tactics", []),
                "overall_assessment":        ai_insights.get("overall_assessment"),
                "priority_action_plan":      ai_insights.get("priority_action_plan", []),
            }
        else:
            ai_block = {
                "status": "not_available",
                "reason": ai_insights.get("error", "AI analysis not enabled."),
            }

        grade         = _grade(final_score)
        status        = _status_from_score(final_score)
        contact_score, _, _, _ = _score_contact(contact_built)

        return {
            "ats_score":      final_score,
            "score_status":   status,
            "grade":          grade,
            "ready_to_apply": final_score >= 72,
            "ready_to_apply_verdict": (
                "Ready to apply — resume will pass ATS screening."
                if final_score >= 72 else
                "Not ready — fix critical and high-priority issues first."
            ),
            "score_breakdown": {
                "resume_quality_score": rules_score.total_score,
                "keyword_match_score":  keyword_score if has_jd else None,
                "final_ats_score":      final_score,
                "format_compliance":    rules_score.format_score,
                "structure_quality":    rules_score.structure_score,
                "content_quality":      rules_score.content_score,
                "ats_compliance":       rules_score.ats_compliance_score,
                "dimension_breakdown":  dim_breakdown,
            },
            "score_explanation": _score_explanation(
                rules_score.total_score, keyword_score, final_score, has_jd, dim_breakdown
            ),
            "contact_detected": {
                "name":     contact_built.get("name"),
                "email":    contact_built.get("email"),
                "phone":    contact_built.get("phone"),
                "location": contact_built.get("location"),
                "linkedin": contact_built.get("linkedin"),
                "github":   contact_built.get("github"),
                "score":    contact_score,
                "status":   _status_from_score(contact_score),
            },
            "issues":                self._format_issues(rules_score.all_issues),
            "critical_issues_count": rules_score.critical_issues_count,
            "section_analysis":      section_output,
            "keyword_analysis":      keyword_output,
            "ai_analysis":           ai_block,
            "recommendations": {
                "top_3_priorities":    detailed_feedback.top_3_priorities,
                "quick_wins":          detailed_feedback.quick_wins_summary,
                "improvement_roadmap": detailed_feedback.improvement_roadmap,
                "ats_passing_tactics": detailed_feedback.ats_passing_tactics,
                "recruiter_tips":      detailed_feedback.recruiter_tips,
                "estimated_improvement": detailed_feedback.estimated_improvement_potential,
            },
            "summary": {
                "ready_to_apply":          final_score >= 72,
                "grade":                   grade,
                "percentile_estimate":     detailed_feedback.percentile_estimate,
                "ats_compatibility_level": _compatibility_label(final_score),
                "ats_verdict":             _ats_verdict(final_score),
                "main_strengths":          detailed_feedback.strengths_summary,
                "main_weaknesses":         detailed_feedback.top_3_priorities,
                "key_findings":            self._key_findings(final_score, rules_score.critical_issues_count),
                "next_steps":              self._next_steps(final_score, rules_score.critical_issues_count),
            },
        }

    def _format_issues(self, issues) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {"critical": [], "high": [], "medium": [], "low": []}
        for issue in issues:
            sev   = issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity)
            entry = {
                "section":    issue.section,
                "message":    issue.message,
                "suggestion": issue.suggestion,
                "impact":     issue.impact_score,
            }
            if getattr(issue, "specific_example",    None): entry["example"]     = issue.specific_example
            if getattr(issue, "improvement_example", None): entry["improvement"] = issue.improvement_example
            if sev in out:
                out[sev].append(entry)
        return out

    @staticmethod
    def _key_findings(score: int, critical: int) -> List[str]:
        out = []
        if critical > 0:
            out.append(f"{critical} critical ATS issue(s) — fix before applying")
        if score >= 85:
            out.append("Highly optimised — strong ATS compatibility")
        elif score >= 72:
            out.append("Passes ATS screening — minor refinements recommended")
        elif score >= 55:
            out.append("Significant ATS weaknesses — improvement needed")
        else:
            out.append("Likely rejected by ATS — major revision required")
        return out

    @staticmethod
    def _next_steps(score: int, critical: int) -> List[str]:
        steps = []
        if critical > 0:
            steps.append(f"Fix {critical} critical issue(s) immediately (see issues → critical)")
        if score < 55:
            steps.append("Follow the improvement roadmap section by section")
            steps.append("Re-scan after each batch of changes to track progress")
        elif score < 72:
            steps.append("Address all High-severity issues (see issues → high)")
            steps.append("Apply Quick Wins — each takes under 15 minutes")
        else:
            steps.append("Ready to apply — tailor keywords to each specific job posting")
        return steps

    @staticmethod
    def _guess_target_role(resume: Dict) -> Optional[str]:
        for exp in _get_experience(resume)[:1]:
            title = _safe(exp.get("title") or exp.get("position", ""))
            if title:
                return title
        return None


async def create_ats_scan(
    resume:          Dict,
    job_description: Optional[str] = None,
    llm_client:      Optional[Callable] = None,
    db:              Optional[AsyncSession] = None,
    include_ai:      bool = True,
) -> Dict:
    scanner = ATSScannerService()
    return await scanner.scan(
        resume=resume,
        job_description=job_description,
        db=db,
        include_ai=include_ai and db is not None,
    )