from __future__ import annotations

import json
import logging
import math
import re
import asyncio
import hashlib
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.ats_scanner.utils.ats_rules_advanced import ATSRulesEngine, SeverityLevel
from app.modules.ats_scanner.utils.ats_keyword_engine import (
    KeywordEngine, KeywordAnalysis, UNIVERSAL_SKILLS,
)
from app.modules.ats_scanner.utils.ats_feedback_generator import (
    DetailedFeedbackGenerator, ComprehensiveFeedback,
    GLOBAL_ATS_TACTICS, RECRUITER_TIPS,
)
from app.modules.ats_scanner.utils.ats_normalizer import (
    NormalizedResume, NormalizedContact,
    normalize_resume,
)
from app.modules.resume_builder.ai_client import call_ai

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

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
    "executed", "planned", "coordinated", "monitored", "tracked",
    "conducted", "supported", "created", "prepared", "managed",
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

# Hard caps are now keyed by condition label → cap value
# They are applied ONLY when the normalized model confirms the condition.
_HARD_CAPS: Dict[str, int] = {
    "no_experience": 40,
    "no_skills":     50,
    "no_email":      85,
}

# ── Fresher / candidate-type detection signals ─────────────────────────────

_FRESHER_KEYWORDS_RE = re.compile(
    r"\b(fresher|fresh graduate|recent graduate|recently graduated|"
    r"currently pursuing|pursuing\s+(?:b\.?tech|b\.?e|bachelor|master|m\.?tech|mba)|"
    r"final[\s\-]year|seeking\s+(?:an?\s+)?entry[\s\-]level|"
    r"entry[\s\-]level\s+(?:position|role|opportunity)|"
    r"0\s+years?\s+of\s+experience|no\s+(?:prior|professional|formal)\s+experience|"
    r"aspiring|career\s+starter|first\s+job|looking\s+for\s+my\s+first)\b", re.I,
)

_SENIOR_TITLE_RE = re.compile(
    r"\b(senior|lead|principal|staff|director|head of|vp\b|vice president|"
    r"chief|manager|ceo|cto|cfo|coo)\b", re.I,
)

_INTERN_TITLE_RE = re.compile(r"\b(intern|internship|trainee|apprentice)\b", re.I)

_YEARS_EXPERIENCE_CLAIM_RE = re.compile(
    r"(\d{1,2})\+?\s*years?\s+(?:of\s+)?(?:professional\s+|relevant\s+|industry\s+)?experience",
    re.I,
)

AI_ANALYSIS_PROMPT = """You are a senior ATS expert. Analyse this resume and respond ONLY with valid JSON.

RESUME:
Name: {name}
Role: {target_role}
Industry: {industry}
Candidate Type: {candidate_type} (fresher = early-career/student — do NOT recommend "add work experience"; instead focus on projects, internships, certifications, and education. experienced = has a formal work history.)
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
# UTILITY HELPERS
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


def _has_metric(text: str) -> bool:
    return any(p.search(text) for p in _METRIC_PATS)


def _has_strong_verb(text: str) -> bool:
    words = re.findall(r"\b\w+\b", text.lower())
    return bool(words and words[0] in _STRONG_VERBS)


def _has_weak_opener(text: str) -> bool:
    lower = text.lower()
    return any(lower.startswith(w) for w in _WEAK_OPENERS)


# ─────────────────────────────────────────────────────────────────────────────
# DATE PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date_to_months(raw: str) -> Optional[int]:
    raw = _safe(raw).lower()
    if not raw or re.match(r"present|current|now|till", raw, re.I):
        return 12 * 2026 + 6          # "now" → June 2026

    for abbr, num in _MONTH_MAP.items():
        if abbr in raw:
            ym = re.search(r"(19|20)\d{2}", raw)
            if ym:
                return int(ym.group(0)) * 12 + num

    ym = re.search(r"(19|20)\d{2}", raw)
    if ym:
        return int(ym.group(0)) * 12 + 6   # year only → mid-year

    return None


def _duration_months(start: str, end: str) -> int:
    s = _parse_date_to_months(start)
    e = _parse_date_to_months(end)
    if s is None or e is None:
        return 0
    return max(0, e - s)


def _quick_total_experience_months(nr: NormalizedResume) -> int:
    """Cheap pre-scoring estimate of total experience duration in months,
    used only to feed candidate-type detection before full dimension
    scoring runs."""
    total = 0
    for exp in nr.experience:
        dur = _duration_months(exp.start_date, exp.end_date)
        if dur == 0 and exp.start_date:
            dur = 12
        total += dur
    return total


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE TYPE DETECTION (fresher vs. experienced)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_candidate_type(
    nr: NormalizedResume, resume_raw: Dict, total_exp_months: int
) -> str:
    """
    Heuristic fresher/experienced classification.

    Defaults to "experienced" unless there is a positive corroborating
    signal of early-career status (fresher language, recent/ongoing
    education, or an internship-only work history). A resume that simply
    has no listed experience but shows NO other early-career signal is
    still treated as "experienced" (e.g. an employment section omitted
    by mistake) — this preserves the existing hard-cap protection for
    that scenario instead of letting every blank experience section
    bypass it.
    """
    text_blob = " ".join(filter(None, [
        nr.summary or "",
        (nr.raw_text or "")[:4000],
    ]))

    has_fresher_kw = bool(_FRESHER_KEYWORDS_RE.search(text_blob))

    non_intern_roles = [e for e in nr.experience if not _INTERN_TITLE_RE.search(e.title or "")]

    intern_roles      = [e for e in nr.experience if _INTERN_TITLE_RE.search(e.title or "")]

    has_only_internships = bool(nr.experience) and not non_intern_roles and bool(intern_roles)
    has_senior_title = any(_SENIOR_TITLE_RE.search(e.title or "") for e in nr.experience)

    years_claim_match = _YEARS_EXPERIENCE_CLAIM_RE.search(text_blob)
    claims_multi_year_experience = bool(
        years_claim_match and int(years_claim_match.group(1)) >= 2
    )

    # Recent / in-progress education (no graduation year, or a graduation
    # year within the last ~2 years) is a weak positive fresher signal —
    # only counted when combined with thin/no work history.
    recent_or_ongoing_education = False
    for edu in nr.education:
        year_str = (edu.year or "").strip()
        if not year_str:
            recent_or_ongoing_education = True
            continue
        ym = re.search(r"(19|20)\d{2}", year_str)
        if ym:
            try:
                if int(ym.group(0)) >= 2024:
                    recent_or_ongoing_education = True
            except ValueError:
                pass

    thin_work_history = total_exp_months <= 12 and not has_senior_title and not claims_multi_year_experience

    is_fresher = (
        not claims_multi_year_experience
        and not has_senior_title
        and (
            has_fresher_kw
            or has_only_internships
            or (not nr.experience and recent_or_ongoing_education)
            or (thin_work_history and recent_or_ongoing_education and (nr.has_projects or has_only_internships))
        )
    )

    return "fresher" if is_fresher else "experienced"


def _score_contact(contact: NormalizedContact) -> Tuple[int, List[str], List[str], List[str]]:

    """

    Score the contact section from the canonical NormalizedContact object.
    Returns (score, missing_fields, quality_issues, strengths).
    All feedback reflects ACTUAL parsed values — never contradicts them.
    """
    score     = 100
    missing:   List[str] = []
    quality:   List[str] = []
    strengths: List[str] = []

    if not contact.name:
        missing.append("Full name")
        score -= 25
    else:
        strengths.append(f"Name present: {contact.name}")

    if not contact.email:
        missing.append("Email address")
        score -= 25
    else:
        if re.search(r"@(hotmail|yahoo|rediffmail|ymail)\.", contact.email, re.I):
            quality.append(f"Unprofessional email domain: {contact.email} — consider Gmail or custom domain")
        else:
            strengths.append(f"Professional email: {contact.email}")

    if not contact.phone:
        missing.append("Phone number")
        score -= 15
    else:
        strengths.append("Phone number present")

    if not contact.location:
        quality.append("Location missing — many ATS filter by city/country")
        score -= 10
    else:
        strengths.append(f"Location: {contact.location}")

    if not contact.linkedin:
        quality.append("No LinkedIn URL — 90% of recruiters check LinkedIn before contacting")
        score -= 5
    else:
        strengths.append(f"LinkedIn linked: {contact.linkedin}")

    if contact.github:
        strengths.append(f"GitHub profile linked: {contact.github}")

    return max(score, 0), missing, quality, strengths
class _DictProxy:

    """Thin dict wrapper that also supports attribute access."""

    def __init__(self, d: Dict) -> None:

        self._d = d
        def __getattr__(self, name: str):
            try:
                return self._d[name]
            except KeyError:
                return None

    def get(self, key, default=None):
        return self._d.get(key, default)
    
def _build_contact_section_proxy(contact: NormalizedContact) -> _DictProxy:
    """
    Build the SectionAnalysis-compatible proxy for the contact section.
    score/status/is_present are all derived from the SAME contact object
    so they are always internally consistent (BUG 5).
    """

    score, missing, quality, strengths = _score_contact(contact)

    ats_tips = [
        "Keep contact info in the resume body — NOT a header/footer.",
        "Use City, State/Country only — not your full street address.",
        "Always include a country code in your phone number.",
        "Add your LinkedIn URL — recruiters verify before scheduling interviews.",
        "Professional email format: firstname.lastname@gmail.com",
    ]
    # is_present = True when at least name or email is present
    is_present = bool(contact.name or contact.email)
    status     = "missing" if not is_present else _status_from_score(score)

    data = {
        "section_name":   "contact",
        "current_score":  score,
        "missing_fields": missing,
        "quality_issues": quality,
        "strengths":      strengths,
        "ats_tips":       ats_tips,
        "improvements":   [f"Add {m}" for m in missing] + quality,
        "rewrite_examples": [],
        "current_status": [status],
        "complete":       not missing,
        # These are used by DetailedFeedbackGenerator
        "is_present":     is_present,
        "is_complete":    not missing,
    }
    return _DictProxy(data)


_KNOWN_SKILLS_SET: Set[str] = set()

for _cats in UNIVERSAL_SKILLS.values():

    for _skills in _cats.values():

        _KNOWN_SKILLS_SET.update(s.lower() for s in _skills)
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

        "social media marketing": {"smm"},

        }
_REV_SYN: Dict[str, str] = {}

for _can, _vars in _SYNONYMS.items():

    for _v in _vars:

        _REV_SYN[_v.lower()] = _can.lower()

def _dim_keyword(nr: NormalizedResume, job_description: Optional[str]) -> Dict:

    """

    Keyword / skill density dimension.
    Skill count = len(nr.skills) — the already-deduplicated canonical list.
    NO additional regex scanning of raw_text.  This eliminates inflation.
    """
    skill_count = len(nr.skills)
    # Build a set of canonical lowercase skill names for matching
    resume_skills_set: Set[str] = set()
    for s in nr.skills:
        lower = s.lower()
        resume_skills_set.add(lower)
        # Also add canonical synonym mapping
        canon = _REV_SYN.get(lower, lower)
        resume_skills_set.add(canon)

    if not job_description or not job_description.strip():
        raw = _clamp(100.0 / (1.0 + math.exp(-0.2 * (skill_count - 12))))
        return {
            "raw_score":   raw,
            "weight":      0.30,
            "skill_count": skill_count,
            "jd_provided": False,
            "penalties":   (["No job description provided"] if skill_count < 5 else []),
            "bonuses":     ([f"{skill_count} skills listed"] if skill_count >= 8 else []),
        }

    jd_lower  = job_description.lower()
    jd_words  = re.findall(r"\b[\w\+#\.]+\b", jd_lower)
    jd_bigrams = [f"{jd_words[i]} {jd_words[i+1]}" for i in range(len(jd_words) - 1)]
    jd_tokens  = list(set(jd_words + jd_bigrams))

    freq: Dict[str, int] = {}
    for tok in jd_tokens:
        canon = _REV_SYN.get(tok, tok)
        if canon in _KNOWN_SKILLS_SET or tok in _KNOWN_SKILLS_SET:
            freq[canon] = freq.get(canon, 0) + jd_lower.count(tok)

    if not freq:
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
        found = skill in resume_skills_set
        if not found:
            for var in _SYNONYMS.get(skill, set()):
                if var.lower() in resume_skills_set:
                    found = True
                    break
        if found:
            matched_weight += w
            matched_skills.append(skill)
        elif cnt >= 3:
            missing_critical.append(skill)

    match_ratio = matched_weight / total_weight if total_weight > 0 else 0.0
    raw = _clamp(match_ratio * 100)

    total_words = len(nr.raw_text.split()) if nr.raw_text else 1
    density = skill_count / max(total_words, 1)
    if density > 0.06:
        raw = min(raw + 5, 100)

    return {
        "raw_score":        raw,
        "weight":           0.30,
        "jd_skills_detected": len(freq),
        "matched":          len(matched_skills),
        "match_ratio":      round(match_ratio, 3),
        "skill_count":      skill_count,
        "missing_critical": missing_critical,
        "keyword_density":  round(density, 4),
        "penalties":        [f'Missing critical keyword: "{s}"' for s in missing_critical[:5]],
        "bonuses":          [f'Matched: "{s}"' for s in matched_skills[:5]],
    }


def _dim_experience(nr: NormalizedResume, candidate_type: str) -> Dict:

    """

    Score work experience from NormalizedResume.
    NormalizedExperience always has .title / .company / .start_date /
    .end_date / .bullets — regardless of which parser produced the input.
    The hard cap check in _apply_hard_caps() uses nr.has_experience AND
    candidate_type, not this score, eliminating false-cap BUG 9 while
    also no longer penalising fresher profiles as heavily as experienced
    ones for a thin/absent work history (v6.2).
    """
    # Freshers carry far less weight on the experience dimension — their
    # standing is judged mainly through fresher_achievements instead.
    weight = 0.10 if candidate_type == "fresher" else 0.25

    if not nr.has_experience:
        return {
            "raw_score": 0.0, "weight": weight,
            "total_months": 0, "roles": 0,
            "penalties": (
                ["No internships, freelance, or work experience entries found"]
                if candidate_type == "fresher" else
                ["No work experience entries found"]
            ),
            "bonuses": [],
        }

    total_months   = 0
    complete_roles = 0
    role_count     = len(nr.experience)
    penalties: List[str] = []
    bonuses:   List[str] = []

    for idx, exp in enumerate(nr.experience):
        dur = _duration_months(exp.start_date, exp.end_date)
        if dur == 0 and exp.start_date:
            dur = 12              # assume ~1 year if start is given but end unclear
        total_months += dur

        complete = bool(exp.title and exp.company and exp.bullets)
        if complete:
            complete_roles += 1
        else:
            miss = []
            if not exp.title:   miss.append("title")
            if not exp.company: miss.append("company")
            if not exp.bullets: miss.append("bullets")
            if miss:
                label = exp.title or exp.company or f"Role {idx+1}"
                penalties.append(f"{label}: missing {', '.join(miss)}")

        title_lower = exp.title.lower()
        if any(t in title_lower for t in ("senior", "lead", "principal", "head", "director", "vp", "chief", "manager")):
            bonuses.append(f"Senior role: {exp.title}")
        if _INTERN_TITLE_RE.search(title_lower):
            bonuses.append(f"Internship experience: {exp.title}")

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
        "weight":           weight,
        "total_months":     total_months,
        "total_years":      round(total_months / 12, 1),
        "roles":            role_count,
        "complete_roles":   complete_roles,
        "completeness_pct": round(completeness * 100),
        "penalties":        penalties,
        "bonuses":          bonuses,
    }


def _dim_fresher_achievements(nr: NormalizedResume, candidate_type: str) -> Dict:

    """

    Credits projects, internships, and certifications as positive signal

    for early-career candidates. Carries ZERO weight for experienced

    candidates (computed for transparency/debugging only), so existing

    scoring for experienced resumes is completely unaffected.

    """

    if candidate_type != "fresher":

        return {

        "raw_score": 0.0, "weight": 0.0,

        "penalties": [], "bonuses": [],

        "note": "Not applied — candidate classified as experienced.",

        }
    project_count = len(nr.projects)
    intern_count  = sum(1 for e in nr.experience if _INTERN_TITLE_RE.search(e.title or ""))
    cert_count    = len(nr.certifications)

    bonuses: List[str] = []
    penalties: List[str] = []

    project_pts = _clamp(min(project_count * 12.0, 40.0))
    if project_count:
        bonuses.append(f"{project_count} project(s) demonstrating practical skills")
    else:
        penalties.append("No projects listed — add academic, personal, or capstone projects")

    intern_pts = _clamp(min(intern_count * 20.0, 35.0))
    if intern_count:
        bonuses.append(f"{intern_count} internship(s) — directly relevant experience")
    else:
        penalties.append("No internships listed — consider adding any internship, training, or apprenticeship")

    cert_pts = _clamp(min(cert_count * 6.0, 15.0))
    if cert_count:
        bonuses.append(f"{cert_count} certification(s) supporting skills claims")

    edu_pts = 10.0 if nr.has_education else 0.0

    raw = _clamp(project_pts + intern_pts + cert_pts + edu_pts)

    return {
        "raw_score":     raw,
        "weight":        0.15,
        "project_count": project_count,
        "intern_count":  intern_count,
        "cert_count":    cert_count,
        "penalties":     penalties,
        "bonuses":       bonuses,
    }

def _dim_quality(nr: NormalizedResume) -> Dict:

    """

    Score content quality (bullet strength, summary depth, skill specificity).
    nr.all_bullets aggregates bullets from ALL normalized experience entries.
    nr.summary is always a plain string.
    """
    all_bullets = nr.all_bullets     # List[str] — normalized field

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

    # nr.summary is always a plain string after normalization
    summary_score = 0.0
    if not nr.summary:
        penalties.append("Professional summary missing")
    else:
        words = len(nr.summary.split())
        if words < 20:
            summary_score = 8.0
            penalties.append(f"Summary too brief ({words} words)")
        elif words > 150:
            summary_score = 15.0
            penalties.append("Summary too long — trim to 50–80 words")
        else:
            summary_score = 20.0
        if _has_metric(nr.summary):
            summary_score = min(summary_score + 5.0, 25.0)
            bonuses.append("Summary contains quantified achievement")

    generic_terms = {
        "ms office", "microsoft office", "computers", "communication",
        "teamwork", "adaptability", "hardworking", "detail oriented",
        "fast learner", "quick learner", "team player",
    }
    specific_count = sum(
        1 for s in nr.skills
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

def _dim_structure(nr: NormalizedResume) -> Dict:

    earned   = 0.0

    max_pts  = 85.0

    penalties: List[str] = []

    bonuses:   List[str] = []
    # Summary (20 pts)
    if nr.has_summary and len(nr.summary.split()) >= 15:
        earned += 20
        bonuses.append("Professional summary present")
    elif nr.has_summary:
        earned += 10
        penalties.append("Summary present but too brief")
    else:
        penalties.append("Professional summary missing")

    # Experience (20 pts)
    if nr.has_experience:
        earned += 20
        bonuses.append(f"{len(nr.experience)} work experience role(s) present")
    else:
        penalties.append("Work experience section missing")

    # Skills (20 pts)
    skill_count = len(nr.skills)
    if skill_count >= 5:
        earned += 20
        bonuses.append(f"{skill_count} skills listed")
    elif skill_count > 0:
        earned += 10
        penalties.append(f"Too few skills ({skill_count} — add at least 5)")
    else:
        penalties.append("Skills section missing or empty")

    # Education (15 pts)
    if nr.has_education:
        earned += 15
        bonuses.append(f"{len(nr.education)} education entr(ies) present")
    else:
        penalties.append("Education section not found")

    # Contact email (5 pts)
    if nr.has_contact_email:
        earned += 5
        bonuses.append("Email present")
    else:
        penalties.append("Email address missing")

    # Projects — small bonus (5 pts)  BUG 7
    if nr.has_projects:
        earned += 5
        bonuses.append(f"{len(nr.projects)} project(s) present")

    raw = _clamp((earned / max_pts) * 100)
    return {
        "raw_score":  raw,
        "weight":     0.15,
        "pts_earned": round(earned),
        "pts_max":    round(max_pts),
        "penalties":  penalties,
        "bonuses":    bonuses,
    }


def _dim_format(resume_raw: Dict, nr: NormalizedResume) -> Dict:

    score    = 100.0

    penalties: List[str] = []

    bonuses:   List[str] = []
    if resume_raw.get("uses_table") or resume_raw.get("has_tables"):
        score -= 15
        penalties.append("Tables detected — ATS cannot parse table content")
    if resume_raw.get("uses_columns") or resume_raw.get("multi_column"):
        score -= 12
        penalties.append("Multi-column layout — ATS reads columns out of order")
    if resume_raw.get("has_images") or resume_raw.get("uses_graphics"):
        score -= 8
        penalties.append("Images/graphics detected — ATS cannot read visual elements")

    if nr.contact.linkedin:
        bonuses.append("LinkedIn profile linked")
        score = min(score + 3, 100)
    if nr.contact.github:
        bonuses.append("GitHub profile linked")
        score = min(score + 2, 100)

    return {
        "raw_score": _clamp(score),
        "weight":    0.10,
        "penalties": penalties,
        "bonuses":   bonuses,
    }


def _apply_hard_caps(score: float, nr: NormalizedResume, candidate_type: str) -> Tuple[float, List[str]]:

    """

    Apply hard caps ONLY when the NORMALIZED model confirms the condition.

    A mapping failure in a dimension can no longer trigger a false cap.
    v6.2: the "no_experience" cap is skipped for candidates classified as
    "fresher" — they're evaluated via fresher_achievements instead, so a
    genuine early-career resume with strong projects/internships is no
    longer artificially ceilinged at 40.
    """
    applied: List[str] = []

    if not nr.has_experience and candidate_type != "fresher":
        if score > _HARD_CAPS["no_experience"]:
            score = float(_HARD_CAPS["no_experience"])
            applied.append(f"No work experience — score capped at {_HARD_CAPS['no_experience']}")

    if not nr.has_skills:
        if score > _HARD_CAPS["no_skills"]:
            score = float(_HARD_CAPS["no_skills"])
            applied.append(f"No skills section — score capped at {_HARD_CAPS['no_skills']}")

    if not nr.has_contact_email:
        if score > _HARD_CAPS["no_email"]:
            score = float(_HARD_CAPS["no_email"])
            applied.append(f"Missing email — score capped at {_HARD_CAPS['no_email']}")

    return score, applied


def _calculate_dynamic_score(

    resume_raw:      Dict,

    nr:              NormalizedResume,

    job_description: Optional[str],

    ai_insights:     Dict,

    candidate_type:  str,

    ) -> Tuple[int, Dict]:

    d_kw      = _dim_keyword(nr, job_description)

    d_exp     = _dim_experience(nr, candidate_type)

    d_fresher = _dim_fresher_achievements(nr, candidate_type)

    d_quality = _dim_quality(nr)

    d_struct  = _dim_structure(nr)

    d_format  = _dim_format(resume_raw, nr)
    raw_weighted = (
        d_kw["raw_score"]      * d_kw["weight"]      +
        d_exp["raw_score"]     * d_exp["weight"]      +
        d_fresher["raw_score"] * d_fresher["weight"]  +
        d_quality["raw_score"] * d_quality["weight"]  +
        d_struct["raw_score"]  * d_struct["weight"]   +
        d_format["raw_score"]  * d_format["weight"]
    )

    logger.info(
        f"  type={candidate_type} kw={d_kw['raw_score']:.1f}  exp={d_exp['raw_score']:.1f}  "
        f"fresher={d_fresher['raw_score']:.1f}  qual={d_quality['raw_score']:.1f}  "
        f"struct={d_struct['raw_score']:.1f}  fmt={d_format['raw_score']:.1f}  "
        f"→ weighted={raw_weighted:.1f}"
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
    capped, caps_applied = _apply_hard_caps(raw_with_ai, nr, candidate_type)
    final = max(0, min(100, round(capped)))

    breakdown = {
        "candidate_type":              candidate_type,
        "keyword_skill_density":      {"raw": round(d_kw["raw_score"],      1), "weight": d_kw["weight"],      "weighted": round(d_kw["raw_score"]      * d_kw["weight"],      2), **{k: v for k, v in d_kw.items()      if k not in ("raw_score", "weight")}},
        "experience_depth_duration":  {"raw": round(d_exp["raw_score"],     1), "weight": d_exp["weight"],     "weighted": round(d_exp["raw_score"]     * d_exp["weight"],     2), **{k: v for k, v in d_exp.items()     if k not in ("raw_score", "weight")}},
        "fresher_achievements":       {"raw": round(d_fresher["raw_score"], 1), "weight": d_fresher["weight"], "weighted": round(d_fresher["raw_score"] * d_fresher["weight"], 2), **{k: v for k, v in d_fresher.items() if k not in ("raw_score", "weight")}},
        "achievement_quality":        {"raw": round(d_quality["raw_score"], 1), "weight": d_quality["weight"], "weighted": round(d_quality["raw_score"] * d_quality["weight"], 2), **{k: v for k, v in d_quality.items() if k not in ("raw_score", "weight")}},
        "structure_completeness":     {"raw": round(d_struct["raw_score"],  1), "weight": d_struct["weight"],  "weighted": round(d_struct["raw_score"]  * d_struct["weight"],  2), **{k: v for k, v in d_struct.items()  if k not in ("raw_score", "weight")}},
        "format_ats_compliance":      {"raw": round(d_format["raw_score"],  1), "weight": d_format["weight"],  "weighted": round(d_format["raw_score"]  * d_format["weight"],  2), **{k: v for k, v in d_format.items()  if k not in ("raw_score", "weight")}},
        "hard_caps_applied":          caps_applied,
        "ai_bonus":                   ai_bonus,
    }

    return final, breakdown


def _build_section_scores(nr: NormalizedResume) -> Dict[str, int]:
    """
    Build per-section scores consumed by DetailedFeedbackGenerator.
    Scores for sections not handled by ATSRulesEngine (projects, languages,
    certifications) are derived directly from normalized presence.
    Fixes BUG 7 (projects) and BUG 8 (languages).
    """
    def _presence_score(present: bool, count: int = 0) -> int:
        """
        Returns a score that reflects actual presence.
        Never returns 0 when the section exists (BUG 8).
        """
        if not present:
            return 0
        if count >= 3:
            return 85
        if count >= 1:
            return 70
        return 60

    return {
        "contact":         0,    # overwritten below from _score_contact
        "summary":         85 if nr.has_summary and len(nr.summary.split()) >= 20 else (40 if nr.has_summary else 0),
        "experience":      0,    # will be overwritten from rules engine + dim
        "education":       85 if nr.has_education else 0,
        "skills":          85 if len(nr.skills) >= 8 else (60 if nr.has_skills else 0),
        "projects":        _presence_score(nr.has_projects, len(nr.projects)),
        "certifications":  _presence_score(nr.has_certifications, len(nr.certifications)),
        "languages":       _presence_score(nr.has_languages, len(nr.languages)),
        "volunteer":       _presence_score(bool(nr.volunteer), len(nr.volunteer)),
        "publications":    _presence_score(bool(nr.publications), len(nr.publications)),
        "awards":          _presence_score(bool(nr.awards), len(nr.awards)),
        "hobbies":         _presence_score(bool(nr.hobbies), len(nr.hobbies)),
        "references":      0,
    }

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
            "score":          keyword_score if has_jd else None,
            "grade":          _grade(keyword_score) if has_jd else "N/A (no JD provided)",
            "what_it_means":  "How many keywords from the job description appear in the resume.",
            "interpretation": (
                "Not calculated — provide a job description to measure keyword match." if not has_jd else
                "High keyword alignment."              if keyword_score >= 70 else
                "Low keyword match — resume will likely be filtered out." if keyword_score < 50 else
                "Moderate keyword match — add more JD keywords."
            ),
        },
        "final_ats_score": {
            "score":          final_score,
            "grade":          _grade(final_score),
            "what_it_means":  "Multi-dimensional score: keyword density 30% + experience/fresher-achievements 25-35% + achievement quality 20% + structure 15% + format 10%.",
            "interpretation": _ats_verdict(final_score),
        },
        "dimension_breakdown": dim_breakdown,
    }

def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text)
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
    result: List[str] = []
    in_string   = False
    escape_next = False

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
            if ch == "\n":
                result.append("\\n")
            elif ch == "\r":
                result.append("\\r")
            elif ch == "\t":
                result.append("\\t")
            else:
                result.append(ch)
        else:
            result.append(ch)

    return "".join(result)

def _repair_json(raw: str) -> str:
    text = re.sub(r",\s*([]}])", r"\1", raw)
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
        swapped = re.sub(r"(?<![\\])'", '\"', repaired)
        return json.loads(swapped)
    except Exception:
        pass

    return None

def _extract_fields_by_regex(text: str) -> Dict:
    result: Dict = {}
    for field in ("industry_detected", "role_level", "ats_compatibility_verdict", "overall_assessment"):
        m = re.search(rf'"({field})"\s*:\s*"([^\"{{}}\[\]]+)"', text)
        if m:
            result[field] = m.group(2).strip()

    for field in ("content_strengths", "ats_passing_tactics", "priority_action_plan"):
        m = re.search(rf'"({field})"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if m:
            items = re.findall(r'"([^"]+)"', m.group(2))
            if items:
                result[field] = items

    section_scores = {}
    for sec in ("summary", "experience", "skills", "education"):
        m = re.search(rf'"({sec})"\s*:\s*{{\s*"score"\s*:\s*(\d+)', text)
        if m:
            section_scores[sec] = {"score": int(m.group(2)), "verdict": ""}
    
    if section_scores:
        result["ai_section_scores"] = section_scores

    return result


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
        logger.info("=== ATS Scan v6.2 Starting ===")

        # ── 0. Normalize — single source of truth for all downstream ─────────
        nr = normalize_resume(resume)
        logger.info(
            f"  name='{nr.contact.name}' "
            f"edu={len(nr.education)} exp={len(nr.experience)} "
            f"skills={len(nr.skills)} certs={len(nr.certifications)} "
            f"proj={len(nr.projects)} langs={len(nr.languages)}"
        )

        # ── 0.5. Candidate-type detection (fresher vs experienced) ────────────
        total_exp_months = _quick_total_experience_months(nr)
        candidate_type = _detect_candidate_type(nr, resume, total_exp_months)
        logger.info(f"  candidate_type={candidate_type} (total_exp_months={total_exp_months})")

        # ── 1. Rules engine (format/structure checks) ─────────────────────────
        logger.info("[Stage 1] ATSRulesEngine")
        # Build a rules-compatible dict from the normalized model
        rules_input = _nr_to_rules_dict(nr, resume)
        rules_score = self.rules_engine.analyze(rules_input, candidate_type=candidate_type)

        # ── 2. Keyword engine ─────────────────────────────────────────────────
        logger.info("[Stage 2] KeywordEngine")
        keyword_analysis: Optional[KeywordAnalysis] = None
        keyword_score = 0
        has_jd = bool(job_description and job_description.strip())

        if has_jd:
            try:
                # Build a flat resume dict the keyword engine can consume
                kw_input = _nr_to_keyword_dict(nr)
                keyword_analysis = self.keyword_engine.match_skills(kw_input, job_description)
                keyword_score    = self.keyword_engine.calculate_keyword_score(keyword_analysis)
                logger.info(f"  match={keyword_analysis.match_percentage}% score={keyword_score}")
            except Exception as e:
                logger.warning(f"  Keyword engine error: {e}")

        # ── 3. AI analysis (via Resume Builder's shared AI client) ────────────
        ai_insights: Dict = {}
        if include_ai and db:
            logger.info("[Stage 3] AI analysis")
            try:
                ai_insights = await self._run_ai_analysis(
                    nr, resume, job_description,
                    rules_score.total_score,
                    keyword_analysis, db, candidate_type,
                )
                logger.info(f"  AI success={ai_insights.get('success')}")
            except Exception as e:
                logger.warning(f"  AI failed (graceful fallback): {e}")
                ai_insights = {"success": False, "error": str(e)}

        # ── 3.5. Dynamic multi-dimensional scoring ────────────────────────────
        logger.info("[Stage 3.5] Dynamic multi-dimensional scoring")
        final_score, dim_breakdown = _calculate_dynamic_score(
            resume, nr, job_description, ai_insights, candidate_type
        )
        logger.info(f"  final={final_score}")

        # ── 4. Feedback generation ────────────────────────────────────────────
        logger.info("[Stage 4] Feedback generation")

        # Build section scores entirely from normalized data
        section_scores = _build_section_scores(nr)

        # Overwrite contact score from canonical contact scorer
        contact_score, _, _, _ = _score_contact(nr.contact)
        section_scores["contact"] = contact_score

        # Overwrite experience from rules engine (it inspects normalized input)
        rules_score.section_issues["contact"] = _build_contact_section_proxy(nr.contact)
        for sn in ("summary", "experience", "education", "skills"):
            sa = rules_score.section_issues.get(sn)
            if sa:
                rs = getattr(sa, "current_score", 0)
                if rs > 0:
                    section_scores[sn] = max(section_scores[sn], rs)

        # Blend AI section scores
        if ai_insights.get("success") and ai_insights.get("ai_section_scores"):
            for sec, ai_data in ai_insights["ai_section_scores"].items():
                if isinstance(ai_data, dict) and "score" in ai_data:
                    r_s = section_scores.get(sec, 0)
                    a_s = int(ai_data.get("score", r_s))
                    section_scores[sec] = int(r_s * 0.6 + a_s * 0.4)

        # Build a resume dict the feedback generator can inspect
        feedback_resume = _nr_to_feedback_dict(nr)

        detailed_feedback = self.feedback_generator.generate_detailed_feedback(
            ats_score         = final_score,
            section_scores    = section_scores,
            resume            = feedback_resume,
            ats_issues        = rules_score.all_issues,
            section_analyses  = rules_score.section_issues,
            candidate_type     = candidate_type,
        )

        # ── 5. Assemble response ──────────────────────────────────────────────
        logger.info("[Stage 5] Assembling response")
        response = self._build_response(
            rules_score, keyword_analysis, keyword_score, final_score,
            detailed_feedback, ai_insights, nr, section_scores,
            has_jd, dim_breakdown, candidate_type,
        )
        logger.info(f"=== Scan complete — final={final_score}/100 type={candidate_type} ===")
        return response

    # ── AI analysis ───────────────────────────────────────────────────────────

    async def _run_ai_analysis(
        self,
        nr:              NormalizedResume,
        resume_raw:      Dict,
        job_description: Optional[str],
        ats_score:       int,
        keyword_analysis: Optional[KeywordAnalysis],
        db,
        candidate_type:  str,
    ) -> Dict:
        """
        Runs AI analysis through the Resume Builder's shared AI client
        (Gemini-first, Groq fallback). `db` is accepted only to preserve
        the `include_ai and db` gating contract used by the caller and by
        /scan-quick — it is not required by call_ai itself.

        candidate_type is fed into the prompt so the AI stops recommending
        "add work experience" to early-career profiles, and — combined with
        the markdown-parser fix — stops recommending it to anyone whose
        experience section was simply parsed correctly this time.
        """
        industry = (
            keyword_analysis.detected_industry if keyword_analysis
            else self.keyword_engine.detect_industry(_nr_to_keyword_dict(nr))
        )

        exp_lines: List[str] = []
        for exp in nr.experience[:3]:
            for b in exp.bullets[:2]:
                clean = _safe_for_prompt(b, 120)
                if clean:
                    exp_lines.append(clean)
        experience_bullets = "; ".join(exp_lines[:6]) or "Not provided"

        edu_parts: List[str] = []
        for edu in nr.education[:2]:
            d = _safe_for_prompt(edu.degree, 60)
            i = _safe_for_prompt(edu.institution, 80)
            y = _safe_for_prompt(edu.year, 10)
            if d or i:
                edu_parts.append(f"{d} at {i} ({y})".strip())
        education = "; ".join(edu_parts) or "Not provided"

        certs  = ", ".join(_safe_for_prompt(c.title, 60) for c in nr.certifications[:5]) or "None"
        projs  = ", ".join(_safe_for_prompt(p.title, 60) for p in nr.projects[:3]) or "None"
        addl   = ", ".join(s for s in ["languages", "volunteer", "publications", "awards"]
                        if getattr(nr, s)) or "None"
        target = nr.experience[0].title if nr.experience else "Not specified"

        prompt = AI_ANALYSIS_PROMPT.format(
            name                = _safe_for_prompt(nr.contact.name, 60) or "Candidate",
            target_role         = _safe_for_prompt(target, 60),
            industry            = _safe_for_prompt(industry, 40),
            candidate_type      = candidate_type,
            summary             = _safe_for_prompt(nr.summary, 250) or "Not provided",
            skills              = _safe_for_prompt(", ".join(nr.skills[:15]), 200) or "Not provided",
            experience_bullets  = experience_bullets,
            education           = education,
            certifications      = certs,
            projects            = projs,
            additional_sections = addl,
            job_description     = _safe_for_prompt(job_description or "Not provided", 600),
            ats_score           = ats_score,
        )

        fingerprint_src = json.dumps({
            "name":    nr.contact.name,
            "email":   nr.contact.email,
            "summary": nr.summary,
            "skills":  sorted(nr.skills),
            "exp":     [(e.title, e.company, e.start_date, e.end_date) for e in nr.experience],
            "edu":     [(e.degree, e.institution, e.year) for e in nr.education],
            "jd":      job_description or "",
            "score":   ats_score,
            "type":    candidate_type,
        }, sort_keys=True, default=str)
        resume_cache_key = hashlib.sha256(fingerprint_src.encode("utf-8")).hexdigest()

        raw = await call_ai(
            prompt=prompt,
            system_prompt=(
                "You are a senior ATS resume expert. Respond ONLY with valid JSON "
                "matching the requested schema — no markdown fences, no preamble, no extra text."
            ),
            max_output_tokens=2048,
            cache_key=resume_cache_key,
        )
        return self._parse_ai_response(raw)

    def _parse_ai_response(self, raw: str) -> Dict:
        if not raw or not raw.strip():
            return {"success": False, "error": "Empty AI response"}
        parsed = _try_parse_json(raw)
        if parsed and isinstance(parsed, dict):
            parsed["success"] = True
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

    # ── Response assembly ─────────────────────────────────────────────────────

    def _build_response(
        self,
        rules_score,
        keyword_analysis: Optional[KeywordAnalysis],
        keyword_score:    int,
        final_score:      int,
        detailed_feedback: ComprehensiveFeedback,
        ai_insights:       Dict,
        nr:                NormalizedResume,
        section_scores:    Dict[str, int],
        has_jd:            bool,
        dim_breakdown:     Dict,
        candidate_type:    str,
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

        grade  = _grade(final_score)
        status = _status_from_score(final_score)
        contact_score, _, _, _ = _score_contact(nr.contact)

        return {
            "ats_score":      final_score,
            "score_status":   status,
            "grade":          grade,
            "candidate_type": candidate_type,
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
                "name":     nr.contact.name,
                "email":    nr.contact.email,
                "phone":    nr.contact.phone,
                "location": nr.contact.location,
                "linkedin": nr.contact.linkedin,
                "github":   nr.contact.github,
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
                "candidate_type":          candidate_type,
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


def _nr_to_rules_dict(nr: NormalizedResume, raw: Dict) -> Dict:

    """

    Build a dict the ATSRulesEngine can consume.

    Uses normalized data so the rules engine always has valid field names.

    """

    return {

    "name":     nr.contact.name,

    "email":    nr.contact.email,

    "phone":    nr.contact.phone,

    "location": nr.contact.location,

    "summary":  nr.summary,

    "skills":   nr.skills,

    "experience": [

    {

    "title":      e.title,

    "company":    e.company,

    "location":   e.location,

    "start_date": e.start_date,

    "end_date":   e.end_date,

    "bullets":    e.bullets,

    }

    for e in nr.experience

    ],

    "education": [

    {

    "degree":      e.degree,

    "institution": e.institution,

    "college":     e.institution,

    "year":        e.year,

    "gpa":         e.gpa,

    }

    for e in nr.education

    ],

    "certifications": [

    {"title": c.title, "issuer": c.issuer, "year": c.year}

    for c in nr.certifications

    ],

    "projects": [

    {"title": p.title, "description": p.description}

    for p in nr.projects

    ],

    "languages":    nr.languages,

    "awards":       nr.awards,

    "volunteer":    nr.volunteer,

    "publications": nr.publications,

    # Pass-through format metadata from raw dict

    "uses_table":   raw.get("uses_table") or raw.get("has_tables", False),

    "uses_columns": raw.get("uses_columns") or raw.get("multi_column", False),

    "has_images":   raw.get("has_images", False),

    "file_type":    raw.get("file_type", ""),

    "font":         raw.get("font", ""),

    }
def _nr_to_keyword_dict(nr: NormalizedResume) -> Dict:

    """Build the dict shape KeywordEngine.match_skills() expects."""

    return {

    "summary": nr.summary,

    "skills":  nr.skills,

    "experience": [

    {

    "title":   e.title,

    "company": e.company,

    "bullets": e.bullets,

    }

    for e in nr.experience

    ],

    "education": [

    {"degree": e.degree, "institution": e.institution}

    for e in nr.education

    ],

    "certifications": [c.title for c in nr.certifications],

    "projects": [p.title for p in nr.projects],

    }

def _nr_to_feedback_dict(nr: NormalizedResume) -> Dict:

    """Build the dict DetailedFeedbackGenerator.generate_detailed_feedback() reads."""

    return {

    "name":     nr.contact.name,

    "email":    nr.contact.email,

    "phone":    nr.contact.phone,

    "location": nr.contact.location,

    "linkedin": nr.contact.linkedin,

    "github":   nr.contact.github,

    "summary":  nr.summary,

    "skills":   nr.skills,

    "experience": [

    {

    "title":      e.title,

    "company":    e.company,

    "location":   e.location,

    "start_date": e.start_date,

    "end_date":   e.end_date,

    "bullets":    e.bullets,

    }

    for e in nr.experience

    ],

    "education": [

    {

    "degree":      e.degree,

    "institution": e.institution,

    "college":     e.institution,

    "year":        e.year,

    }

    for e in nr.education

    ],

    "certifications": [

    {"title": c.title, "name": c.title, "issuer": c.issuer, "year": c.year}

    for c in nr.certifications

    ],

    "projects": [

        {

            "title":        p.title,

            "description":  p.description,

            "technologies": p.technologies,

            "bullets":      p.bullets,

            "url":          p.url,

        }

    for p in nr.projects

        ],

        "languages":    nr.languages,

        "awards":       nr.awards,

        "volunteer":    nr.volunteer,

        "publications": nr.publications,

        "hobbies":      nr.hobbies,

        }


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
