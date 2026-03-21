# /home/aryu_user/Arun/aiproject_staging/app/modules/ats_scanner/service.py
"""
Production ATS Scanner Service v3.1  —  ALL BUGS FIXED
═══════════════════════════════════════════════════════════════════════════════

BUG 1 — Grade/Status contradiction  (score=52 but grade="A+", status="excellent")
─────────────────────────────────────────────────────────────────────────────────
Root cause: generate_detailed_feedback() was called with ats_score=rules_score
            .total_score (97) because final_score had not been computed yet.
            Every grade/status label was derived from 97, not the real 52.
Fix:        Compute final_score FIRST (Stage 3.5), then pass it to Stage 4.

BUG 2 — Contact section always score=0 / status="missing"
──────────────────────────────────────────────────────────
Root cause: The old ATSRulesEngine only analyses summary/skills/experience/
            education. Contact fields (name, email, phone) sit at the top-level
            of the resume dict, not inside a "contact" key, so they were never
            seen by the feedback generator.
Fix:        _build_contact_from_resume() assembles a contact dict from all
            possible locations (top-level, nested, raw_text). We inject the
            result as a _DictProxy into section_issues["contact"] BEFORE the
            feedback generator runs.

BUG 3 — Education "missing" even though resume has it
──────────────────────────────────────────────────────
Root cause: Canva two-column PDFs produce scrambled text order. pdfplumber
            reads right-column text first, so degree/institution end up
            scattered. The parser returns an empty list for resume["education"].
Fix:        _recover_education_from_text() scans raw_text for degree + institution
            + year patterns and reconstructs the education list when the
            structured parser returns nothing.

BUG 4 — parsed_name = "Agile Delivery" (subtitle extracted as name)
────────────────────────────────────────────────────────────────────
Root cause: Canva PDFs embed the subtitle ("Full Stack Software Engineer |
            MERN Stack | Agile Delivery | Performance Optimization") before
            the actual name in reading order.
Fix:        _clean_name() rejects strings containing subtitle tokens or "|".
            _extract_name_from_raw_text() finds the first short, title-cased,
            non-digit line that cannot be a subtitle.

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import logging
import re
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.ats_scanner.utils.ats_rules_advanced import ATSRulesEngine, SeverityLevel
from app.modules.ats_scanner.utils.ats_keyword_engine import KeywordEngine, KeywordAnalysis
from app.modules.ats_scanner.utils.ats_feedback_generator import (
    DetailedFeedbackGenerator,
    ComprehensiveFeedback,
    GLOBAL_ATS_TACTICS,
    RECRUITER_TIPS,
)
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I)
PHONE_RE    = re.compile(r"(\+\d{1,3}[\s\-]?)?\(?\d{3,5}\)?[\s\-]?\d{3,5}[\s\-]?\d{4,6}")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.I)
GITHUB_RE   = re.compile(r"github\.com/[\w\-]+", re.I)

# Tokens that indicate a line is a job-title subtitle, NOT a person's name
_SUBTITLE_TOKENS = {
    "full stack", "software engineer", "mern", "developer", "manager",
    "agile", "delivery", "performance", "optimization", "stack", "frontend",
    "backend", "engineer", "architect", "analyst", "consultant", "specialist",
    "designer", "director", "officer", "lead", "head of",
}

# Groq prompt (unchanged from v3)
AI_ANALYSIS_PROMPT = """You are a senior resume coach and ATS expert with 15+ years of experience 
helping candidates pass applicant tracking systems across ALL industries.

RESUME DATA:
Name: {name}
Target Role: {target_role}
Industry: {industry}
Summary: {summary}
Skills: {skills}
Experience Bullets (sample): {experience_bullets}
Education: {education}
Certifications: {certifications}
Projects: {projects}
Additional Sections Present: {additional_sections}

JOB DESCRIPTION:
{job_description}

CURRENT ATS SCORE: {ats_score}/100

Respond ONLY with valid JSON. No markdown fences. No preamble.

{{
  "industry_detected": "Industry and role type",
  "role_level": "Entry / Mid / Senior / Executive / Specialist",
  "ats_compatibility_verdict": "1-2 sentence ATS verdict",
  "ai_section_scores": {{
    "summary":    {{"score": 0, "verdict": "1 sentence"}},
    "experience": {{"score": 0, "verdict": "1 sentence"}},
    "skills":     {{"score": 0, "verdict": "1 sentence"}},
    "education":  {{"score": 0, "verdict": "1 sentence"}}
  }},
  "content_strengths": ["strength 1","strength 2","strength 3"],
  "critical_improvements": [
    {{"section":"","issue":"","current_example":"","rewritten_example":"","estimated_score_gain":5}}
  ],
  "keyword_gaps": [
    {{"missing_keyword":"","importance":"critical/preferred","add_to_section":"","how_to_use":""}}
  ],
  "ats_passing_tactics": ["tactic 1","tactic 2","tactic 3","tactic 4","tactic 5"],
  "bullet_rewrites": [
    {{"original":"","rewritten":"","why_better":""}}
  ],
  "summary_rewrite": {{"current":"","suggested":"","why_better":""}},
  "missing_sections": [
    {{"section":"","why_important":"","example_content":""}}
  ],
  "overall_assessment": "3-4 sentence assessment",
  "priority_action_plan": ["1. action +X pts","2. action","3. action","4. action","5. action"]
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _safe(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return bool(value)


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


# ─────────────────────────────────────────────────────────────────────────────
# BUG 4 FIX — NAME CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _clean_name(raw: str) -> str:
    """
    Reject subtitle strings that Canva PDF parsers return as the 'name'.
    A valid name has no subtitle tokens and no pipe separators.
    """
    if not raw:
        return ""
    lower = raw.lower()
    hits  = sum(1 for t in _SUBTITLE_TOKENS if t in lower)
    if hits >= 2 or "|" in raw or len(raw) > 60:
        return ""
    return raw.strip()


def _extract_name_from_raw_text(raw_text: str) -> str:
    """
    Scan the first 15 lines for the first short Title-Case line
    that cannot be a subtitle, role, or contact line.
    """
    if not raw_text:
        return ""
    for line in raw_text.split("\n")[:15]:
        s = line.strip()
        if not s or len(s) > 50:
            continue
        if re.search(r"[\d@|]", s):
            continue
        words = s.split()
        if len(words) < 1 or len(words) > 4:
            continue
        if s[0].isupper() and not s.isupper():
            lower = s.lower()
            if not any(t in lower for t in _SUBTITLE_TOKENS):
                return s
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# BUG 2 FIX — CONTACT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_contact_from_resume(resume: Dict) -> Dict:
    """
    Assemble contact info from all possible parser output locations:
    top-level fields, nested 'contact' dict, and raw_text fallback.
    """
    nested   = resume.get("contact") or {}
    raw_text = _safe(resume.get("raw_text") or resume.get("_raw_text", ""))

    # helper: try multiple sources in order, return first non-empty
    def _first(*sources):
        for s in sources:
            v = _safe(s)
            if v:
                return v
        return ""

    name = (
        _clean_name(_first(nested.get("name"), resume.get("name"))) or
        _extract_name_from_raw_text(raw_text)
    )

    email_match   = EMAIL_RE.search(raw_text)
    phone_match   = PHONE_RE.search(raw_text)
    linkedin_match = LINKEDIN_RE.search(raw_text)
    github_match  = GITHUB_RE.search(raw_text)

    return {
        "name":     name,
        "email":    _first(nested.get("email"),    resume.get("email"),    email_match and email_match.group(0)),
        "phone":    _first(nested.get("phone"),    resume.get("phone"),    phone_match and phone_match.group(0)),
        "location": _first(nested.get("location"), resume.get("location")),
        "linkedin": _first(nested.get("linkedin"), resume.get("linkedin"), linkedin_match and linkedin_match.group(0)),
        "github":   _first(nested.get("github"),   resume.get("github"),   github_match  and github_match.group(0)),
    }


def _score_contact(contact: Dict) -> Tuple[int, List[str], List[str], List[str]]:
    """Return (score, missing_fields, quality_issues, strengths)."""
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
        if EMAIL_RE.match(contact["email"]):
            # Flag unprofessional domains
            if re.search(r"@(hotmail|yahoo|rediffmail|ymail)\.", contact["email"], re.I):
                quality.append(f"Unprofessional email: {contact['email']} — use Gmail or custom domain")
            else:
                strengths.append(f"Professional email: {contact['email']}")
        else:
            quality.append(f"Email format may be incorrect: {contact['email']}")
            score -= 5

    if not contact.get("phone"):
        missing.append("Phone number")
        score -= 15
    else:
        strengths.append("Phone number present")

    if not contact.get("location"):
        quality.append("Location missing — many ATS filter by city/country")
        score -= 10
    else:
        strengths.append(f"Location: {contact['location']}")

    if not contact.get("linkedin"):
        quality.append("No LinkedIn URL — 90 % of recruiters check LinkedIn before contacting")
        score -= 5

    if contact.get("github"):
        strengths.append("GitHub profile linked — strong proof of work for tech roles")

    return max(score, 0), missing, quality, strengths


def _build_contact_section_proxy(contact: Dict) -> "_DictProxy":
    """Build a _DictProxy that the feedback generator can use as a SectionIssue."""
    score, missing, quality, strengths = _score_contact(contact)

    ats_tips = [
        "Keep contact info in the resume body — NOT a header/footer (ATS skips those).",
        "Use City, State/Country only — not your full street address (privacy).",
        "Always include a country code in your phone number: +91, +1, +44 etc.",
        "Add your LinkedIn URL — recruiters verify before scheduling interviews.",
        "Professional email format: firstname.lastname@gmail.com",
    ]

    data = {
        "section_name":    "contact",
        "current_score":   score,
        "missing_fields":  missing,
        "quality_issues":  quality,
        "strengths":       strengths,
        "ats_tips":        ats_tips,
        "improvements":    [f"Add {m}" for m in missing] + quality,
        "rewrite_examples": [{
            "before": "Phone No : 91 6383178328 | Email : venu191202@gmail.com | Tiruvallur District,Tamil Nadu,India",
            "after":  "Chennai, Tamil Nadu, India  ·  +91 63831 78328  ·  venu191202@gmail.com  ·  linkedin.com/in/venud  ·  github.com/venud",
        }],
        "current_status":  ["good"] if score >= 70 else (["needs_improvement"] if score >= 40 else ["critical"]),
        "complete":        not missing,
    }
    return _DictProxy(data)


# ─────────────────────────────────────────────────────────────────────────────
# BUG 3 FIX — EDUCATION RECOVERY
# ─────────────────────────────────────────────────────────────────────────────

def _recover_education_from_text(raw_text: str) -> List[Dict]:
    """
    When the structured parser returns an empty education list, scan raw_text
    for degree / institution / year / CGPA patterns.
    """
    DEGREE_RE = re.compile(
        r"(Bachelor[^,\n|]{0,60}|B\.?[ESTech]{1,5}\.?[^,\n|]{0,40}|"
        r"Master[^,\n|]{0,60}|M\.?[ESTech]{1,5}\.?[^,\n|]{0,40}|"
        r"Ph\.?D\.?[^,\n|]{0,40}|MBA[^,\n|]{0,30}|"
        r"B\.?Sc\.?[^,\n|]{0,40}|M\.?Sc\.?[^,\n|]{0,40}|"
        r"Associate[^,\n|]{0,40}|Diploma[^,\n|]{0,40})",
        re.IGNORECASE,
    )
    INST_RE  = re.compile(
        r"([\w\s&'\-\.]+(?:University|College|Institute|School|Academy|Engineering College)[\w\s&'\-\.]{0,40})",
        re.IGNORECASE,
    )
    YEAR_RE  = re.compile(r"\b(19|20)\d{2}\b")
    CGPA_RE  = re.compile(r"(?:CGPA|GPA)[:\s]*([0-9]\.[0-9]{1,2})", re.IGNORECASE)

    entries = []
    for m in DEGREE_RE.finditer(raw_text):
        degree = m.group(0).strip().rstrip(",;|")
        start  = max(0, m.start() - 50)
        end    = min(len(raw_text), m.end() + 400)
        snip   = raw_text[start:end]

        institution = ""
        im = INST_RE.search(snip)
        if im:
            institution = im.group(1).strip()

        year = ""
        ym = YEAR_RE.search(snip)
        if ym:
            year = ym.group(0)

        gpa = ""
        gm = CGPA_RE.search(snip)
        if gm:
            gpa = gm.group(1)

        if degree or institution:
            entries.append({
                "degree":      degree,
                "institution": institution,
                "year":        year,
                "gpa":         gpa,
            })
            break   # one entry is sufficient for the ATS scanner

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# RESUME ENRICHMENT (runs all fixes before analysis)
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_resume(resume: Dict) -> Dict:
    """
    Apply all pre-processing fixes to the parsed resume dict BEFORE analysis.
    Returns a new dict — does NOT mutate the original.
    """
    r = dict(resume)  # shallow copy

    # BUG 4 FIX: clean name
    raw_text = _safe(r.get("raw_text") or r.get("_raw_text", ""))
    clean_n  = _clean_name(_safe(r.get("name")))
    if not clean_n and raw_text:
        clean_n = _extract_name_from_raw_text(raw_text)
    if clean_n:
        r["name"] = clean_n

    # BUG 3 FIX: recover education
    education = r.get("education") or []
    has_valid = (
        isinstance(education, list) and
        any(isinstance(e, dict) and (e.get("degree") or e.get("institution"))
            for e in education)
    )
    if not has_valid and raw_text:
        recovered = _recover_education_from_text(raw_text)
        if recovered:
            r["education"] = recovered
            logger.info(f"Education recovered from raw_text: {recovered}")

    return r


# ─────────────────────────────────────────────────────────────────────────────
# _DictProxy — lets getattr() work on plain dicts
# ─────────────────────────────────────────────────────────────────────────────

class _DictProxy:
    """Thin wrapper so getattr(proxy, key) works like dict.get(key, None)."""
    def __init__(self, d: Dict) -> None:
        self._d = d

    def __getattr__(self, name: str):
        try:
            return self._d[name]
        except KeyError:
            return None

    def get(self, key, default=None):
        return self._d.get(key, default)


# ─────────────────────────────────────────────────────────────────────────────
# SCORE EXPLANATION (plain language)
# ─────────────────────────────────────────────────────────────────────────────

def _score_explanation(
    rule_score:    int,
    keyword_score: int,
    final_score:   int,
    has_jd:        bool,
) -> Dict:
    return {
        "resume_quality_score": {
            "score":  rule_score,
            "grade":  _grade(rule_score),
            "what_it_means": (
                "How well-written the resume is: structure, formatting, "
                "action verbs, bullet quality. Does NOT measure job fit."
            ),
            "interpretation": (
                "Excellent resume writing quality."
                if rule_score >= 85 else
                "Good quality with minor improvements needed."
                if rule_score >= 70 else
                "Quality issues that need addressing."
            ),
        },
        "keyword_match_score": {
            "score":  keyword_score if has_jd else None,
            "grade":  _grade(keyword_score) if has_jd else "N/A (no JD provided)",
            "what_it_means": (
                "How many keywords from the job description appear in the resume. "
                "This is the PRIMARY ATS filter — a well-written resume with the wrong "
                "keywords will still be rejected before a human reads it."
            ),
            "interpretation": (
                "Not calculated — provide a job description to measure keyword match."
                if not has_jd else
                "High keyword alignment — resume will pass ATS keyword filters."
                if keyword_score >= 70 else
                "Low keyword match — resume will likely be filtered out by ATS. "
                "Add the missing keywords from the job description immediately."
                if keyword_score < 50 else
                "Moderate keyword match — adding more JD keywords will significantly improve ATS score."
            ),
        },
        "final_ats_score": {
            "score":  final_score,
            "grade":  _grade(final_score),
            "what_it_means": (
                "Weighted final score: 40% resume quality + 50% keyword match + up to 10% AI quality bonus. "
                "This is the score that predicts your real-world ATS pass rate."
                if has_jd else
                "Resume quality score (no job description provided — "
                "keyword match weighting cannot be applied)."
            ),
            "interpretation": _ats_verdict(final_score),
        },
        "why_scores_differ": (
            "Your resume is well-written (quality score high) but does not match the job "
            "description well (keyword score low). ATS systems primarily filter on keyword match, "
            "so a high quality score is meaningless if keywords are missing. "
            "Focus immediately on adding the missing keywords listed in keyword_analysis."
            if has_jd and rule_score >= 70 and keyword_score < 50 else
            "Scores are consistent across all dimensions."
        ),
    }


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
        logger.info("=== ATS Scan v3.1 Starting ===")

        # ── PRE-PROCESS ───────────────────────────────────────────────────────
        resume        = _enrich_resume(resume)
        contact_built = _build_contact_from_resume(resume)
        logger.info(f"  name='{resume.get('name')}' | "
                    f"email='{contact_built.get('email')}' | "
                    f"edu_entries={len(resume.get('education') or [])}")

        # ── Stage 1: Rules ────────────────────────────────────────────────────
        logger.info("[Stage 1] ATSRulesEngine")
        rules_score = self.rules_engine.analyze(resume)

        # ── Stage 2: Keywords ─────────────────────────────────────────────────
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

        # ── Stage 3: Groq AI ──────────────────────────────────────────────────
        ai_insights: Dict = {}
        if include_ai and db:
            logger.info("[Stage 3] Groq AI call")
            try:
                ai_insights = await self._run_ai_analysis(
                    resume, job_description, rules_score.total_score,
                    keyword_analysis, db,
                )
            except Exception as e:
                logger.warning(f"  AI failed (graceful fallback): {e}")
                ai_insights = {"success": False, "error": str(e)}

        # ── BUG 1 FIX: Stage 3.5 — compute final_score BEFORE feedback ────────
        final_score = self._calculate_final_score(
            rules_score.total_score, keyword_score, ai_insights
        )
        logger.info(f"  final_score={final_score} "
                    f"(rules={rules_score.total_score} kw={keyword_score})")

        # ── Stage 4: Feedback ─────────────────────────────────────────────────
        logger.info("[Stage 4] Feedback generation")

        # BUG 2 FIX: inject contact section
        rules_score.section_issues["contact"] = _build_contact_section_proxy(contact_built)

        section_scores: Dict[str, int] = {}
        for s in ["contact","summary","experience","education","skills",
                  "projects","certifications","languages","volunteer",
                  "publications","awards","hobbies","references"]:
            sa = rules_score.section_issues.get(s)
            section_scores[s] = getattr(sa, "current_score", 0) if sa else 0

        # Blend AI section scores
        if ai_insights.get("success") and ai_insights.get("ai_section_scores"):
            for sec, ai_data in ai_insights["ai_section_scores"].items():
                if isinstance(ai_data, dict) and "score" in ai_data:
                    r_s = section_scores.get(sec, 0)
                    a_s = int(ai_data.get("score", r_s))
                    section_scores[sec] = int(r_s * 0.6 + a_s * 0.4)

        # BUG 1 FIX: pass final_score, NOT rules_score.total_score
        detailed_feedback = self.feedback_generator.generate_detailed_feedback(
            ats_score       = final_score,
            section_scores  = section_scores,
            resume          = resume,
            ats_issues      = rules_score.all_issues,
            section_analyses= rules_score.section_issues,
        )

        # ── Stage 5: Assemble ─────────────────────────────────────────────────
        logger.info("[Stage 5] Assembling response")
        response = self._build_response(
            rules_score, keyword_analysis, keyword_score, final_score,
            detailed_feedback, ai_insights, resume,
            section_scores, contact_built, has_jd,
        )
        logger.info(f"=== Scan complete — final={final_score}/100 ===")
        return response

    # ── AI ────────────────────────────────────────────────────────────────────

    async def _run_ai_analysis(
        self,
        resume, job_description, ats_score,
        keyword_analysis, db,
    ) -> Dict:
        contact  = resume.get("_contact_built") or _build_contact_from_resume(resume)
        name     = _safe(contact.get("name") or resume.get("name"))
        industry = (keyword_analysis.detected_industry if keyword_analysis
                    else self.keyword_engine.detect_industry(resume))
        summary  = _safe(resume.get("summary"))[:300]
        skills   = ", ".join([_safe(s) for s in (resume.get("skills") or [])[:20]])

        exp_bullets: List[str] = []
        for exp in (resume.get("experience") or [])[:3]:
            if isinstance(exp, dict):
                for b in (exp.get("bullets") or exp.get("responsibilities") or [])[:2]:
                    exp_bullets.append(_safe(b))

        edu_parts: List[str] = []
        for edu in (resume.get("education") or [])[:2]:
            if isinstance(edu, dict):
                d = _safe(edu.get("degree"))
                i = _safe(edu.get("institution") or edu.get("college"))
                y = _safe(edu.get("year"))
                if d or i:
                    edu_parts.append(f"{d} | {i} | {y}".strip(" |"))

        certs    = (resume.get("certifications") or [])[:5]
        projects = (resume.get("projects") or [])[:3]
        additional = [s for s in ["languages","volunteer","publications","awards","hobbies"]
                      if _is_present(resume.get(s))]

        prompt = AI_ANALYSIS_PROMPT.format(
            name               = name or "Candidate",
            target_role        = _safe(resume.get("target_role") or
                                       self._guess_target_role(resume) or "Not specified"),
            industry           = industry,
            summary            = summary or "Not provided",
            skills             = skills or "Not provided",
            experience_bullets = "\n".join([f"• {b}" for b in exp_bullets[:6]]) or "Not provided",
            education          = "; ".join(edu_parts) or "Not provided",
            certifications     = ", ".join([_safe(c.get("name") if isinstance(c,dict) else c) for c in certs]) or "None",
            projects           = ", ".join([_safe(p.get("name") if isinstance(p,dict) else p) for p in projects]) or "None",
            additional_sections= ", ".join(additional) or "None",
            job_description    = (job_description or "Not provided")[:800],
            ats_score          = ats_score,
        )

        raw = await call_llm(user_message=prompt, agent_name="ats_scanner", db=db)
        return self._parse_ai_response(raw)

    def _parse_ai_response(self, raw: str) -> Dict:
        if not raw:
            return {"success": False, "error": "Empty AI response"}
        clean = raw.strip()
        if "```" in clean:
            parts = clean.split("```")
            if len(parts) >= 2:
                clean = parts[1]
                if clean.startswith("json"):
                    clean = clean[4:]
        clean = clean.strip()
        m = re.search(r"\{[\s\S]*\}", clean)
        if m:
            clean = m.group(0)
        try:
            data = json.loads(clean)
            data["success"] = True
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"AI JSON parse failed: {e}")
            result: Dict = {"success": False, "partial": True}
            m2 = re.search(r'"overall_assessment"\s*:\s*"([^"]{20,})"', clean)
            if m2:
                result["overall_assessment"] = m2.group(1)
                result["success"] = True
            actions = re.findall(r'"(\d\..{10,80})"', clean)
            if actions:
                result["priority_action_plan"] = actions[:5]
            return result

    # ── SCORING ───────────────────────────────────────────────────────────────

    def _calculate_final_score(
        self, rule_score: int, keyword_score: int, ai_insights: Dict
    ) -> int:
        ai_bonus = 0
        if ai_insights.get("success"):
            ai_scores = ai_insights.get("ai_section_scores") or {}
            if ai_scores:
                vals = [v.get("score", 70) for v in ai_scores.values() if isinstance(v, dict)]
                if vals:
                    ai_bonus = int((sum(vals) / len(vals) - 70) * 0.1)

        if keyword_score > 0:
            final = (rule_score * 0.40) + (keyword_score * 0.50) + ai_bonus
        else:
            final = (rule_score * 0.80) + ai_bonus

        return min(max(int(final), 0), 100)

    # ── RESPONSE ──────────────────────────────────────────────────────────────

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
    ) -> Dict:

        # Section output
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

        # Keyword output
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
                            "how_to_add":        ai_gap.get("how_to_use", ""),
                            "source":            "ai",
                        })

        # AI block — clear status when not available
        if ai_insights.get("success"):
            ai_block = {
                "status":                    "success",
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
                "how_to_enable": "Pass include_ai=true with a valid database session.",
            }

        # BUG 1 FIX: grade and status derived from final_score ONLY
        grade  = _grade(final_score)
        status = _status_from_score(final_score)

        contact_score, _, _, _ = _score_contact(contact_built)

        return {
            # ── Primary result ────────────────────────────────────────────────
            "ats_score":      final_score,
            "score_status":   status,           # BUG 1 FIX
            "grade":          grade,            # BUG 1 FIX
            "ready_to_apply": final_score >= 72,
            "ready_to_apply_verdict": (
                "✅ Ready to apply — resume will pass ATS screening."
                if final_score >= 72 else
                "❌ Not ready — fix critical and high-priority issues first."
            ),

            # ── Score breakdown ───────────────────────────────────────────────
            "score_breakdown": {
                "resume_quality_score": rules_score.total_score,
                "keyword_match_score":  keyword_score if has_jd else None,
                "final_ats_score":      final_score,
                "format_compliance":    rules_score.format_score,
                "structure_quality":    rules_score.structure_score,
                "content_quality":      rules_score.content_score,
                "ats_compliance":       rules_score.ats_compliance_score,
            },

            # ── Plain-language explanation of the 3 scores ────────────────────
            "score_explanation": _score_explanation(
                rules_score.total_score, keyword_score, final_score, has_jd
            ),

            # ── Contact info actually found in resume ─────────────────────────
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

            # ── Issues ────────────────────────────────────────────────────────
            "issues":                self._format_issues(rules_score.all_issues),
            "critical_issues_count": rules_score.critical_issues_count,

            # ── Section analysis (all 13 sections) ───────────────────────────
            "section_analysis": section_output,

            # ── Keyword intelligence ──────────────────────────────────────────
            "keyword_analysis": keyword_output,

            # ── AI insights ───────────────────────────────────────────────────
            "ai_analysis": ai_block,

            # ── Recommendations ───────────────────────────────────────────────
            "recommendations": {
                "top_3_priorities":    detailed_feedback.top_3_priorities,
                "quick_wins":          detailed_feedback.quick_wins_summary,
                "improvement_roadmap": detailed_feedback.improvement_roadmap,
                "ats_passing_tactics": detailed_feedback.ats_passing_tactics,
                "recruiter_tips":      detailed_feedback.recruiter_tips,
                "estimated_improvement": detailed_feedback.estimated_improvement_potential,
            },

            # ── Executive summary ─────────────────────────────────────────────
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

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _format_issues(self, issues) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {"critical": [], "high": [], "medium": [], "low": []}
        for issue in issues:
            sev = (issue.severity.value if hasattr(issue.severity, "value")
                   else str(issue.severity))
            entry = {
                "section":    issue.section,
                "message":    issue.message,
                "suggestion": issue.suggestion,
                "impact":     issue.impact_score,
            }
            if getattr(issue, "specific_example",   None): entry["example"]     = issue.specific_example
            if getattr(issue, "improvement_example", None): entry["improvement"] = issue.improvement_example
            if sev in out:
                out[sev].append(entry)
        return out

    @staticmethod
    def _key_findings(score: int, critical: int) -> List[str]:
        out = []
        if critical > 0:
            out.append(f"⚠️ {critical} critical ATS issue(s) — fix before applying")
        if score >= 85:
            out.append("✅ Highly optimised — strong ATS compatibility")
        elif score >= 72:
            out.append("✅ Passes ATS screening — minor refinements recommended")
        elif score >= 55:
            out.append("⚠️ Significant ATS weaknesses — improvement needed")
        else:
            out.append("❌ Likely rejected by ATS — major revision required")
        return out

    @staticmethod
    def _next_steps(score: int, critical: int) -> List[str]:
        steps = []
        if critical > 0:
            steps.append(f"1. Fix {critical} critical issue(s) immediately (see issues → critical)")
        if score < 55:
            steps.append("2. Follow the improvement roadmap section by section")
            steps.append("3. Re-scan after each batch of changes to track progress")
        elif score < 72:
            steps.append("2. Address all High-severity issues (see issues → high)")
            steps.append("3. Apply Quick Wins — each takes under 15 minutes")
        else:
            steps.append("1. ✅ Ready to apply — tailor keywords to each specific job posting")
        return steps

    @staticmethod
    def _guess_target_role(resume: Dict) -> Optional[str]:
        exp = resume.get("experience") or []
        if isinstance(exp, list) and exp:
            first = exp[0]
            if isinstance(first, dict):
                return _safe(first.get("title") or first.get("job_title"))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

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