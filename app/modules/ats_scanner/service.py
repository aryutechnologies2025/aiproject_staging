# /home/aryu_user/Arun/aiproject_staging/app/modules/ats_scanner/service.py
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

_SUBTITLE_TOKENS = {
    "full stack", "software engineer", "mern", "developer", "manager",
    "agile", "delivery", "performance", "optimization", "stack", "frontend",
    "backend", "engineer", "architect", "analyst", "consultant", "specialist",
    "designer", "director", "officer", "lead", "head of",
}

# ─────────────────────────────────────────────────────────────────────────────
# AI PROMPT  — uses numbered placeholders so curly braces in JSON schema
# do NOT clash with .format() substitution
# ─────────────────────────────────────────────────────────────────────────────

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
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _safe(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_for_prompt(value: Any, max_len: int = 200) -> str:
    """
    Sanitise a value for injection into an LLM prompt that will ask
    the LLM to respond with JSON.  Removes characters that commonly
    break JSON generation when embedded in prompt context.
    """
    text = _safe(value)[:max_len]
    # Remove actual newlines — replace with space
    text = text.replace("\n", " ").replace("\r", " ")
    # Remove or escape characters that confuse JSON parsing in LLM output
    text = text.replace("\\", "")
    # Collapse multiple spaces
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
# NAME CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _clean_name(raw: str) -> str:
    if not raw:
        return ""
    lower = raw.lower()
    hits  = sum(1 for t in _SUBTITLE_TOKENS if t in lower)
    if hits >= 2 or "|" in raw or len(raw) > 60:
        return ""
    return raw.strip()


def _extract_name_from_raw_text(raw_text: str) -> str:
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
# CONTACT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_contact_from_resume(resume: Dict) -> Dict:
    nested   = resume.get("contact") or {}
    raw_text = _safe(resume.get("raw_text") or resume.get("_raw_text", ""))

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

    email_match    = EMAIL_RE.search(raw_text)
    phone_match    = PHONE_RE.search(raw_text)
    linkedin_match = LINKEDIN_RE.search(raw_text)
    github_match   = GITHUB_RE.search(raw_text)

    return {
        "name":     name,
        "email":    _first(nested.get("email"),    resume.get("email"),    email_match    and email_match.group(0)),
        "phone":    _first(nested.get("phone"),    resume.get("phone"),    phone_match    and phone_match.group(0)),
        "location": _first(nested.get("location"), resume.get("location")),
        "linkedin": _first(nested.get("linkedin"), resume.get("linkedin"), linkedin_match and linkedin_match.group(0)),
        "github":   _first(nested.get("github"),   resume.get("github"),   github_match   and github_match.group(0)),
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
        if EMAIL_RE.match(contact["email"]):
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
        quality.append("No LinkedIn URL — 90% of recruiters check LinkedIn before contacting")
        score -= 5

    if contact.get("github"):
        strengths.append("GitHub profile linked — strong proof of work for tech roles")

    return max(score, 0), missing, quality, strengths


def _build_contact_section_proxy(contact: Dict) -> "_DictProxy":
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
        "rewrite_examples": [],
        "current_status":  ["good"] if score >= 70 else (["needs_improvement"] if score >= 40 else ["critical"]),
        "complete":        not missing,
    }
    return _DictProxy(data)


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION RECOVERY
# ─────────────────────────────────────────────────────────────────────────────

def _recover_education_from_text(raw_text: str) -> List[Dict]:
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
                "college":     institution,
                "year":        year,
                "gpa":         gpa,
            })
            break

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# RESUME ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_resume(resume: Dict) -> Dict:
    r = dict(resume)

    raw_text = _safe(r.get("raw_text") or r.get("_raw_text", ""))
    clean_n  = _clean_name(_safe(r.get("name")))
    if not clean_n and raw_text:
        clean_n = _extract_name_from_raw_text(raw_text)
    if clean_n:
        r["name"] = clean_n

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
# _DictProxy
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# SCORE EXPLANATION
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
                "How many keywords from the job description appear in the resume."
            ),
            "interpretation": (
                "Not calculated — provide a job description to measure keyword match."
                if not has_jd else
                "High keyword alignment — resume will pass ATS keyword filters."
                if keyword_score >= 70 else
                "Low keyword match — resume will likely be filtered out by ATS."
                if keyword_score < 50 else
                "Moderate keyword match — adding more JD keywords will improve ATS score."
            ),
        },
        "final_ats_score": {
            "score":  final_score,
            "grade":  _grade(final_score),
            "what_it_means": (
                "Weighted final score: 40% resume quality + 50% keyword match."
                if has_jd else
                "Resume quality score (no job description provided)."
            ),
            "interpretation": _ats_verdict(final_score),
        },
        "why_scores_differ": (
            "Resume is well-written but does not match the job description keywords. "
            "Add the missing keywords listed in keyword_analysis."
            if has_jd and rule_score >= 70 and keyword_score < 50 else
            "Scores are consistent across all dimensions."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# JSON REPAIR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    text = text.strip()
    # Remove opening fence
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str:
    """
    Extract the outermost { ... } block.
    Handles cases where the LLM adds preamble or postamble text.
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False
    end = start

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


def _repair_json_string(raw: str) -> str:
    """
    Multi-pass JSON repair for common LLM output issues:
    1. Trailing commas before } or ]
    2. Unescaped newlines inside string values
    3. Unescaped double quotes inside string values (basic heuristic)
    4. Truncated JSON — add missing closing brackets
    """
    # Pass 1: remove trailing commas
    text = re.sub(r",\s*([\]}])", r"\1", raw)

    # Pass 2: replace literal newlines inside strings with \n escape
    # We do this by scanning char by char to find string boundaries
    text = _escape_newlines_in_strings(text)

    # Pass 3: fix unescaped control characters
    # Replace tab, carriage return inside strings
    text = re.sub(r'(?<=": ")(.*?)(?="(?:\s*[,}\]]))', _sanitise_string_value, text)

    # Pass 4: attempt to close unclosed structure
    text = _close_unclosed_json(text)

    return text


def _escape_newlines_in_strings(text: str) -> str:
    """Replace literal \n and \r inside JSON string values with their escape sequences."""
    result = []
    in_string = False
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


def _sanitise_string_value(m: re.Match) -> str:
    """Escape any bare double-quotes found inside a captured string value."""
    content = m.group(0)
    # This is a rough pass — only process if needed
    return content


def _close_unclosed_json(text: str) -> str:
    """Add missing ] and } to close a truncated JSON string."""
    open_braces   = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    # Remove trailing comma before we close
    text = text.rstrip().rstrip(",")

    text += "]" * max(open_brackets, 0)
    text += "}" * max(open_braces, 0)

    return text


def _try_parse_json(text: str) -> Optional[Dict]:
    """
    Attempt JSON parsing with progressive repair steps.
    Returns parsed dict or None.
    """
    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip fences + extract object
    cleaned = _strip_markdown_fences(text)
    cleaned = _extract_json_object(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: repair then parse
    repaired = _repair_json_string(cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Attempt 4: use json5-style lenient parser via ast.literal_eval approximation
    # Replace single-quoted strings with double-quoted (common LLM mistake)
    try:
        swapped = re.sub(r"(?<![\\])'", '"', repaired)
        return json.loads(swapped)
    except Exception:
        pass

    return None


def _extract_fields_by_regex(text: str) -> Dict:
    """
    Last-resort: extract known fields from malformed JSON using regex.
    Returns a partial dict — always succeeds.
    """
    result: Dict = {}

    # String fields
    for field in (
        "industry_detected", "role_level", "ats_compatibility_verdict",
        "overall_assessment",
    ):
        m = re.search(rf'"{field}"\s*:\s*"([^"{{}}[\]]*)"', text)
        if m:
            result[field] = m.group(1).strip()

    # Array of strings
    for field in ("content_strengths", "ats_passing_tactics", "priority_action_plan"):
        m = re.search(rf'"{field}"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
        if m:
            items = re.findall(r'"([^"]*)"', m.group(1))
            if items:
                result[field] = items

    # ai_section_scores — extract scores only
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
        logger.info("=== ATS Scan v4.0 Starting ===")

        # ── PRE-PROCESS ───────────────────────────────────────────────────────
        resume        = _enrich_resume(resume)
        contact_built = _build_contact_from_resume(resume)
        logger.info(
            f"  name='{resume.get('name')}' | "
            f"email='{contact_built.get('email')}' | "
            f"edu={len(resume.get('education') or [])} | "
            f"exp={len(resume.get('experience') or [])} | "
            f"skills={len(resume.get('skills') or [])}"
        )

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

        # ── Stage 3: AI Analysis ──────────────────────────────────────────────
        ai_insights: Dict = {}
        if include_ai and db:
            logger.info("[Stage 3] AI analysis")
            try:
                ai_insights = await self._run_ai_analysis(
                    resume, job_description, rules_score.total_score,
                    keyword_analysis, db,
                )
                logger.info(f"  AI success={ai_insights.get('success')} "
                            f"partial={ai_insights.get('partial', False)}")
            except Exception as e:
                logger.warning(f"  AI failed (graceful fallback): {e}")
                ai_insights = {"success": False, "error": str(e)}

        # ── Stage 3.5: Final score ────────────────────────────────────────────
        final_score = self._calculate_final_score(
            rules_score.total_score, keyword_score, ai_insights
        )
        logger.info(
            f"  final={final_score} rules={rules_score.total_score} kw={keyword_score}"
        )

        # ── Stage 4: Feedback ─────────────────────────────────────────────────
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
            ats_score        = final_score,
            section_scores   = section_scores,
            resume           = resume,
            ats_issues       = rules_score.all_issues,
            section_analyses = rules_score.section_issues,
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

    # ─────────────────────────────────────────────────────────────────────────
    # AI
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_ai_analysis(
        self,
        resume, job_description, ats_score,
        keyword_analysis, db,
    ) -> Dict:
        contact  = _build_contact_from_resume(resume)
        name     = _safe_for_prompt(contact.get("name") or resume.get("name"), 60)
        industry = (
            keyword_analysis.detected_industry if keyword_analysis
            else self.keyword_engine.detect_industry(resume)
        )
        summary = _safe_for_prompt(resume.get("summary"), 250)
        skills  = _safe_for_prompt(
            ", ".join([_safe(s) for s in (resume.get("skills") or [])[:15]]), 200
        )

        # Experience bullets — sanitised, max 6 bullets
        exp_lines: List[str] = []
        for exp in (resume.get("experience") or [])[:3]:
            if isinstance(exp, dict):
                for b in (exp.get("bullets") or [])[:2]:
                    clean = _safe_for_prompt(b, 120)
                    if clean:
                        exp_lines.append(clean)
        experience_bullets = "; ".join(exp_lines[:6]) or "Not provided"

        # Education — sanitised
        edu_parts: List[str] = []
        for edu in (resume.get("education") or [])[:2]:
            if isinstance(edu, dict):
                d = _safe_for_prompt(edu.get("degree"), 60)
                i = _safe_for_prompt(edu.get("institution") or edu.get("college"), 80)
                y = _safe_for_prompt(edu.get("year"), 10)
                if d or i:
                    edu_parts.append(f"{d} at {i} ({y})".strip())
        education = "; ".join(edu_parts) or "Not provided"

        # Certs and projects — sanitised
        certs = ", ".join([
            _safe_for_prompt(c.get("name") if isinstance(c, dict) else c, 60)
            for c in (resume.get("certifications") or [])[:5]
            if c
        ]) or "None"

        projects = ", ".join([
            _safe_for_prompt(p.get("name") or p.get("title") if isinstance(p, dict) else p, 60)
            for p in (resume.get("projects") or [])[:3]
            if p
        ]) or "None"

        additional = ", ".join([
            s for s in ["languages", "volunteer", "publications", "awards", "hobbies"]
            if _is_present(resume.get(s))
        ]) or "None"

        jd_text = _safe_for_prompt(job_description or "Not provided", 600)

        prompt = AI_ANALYSIS_PROMPT.format(
            name               = name or "Candidate",
            target_role        = _safe_for_prompt(
                resume.get("target_role") or self._guess_target_role(resume) or "Not specified", 60
            ),
            industry           = _safe_for_prompt(industry, 40),
            summary            = summary or "Not provided",
            skills             = skills or "Not provided",
            experience_bullets = experience_bullets,
            education          = education,
            certifications     = certs,
            projects           = projects,
            additional_sections= additional,
            job_description    = jd_text,
            ats_score          = ats_score,
        )

        raw = await call_llm(user_message=prompt, agent_name="ats_scanner", db=db)
        return self._parse_ai_response(raw)

    def _parse_ai_response(self, raw: str) -> Dict:
        """
        Robust multi-pass JSON parser for LLM responses.
        Never raises — always returns a dict with success flag.
        """
        if not raw or not raw.strip():
            return {"success": False, "error": "Empty AI response"}

        # Attempt full parse with progressive repair
        parsed = _try_parse_json(raw)

        if parsed and isinstance(parsed, dict):
            parsed["success"] = True
            logger.info("AI JSON parsed successfully")
            return parsed

        # Full parse failed — extract whatever fields we can via regex
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

    # ─────────────────────────────────────────────────────────────────────────
    # SCORING
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_final_score(
        self, rule_score: int, keyword_score: int, ai_insights: Dict
    ) -> int:
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
                    ai_bonus = int((sum(vals) / len(vals) - 70) * 0.1)

        if keyword_score > 0:
            final = (rule_score * 0.40) + (keyword_score * 0.50) + ai_bonus
        else:
            final = (rule_score * 0.80) + ai_bonus

        return min(max(int(final), 0), 100)

    # ─────────────────────────────────────────────────────────────────────────
    # RESPONSE BUILDER
    # ─────────────────────────────────────────────────────────────────────────

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
                            "source":            "ai",
                        })

        # AI block
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
            },
            "score_explanation": _score_explanation(
                rules_score.total_score, keyword_score, final_score, has_jd
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

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _format_issues(self, issues) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {"critical": [], "high": [], "medium": [], "low": []}
        for issue in issues:
            sev = (
                issue.severity.value if hasattr(issue.severity, "value")
                else str(issue.severity)
            )
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
        exp = resume.get("experience") or []
        if isinstance(exp, list) and exp:
            first = exp[0]
            if isinstance(first, dict):
                return _safe(first.get("title") or first.get("job_title"))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE
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