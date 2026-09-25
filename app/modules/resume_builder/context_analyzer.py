"""
context_analyzer.py — Intelligent Domain Detection, Context Grounding & Prompt Synthesis

Provides:
1. Multi-signal domain detection with conflict resolution across ambiguous roles (e.g. QC, QA, Engineer).
2. Strict factual grounding ensuring zero invented academic subjects, responsibilities, or tools.
3. User prompt parsing for tone/style guidance without overriding factual boundaries.
4. Grounded prompt builders and sanitization post-processors for Education, Experience, Skills, and Summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN TAXONOMY & SIGNAL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class DomainType(str, Enum):
    SOFTWARE_TESTING = "software_testing"
    MANUFACTURING_QC = "manufacturing_qc"
    CONSTRUCTION_QC = "construction_qc"
    PHARMA_QC = "pharma_qc"
    SOFTWARE_DEV = "software_dev"
    DATA_SCIENCE = "data_science"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    AMBIGUOUS_NEUTRAL = "ambiguous_neutral"
    GENERAL = "general"


# Multi-word signals listed first for greedy matching
SOFTWARE_TESTING_SIGNALS = [
    "rest api testing", "api testing", "regression testing", "functional testing",
    "integration testing", "system testing", "smoke testing", "sanity testing",
    "user acceptance testing", "uat", "defect tracking", "bug reporting",
    "bug tracking", "test cases", "test scenarios", "test suites",
    "test execution", "test plan", "automation testing", "automated testing",
    "manual testing", "sql testing", "database testing", "performance testing",
    "load testing", "selenium webdriver", "selenium", "cypress", "playwright",
    "postman", "jira", "testrail", "jmeter", "appium", "cucumber", "bdd", "tdd",
    "pytest", "junit", "testng", "software quality", "software testing",
    "web application testing", "mobile testing", "sdlc", "stlc", "black box testing",
    "white box testing", "qa engineer", "qa analyst", "qa tester", "software qa"
]

MANUFACTURING_QC_SIGNALS = [
    "raw materials", "finished goods", "production line", "assembly line",
    "shop floor", "dimensional inspection", "quality inspection", "process inspection",
    "first article inspection", "fai", "statistical process control", "spc",
    "six sigma", "iso 9001", "iso manufacturing standards", "calipers", "micrometers",
    "cmm", "gd&t", "geometric dimensioning", "defect rate", "batch inspection",
    "sampling plan", "aql", "non-conformance", "ncr", "manufacturing", "production floor",
    "factory inspection", "incoming inspection", "production quality"
]

CONSTRUCTION_QC_SIGNALS = [
    "site inspection", "civil engineering", "structural inspection", "boq",
    "bill of quantities", "concrete testing", "slump test", "material testing",
    "site quality", "compaction test", "rebar inspection", "mep inspection",
    "architectural drawings", "building codes", "construction safety", "qa/qc civil",
    "construction quality", "structural drawings", "as-built drawings"
]

PHARMA_QC_SIGNALS = [
    "pharmaceutical", "laboratory testing", "gmp", "glp", "cgmp", "hplc",
    "gas chromatography", "analytical testing", "stability testing", "batch release",
    "wet chemistry", "titration", "dissolution testing", "usp", "pharmacopeia",
    "fda compliance", "cleanroom standards", "microbiology testing", "qc chemist",
    "analytical chemist"
]

SOFTWARE_DEV_SIGNALS = [
    "python", "javascript", "typescript", "java", "c++", "golang", "rust",
    "react", "angular", "vue", "node.js", "django", "fastapi", "flask",
    "spring boot", "docker", "kubernetes", "aws", "azure", "gcp",
    "microservices", "rest api", "graphql", "postgresql", "mysql", "mongodb",
    "redis", "git", "ci/cd", "full stack", "backend", "frontend"
]

DATA_SCIENCE_SIGNALS = [
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "pytorch", "tensorflow", "pandas", "numpy", "scikit-learn",
    "etl pipeline", "data pipeline", "power bi", "tableau", "data warehouse",
    "snowflake", "bigquery", "data modeling"
]

FINANCE_SIGNALS = [
    "gaap", "ifrs", "auditing", "tax compliance", "financial accounting",
    "corporate finance", "quickbooks", "general ledger", "balance sheet",
    "profit and loss", "reconciliation", "accounts payable", "accounts receivable",
    "financial modeling"
]

HEALTHCARE_SIGNALS = [
    "patient care", "clinical", "hospital", "nursing", "icu", "bls certified",
    "acls", "triage", "electronic health records", "ehr", "emr", "vital signs",
    "phlebotomy", "medical terminology"
]

# Titles that are ambiguous without context
AMBIGUOUS_TITLES = {
    "qc", "qa", "qa/qc", "quality control", "quality assurance",
    "quality engineer", "quality control engineer", "qc engineer",
    "quality inspector", "qc inspector", "quality analyst", "qc analyst",
    "inspector", "tester", "analyst", "engineer", "specialist",
    "technician", "associate", "coordinator", "officer", "supervisor"
}

# Generic filler phrases to avoid in outputs
GENERIC_FILLER_PHRASES = [
    "results-driven professional",
    "dynamic individual",
    "hardworking professional",
    "passionate individual",
    "motivated self-starter",
    "team player with a can-do attitude",
    "proven track record of success",
    "experienced professional with a demonstrated history",
    "detail-oriented professional"
]

# Generic soft skills to avoid unless provided in candidate source data
GENERIC_SOFT_SKILLS = {
    "leadership", "teamwork", "communication", "problem solving",
    "time management", "interpersonal skills", "adaptability",
    "work ethic", "collaboration", "critical thinking", "creativity",
    "conflict resolution", "multitasking"
}


# ─────────────────────────────────────────────────────────────────────────────
# RESUME CONTEXT DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResumeContext:
    inferred_domain: DomainType
    domain_label: str
    resolved_role: str
    raw_role: str
    is_ambiguous: bool
    domain_scores: Dict[str, int] = field(default_factory=dict)
    verified_skills: List[str] = field(default_factory=list)
    verified_tools: List[str] = field(default_factory=list)
    verified_responsibilities: List[str] = field(default_factory=list)
    verified_education: Dict[str, Any] = field(default_factory=dict)
    user_prompt: str = ""
    style_guidelines: str = ""
    prohibited_domains: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN DETECTION & CONTEXT EXTRACTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _score_signals(text: str, signals: List[str]) -> Tuple[int, List[str]]:
    """Count occurrences of domain signals in text using word boundaries."""
    text_lower = text.lower()
    score = 0
    matched = []
    for sig in signals:
        pattern = r'\b' + re.escape(sig) + r'\b'
        matches = len(re.findall(pattern, text_lower))
        if matches > 0:
            score += matches
            matched.append(sig)
    return score, matched


def _extract_all_text(data: Dict[str, Any]) -> str:
    """Aggregate all relevant text fields from payload for domain analysis."""
    parts = []

    # Job title / target roles
    for key in ("job_title", "target_title", "title", "role"):
        if data.get(key) and isinstance(data[key], str):
            parts.append(data[key])
    if isinstance(data.get("job_titles"), list):
        parts.extend([str(t) for t in data["job_titles"] if t])

    # Experience / responsibilities
    for key in ("experience", "responsibilities", "description", "existing_experience"):
        val = data.get(key)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            parts.extend([str(item) for item in val if item])

    # Experiences list of dicts
    if isinstance(data.get("experiences"), list):
        for exp in data["experiences"]:
            if isinstance(exp, dict):
                for k in ("job_title", "description", "responsibilities", "bullets"):
                    v = exp.get(k)
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, list):
                        parts.extend([str(i) for i in v if i])

    # Skills / tools
    skills = data.get("skills") or data.get("existing_skills") or data.get("technologies")
    if isinstance(skills, list):
        parts.extend([str(s) for s in skills if s])
    elif isinstance(skills, str):
        parts.append(skills)

    # Projects
    if isinstance(data.get("projects"), list):
        for proj in data["projects"]:
            if isinstance(proj, dict):
                for k in ("title", "description", "technologies", "bullets"):
                    v = proj.get(k)
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, list):
                        parts.extend([str(i) for i in v if i])

    # Education
    for key in ("degree", "college", "institution", "specialization", "coursework"):
        val = data.get(key)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            parts.extend([str(item) for item in val if item])

    # User prompt
    prompt = data.get("prompt") or data.get("instruction") or data.get("user_prompt")
    if isinstance(prompt, str):
        parts.append(prompt)

    # Resume data blob if present
    if isinstance(data.get("resume_data"), dict):
        parts.append(_extract_all_text(data["resume_data"]))

    return " ".join(parts)


def _parse_user_prompt_style(prompt: str) -> str:
    """Extract writing style and tone constraints from user prompt without altering facts."""
    if not prompt:
        return "Professional, concise, direct ATS-friendly resume style."

    p_lower = prompt.lower()
    style_parts = []

    if "human" in p_lower:
        style_parts.append("Natural, authentic human-written tone (no robotic cadence)")
    if "professional" in p_lower:
        style_parts.append("Polished professional tone")
    if "concise" in p_lower or "short" in p_lower or "brief" in p_lower:
        style_parts.append("Strictly concise and direct")
    if "impact" in p_lower or "achievement" in p_lower:
        style_parts.append("Emphasize impact and active ownership")
    if "formal" in p_lower:
        style_parts.append("Formal professional language")

    if not style_parts:
        style_parts.append(f"Follow requested style instruction: '{prompt}'")

    return "; ".join(style_parts)


def analyze_resume_context(data: Dict[str, Any]) -> ResumeContext:
    """
    Analyzes all candidate data to infer the true professional domain,
    resolve role ambiguities (e.g. QC in software vs manufacturing), and
    extract verified grounding facts.
    """
    raw_role = (
        data.get("job_title")
        or (data.get("job_titles")[0] if isinstance(data.get("job_titles"), list) and data.get("job_titles") else "")
        or data.get("target_title")
        or data.get("title")
        or ""
    ).strip()

    # Extract user prompt
    user_prompt = (
        data.get("prompt")
        or data.get("instruction")
        or data.get("user_prompt")
        or ""
    ).strip()
    style_guidelines = _parse_user_prompt_style(user_prompt)

    # Full context text aggregation
    full_text = _extract_all_text(data)

    # Non-title text context (crucial for conflict resolution when title is ambiguous)
    title_words = set(re.findall(r'\w+', raw_role.lower()))
    non_title_data = dict(data)
    for k in ("job_title", "job_titles", "target_title", "title", "role"):
        non_title_data.pop(k, None)
    context_text = _extract_all_text(non_title_data)

    # Calculate domain signal scores across full text and context text
    scores = {
        DomainType.SOFTWARE_TESTING.value: _score_signals(full_text, SOFTWARE_TESTING_SIGNALS)[0],
        DomainType.MANUFACTURING_QC.value: _score_signals(full_text, MANUFACTURING_QC_SIGNALS)[0],
        DomainType.CONSTRUCTION_QC.value: _score_signals(full_text, CONSTRUCTION_QC_SIGNALS)[0],
        DomainType.PHARMA_QC.value: _score_signals(full_text, PHARMA_QC_SIGNALS)[0],
        DomainType.SOFTWARE_DEV.value: _score_signals(full_text, SOFTWARE_DEV_SIGNALS)[0],
        DomainType.DATA_SCIENCE.value: _score_signals(full_text, DATA_SCIENCE_SIGNALS)[0],
        DomainType.FINANCE.value: _score_signals(full_text, FINANCE_SIGNALS)[0],
        DomainType.HEALTHCARE.value: _score_signals(full_text, HEALTHCARE_SIGNALS)[0],
    }

    # Context-specific scores (excluding ambiguous title itself)
    context_sw_testing_score, _ = _score_signals(context_text, SOFTWARE_TESTING_SIGNALS)
    context_mfg_qc_score, _ = _score_signals(context_text, MANUFACTURING_QC_SIGNALS)
    context_con_qc_score, _ = _score_signals(context_text, CONSTRUCTION_QC_SIGNALS)
    context_pharma_qc_score, _ = _score_signals(context_text, PHARMA_QC_SIGNALS)

    raw_role_clean = raw_role.lower().strip()
    is_ambiguous_title = (
        raw_role_clean in AMBIGUOUS_TITLES
        or any(raw_role_clean == amb for amb in AMBIGUOUS_TITLES)
        or (
            any(kw in raw_role_clean for kw in ("qc", "quality", "inspector", "tester"))
            and not any(clear in raw_role_clean for clear in (
                "software qa", "software test", "civil", "structural", "pharmaceutical", "pharma", "clinical"
            ))
        )
    )

    inferred_domain = DomainType.GENERAL
    domain_label = "General Professional"
    resolved_role = raw_role or "Professional"
    is_ambiguous = False
    prohibited_domains: List[str] = []

    # ─────────────────────────────────────────────────────────────────────────
    # CONFLICT RESOLUTION & DOMAIN INFERENCE
    # ─────────────────────────────────────────────────────────────────────────
    if is_ambiguous_title:
        # Check context signals to disambiguate the role
        qc_scores = {
            DomainType.SOFTWARE_TESTING: context_sw_testing_score,
            DomainType.MANUFACTURING_QC: context_mfg_qc_score,
            DomainType.CONSTRUCTION_QC: context_con_qc_score,
            DomainType.PHARMA_QC: context_pharma_qc_score,
        }
        max_qc_score = max(qc_scores.values())

        if max_qc_score > 0:
            # Clear domain signal present in context
            if context_sw_testing_score == max_qc_score and context_sw_testing_score > context_mfg_qc_score:
                inferred_domain = DomainType.SOFTWARE_TESTING
                domain_label = "Software Quality Assurance & Testing"
                resolved_role = "Software QA / Test Engineer"
                prohibited_domains = [
                    "manufacturing", "factory inspection", "production quality",
                    "raw materials", "finished goods", "construction", "civil inspection", "pharmaceutical"
                ]
            elif context_mfg_qc_score == max_qc_score and context_mfg_qc_score > context_sw_testing_score:
                inferred_domain = DomainType.MANUFACTURING_QC
                domain_label = "Manufacturing Quality Control"
                resolved_role = raw_role if "inspector" in raw_role_clean else "Manufacturing QC Inspector"
                prohibited_domains = [
                    "selenium", "postman", "api testing", "software testing",
                    "jira defect tracking", "web applications", "automation testing"
                ]
            elif context_con_qc_score == max_qc_score:
                inferred_domain = DomainType.CONSTRUCTION_QC
                domain_label = "Construction Quality Control"
                resolved_role = "Construction QA/QC Inspector"
                prohibited_domains = ["software testing", "selenium", "raw materials manufacturing", "pharma"]
            elif context_pharma_qc_score == max_qc_score:
                inferred_domain = DomainType.PHARMA_QC
                domain_label = "Pharmaceutical Quality Control"
                resolved_role = "Pharmaceutical QC Analyst"
                prohibited_domains = ["software testing", "selenium", "construction"]
        else:
            # Genuinely ambiguous with NO context signals: DO NOT GUESS!
            inferred_domain = DomainType.AMBIGUOUS_NEUTRAL
            domain_label = "Quality Control (Neutral / General)"
            resolved_role = raw_role or "Quality Control (QC)"
            is_ambiguous = True
            prohibited_domains = [
                "selenium", "api testing", "postman", "software testing",
                "raw materials manufacturing", "factory floor", "civil construction", "laboratory pharma"
            ]
    else:
        # Non-ambiguous title, or general domain resolution
        top_domain = max(scores, key=scores.get)
        if scores[top_domain] > 1:
            inferred_domain = DomainType(top_domain)
            domain_label = top_domain.replace("_", " ").title()
            resolved_role = raw_role
        else:
            inferred_domain = DomainType.GENERAL
            domain_label = "General Professional"
            resolved_role = raw_role or "Professional"

    # Extract verified skills
    raw_skills = data.get("skills") or data.get("existing_skills") or data.get("technologies") or []
    if isinstance(raw_skills, str):
        verified_skills = [s.strip() for s in raw_skills.split(",") if s.strip()]
    elif isinstance(raw_skills, list):
        verified_skills = [str(s).strip() for s in raw_skills if str(s).strip()]
    else:
        verified_skills = []

    # Extract verified education
    verified_education = {
        "degree": str(data.get("degree") or "").strip(),
        "college": str(data.get("college") or data.get("institution") or data.get("university") or "").strip(),
        "location": str(data.get("location") or "").strip(),
        "year": str(data.get("year") or data.get("toYear") or data.get("graduation_year") or "").strip(),
        "specialization": str(data.get("specialization") or data.get("major") or "").strip(),
        "coursework": data.get("coursework") if isinstance(data.get("coursework"), list) else [],
        "achievements": data.get("achievements") if isinstance(data.get("achievements"), list) else [],
    }

    # Extract verified responsibilities/experience
    verified_responsibilities: List[str] = []
    exp_val = data.get("experience") or data.get("responsibilities") or data.get("description")
    if isinstance(exp_val, str) and exp_val.strip():
        verified_responsibilities.append(exp_val.strip())
    elif isinstance(exp_val, list):
        verified_responsibilities.extend([str(x).strip() for x in exp_val if str(x).strip()])

    if isinstance(data.get("experiences"), list):
        for exp in data["experiences"]:
            if isinstance(exp, dict):
                for k in ("description", "responsibilities", "bullets"):
                    v = exp.get(k)
                    if isinstance(v, str) and v.strip():
                        verified_responsibilities.append(v.strip())
                    elif isinstance(v, list):
                        verified_responsibilities.extend([str(i).strip() for i in v if str(i).strip()])

    return ResumeContext(
        inferred_domain=inferred_domain,
        domain_label=domain_label,
        resolved_role=resolved_role,
        raw_role=raw_role,
        is_ambiguous=is_ambiguous,
        domain_scores=scores,
        verified_skills=verified_skills,
        verified_tools=list(set(verified_skills)),
        verified_responsibilities=verified_responsibilities,
        verified_education=verified_education,
        user_prompt=user_prompt,
        style_guidelines=style_guidelines,
        prohibited_domains=prohibited_domains,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS (REUSABLE, GROUNDED, ATS-OPTIMIZED)
# ─────────────────────────────────────────────────────────────────────────────

def build_grounded_education_prompt(data: Dict[str, Any], ctx: ResumeContext) -> str:
    """
    Builds a strictly grounded prompt for education suggestions.
    Guarantees zero invented coursework or subjects unless provided in data.
    """
    edu = ctx.verified_education
    degree = edu.get("degree", "")
    college = edu.get("college", "")
    location = edu.get("location", "")
    year = edu.get("year", "")
    specialization = edu.get("specialization", "")
    coursework = edu.get("coursework", [])
    achievements = edu.get("achievements", [])

    has_extra_details = bool(specialization or coursework or achievements)

    prompt = f"""[CRITICAL INTEGRITY & FACTUAL GROUNDING CONSTRAINT]
You are a precise ATS resume formatting engine. You must format the candidate's education record using ONLY the verified facts supplied below.

VERIFIED EDUCATION DATA:
- Degree: {degree or "Not specified"}
- Institution: {college or "Not specified"}
- Location: {location or "Not specified"}
- Graduation Year: {year or "Not specified"}
- Specialization: {specialization or "None supplied"}
- Coursework: {', '.join(coursework) if coursework else "None supplied"}
- Honors/Achievements: {', '.join(achievements) if achievements else "None supplied"}

WRITING STYLE & TONE GUIDELINES:
{ctx.style_guidelines}

STRICT EXECUTION RULES:
1. DO NOT invent subjects, courses, modules, or academic theories (e.g. Do NOT invent marketing, finance, corporate accounting, HR, operations, or supply chain concepts).
2. DO NOT invent fake GPA, grades, rankings, scholarships, or extracurricular achievements.
3. If ONLY degree, institution, location, and year are provided, produce a single concise, factual, ATS-friendly statement line in standard resume format:
   Example: {degree}, {college}{', ' + location if location else ''} | {year}
4. If specialization, coursework, or achievements are explicitly supplied above, produce at most 1 to 2 concise ATS-friendly bullet points utilizing ONLY those supplied details.
5. Never output generic academic filler or textbook curriculum summaries.
6. Return ONLY the formatted text. No intro, no conversational text, no markdown styling."""

    return prompt


def build_grounded_experience_prompt(data: Dict[str, Any], ctx: ResumeContext) -> str:
    """
    Builds a domain-aware, grounded prompt for experience bullet suggestions.
    Respects resolved domain (e.g. software QC vs manufacturing QC) and candidate evidence.
    """
    job_title = ctx.raw_role or data.get("job_title", "").strip()
    company = data.get("company", "").strip()
    start = data.get("start_date", "")
    end = data.get("end_date", "")
    if data.get("duration"):
        duration = str(data["duration"]).strip()
    elif start or end:
        duration = f"{start} - {end or 'Present'}".strip(" -")
    else:
        duration = "Not specified"
    location = data.get("location", "").strip()

    skills_str = ", ".join(ctx.verified_skills) if ctx.verified_skills else "None specified"
    resp_str = " | ".join(ctx.verified_responsibilities) if ctx.verified_responsibilities else "Standard operational duties"

    prohibited_str = ""
    if ctx.prohibited_domains:
        prohibited_str = f"- STRICTLY PROHIBITED TOPICS: Do NOT mention or invent: {', '.join(ctx.prohibited_domains)}.\n"

    prompt = f"""[CRITICAL INTEGRITY & ATS GROUNDING CONSTRAINT]
You are an expert ATS resume writer specializing in verified candidate experience.

ROLE & DOMAIN CONTEXT:
- Target / Job Title: {job_title}
- Inferred Professional Domain: {ctx.domain_label}
- Resolved Professional Role: {ctx.resolved_role}
- Company: {company or "Not specified"}
- Duration: {duration}
- Location: {location or "Not specified"}

VERIFIED CANDIDATE SOURCE DATA:
- Verified Skills / Tools: {skills_str}
- Provided Responsibilities / Experience: {resp_str}

STYLE & TONE INSTRUCTIONS:
{ctx.style_guidelines}

DOMAIN-SPECIFIC RULES:
{prohibited_str}
- If domain is Software QA / Testing, bullets MUST focus on software validation, test cases, regression testing, API testing, or defect reporting using verified tools ({skills_str}).
- If domain is Manufacturing QC, bullets MUST focus on production floor inspection, quality standards, and defect reduction without software testing terms.
- If domain is Ambiguous / Neutral QC, use neutral quality control terminology without assuming software or manufacturing.

EXECUTION RULES:
1. Generate exactly 3 to 5 concise, high-impact resume experience bullets.
2. Start every bullet with a strong action verb (e.g., Performed, Executed, Documented, Validated, Coordinated).
3. Ground every bullet in the candidate's verified responsibilities, tools, and domain.
4. Preserve exact technical keywords from the input (e.g., {skills_str}).
5. DO NOT fabricate metrics, revenue figures, percentage improvements, or tools not present in candidate data.
6. Keep each bullet between 15 and 30 words.
7. Return ONLY the bullet points, one per line. No symbols, no bullets, no numbers, no explanations."""

    return prompt


def build_grounded_skills_prompt(data: Dict[str, Any], ctx: ResumeContext) -> str:
    """
    Builds a domain-aware skills prompt prioritizing candidate evidence,
    preserving exact technical keywords, and suppressing generic soft skills.
    """
    roles_str = ctx.resolved_role or ctx.raw_role or "Professional"
    skills_str = ", ".join(ctx.verified_skills) if ctx.verified_skills else "None provided"
    career_level = data.get("career_level", "experienced")

    prohibited_str = ""
    if ctx.prohibited_domains:
        prohibited_str = f"- Strictly avoid keywords from wrong domains: {', '.join(ctx.prohibited_domains)}.\n"

    prompt = f"""[CRITICAL INTEGRITY & ATS SKILL EXTRACTION]
You are an ATS skill indexing engine. Extract, normalize, and suggest core domain-specific competencies.

ROLE CONTEXT:
- Target Role: {roles_str}
- Inferred Domain: {ctx.domain_label}
- Career Level: {career_level}

VERIFIED CANDIDATE EVIDENCE:
- Provided Skills / Keywords: {skills_str}
- Experience Evidence: {' '.join(ctx.verified_responsibilities)[:300]}

STYLE INSTRUCTION:
{ctx.style_guidelines}

STRICT EXECUTION RULES:
1. Output EXACTLY 5 to 8 domain-specific technical skills / competencies.
2. PRIORITY RULE: You MUST preserve and prioritize all relevant verified skills provided in candidate data ({skills_str}).
3. Add relevant domain competencies (e.g., methodologies, frameworks) ONLY if directly supported by the candidate's domain evidence.
4. DO NOT generate generic soft skills (e.g., Leadership, Teamwork, Communication, Problem Solving, Time Management).
{prohibited_str}5. Return ONLY one distinct skill per line.
6. Output raw plain text only. No numbers, no bullets, no punctuation at start of lines, no explanations."""

    return prompt


def build_grounded_summary_prompt(data: Dict[str, Any], ctx: ResumeContext) -> str:
    """
    Builds a grounded professional summary prompt (2-4 concise sentences)
    using only verified domain facts and candidate skills.
    """
    skills_str = ", ".join(ctx.verified_skills[:8]) if ctx.verified_skills else "professional skills"
    exp_str = " ".join(ctx.verified_responsibilities)[:400]
    
    deg = ctx.verified_education.get('degree', '').strip()
    col = ctx.verified_education.get('college', '').strip()
    if deg and col:
        edu_str = f"{deg} from {col}"
    elif deg:
        edu_str = deg
    elif col:
        edu_str = col
    else:
        edu_str = "Not specified"

    # Estimate career tenure if available
    years_exp = "experienced"
    experiences = data.get("experiences") or []
    if isinstance(experiences, list) and experiences:
        try:
            start_years = [
                int(exp.get("start_date", "")[:4])
                for exp in experiences
                if isinstance(exp, dict) and exp.get("start_date")
            ]
            if start_years:
                from datetime import datetime
                diff = datetime.now().year - min(start_years)
                years_exp = f"{diff}+ years of experience" if diff > 0 else "1+ years of experience"
        except Exception:
            years_exp = "experienced"

    prohibited_str = ""
    if ctx.prohibited_domains:
        prohibited_str = f"- STRICTLY AVOID: Do not introduce topics from: {', '.join(ctx.prohibited_domains)}.\n"

    prompt = f"""[CRITICAL INTEGRITY & ATS SUMMARY GENERATION]
You are a professional ATS resume writer. Write a concise, factual 2 to 4 sentence professional summary.

CANDIDATE FACTUAL CONTEXT:
- Professional Role: {ctx.resolved_role}
- Domain: {ctx.domain_label}
- Career Tenure: {years_exp}
- Verified Technical Skills / Tools: {skills_str}
- Work Experience Context: {exp_str or "Professional execution in the domain"}
- Education Background: {edu_str or "Not specified"}

STYLE & TONE INSTRUCTIONS:
{ctx.style_guidelines}

STRICT EXECUTION RULES:
1. Target length: EXACTLY 2 to 4 concise sentences unified into a single plain-text paragraph.
2. Ground the summary strictly in the candidate's actual domain, verified skills, and experience.
3. Seamlessly integrate the verified keywords ({skills_str}) for ATS matching.
4. NO first-person pronouns (No: I, me, my, we, our).
5. DO NOT use generic clichés like "hardworking professional", "passionate individual", "results-driven professional", "dynamic individual".
{prohibited_str}6. Return ONLY the plain-text paragraph. No introduction, no markdown, no quotes."""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# SANITIZERS & POST-PROCESSORS
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_education_output(raw_response: str, data: Dict[str, Any], ctx: ResumeContext) -> List[str]:
    """
    Validates and sanitizes education output.
    If the LLM hallucinated generic curriculum/subjects when none were provided,
    it falls back to a clean, factual ATS-friendly education entry.
    """
    lines = [l.strip().lstrip('•-*123456789. ') for l in raw_response.split('\n') if l.strip()]

    edu = ctx.verified_education
    degree = edu.get("degree") or data.get("degree", "").strip()
    college = edu.get("college") or data.get("college", "").strip() or data.get("institution", "").strip()
    location = edu.get("location") or data.get("location", "").strip()
    year = edu.get("year") or data.get("year", "").strip()

    # Check if input had coursework or subjects
    has_coursework_input = bool(edu.get("coursework") or data.get("coursework") or edu.get("specialization") or data.get("specialization"))

    # Hallucination markers to reject if no coursework was supplied
    academic_hallucination_patterns = [
        r'\bstudied\b', r'\banalyzed\b', r'\bexplored\b', r'\bexamined\b',
        r'\bapplied\b', r'\bframeworks\b', r'\borganizational behavior\b',
        r'\bcorporate finance\b', r'\bfinancial accounting\b', r'\bmarketing strategies\b',
        r'\bhuman resource management\b', r'\bsupply chain management\b',
        r'\boperational research\b', r'\bconsumer behavior\b', r'\bprinciples\b'
    ]

    clean_bullets: List[str] = []
    for line in lines:
        if not has_coursework_input:
            is_hallucinated = any(re.search(pat, line, re.IGNORECASE) for pat in academic_hallucination_patterns)
            if is_hallucinated:
                continue

        # Strip model preamble
        if line.lower().startswith("output:") or line.lower().startswith("education:"):
            line = line.split(":", 1)[1].strip()

        if line:
            clean_bullets.append(line)

    # Fallback to pure factual statement if all were hallucinated or response was empty
    if not clean_bullets or not has_coursework_input:
        factual_parts = []
        if degree and college:
            first_part = f"{degree}, {college}"
            if location:
                first_part += f", {location}"
            factual_parts.append(first_part)
        elif degree:
            factual_parts.append(f"{degree}{', ' + location if location else ''}")
        elif college:
            factual_parts.append(f"{college}{', ' + location if location else ''}")

        if year and factual_parts:
            fallback = f"{factual_parts[0]} | {year}"
        elif factual_parts:
            fallback = factual_parts[0]
        else:
            fallback = "Education details provided"

        # If clean_bullets had a valid concise line containing degree/college, keep it; else fallback
        valid_concise = [b for b in clean_bullets if degree.lower() in b.lower() or college.lower() in b.lower()]
        return valid_concise if valid_concise else [fallback]

    return clean_bullets[:3]


def sanitize_experience_output(raw_response: str, data: Dict[str, Any], ctx: ResumeContext) -> List[str]:
    """
    Sanitizes experience bullets ensuring domain alignment and stripping prohibited terminology.
    """
    raw_lines = raw_response.strip().split('\n')
    bullets: List[str] = []

    for line in raw_lines:
        cleaned = line.strip().strip('*').strip('-').strip('•').strip('"').strip("'").strip()
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
        if cleaned.lower().startswith("output:"):
            cleaned = cleaned[7:].strip()
        if not cleaned:
            continue

        # Prohibit crossed-domain contamination
        is_contaminated = False
        cleaned_lower = cleaned.lower()

        if ctx.inferred_domain == DomainType.SOFTWARE_TESTING:
            # Check for manufacturing contamination
            for bad_term in ("raw materials", "production floor", "shop floor", "dimensional inspection", "factory floor", "finished goods"):
                if bad_term in cleaned_lower:
                    is_contaminated = True
                    break
        elif ctx.inferred_domain == DomainType.MANUFACTURING_QC:
            # Check for software testing contamination
            for bad_term in (
                "selenium", "postman", "api testing", "jira", "web applications",
                "test cases", "software testing", "regression testing", "functional testing",
                "test automation", "qa tester"
            ):
                if bad_term in cleaned_lower:
                    is_contaminated = True
                    break
        elif ctx.is_ambiguous:
            # Ambiguous: reject both software and manufacturing assumptions
            for bad_term in ("selenium", "postman", "api testing", "raw materials", "production line", "shop floor"):
                if bad_term in cleaned_lower:
                    is_contaminated = True
                    break

        if not is_contaminated:
            bullets.append(cleaned)

    # If all were contaminated or empty, synthesize from verified data
    if not bullets and ctx.verified_responsibilities:
        for resp in ctx.verified_responsibilities[:3]:
            bullets.append(resp)

    return bullets[:5]


def sanitize_skills_output(raw_response: str, data: Dict[str, Any], ctx: ResumeContext) -> List[str]:
    """
    Normalizes skills, ensures candidate ATS keywords are preserved,
    and strips generic soft skills unless explicitly supplied.
    """
    raw_lines = [l.strip().lstrip('•-*123456789. ') for l in raw_response.split('\n') if l.strip()]
    seen_lower = set()
    cleaned_skills: List[str] = []

    # Priority 1: Always preserve verified candidate skills from input
    user_skills_lower = {s.lower() for s in ctx.verified_skills}
    for s in ctx.verified_skills:
        s_clean = s.strip()
        if s_clean and s_clean.lower() not in seen_lower:
            seen_lower.add(s_clean.lower())
            cleaned_skills.append(s_clean)

    # Priority 2: Add valid model-suggested domain skills
    for line in raw_lines:
        skill = line.strip().strip('"').strip("'")
        if not skill:
            continue
        skill_lower = skill.lower()

        # Remove generic soft skills unless user explicitly provided them
        if skill_lower in GENERIC_SOFT_SKILLS and skill_lower not in user_skills_lower:
            continue

        # Remove generic headers / markdown
        if skill_lower.startswith("output:") or skill_lower.startswith("skills:") or "competencies" in skill_lower:
            continue

        # Prevent cross-domain hallucination
        if ctx.inferred_domain == DomainType.SOFTWARE_TESTING:
            if any(term in skill_lower for term in ("raw materials", "shop floor", "finished goods", "factory")):
                continue
        elif ctx.inferred_domain == DomainType.MANUFACTURING_QC:
            if any(term in skill_lower for term in ("selenium", "postman", "api testing", "jira")):
                continue

        if skill_lower not in seen_lower:
            seen_lower.add(skill_lower)
            cleaned_skills.append(skill)

    return cleaned_skills[:8]


def sanitize_summary_output(raw_response: str, data: Dict[str, Any], ctx: ResumeContext) -> str:
    """
    Cleans summary output, removes boilerplate fluff, and formats as 2-4 sentence paragraph.
    """
    summary = raw_response.strip()

    # Strip code blocks and markdown fences
    if summary.startswith("```"):
        summary = re.sub(r'^```[a-zA-Z]*\n', '', summary)
        summary = re.sub(r'```$', '', summary).strip()

    summary = summary.strip('"').strip("'")

    # Remove generic AI introduction headers
    for prefix in ("here is a professional summary:", "summary:", "professional summary:"):
        if summary.lower().startswith(prefix):
            summary = summary[len(prefix):].strip()

    # Remove generic filler phrases
    for filler in GENERIC_FILLER_PHRASES:
        pattern = re.compile(re.escape(filler), re.IGNORECASE)
        summary = pattern.sub("", summary).strip()

    # Clean up accidental double spaces or orphan punctuation
    summary = re.sub(r'\s+', ' ', summary).strip()
    summary = re.sub(r'^\s*[,;\.\-]\s*', '', summary).strip()

    return summary
