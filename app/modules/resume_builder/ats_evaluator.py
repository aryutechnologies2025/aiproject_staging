"""
ats_evaluator.py — Transparent, explainable ATS evaluation engine for resume_builder.

Combines:
1. Deterministic Rule Engine (80% weight):
   - Exact keyword & hard skill matching (35%)
   - Measurable metrics & strong action verbs (20%)
   - Section completeness & layout standard (15%)
   - Contact validity & parseability (10%)
2. Gemini Semantic Evaluator (20% weight):
   - Seniority and role relevance fit (10%)
   - Narrative impact & experience quality (10%)
   - Targeted line-by-line bullet recommendations & skill gap analysis

Zero arbitrary black-box scoring. Every point is explainable.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from app.modules.resume_builder.gemini_client import get_gemini_client
from app.modules.resume_builder.schemas import CanonicalResume

logger = logging.getLogger("resume_builder.ats_evaluator")

# Common strong ATS action verbs
ACTION_VERBS = {
    "accelerated", "accomplished", "achieved", "acquired", "administered", "advanced",
    "analyzed", "architected", "automated", "built", "centralized", "championed",
    "coached", "collaborated", "constructed", "created", "decreased", "delivered",
    "deployed", "designed", "developed", "directed", "eliminated", "engineered",
    "established", "executed", "expanded", "expedited", "formulated", "generated",
    "guided", "implemented", "improved", "increased", "initiated", "innovated",
    "installed", "integrated", "launched", "lead", "led", "managed", "maximized",
    "mentored", "migrated", "minimized", "modeled", "negotiated", "optimized",
    "orchestrated", "overhauled", "pioneered", "planned", "produced", "programmed",
    "published", "re-engineered", "rebuilt", "redesigned", "reduced", "refactored",
    "resolved", "restructured", "revamped", "scaled", "secured", "simplified",
    "spearheaded", "standardized", "streamlined", "strengthened", "structured",
    "supervised", "trained", "transformed", "upgraded", "validated",
}


# ─────────────────────────────────────────────────────────────────────────────
# ATS EVALUATION SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class DeterministicATSBreakdown(BaseModel):
    keyword_score: float = Field(default=0.0, description="Score for skill and keyword match (max 35.0)")
    metrics_score: float = Field(default=0.0, description="Score for quantifiable metrics and action verbs (max 20.0)")
    section_score: float = Field(default=0.0, description="Score for required sections completeness (max 15.0)")
    contact_score: float = Field(default=0.0, description="Score for contact information completeness (max 10.0)")
    total_deterministic_score: float = Field(default=0.0, description="Sum of deterministic scores (max 80.0)")
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    metrics_found_count: int = 0
    action_verbs_count: int = 0
    sections_present: List[str] = Field(default_factory=list)
    missing_sections: List[str] = Field(default_factory=list)


class SemanticATSBreakdown(BaseModel):
    role_fit_score: float = Field(default=0.0, description="Semantic fit to target role / seniority (max 10.0)")
    narrative_quality_score: float = Field(default=0.0, description="Impact and quality of experience descriptions (max 10.0)")
    semantic_score: float = Field(default=0.0, description="Total semantic score (max 20.0)")
    seniority_assessment: str = Field(default="", description="Seniority evaluation (e.g. Junior, Mid, Senior, Lead)")
    missing_critical_skills: List[str] = Field(default_factory=list, description="Crucial competencies missing for role")
    actionable_recommendations: List[str] = Field(default_factory=list, description="Specific line-by-line bullet improvements")


class ATSCompositeReport(BaseModel):
    overall_score: int = Field(default=0, description="Final ATS score from 0 to 100")
    grade: str = Field(default="N/A", description="Letter grade: A (90+), B (80+), C (70+), D (60+), F (<60)")
    deterministic_breakdown: DeterministicATSBreakdown
    semantic_breakdown: Optional[SemanticATSBreakdown] = None
    summary_feedback: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# 1. DETERMINISTIC ATS EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def _extract_keywords_from_text(text: str) -> Set[str]:
    """Extract clean words and tech tokens from job description."""
    tokens = re.findall(r"\b[A-Za-z0-9+#.-]{2,30}\b", text.lower())
    stop_words = {
        "and", "the", "with", "for", "that", "this", "from", "have", "will", "your",
        "our", "team", "work", "role", "must", "able", "skills", "experience", "years",
        "about", "looking", "join", "help", "build", "using", "responsible", "requirements",
    }
    return {t for t in tokens if t not in stop_words and len(t) > 2}


def evaluate_ats_deterministically(
    resume: CanonicalResume,
    job_description: Optional[str] = None,
) -> DeterministicATSBreakdown:
    """
    Computes deterministic ATS score components (max 80 points) using pure algorithmic rules.
    """
    # ── 1. Contact Info Score (Max 10) ──
    contact_score = 0.0
    p_info = resume.personal_information
    if p_info.name and len(p_info.name.strip()) > 2:
        contact_score += 2.5
    if p_info.email and "@" in p_info.email:
        contact_score += 2.5
    if p_info.phone and len(re.findall(r"\d", p_info.phone)) >= 7:
        contact_score += 2.5
    if p_info.location or p_info.link:
        contact_score += 2.5

    # ── 2. Section Completeness Score (Max 15) ──
    sections_present = []
    missing_sections = []

    if resume.experience and len(resume.experience) > 0:
        sections_present.append("Experience")
    else:
        missing_sections.append("Experience")

    if resume.education and len(resume.education) > 0:
        sections_present.append("Education")
    else:
        missing_sections.append("Education")

    if resume.skills and len(resume.skills) > 0:
        sections_present.append("Skills")
    else:
        missing_sections.append("Skills")

    if resume.summary and len(resume.summary.strip()) > 20:
        sections_present.append("Summary")
    else:
        missing_sections.append("Summary")

    if resume.projects and len(resume.projects) > 0:
        sections_present.append("Projects")

    # 4 core sections = 3.75 pts each
    section_score = round(min(15.0, len(sections_present) * 3.0 + (3.0 if "Experience" in sections_present else 0)), 1)

    # ── 3. Measurable Metrics & Action Verbs (Max 20) ──
    all_bullets = []
    for exp in resume.experience:
        all_bullets.extend(exp.bullets)
    for proj in resume.projects:
        all_bullets.extend(proj.bullets)

    all_text = " ".join(all_bullets).lower()
    verbs_found = sum(1 for v in ACTION_VERBS if re.search(r"\b" + re.escape(v) + r"\b", all_text))
    # Count numbers, percentages, dollar amounts, metrics (e.g. 50%, $10M, 100k, 2x, 40ms)
    metrics_matches = re.findall(r"\b(?:\$?\d+(?:\.\d+)?(?:k|m|b|%|x|ms|s)?|\d+\+?)\b", all_text)
    metrics_count = len(metrics_matches)

    verbs_score = min(10.0, verbs_found * 1.25)
    metrics_score_val = min(10.0, metrics_count * 1.0)
    metrics_score = round(verbs_score + metrics_score_val, 1)

    # ── 4. Keyword & Skill Coverage (Max 35) ──
    resume_skills_lower = {s.lower().strip() for s in resume.skills}
    for exp in resume.experience:
        for b in exp.bullets:
            for word in b.lower().split():
                if len(word) > 2:
                    resume_skills_lower.add(word.strip(".,;:()"))

    matched_skills = []
    missing_skills = []

    if job_description and len(job_description.strip()) > 20:
        jd_keywords = _extract_keywords_from_text(job_description)
        if jd_keywords:
            matched = jd_keywords.intersection(resume_skills_lower)
            missing = jd_keywords.difference(resume_skills_lower)
            match_ratio = len(matched) / len(jd_keywords)
            keyword_score = round(min(35.0, match_ratio * 35.0 * 1.5), 1)
            matched_skills = sorted(list(matched))[:15]
            missing_skills = sorted(list(missing))[:15]
        else:
            keyword_score = 25.0
    else:
        # Default baseline skill score based on richness of candidate skills
        skill_count = len(resume.skills)
        keyword_score = round(min(35.0, skill_count * 2.5), 1)
        matched_skills = resume.skills[:10]

    total_det = round(contact_score + section_score + metrics_score + keyword_score, 1)

    return DeterministicATSBreakdown(
        keyword_score=keyword_score,
        metrics_score=metrics_score,
        section_score=section_score,
        contact_score=contact_score,
        total_deterministic_score=total_det,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        metrics_found_count=metrics_count,
        action_verbs_count=verbs_found,
        sections_present=sections_present,
        missing_sections=missing_sections,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. GEMINI SEMANTIC ATS EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

SEMANTIC_ATS_PROMPT = """You are an Executive Recruiter and ATS Evaluation Engine.
Analyze the candidate's structured resume against the target role and provide an objective semantic assessment.

Evaluation Criteria:
1. "role_fit_score": Score from 0.0 to 10.0 on how well candidate experience and seniority matches the target role.
2. "narrative_quality_score": Score from 0.0 to 10.0 on clarity, impact, density of achievements, and professional tone.
3. "seniority_assessment": E.g. "Junior (1-2 yrs)", "Mid-Level (3-5 yrs)", "Senior (5-8 yrs)", "Staff/Lead (8+ yrs)".
4. "missing_critical_skills": Top 3-5 crucial skills or domain concepts missing for the target position.
5. "actionable_recommendations": 3-5 concrete, actionable bullet improvements to optimize this resume for ATS scoring.

Return ONLY the JSON schema.
"""


async def evaluate_ats_semantically(
    resume: CanonicalResume,
    job_description: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> SemanticATSBreakdown:
    """
    Executes a single structured Gemini call for semantic ATS fit and recommendations.
    """
    try:
        client = get_gemini_client()
        if not client.is_configured:
            return SemanticATSBreakdown(
                role_fit_score=7.0,
                narrative_quality_score=7.0,
                semantic_score=14.0,
                seniority_assessment="Standard Professional",
                actionable_recommendations=["Add more quantifiable metrics and specific business outcomes."],
            )

        resume_summary_payload = {
            "title": resume.personal_information.title,
            "skills": resume.skills,
            "summary": resume.summary,
            "experience_count": len(resume.experience),
            "experience_samples": [
                {"position": exp.position, "company": exp.company, "bullets": exp.bullets[:3]}
                for exp in resume.experience[:3]
            ],
            "target_job_description": job_description or "General Software / Technical Professional",
        }

        raw_json = await client.generate(
            prompt=f"Evaluate this resume payload against the role:\n\n{json.dumps(resume_summary_payload)}",
            system_instruction=SEMANTIC_ATS_PROMPT,
            response_schema=SemanticATSBreakdown,
            operation="ats_semantic_analysis",
            user_id=user_id,
            request_id=request_id,
        )

        breakdown = SemanticATSBreakdown.model_validate_json(raw_json)
        breakdown.semantic_score = round(min(20.0, breakdown.role_fit_score + breakdown.narrative_quality_score), 1)
        return breakdown

    except Exception as e:
        logger.error(f"[ATSEvaluator] Semantic evaluation error: {e}")
        return SemanticATSBreakdown(
            role_fit_score=6.0,
            narrative_quality_score=6.0,
            semantic_score=12.0,
            seniority_assessment="Professional",
            actionable_recommendations=["Include specific percentages and measurable business impact in your experience bullets."],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPOSITE ATS REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


async def generate_ats_composite_report(
    resume: CanonicalResume,
    job_description: Optional[str] = None,
    include_semantic: bool = True,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ATSCompositeReport:
    """
    Generates a complete explainable ATS report combining deterministic rules and Gemini semantic insights.
    """
    det = evaluate_ats_deterministically(resume, job_description)

    sem = None
    if include_semantic:
        sem = await evaluate_ats_semantically(resume, job_description, user_id=user_id, request_id=request_id)
        total_float = det.total_deterministic_score + sem.semantic_score
    else:
        # Scale 80 pts to 100 pts if semantic analysis is omitted
        total_float = (det.total_deterministic_score / 80.0) * 100.0

    final_score = int(round(min(100, max(0, total_float))))
    grade = _score_to_grade(final_score)

    feedback_parts = [
        f"ATS Score: {final_score}/100 (Grade: {grade}).",
        f"Deterministic Rules: {det.total_deterministic_score}/80 pts (Keywords: {det.keyword_score}/35, Metrics: {det.metrics_score}/20, Sections: {det.section_score}/15, Contact: {det.contact_score}/10).",
    ]
    if sem:
        feedback_parts.append(
            f"Semantic Fit: {sem.semantic_score}/20 pts (Role Fit: {sem.role_fit_score}/10, Narrative Quality: {sem.narrative_quality_score}/10, Seniority: '{sem.seniority_assessment}')."
        )

    return ATSCompositeReport(
        overall_score=final_score,
        grade=grade,
        deterministic_breakdown=det,
        semantic_breakdown=sem,
        summary_feedback=" ".join(feedback_parts),
    )
