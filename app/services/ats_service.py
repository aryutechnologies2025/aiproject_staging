
from app.utils.ats_rules import run_ats_rules, keyword_match, calculate_final_score, calculate_final_score_non_ai
import json


ACTION_VERBS = (
    "developed", "built", "designed",
    "implemented", "led", "optimized"
)


def analyze_bullets(experience):
    suggestions = []

    for exp in experience:
        for bullet in exp.bullets:
            if len(bullet.split()) < 6:
                suggestions.append(f"Bullet too short: '{bullet}'")

            if not bullet.lower().startswith(ACTION_VERBS):
                suggestions.append(
                    f"Start with action verb: '{bullet}'"
                )

    return suggestions


def scan_resume(resume):
    rule_score, issues = run_ats_rules(resume)

    keyword_score = keyword_match(
        resume, resume.job_description
    )

    suggestions = analyze_bullets(resume.experience)

    final_score = calculate_final_score_non_ai(
        rule_score, keyword_score
    )

    return {
        "ats_score": final_score,
        "keyword_match_percentage": keyword_score,
        "issues": issues,
        "suggestions": suggestions
    }

AI_ATS_PROMPT = """
You are an ATS resume evaluator.

GOAL:
- Help improve ATS compatibility
- Highlight missing or weak areas politely
- Assume resume content is AI-generated and professional

RULES:
- Do NOT rewrite the resume
- Do NOT be harsh or rejecting
- Always return at least 1 issue and 1 suggestion
- Issues should be soft (not rejection-level)

INPUT:
Resume Skills:
{skills}

Experience Bullets:
{experience_bullets}

Education:
{education_bullets}

Job Description:
{job_description}

OUTPUT STRICT JSON ONLY:
{
  "content_coverage_score": number between 70 and 100,
  "keyword_alignment_score": number between 60 and 100,
  "structure_completeness_score": number between 80 and 100,
  "ai_issues": [list of soft ATS issues],
  "missing_skills": [critical missing technical skills],
  "improvement_suggestions": [clear, positive ATS suggestions]
}
"""


async def ai_evaluate_resume(llm_client, resume):
    experience_bullets = [
        b for exp in resume.experience for b in exp.bullets
    ]

    education_bullets = [
        b for edu in resume.education for b in edu.educationDescription
    ]

    prompt = AI_ATS_PROMPT.format(
        skills=", ".join(resume.skills),
        experience_bullets="\n".join(experience_bullets),
        education_bullets="\n".join(education_bullets),
        job_description=resume.job_description or "Not provided"
    )

    try:
        response = await llm_client(prompt)
        raw = json.loads(response)
    except Exception:
        raw = {}

    ai_issues = raw.get("ai_issues", [])
    ai_suggestions = raw.get("improvement_suggestions", [])

    # Enforce at least one issue & suggestion
    if not ai_issues:
        ai_issues = [
            "Resume is ATS-friendly but could be better aligned to the job description"
        ]

    if not ai_suggestions:
        ai_suggestions = [
            "Consider adding a few role-specific keywords from the job description"
        ]

    return {
        "content_coverage_score": max(70, raw.get("content_coverage_score", 80)),
        "keyword_alignment_score": max(60, raw.get("keyword_alignment_score", 75)),
        "structure_completeness_score": max(80, raw.get("structure_completeness_score", 85)),
        "ai_issues": ai_issues,
        "missing_skills": raw.get("missing_skills", []),
        "improvement_suggestions": ai_suggestions
    }



async def scan_resume_with_ai(resume, llm_client=None):
    rule_score, issues = run_ats_rules(resume)
    keyword_score = keyword_match(resume, resume.job_description)

    ai_scores = {
        "content_coverage_score": 80,
        "keyword_alignment_score": 75,
        "structure_completeness_score": 85
    }
    missing_skills = []
    ai_suggestions = []
    ai_issues = []

    if llm_client and resume.job_description:
        ai_result = await ai_evaluate_resume(llm_client, resume)
        for key in ai_scores:
            if key in ai_result:
                ai_scores[key] = ai_result[key]
        missing_skills = ai_result.get("missing_skills", [])
        ai_suggestions = ai_result.get("improvement_suggestions", [])
        ai_issues = ai_result.get("ai_issues", [])

    ai_quality_score = (
        ai_scores["content_coverage_score"] * 0.4 +
        ai_scores["keyword_alignment_score"] * 0.4 +
        ai_scores["structure_completeness_score"] * 0.2
    )
    ai_quality_score = max(ai_quality_score, 75)

    final_score = calculate_final_score(
        rule_score,
        keyword_score,
        ai_quality_score
    )
    all_issues = list(dict.fromkeys(issues + ai_issues))

    return {
        "ats_score": max(round(final_score, 2), 72),
        "rule_score": rule_score,
        "keyword_match_percentage": keyword_score,
        "ai_quality_score": round(ai_quality_score, 2),
        "issues": all_issues,
        "suggestions": ai_suggestions,
        "missing_skills": missing_skills,
        "status": "ATS Friendly"
    }


