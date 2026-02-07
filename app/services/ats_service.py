
from app.utils.ats_rules import build_sections_array, run_ats_rules, keyword_match, calculate_final_score_non_ai
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
{{
  "content_coverage_score": 80,
  "keyword_alignment_score": 75,
  "structure_completeness_score": 85,
  "ai_issues": ["Example soft issue"],
  "missing_skills": ["Example skill"],
  "improvement_suggestions": ["Example suggestion"]
}}
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
        "ai_issues": ai_issues,
        "improvement_suggestions": ai_suggestions,
        "missing_skills": raw.get("missing_skills", []),
    }



async def scan_resume_with_ai(resume, llm_client=None):
    # 1. Rule-based ATS
    rule_score, rule_issues = run_ats_rules(resume)

    # 2. Keyword match
    keyword_score = keyword_match(resume, resume.job_description)

    # 3. Final score (non-AI, predictable)
    final_score = calculate_final_score_non_ai(
        rule_score,
        keyword_score
    )

    # 4. Missing skills (AI optional)
    ai_issues = []
    ai_suggestions = []
    if llm_client and resume.job_description:
        ai_result = await ai_evaluate_resume(llm_client, resume)
        missing_skills = ai_result.get("missing_skills", [])
        ai_issues = ai_result.get("ai_issues", [])
        ai_suggestions = ai_result.get("improvement_suggestions", [])

    res = {
        "ats_score": max(final_score, 0),
        "sections": build_sections_array(rule_issues),
        "missing_skills": missing_skills,
        "ai_issues": ai_issues,
        "recommendations": ai_suggestions
    }
    print(res)
    # 5. Build FINAL response
    return res


