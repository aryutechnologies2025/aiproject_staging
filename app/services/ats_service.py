
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
You are an ATS resume quality evaluator.

RULES:
- Do NOT rewrite the resume
- Do NOT change formatting rules
- Evaluate ONLY content quality and relevance
- Be conservative in scoring

INPUT:
Resume Skills:
{skills}

Resume Experience Bullets:
{bullets}

Job Description:
{job_description}

OUTPUT STRICT JSON ONLY:
{
  "bullet_quality_score": number between 0 and 100,
  "missing_skills": [list of skills missing],
  "improvement_suggestions": [list of short suggestions]
}
"""

async def ai_evaluate_resume(llm_client, resume):
    bullets = [
        b for exp in resume.experience for b in exp.bullets
    ]

    prompt = AI_ATS_PROMPT.format(
        skills=", ".join(resume.skills),
        bullets="\n".join(bullets),
        job_description=resume.job_description or "Not provided"
    )

    response = await llm_client(prompt)

    try:
        return json.loads(response)
    except Exception:
        return {
            "bullet_quality_score": 50,
            "missing_skills": [],
            "improvement_suggestions": []
        }
    


async def scan_resume_with_ai(resume, llm_client=None):
    rule_score, issues = run_ats_rules(resume)
    keyword_score = keyword_match(resume, resume.job_description)

    ai_quality_score = 50
    ai_suggestions = []
    missing_skills = []

    if llm_client and resume.job_description:
        ai_result = await ai_evaluate_resume(llm_client, resume)
        ai_quality_score = ai_result["bullet_quality_score"]
        ai_suggestions = ai_result["improvement_suggestions"]
        missing_skills = ai_result["missing_skills"]

    final_score = calculate_final_score(
        rule_score,
        keyword_score,
        ai_quality_score
    )

    return {
        "ats_score": final_score,
        "rule_score": rule_score,
        "keyword_match_percentage": keyword_score,
        "ai_quality_score": ai_quality_score,
        "issues": issues,
        "suggestions": ai_suggestions,
        "missing_skills": missing_skills
    }

