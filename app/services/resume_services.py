from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.llm_client import call_llm
from app.services.prompt_service import get_prompt
import re
import json

async def suggest_experience(data: dict, db: AsyncSession):
    """
    Generate AI-optimized experience bullet points.
    """
    system_prompt = await get_prompt(db, "resume_builder")

    user_input = f"""
Generate ATS-optimized resume experience bullet points.

Job Title: {data.get("job_title")}
Company: {data.get("company")}
Duration: {data.get("duration")}
Location: {data.get("location")}
Current Description: {data.get("description") or ""}
Tone: {data.get("tone", "professional")}

Rules:
- Use strong action verbs.
- Convert duties into achievements.
- Add measurable impact if possible.
- Do NOT invent fake details.
- Output 6-8 bullet points only.
"""

    response = await call_llm(
        model="gemma",
        user_message=user_input,
        agent_name="resume_builder",
        db=db,
    )

    return {"bullets": response}

def build_summary_input(data: dict) -> str:
    experiences_text = []
    for exp in data.get("experiences", []):
        experiences_text.append(
            f"- {exp.get('job_title')} at {exp.get('company')} "
            f"({exp.get('start_date')} to {exp.get('end_date')}), "
            f"Skills: {', '.join(exp.get('skills', []))}"
        )

    education_text = []
    for edu in data.get("education", []):
        education_text.append(
            f"- {edu.get('degree')} from {edu.get('institution')} ({edu.get('year')})"
        )

    return f"""
Experience:
{chr(10).join(experiences_text)}

Education:
{chr(10).join(education_text)}

Core Skills:
{', '.join(data.get("skills", []))}
""".strip()

async def suggest_summary(data: dict, db: AsyncSession):
    system_prompt = await get_prompt(db, "resume_builder")

    structured_input = build_summary_input(data)

    user_input = f"""
Write a polished 2–4 line professional summary.

{structured_input}

Tone: {data.get("tone", "modern professional")}

Rules:
- ATS-friendly.
- No personal pronouns.
- No placeholders.
- No bullet points.
- Write like a senior resume writer.
"""

    response = await call_llm(
        model="gemma",
        user_message=user_input,
        agent_name="resume_builder",
        db=db,
    )

    return {"summary": response}


def build_skills_prompt(summary, experience, education, title):
    return f"""
Generate the Skills section only.

STRICT RULES:
- Output valid JSON only
- JSON must contain a single key "skills"
- Include 5 to 8 skills only
- Skills must be HARD, TECHNICAL, or ROLE-SPECIFIC
- DO NOT include soft skills or personality traits
- DO NOT include generic skills (communication, teamwork, leadership, etc.)
- DO NOT explain anything

Inference Rules:
- If detailed experience is provided, extract skills ONLY from that content
- If experience is minimal or missing, infer STANDARD INDUSTRY HARD SKILLS
  that are commonly expected for the given job title
- Use the job title as the ONLY inference signal
- Do NOT guess tools unless they are industry-standard for the role

Job Title:
{title}

User Content:
Summary:
{summary}

Experience:
{experience}

Education:
{education}
"""

def extract_skills_json(response: str):
    try:
        return json.loads(response)
    except Exception:
        return {"skills": []}


def validate_skills_output(data: dict):
    if "skills" not in data:
        raise ValueError("Missing skills key")

    skills = data["skills"]

    if not isinstance(skills, list):
        raise ValueError("Skills must be a list")

    if not 5 <= len(skills) <= 8:
        raise ValueError("Skills must contain 5–8 items")

    for skill in skills:
        if not isinstance(skill, str):
            raise ValueError("Skill must be a string")


async def suggest_skills(
    db: AsyncSession,
    summary: str = "",
    experience: str = "",
    education: str = "",
    title: str = "",
    model: str = "gemma"
) -> dict:

    user_prompt = build_skills_prompt(
        summary=summary,
        experience=experience,
        education=education,
        title=title
    )

    raw_response = await call_llm(
        user_message=user_prompt,
        agent_name="resume_skills",
        db=db,
        model=model,
        expect_json=False
    )

    skills_json = extract_skills_json(raw_response)
    validate_skills_output(skills_json)
    return skills_json


async def suggest_education(data: dict, db: AsyncSession):
    system_prompt = await get_prompt(db, "resume_builder")

    education_list = data.get("education", [])

    if not isinstance(education_list, list):
        raise HTTPException(400, "education must be a list")

    if len(education_list) == 0:
        # user sent no education → still allow, handle gracefully
        user_input = """
User provided no education details.
Generate a general placeholder education bullet point section.
"""
    else:
        formatted_entries = ""

        for edu in education_list:
            formatted_entries += f"""
Degree: {edu.get("degree")}
Institution: {edu.get("college")}
Year: {edu.get("year")}
Location: {edu.get("location")}
Grade: {edu.get("grade")}
Achievements: {edu.get("achievements")}
----
"""

        user_input = f"""
Generate resume education bullet points. All fields are optional.

Education Details:
{formatted_entries}

Rules:
- If a field is missing, skip it gracefully.
- Do NOT invent any information.
- Write 2–3 bullet points for each entry.
"""

    response = await call_llm(
        model="gemma",
        user_message=user_input,
        agent_name="resume_builder",
        db=db,
    )

    return {"education_bullets": response}


