from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.llm_client import call_llm
from app.services.prompt_service import get_prompt
import re
import json

async def suggest_experience(data: dict, db: AsyncSession) -> dict:
    """
    Generate resume-ready experience bullet points ONLY.
    """

    user_prompt = f"""
Generate ATS-optimized resume experience bullet points.

Job Title: {data.get("job_title")}
Company: {data.get("company")}
Duration: {data.get("duration")}
Location: {data.get("location")}
Current Description: {data.get("description") or ""}
Tone: {data.get("tone", "professional")}

RULES:
- Output ONLY bullet points
- Output 15 to 20 bullet points ONLY
- Each bullet must start with an action verb
- Resume language only
- No introductions, explanations, or questions
- Do NOT invent information
"""

    response = await call_llm(
        model="groq",
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
    )

    return {
        "experience_bullets": response.strip()
    }


async def suggest_summary(data: dict, db: AsyncSession) -> dict:
    # DB work FIRST
    system_prompt = await get_prompt(db, "resume_builder")
    if not system_prompt:
        system_prompt = "You are YURA, a helpful AI assistant built by Aryu Enterprises."

    # Build user prompt
    user_prompt = f"""
Write a professional resume summary.

Job Title:
{data.get("job_title", "")}

Experience:
{data.get("experience", "")}

Skills:
{", ".join(data.get("skills", []))}

Education:
{data.get("education", "")}

Tone:
{data.get("tone", "modern professional")}

STRICT RULES:
- Output ONLY the summary text
- 2 to 4 lines maximum
- ATS-friendly wording
- No personal pronouns
- No bullet points
"""

    # LLM call (NO DB here)
    response = await call_llm(
        user_message=user_prompt,
        system_prompt=system_prompt,
    )

    return {"summary": response}



def build_skills_prompt(job_titles: list[str], career_level: str = "experienced") -> str:
    roles = ", ".join(job_titles) if job_titles else "Entry-level professional"

    return f"""
Generate role-specific technical skills.

Roles:
{roles}

Career Level:
{career_level}

Rules:
- Output ONLY skill names
- One skill per line
- 5 to 8 skills only
- HARD, TECHNICAL, or ROLE-SPECIFIC skills only
- No soft skills
- No explanations
- No formatting
- No extra text
"""


async def suggest_education(data: dict, db: AsyncSession):

    education_list = data.get("education", [])

    if not isinstance(education_list, list):
        raise HTTPException(400, "education must be a list")

    if len(education_list) == 0:
        user_input = """
Generate education bullet points for a candidate with no formal education details.

Rules:
- Output ONLY bullet points
- 3 to 5 bullet points only
- No invented degree names
- Generic, ATS-safe phrasing
"""
    else:
        formatted_entries = ""

        for edu in education_list:
            formatted_entries += f"""
Degree: {edu.get("degree", "")}
Institution: {edu.get("college", "")}
Year: {edu.get("year", "")}
Location: {edu.get("location", "")}
Grade: {edu.get("grade", "")}
Achievements: {edu.get("achievements", "")}
----
"""

        user_input = f"""
Generate resume education bullet points.

Education Details:
{formatted_entries}

Rules:
- Output ONLY bullet points
- Write 2–3 bullet points per education entry
- If a field is missing, skip it
- Do NOT invent any information
- ATS-optimized language only
"""

    response = await call_llm(
        model="groq",
        user_message=user_input,
        agent_name="resume_builder",
        db=db,
    )

    return {"education_bullets": response}


def build_ats_resume_json_prompt(
    job_title: str,
    company: str,
    job_description: str
) -> str:
    return f"""
Generate an ATS-optimized resume in STRICT JSON format.

Job Title:
{job_title}

Company:
{company}

Job Description:
{job_description}

OUTPUT RULES (MANDATORY):
- Output VALID JSON ONLY
- No markdown
- No explanations
- No extra text
- No trailing commas

JSON SCHEMA (STRICT):

{{
  "summary": "2–3 line professional summary, no personal pronouns",
  "experience": [
    "5–7 achievement-based bullet points aligned to job description"
  ],
  "skills": [
    "6–8 hard technical or role-specific skills only"
  ]
}}

CONTENT RULES:
- ATS-friendly
- No invented metrics
- No soft skills
- No assumptions beyond provided job description
"""

async def generate_ats_resume_json(data: dict, db: AsyncSession):
    """
    Generate ATS resume (summary + experience + skills) in JSON.
    """

    job_title = data.get("job_title")
    job_description = data.get("job_description")
    company = data.get("company", "")

    if not job_title or not job_description:
        raise ValueError("job_title and job_description are required")

    user_prompt = build_ats_resume_json_prompt(
        job_title=job_title,
        company=company,
        job_description=job_description
    )

    raw_response = await call_llm(
        model="groq",
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
    )

    try:
        return json.loads(raw_response)
    except Exception:
        raise ValueError("LLM returned invalid JSON")
    

async def refine_resume_section(
    *,
    section_name: str,
    existing_content: str,
    user_instruction: str,
    experience_level: str,
    db: AsyncSession
) -> dict:
    """
    Refines existing resume content based on user instruction.
    NOT a chatbot. Output is resume-ready only.
    """

    prompt = f"""
You are a professional resume editor.

SECTION:
{section_name}

EXISTING CONTENT:
{existing_content}

USER INSTRUCTION:
{user_instruction}

EXPERIENCE LEVEL:
{experience_level}

STRICT RULES:
- Modify ONLY based on user instruction
- Preserve resume format
- Do NOT add explanations
- Do NOT ask questions
- Do NOT invent experience
- ATS-optimized language only
- Output ONLY the updated content
"""

    response = await call_llm(
        model="groq",
        user_message=prompt,
        agent_name="resume_builder",
        db=db,
    )

    return {
        "section": section_name,
        "updated_content": response.strip()
    }

