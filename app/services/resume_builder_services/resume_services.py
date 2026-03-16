# /home/aryu_user/Arun/aiproject_staging/app/services/resume_services.py
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.llm_client import call_llm
from app.services.prompt_service import get_prompt
import re
import json

async def suggest_experience(data: dict, db: AsyncSession) -> dict:
    """
    Generate ATS-optimized resume experience bullet points with impact metrics.
    Produces Claude-like thoughtful, detailed reasoning.
    """

    job_title = data.get("job_title", "")
    company = data.get("company", "")
    duration = data.get("duration", "")
    location = data.get("location", "")
    description = data.get("description", "")
    tone = data.get("tone", "professional")

    user_prompt = f"""You are a professional resume writer creating compelling experience bullets.

ROLE CONTEXT:
Title: {job_title}
Company: {company}
Duration: {duration}
Location: {location}
Description: {description}
Style: {tone}

YOUR TASK:
Generate 15-20 impact-focused experience bullets that:
1. Start with strong action verbs (developed, built, designed, implemented, led, optimized, created, launched)
2. Include quantifiable results where possible (%, numbers, time saved, scale)
3. Show business or team impact, not just tasks
4. Use professional resume language (no "I", "we", personal pronouns)
5. Follow format: [Verb] [what] [impact/metric]

EXAMPLES OF STRONG BULLETS:
• Developed microservices architecture handling 1M+ daily transactions, reducing latency by 40%
• Led cross-functional team of 6 to deliver Q4 roadmap 2 weeks early, exceeding targets by 15%
• Optimized database queries improving application performance by 60%, enhancing user experience

OUTPUT EXACTLY 15-20 BULLETS. NO INTRO, NO EXPLANATION. ONLY BULLETS."""

    response = await call_llm(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
    )

    bullets = [line.strip().lstrip("•").strip() for line in response.split('\n') if line.strip()]
    
    return {
        "experience_bullets": "\n".join(f"• {b}" for b in bullets if b),
        "count": len(bullets),
        "quality_notes": "Each bullet emphasizes measurable impact and business outcomes"
    }


async def suggest_summary(data: dict, db: AsyncSession) -> dict:
    """
    Generate compelling 2-4 line professional summary.
    Focuses on unique value proposition and differentiators.
    """
    
    system_prompt = await get_prompt(db, "resume_builder")
    if not system_prompt:
        system_prompt = "You are a professional resume writer for students and professionals."

    job_title = data.get("job_title", "")
    experience = data.get("experience", "")
    skills = data.get("skills", [])
    education = data.get("education", "")
    tone = data.get("tone", "modern professional")

    user_prompt = f"""Create a powerful professional summary that gets recruiter attention.

CANDIDATE PROFILE:
Position: {job_title}
Background: {experience[:150]}
Key Skills: {', '.join(skills[:5]) if isinstance(skills, list) else skills}
Education: {education[:100]}
Tone: {tone}

YOUR TASK:
Write 2-4 lines that:
1. Start with strongest differentiator or achievement
2. Show years of experience and domain expertise
3. Highlight 2-3 core technical competencies
4. Demonstrate track record of results
5. Use power words: proven, expert, accomplished, pioneered, transformed
6. No personal pronouns (I, me, my, we)
7. Include ATS keywords naturally

STRUCTURE:
Line 1: [Title] professional with [X years] expertise in [specialization]
Line 2: Proven track record in [2-3 key skills]. Delivered [specific achievement/metric].
Line 3 (optional): [Additional competitive advantage or forward-looking statement]

OUTPUT ONLY THE SUMMARY. NO EXPLANATION."""

    response = await call_llm(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
    )

    summary = response.strip()
    lines = len([l for l in summary.split('\n') if l.strip()])

    return {
        "summary": summary,
        "line_count": lines,
        "quality_notes": "Summary positions candidate on unique value and key differentiators"
    }


def build_skills_prompt(job_titles: list[str], career_level: str = "experienced") -> str:
    """Build optimized skills generation prompt"""
    
    roles = ", ".join(job_titles) if job_titles else "Entry-level professional"

    return f"""Generate the most valuable technical skills for these roles.

TARGET ROLES: {roles}
CAREER LEVEL: {career_level}

OUTPUT EXACTLY 5-8 SKILLS:
- ONLY hard technical/domain skills (no soft skills)
- Tools, frameworks, platforms actually used in these roles
- Currently in-demand and market-relevant
- One skill per line
- No descriptions, explanations, or formatting
- No numbers or versions

EXAMPLES OF GOOD SKILLS:
Python, React, AWS, Docker, PostgreSQL, Kubernetes, GraphQL, TensorFlow

OUTPUT: ONE SKILL PER LINE. NOTHING ELSE."""


async def suggest_education(data: dict, db: AsyncSession) -> dict:
    """
    Generate education section with achievement focus.
    """

    education_list = data.get("education", [])

    if not isinstance(education_list, list):
        raise HTTPException(400, "education must be a list")

    if len(education_list) == 0:
        user_prompt = """Generate education bullets for candidate with non-traditional background.

Your task: Create 3-5 achievement-focused bullets covering:
- Relevant certifications or bootcamps completed
- Online courses or technical training
- Self-taught skills or demonstrated competencies
- Professional development or continuous learning

RULES:
- No invented institutions or programs
- Generic, ATS-safe language
- Achievement-focused (what was learned/accomplished, not just attended)
- Format: [Program/Area] - [Key Competency/Outcome]
- 3-5 bullets maximum

OUTPUT ONLY BULLETS. NO EXPLANATION."""
    else:
        formatted_entries = ""
        for i, edu in enumerate(education_list, 1):
            formatted_entries += f"""
EDUCATION #{i}:
Degree: {edu.get("degree", "")}
Institution: {edu.get("college", "")}
Year: {edu.get("year", "")}
Location: {edu.get("location", "")}
GPA/Grade: {edu.get("grade", "")}
Achievements: {edu.get("achievements", "")}
---"""

        user_prompt = f"""Generate compelling education section bullets.

{formatted_entries}

Your task: Create 2-3 bullets per education entry that highlight:
1. Academic honors/distinctions (only if GPA 3.5+)
2. Relevant coursework or specializations
3. Scholarships, awards, or achievements
4. Leadership or involvement if applicable

RULES:
- 2-3 bullets per education entry
- Only include provided information (NO invention)
- Achievement-focused language
- Format: [Degree], [Institution] | [Key Achievement]
- Skip empty/missing fields
- ATS-optimized wording

OUTPUT ONLY BULLETS. NO EXPLANATION."""

    response = await call_llm(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
    )

    bullets = [line.strip().lstrip("•").strip() for line in response.split('\n') if line.strip()]

    return {
        "education_bullets": "\n".join(f"• {b}" for b in bullets if b),
        "count": len(bullets),
        "quality_notes": "Education section emphasizes academic achievements and relevant learning"
    }


def build_ats_resume_json_prompt(
    job_title: str,
    company: str,
    job_description: str
) -> str:
    """Build optimized ATS resume JSON prompt"""
    
    return f"""Generate an ATS-optimized resume specifically for this job posting.

JOB POSTING:
Title: {job_title}
Company: {company}
Description (key requirements):
{job_description[:1500]}

YOUR TASK:
Create a resume that:
1. Professional summary (2-3 lines) directly addressing job requirements
2. 5-7 achievement bullets clearly aligned to job description
3. 6-8 technical skills matching job requirements
4. Includes keywords from job posting naturally

OPTIMIZATION RULES:
- Match candidate strength to job needs
- Use job description keywords without forcing
- Achievement-based language (metrics when possible)
- ATS-friendly format
- No invented information or metrics

RETURN VALID JSON ONLY - NO MARKDOWN OR EXPLANATION:

{{
  "summary": "2-3 line summary directly addressing job requirements",
  "experience": [
    "Achievement bullet aligned to job description",
    "Achievement bullet with relevant skill"
  ],
  "skills": [
    "Required technical skill 1",
    "Required technical skill 2"
  ],
  "optimization_notes": "How resume is optimized for this specific role"
}}"""


async def generate_ats_resume_json(data: dict, db: AsyncSession) -> dict:
    """
    Generate complete ATS-optimized resume for specific job.
    """

    job_title = data.get("job_title")
    job_description = data.get("job_description")
    company = data.get("company", "")

    if not job_title or not job_description:
        raise HTTPException(status_code=400, detail="job_title and job_description are required")

    user_prompt = build_ats_resume_json_prompt(
        job_title=job_title,
        company=company,
        job_description=job_description
    )

    try:
        raw_response = await call_llm(
            user_message=user_prompt,
            agent_name="resume_builder",
            db=db,
        )

        # Clean response
        clean = raw_response.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        
        result = json.loads(clean.strip())
        return result
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Failed to generate valid resume JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume generation failed: {str(e)}")


async def refine_resume_section(
    *,
    section_name: str,
    existing_content: str,
    user_instruction: str,
    experience_level: str,
    db: AsyncSession
) -> dict:
    """
    Refine resume section based on user feedback.
    Improves clarity, impact, and professionalism.
    """

    prompt = f"""You are a professional resume editor. Refine this section based on feedback.

SECTION NAME: {section_name}
EXPERIENCE LEVEL: {experience_level}

CURRENT CONTENT:
{existing_content}

USER REQUEST:
{user_instruction}

YOUR TASK:
Refine the content by:
1. Understanding the specific improvement requested
2. Strengthening impact language if requested
3. Adding metrics or context where applicable
4. Maintaining professional resume format
5. Keeping ATS-friendly language

REFINEMENT RULES:
- Modify ONLY what was specifically requested
- Preserve section format and structure
- Use stronger action verbs if improving impact
- Keep resume-appropriate tone
- No explanations or questions
- Output ONLY the refined content

OUTPUT: REFINED CONTENT ONLY."""

    response = await call_llm(
        user_message=prompt,
        agent_name="resume_builder",
        db=db,
    )

    return {
        "section": section_name,
        "updated_content": response.strip(),
        "quality_notes": "Section refined based on feedback while maintaining professional standards"
    }