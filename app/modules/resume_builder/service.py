# /home/aryu_user/Arun/aiproject_staging/app/modules/resume_builder/service.py
from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.llm_client import call_llm
from app.utils.prompt_service import get_prompt
from typing import Dict, Any, Optional
import json
import re
import time
from app.modules.resume_builder.ai_client import call_ai
from app.modules.resume_builder.telemetry import log_ai_usage
from app.modules.resume_builder.context_analyzer import (
    analyze_resume_context,
    build_grounded_education_prompt,
    build_grounded_experience_prompt,
    build_grounded_skills_prompt,
    build_grounded_summary_prompt,
    sanitize_education_output,
    sanitize_experience_output,
    sanitize_skills_output,
    sanitize_summary_output,
)
import logging

logger = logging.getLogger(__name__)


async def _call_llm_with_telemetry(
    *,
    user_message: str,
    agent_name: str,
    db: AsyncSession,
    operation: str = "suggestion",
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> str:
    """Wrapper that records structured telemetry for Ollama LLM calls."""
    start_time = time.time()
    input_est_tokens = max(1, len(user_message.split()) * 4 // 3)
    try:
        response = await call_llm(
            user_message=user_message,
            agent_name=agent_name,
            db=db,
        )
        latency_ms = (time.time() - start_time) * 1000
        output_est_tokens = max(1, len(response.split()) * 4 // 3) if response else 0

        log_ai_usage(
            provider="ollama",
            model="ollama-default",
            operation=operation,
            input_tokens=input_est_tokens,
            output_tokens=output_est_tokens,
            total_tokens=input_est_tokens + output_est_tokens,
            latency_ms=latency_ms,
            status="success",
            user_id=user_id,
            request_id=request_id,
        )
        return response
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        log_ai_usage(
            provider="ollama",
            model="ollama-default",
            operation=operation,
            input_tokens=input_est_tokens,
            latency_ms=latency_ms,
            status="failed",
            error_type=type(e).__name__,
            user_id=user_id,
            request_id=request_id,
        )
        raise

def _strip_leading_symbol(text: str) -> str:
    """Remove leading bullet symbols, dashes, asterisks, dots from a line."""
    if not text:
        return text
    
    return re.sub(r'^[\s]*[•\-\*\.\u2022\u2023\u25E6\u2043\u2219►▶→]+[\s]*', '', text.strip()).strip()


def _clean_bullets(response: str) -> list:
    """Split response into lines and strip all leading symbols."""
    lines = []
    for line in response.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = _strip_leading_symbol(stripped)
        if cleaned:
            lines.append(cleaned)
    return lines


async def suggest_experience(data: dict, db: AsyncSession) -> dict:
    if not isinstance(data, dict):
        raise HTTPException(400, "invalid payload")

    ctx = analyze_resume_context(data)
    user_prompt = build_grounded_experience_prompt(data, ctx)

    response = await _call_llm_with_telemetry(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
        operation="suggest_experience",
    )

    bullets = sanitize_experience_output(response, data, ctx)

    return {
        "experience_bullets": "\n".join(bullets),
        "count": len(bullets),
        "quality_notes": f"Experience statements grounded in {ctx.domain_label} domain context and verified candidate responsibilities."
    }


async def suggest_summary(data: dict, db: AsyncSession) -> dict:
    if not isinstance(data, dict):
        raise HTTPException(400, "invalid payload")

    ctx = analyze_resume_context(data)
    user_prompt = build_grounded_summary_prompt(data, ctx)

    response = await _call_llm_with_telemetry(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
        operation="suggest_summary",
    )

    summary = sanitize_summary_output(response, data, ctx)
    lines = len([l for l in summary.split('\n') if l.strip()])

    return {
        "summary": summary,
        "line_count": lines,
        "quality_notes": f"Summary concisely grounded in {ctx.domain_label} domain context and verified candidate ATS keywords."
    }


def build_skills_prompt(
    job_titles: list[str],
    career_level: str = "experienced",
    data: Optional[dict] = None,
) -> str:
    payload = dict(data) if isinstance(data, dict) else {}
    if job_titles:
        payload["job_titles"] = job_titles
    if career_level:
        payload["career_level"] = career_level

    ctx = analyze_resume_context(payload)
    return build_grounded_skills_prompt(payload, ctx)


async def suggest_skills(data: dict, db: AsyncSession) -> dict:
    if not isinstance(data, dict):
        raise HTTPException(400, "invalid payload")

    job_titles = data.get("job_titles", [])
    if isinstance(job_titles, str):
        job_titles = [job_titles]
    elif not job_titles and data.get("job_title"):
        job_titles = [data.get("job_title")]

    career_level = data.get("career_level", "experienced")

    ctx = analyze_resume_context(data)
    user_prompt = build_grounded_skills_prompt(data, ctx)

    response = await _call_llm_with_telemetry(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
        operation="generate_skills",
    )

    skills = sanitize_skills_output(response, data, ctx)

    return {
        "skills": skills,
        "count": len(skills),
        "quality_notes": f"Skills extracted and normalized for {ctx.domain_label}, preserving candidate ATS keywords."
    }


async def suggest_education(data: dict, db: AsyncSession) -> dict:
    if not isinstance(data, dict):
        raise HTTPException(400, "invalid payload")

    ctx = analyze_resume_context(data)
    user_prompt = build_grounded_education_prompt(data, ctx)

    response = await _call_llm_with_telemetry(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
        operation="suggest_education",
    )

    bullets = sanitize_education_output(response, data, ctx)

    return {
        "education_bullets": "\n".join(bullets),
        "count": len(bullets),
        "quality_notes": "Education entry strictly formatted according to verified candidate credentials and ATS standards."
    }

async def suggest_project(data: dict, db: AsyncSession) -> dict:
    project_title = data.get("project_title", "").strip()
    tech_stack = data.get("tech_stack", [])

    if isinstance(tech_stack, list):
        tech_stack_str = ", ".join(tech_stack)
    else:
        tech_stack_str = str(tech_stack).strip()

    user_prompt = f"""
You are an expert ATS resume writer and senior software architect.

Generate a professional resume-ready project description.

Project Information:
- Project Title: {project_title}
- Technologies: {tech_stack_str or "Not specified"}

The user has NOT provided a project description.

Your responsibility is to infer the most realistic functionality,
features, architecture, and business purpose of this project based
on its title and technology stack.

Rules:

1. Generate exactly 5 concise resume bullet points.
2. Begin every bullet with a strong action verb.
3. Make every bullet ATS-friendly.
4. Keep every bullet between 15 and 30 words.
5. Mention technologies naturally where appropriate.
6. Do NOT use first-person pronouns.
7. Do NOT invent impossible metrics.
8. Do NOT invent fake users or revenue.
9. Focus on implementation, architecture, optimization, integrations,
   APIs, security, scalability, and user experience.
10. Return ONLY the bullet statements.
11. Do not return markdown, numbering, or explanations.
"""

    response = await _call_llm_with_telemetry(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
        operation="suggest_project",
    )

    raw_lines = response.strip().split("\n")
    bullets = []

    for line in raw_lines:
        cleaned = (
            line.strip()
            .strip("*")
            .strip("-")
            .strip("•")
            .strip('"')
            .strip("'")
            .strip()
        )

        if cleaned.lower().startswith("output:"):
            cleaned = cleaned[7:].strip()

        if cleaned:
            bullets.append(cleaned)

    return {
        "project_description": "\n".join(bullets),
        "count": len(bullets),
        "quality_notes": (
            "AI-generated ATS-optimized project description based on "
            "project title and technology stack."
        ),
    }

def build_ats_resume_json_prompt(
    job_title: str,
    company: str,
    job_description: str
) -> str:
    """Build a concise, multi-industry ATS resume JSON prompt"""
    
    return f"""[CRITICAL INTEGRITY CONSTRAINT]
You are a precision ATS optimization engine. Your task is to extract core requirements and target keywords from the job posting below to outline an optimized resume framework. Do NOT invent explicit historical company names, personal credentials, or unprovided numerical metrics.

JOB POSTING:
- Title: {job_title}
- Organization: {company}
- Core Requirements & Description:
{job_description[:1200]}

EXECUTION RULES:
1. "summary": Write a 2-3 line target profile matching this role without using personal pronouns.
2. "experience": Generate exactly 4 to 6 concise, impact-focused role accomplishment lines that seamlessly integrate primary keywords from the description text.
3. "skills": Extract exactly 6 to 8 core competencies or specialized domain skills found in the posting (No soft skills).
4. Do NOT output any introductory text, trailing explanations, or markdown code blocks. Output the raw JSON structure only.

REQUIRED JSON FORMAT SPECIFICATION:
{{
  "summary": "String paragraph containing the optimized profile.",
  "experience": [
    "Accomplishment line 1 integrating target job keywords.",
    "Accomplishment line 2 showing domain capability."
  ],
  "skills": [
    "Core Skill 1",
    "Core Skill 2"
  ],
  "optimization_notes": "A brief summary sentence detailing why this layout clears the target posting's ATS keywords."
}}"""


async def generate_ats_resume_json(
    data: dict,
    db: AsyncSession,
) -> dict:
    """
    Generate an ATS-optimized resume JSON from the provided job details.
    """

    job_title = (data.get("job_title") or "").strip()
    job_description = (data.get("job_description") or "").strip()
    company = (data.get("company") or "").strip()

    if not job_title:
        raise HTTPException(
            status_code=400,
            detail="job_title is required",
        )

    if not job_description:
        raise HTTPException(
            status_code=400,
            detail="job_description is required",
        )

    user_prompt = build_ats_resume_json_prompt(
        job_title=job_title,
        company=company,
        job_description=job_description,
    )

    try:
        raw_response = await _call_llm_with_telemetry(
            user_message=user_prompt,
            agent_name="resume_builder",
            db=db,
            operation="generate_ats_resume_json",
        )

        clean = raw_response.strip()

        # Remove markdown fences
        clean = re.sub(r"^```(?:json)?", "", clean.strip(), flags=re.IGNORECASE)
        clean = re.sub(r"```$", "", clean.strip())

        # Extract first JSON object
        match = re.search(r"\{.*\}", clean, re.DOTALL)

        if match:
            clean = match.group(0)

        result = json.loads(clean)

        result.setdefault("summary", "")
        result.setdefault("experience", [])
        result.setdefault("skills", [])
        result.setdefault("optimization_notes", "")

        return result

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail="LLM returned invalid JSON.",
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ATS resume generation failed: {str(e)}",
        )


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

    response = await _call_llm_with_telemetry(
        user_message=prompt,
        agent_name="resume_builder",
        db=db,
        operation="refine_resume_section",
    )

    return {
        "section": section_name,
        "updated_content": response.strip(),
        "quality_notes": "Section refined based on feedback while maintaining professional standards"
    }


# =====================================================
# ENHANCED PROMPTS FOR LETTER-QUALITY OUTPUT
# =====================================================
 
def build_professional_cv_prompt(data: Dict[str, Any], level: str) -> str:
    """
    Enhanced prompt for substantial, letter-quality CV output.
    Produces real professional document, not skeleton.
    """
    
    name = data.get("name", "Professional")
    email = data.get("email", "")
    phone = data.get("phone", "")
    location = data.get("location", "")
    summary = data.get("summary", "")
    
    # Extract experience details
    experience_list = data.get("experience", [])
    exp_text = ""
    for i, exp in enumerate(experience_list[:5], 1):
        if isinstance(exp, dict):
            title = exp.get("title", "")
            company = exp.get("company", "")
            duration = exp.get("duration", "")
            location_exp = exp.get("location", "")
            bullets = exp.get("bullets", [])
        else:
            # Handle object format
            title = getattr(exp, "title", "")
            company = getattr(exp, "company", "")
            duration = getattr(exp, "duration", "")
            location_exp = getattr(exp, "location", "")
            bullets = getattr(exp, "bullets", [])
        
        if title and company:
            bullet_text = "\n    ".join([f"• {b}" for b in bullets[:3]])
            exp_text += f"{i}. {title} | {company} ({duration}, {location_exp})\n    {bullet_text}\n"
    
    # Extract education
    education_list = data.get("education", [])
    edu_text = ""
    for i, edu in enumerate(education_list[:3], 1):
        if isinstance(edu, dict):
            degree = edu.get("degree", "")
            college = edu.get("college", "")
            year = edu.get("year", "")
        else:
            degree = getattr(edu, "degree", "")
            college = getattr(edu, "college", "")
            year = getattr(edu, "year", "")
        
        if degree and college:
            edu_text += f"{degree} from {college} ({year})\n"
    
    # Skills
    skills = data.get("skills", [])
    skills_text = ", ".join(skills[:12]) if skills else "Not specified"
    
    # Additional info
    certifications = data.get("certifications", [])
    languages = data.get("languages", [])
    awards = data.get("awards", [])
    
    return f"""You are an expert professional CV writer. Create a comprehensive, letter-quality curriculum vitae.
 
CANDIDATE INFORMATION:
Name: {name}
Email: {email}
Phone: {phone}
Location: {location}
Experience Level: {level}
Professional Summary: {summary if summary else 'Not provided'}
 
PROFESSIONAL EXPERIENCE:
{exp_text if exp_text else 'Not provided'}
 
EDUCATION:
{edu_text if edu_text else 'Not provided'}
 
KEY SKILLS:
{skills_text}
 
ADDITIONAL CREDENTIALS:
Certifications: {', '.join(certifications) if certifications else 'None'}
Languages: {', '.join(languages) if languages else 'None'}
Awards: {', '.join(awards) if awards else 'None'}
 
TASK: Create a comprehensive, professional CV that reads like a real, polished document.
 
STRUCTURE:
1. PROFESSIONAL SUMMARY (4-5 substantial sentences)
   - Who they are professionally
   - What they've accomplished
   - Core competencies (2-3)
   - Career direction
   - Make it compelling and specific, not generic
 
2. CORE COMPETENCIES (8-12 skills, organized by category if diverse)
   - Group logically: Technical | Leadership | Domain Expertise
   - Only include relevant, current skills
   - For {level} level
 
3. PROFESSIONAL EXPERIENCE (detailed, impressive format)
   For each role:
   - [Title] | [Company] | [Location] | [Duration]
   - Brief context: what the role entailed
   - 4-5 achievement bullets with metrics
   - Each bullet: Strong verb + specific action + quantified impact
   - Show progression, growth, and leadership
   - Demonstrate scale, complexity, results
 
4. EDUCATION & CREDENTIALS
   - Degree | Institution | Graduation Year
   - GPA/Honors if relevant (3.7+)
   - Relevant coursework or specializations
   - List certifications with issuing body and year
 
5. LANGUAGES & AWARDS (if applicable)
   - Language proficiency levels
   - Awards, recognitions, distinctions
 
WRITING REQUIREMENTS:
- Professional, sophisticated tone
- Active voice, strong action verbs
- Specific examples and metrics throughout
- Show impact: revenue, scale, team size, efficiency gains
- Demonstrate continuous growth and learning
- 1.5-2 pages for entry/mid-level
- 2-3 pages for senior/executive
- ATS-optimized but sophisticated
 
TONE FOR {level}:
- Entry-level: Eager, detail-oriented, growth-focused, ready to contribute
- Mid-level: Confident, results-driven, strategic-thinking, proven track record
- Senior: Visionary, transformational, mentorship-focused, strategic leader
- Executive: Strategic, innovative, market-aware, P&L focused, board-ready
 
QUALITY REQUIREMENTS:
✓ Read like a polished professional document
✓ Compelling narrative of career progression
✓ Specific achievements with numbers/percentages
✓ Professional formatting with clear sections
✓ Authentic voice - no clichés or generic phrases
✓ Demonstrates pattern of excellence and growth
✓ Ready for immediate submission
 
OUTPUT RULES:
- Create a complete, polished CV (not a skeleton)
- Use clear formatting with section breaks (---)
- Make each section substantial and impressive
- Include all provided information effectively
- NO placeholder brackets or incomplete sections
- NO generic language or clichés
- NO preamble or explanation - ONLY CV TEXT
 
START WRITING THE CV NOW:"""
 
 
def build_targeted_cv_prompt_enhanced(data: Dict[str, Any], job_title: str, job_description: str) -> str:
    """Enhanced targeted CV prompt for job-specific positioning"""
    
    name = data.get("name", "Professional")
    
    # Build brief profile
    experience_list = data.get("experience", [])
    exp_summary = ""
    for exp in experience_list[:3]:
        if isinstance(exp, dict):
            title = exp.get("title", "")
            company = exp.get("company", "")
            bullets = exp.get("bullets", [])
        else:
            title = getattr(exp, "title", "")
            company = getattr(exp, "company", "")
            bullets = getattr(exp, "bullets", [])
        
        if title and company:
            first_bullet = bullets[0] if bullets else ""
            exp_summary += f"- {title} at {company}: {first_bullet}\n"
    
    skills = data.get("skills", [])
    skills_text = ", ".join(skills[:10])
    
    return f"""Create a professional CV specifically for: {job_title}
 
TARGET POSITION: {job_title}
 
JOB REQUIREMENTS & CONTEXT:
{job_description[:800]}
 
CANDIDATE:
Name: {name}
Background: {exp_summary}
Skills: {skills_text}
 
TASK: Create a substantial, impressive CV tailored to this specific role.
 
KEY FOCUS:
1. Open with compelling summary addressing job requirements
2. Emphasize most relevant experience and achievements
3. Highlight skills matching job description naturally
4. Reorder achievements to show best fit
5. Use job keywords authentically throughout
6. Demonstrate clear fit without overstatement
 
STRUCTURE:
1. Professional Summary (3-4 sentences)
   - Address key job requirements
   - Highlight most relevant qualifications
   - Show enthusiasm for this type of role
   - Position as ideal candidate
 
2. Key Competencies (6-8 skills)
   - Prioritize skills from job description
   - Include both listed and implied requirements
   - Show breadth and depth
 
3. Professional Experience (substantial bullets)
   - Lead with most relevant role
   - 4-5 achievement bullets per role
   - Emphasize achievements matching job needs
   - Show scale and impact
 
4. Education & Certifications
 
TONE: Professional, confident, enthusiastic for this specific opportunity
 
QUALITY:
- Real professional document (substantial, not skeleton)
- Compelling narrative showing fit
- Specific metrics and achievements
- Ready to submit to hiring manager
 
OUTPUT: Complete, polished CV text only (no preamble)"""
 
 
def build_cover_letter_enhanced_prompt(
    name: str,
    job_title: str,
    company_name: str,
    job_description: str,
    cv_content: str
) -> str:
    """Enhanced cover letter prompt for real, compelling letter"""
    
    return f"""Write a professional, compelling cover letter.
 
POSITION: {job_title} at {company_name}
 
KEY REQUIREMENTS:
{job_description[:500]}
 
CANDIDATE: {name}
 
CV HIGHLIGHTS (reference for coherence):
{cv_content[:400]}...
 
TASK: Write a 4-paragraph professional cover letter that would impress a hiring manager.
 
PARAGRAPH 1 (Opening - 3-4 sentences):
- State the position you're applying for
- Express specific interest in this role and company
- Briefly mention your strongest relevant qualification
- Make it personal and specific (not templated)
 
PARAGRAPH 2 (Value Proposition - 4-5 sentences):
- Your most relevant achievement from CV
- How it directly applies to job requirements
- What you accomplished and the impact
- Why you're qualified for this specific role
 
PARAGRAPH 3 (Additional Strengths - 4-5 sentences):
- Second major achievement or strength
- How it matches additional job requirements
- Your understanding of company/industry needs
- What you'll bring to the team
 
PARAGRAPH 4 (Closing - 2-3 sentences):
- Enthusiasm and specific call to action
- When you're available to discuss
- How to reach you (or reference to CV)
- Professional sign-off
 
REQUIREMENTS:
✓ Substantial, real letter (200-300 words)
✓ Specific to this job and company (not generic)
✓ Professional yet personable tone
✓ Confident without arrogance
✓ Shows research and genuine interest
✓ Flows naturally, reads well
✓ References specific qualifications
 
OUTPUT: Complete cover letter text only (no salutation/signature/placeholders)"""
 
 
# =====================================================
# PRODUCTION-GRADE CV FUNCTIONS
# =====================================================
 
async def generate_professional_cv_production(data: Dict[str, Any], db: AsyncSession) -> dict:
    """
    Generate comprehensive professional CV with enhanced output.
    Handles both JSON payload and parsed resume data from files.
    """
    
    name = data.get("name", "Professional")
    experience = data.get("experience", [])
    
    # Determine experience level
    exp_count = len(experience) if isinstance(experience, list) else 0
    if exp_count >= 10:
        level = "Executive"
    elif exp_count >= 5:
        level = "Senior"
    elif exp_count >= 2:
        level = "Mid-level"
    else:
        level = "Entry-level"
    
    try:
        logger.info(f"Generating CV ({level})")
        
        # Build enhanced prompt
        prompt = build_professional_cv_prompt(data, level)
        
        # Call LLM with enhanced prompt
        response = await _call_llm_with_telemetry(
            user_message=prompt,
            agent_name="resume_builder",
            db=db,
            operation="generate_professional_cv",
        )
        
        if not response or len(response.strip()) < 200:
            logger.warning(f"Short CV response ({level}): {len(response)} chars")
            raise HTTPException(500, "CV generation produced insufficient content")
        
        logger.info(f"CV generated successfully ({level}) ({len(response)} chars)")
        
        return {
            "status": "success",
            "data": {
                "cv_content": response.strip(),
                "candidate_name": name,
                "experience_level": level,
                "page_estimate": _estimate_cv_pages(response),
                "sections_included": _analyze_cv_sections(response),
                "word_count": len(response.split()),
                "quality_notes": "Comprehensive, professional CV crafted for immediate submission"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CV generation error ({level}): {str(e)}", exc_info=True)
        raise HTTPException(500, f"CV generation failed: {str(e)}")
 
 
async def generate_targeted_cv_production(
    data: Dict[str, Any],
    job_title: str,
    job_description: str,
    db: AsyncSession
) -> dict:
    """Generate impressive CV tailored to specific job"""
    
    name = data.get("name", "Professional")
    
    try:
        logger.info(f"Generating targeted CV for {job_title}")
        
        prompt = build_targeted_cv_prompt_enhanced(data, job_title, job_description)
        
        response = await _call_llm_with_telemetry(
            user_message=prompt,
            agent_name="resume_builder",
            db=db,
            operation="generate_targeted_cv",
        )
        
        if not response or len(response.strip()) < 200:
            raise HTTPException(500, "Targeted CV generation produced insufficient content")
        
        logger.info(f"Targeted CV generated for {job_title}")
        
        return {
            "status": "success",
            "data": {
                "cv_content": response.strip(),
                "targeted_for_role": job_title,
                "candidate_name": name,
                "word_count": len(response.split()),
                "quality_notes": "Professionally tailored CV for specific opportunity"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Targeted CV generation error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Targeted CV generation failed: {str(e)}")
 
 
async def generate_cv_and_cover_letter_production(
    data: Dict[str, Any],
    job_title: str,
    company_name: str,
    job_description: str,
    db: AsyncSession
) -> dict:
    """Generate coordinated CV and cover letter package"""
    
    name = data.get("name", "Professional")
    
    try:
        logger.info(f"Generating application package for {job_title} at {company_name}")
        
        # Generate targeted CV
        cv_result = await generate_targeted_cv_production(
            data,
            job_title,
            job_description,
            db
        )
        
        cv_content = cv_result['data']['cv_content']
        
        # Generate coordinated cover letter
        cover_prompt = build_cover_letter_enhanced_prompt(
            name,
            job_title,
            company_name,
            job_description,
            cv_content
        )
        
        cover_letter = await _call_llm_with_telemetry(
            user_message=cover_prompt,
            agent_name="resume_builder",
            db=db,
            operation="generate_cover_letter",
        )
        
        if not cover_letter or len(cover_letter.strip()) < 150:
            raise HTTPException(500, "Cover letter generation produced insufficient content")
        
        logger.info(f"Application package generated for {company_name}")
        
        return {
            "status": "success",
            "data": {
                "cv_content": cv_content,
                "cover_letter": cover_letter.strip(),
                "candidate_name": name,
                "job_position": job_title,
                "company": company_name,
                "cv_word_count": len(cv_content.split()),
                "letter_word_count": len(cover_letter.split()),
                "quality_notes": "Professional application package ready for submission",
                "application_ready": True
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Application package generation error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Application package generation failed: {str(e)}")
 
 
async def generate_cv_from_parsed_resume(
    parsed_resume: Any,
    db: AsyncSession
) -> dict:
    """
    Generate CV from parsed resume (PDF/DOCX extraction).
    Converts parsed schema to resume data format.
    """
    
    try:
        # Convert parsed resume schema to dict format
        resume_data = _convert_parsed_schema_to_dict(parsed_resume)
        
        logger.info("Generating CV from parsed resume")
        
        # Generate CV using standard function
        return await generate_professional_cv_production(resume_data, db)
    
    except Exception as e:
        logger.error(f"CV generation from parsed resume failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"CV generation from parsed resume failed: {str(e)}")
 
 
# =====================================================
# HELPER FUNCTIONS
# =====================================================
 
def _convert_parsed_schema_to_dict(parsed_resume: Any) -> dict:
    """
    Convert parsed resume schema (from PDF/DOCX) to dict format for CV generation.
    Handles ATSScanRequest or similar schema objects.
    """
    
    try:
        # Try direct dict conversion first
        if hasattr(parsed_resume, 'dict'):
            data = parsed_resume.dict()
        elif hasattr(parsed_resume, '__dict__'):
            data = parsed_resume.__dict__
        else:
            data = dict(parsed_resume)
        
        # Ensure all required fields exist
        resume_dict = {
            "name": data.get("name", "Professional"),
            "email": data.get("email", ""),
            "phone": data.get("phone", ""),
            "location": data.get("location", ""),
            "summary": data.get("summary", ""),
            "skills": data.get("skills", []),
            "certifications": data.get("certifications", []),
            "languages": data.get("languages", []),
            "awards": data.get("awards", []),
            "projects": data.get("projects", []),
            "publications": data.get("publications", []),
        }
        
        # Handle experience (convert objects to dicts if needed)
        experience_list = data.get("experience", [])
        resume_dict["experience"] = []
        for exp in experience_list:
            exp_dict = {}
            if isinstance(exp, dict):
                exp_dict = exp
            elif hasattr(exp, '__dict__'):
                exp_dict = exp.__dict__
            else:
                exp_dict = dict(exp)
            
            # Ensure required fields
            exp_dict = {
                "title": exp_dict.get("title", ""),
                "company": exp_dict.get("company", ""),
                "duration": exp_dict.get("duration", ""),
                "location": exp_dict.get("location", ""),
                "bullets": exp_dict.get("bullets", [])
            }
            resume_dict["experience"].append(exp_dict)
        
        # Handle education
        education_list = data.get("education", [])
        resume_dict["education"] = []
        for edu in education_list:
            edu_dict = {}
            if isinstance(edu, dict):
                edu_dict = edu
            elif hasattr(edu, '__dict__'):
                edu_dict = edu.__dict__
            else:
                edu_dict = dict(edu)
            
            edu_dict = {
                "degree": edu_dict.get("degree", ""),
                "college": edu_dict.get("college", ""),
                "year": edu_dict.get("year", ""),
                "location": edu_dict.get("location", "")
            }
            resume_dict["education"].append(edu_dict)
        
        logger.info(f"Converted parsed schema to dict (exp: {len(resume_dict['experience'])}, edu: {len(resume_dict['education'])})")
        return resume_dict
    
    except Exception as e:
        logger.error(f"Schema conversion error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to convert parsed resume: {str(e)}")
 
 
def _estimate_cv_pages(cv_content: str) -> str:
    """Estimate CV page count from word count"""
    words = len(cv_content.split())
    
    if words < 300:
        return "~0.5 page"
    elif words < 500:
        return "~1 page"
    elif words < 900:
        return "~1-2 pages"
    elif words < 1400:
        return "~2 pages"
    else:
        return "~2-3 pages"
 
 
def _analyze_cv_sections(cv_content: str) -> list:
    """Identify sections present in CV"""
    sections = []
    keywords = {
        "professional summary": "Professional Summary",
        "competencies": "Competencies",
        "experience": "Professional Experience",
        "education": "Education",
        "certifications": "Certifications",
        "languages": "Languages",
        "awards": "Awards",
        "achievements": "Key Achievements"
    }
    
    content_lower = cv_content.lower()
    for keyword, section_name in keywords.items():
        if keyword in content_lower:
            sections.append(section_name)
    
    return sections if sections else ["Professional Summary", "Professional Experience"]


