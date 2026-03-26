# /home/aryu_user/Arun/aiproject_staging/app/modules/resume_builder/service.py
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.llm_client import call_llm
from app.services.prompt_service import get_prompt
import os
from fastapi import HTTPException
from typing import Dict, Any
import json
import traceback
from .extractor import extract_text
from .parser import parse_resume
from .schemas import ResumeResponse
import logging


logger = logging.getLogger(__name__)

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
        logger.info(f"Generating CV for {name} ({level})")
        
        # Build enhanced prompt
        prompt = build_professional_cv_prompt(data, level)
        
        # Call LLM with enhanced prompt
        response = await call_llm(
            user_message=prompt,
            agent_name="resume_builder",
            db=db,
        )
        
        if not response or len(response.strip()) < 200:
            logger.warning(f"Short CV response for {name}: {len(response)} chars")
            raise HTTPException(500, "CV generation produced insufficient content")
        
        logger.info(f"CV generated successfully for {name} ({len(response)} chars)")
        
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
        logger.error(f"CV generation error for {name}: {str(e)}", exc_info=True)
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
        
        response = await call_llm(
            user_message=prompt,
            agent_name="resume_builder",
            db=db,
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
        
        cover_letter = await call_llm(
            user_message=cover_prompt,
            agent_name="resume_builder",
            db=db,
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
        
        logger.info(f"Generating CV from parsed resume for {resume_data.get('name')}")
        
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
        
        logger.info(f"Converted parsed schema to dict: {resume_dict.get('name')}")
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

 
PARSER_VERSION = "2.0.0"
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB
_ALLOWED_EXT = {".pdf", ".docx", ".doc"}

async def parse_resume_service(file: UploadFile) -> ResumeResponse:
    filename = file.filename or "upload"
    suffix = os.path.splitext(filename)[1].lower()

    # Validate
    if suffix not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(sorted(_ALLOWED_EXT))}"
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > _MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large (max {_MAX_BYTES:,} bytes).")

    # Extract text
    try:
        clean_text, ext_meta = extract_text(raw, filename)
    except Exception as e:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=422, detail=f"Extraction failed: {str(e)}")

    if not clean_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    # Parse
    try:
        parsed = parse_resume(clean_text)
    except Exception as e:
        logger.exception("Parsing failed")
        raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")

    # Build meta
    meta = {
        **ext_meta,
        "parser_version": PARSER_VERSION,
        "char_count": len(clean_text),
        "sections_detected": list(parsed.get("raw_sections", {}).keys()),
    }

    return ResumeResponse(
        personal_info=parsed["personal_info"],
        summary=parsed["summary"],
        professional_experience=parsed["professional_experience"],
        education=parsed["education"],
        technical_stack=parsed["technical_stack"],
        projects=parsed["projects"],
        languages=parsed["languages"],
        certifications=parsed["certifications"],
        custom_sections=parsed["custom_sections"],
        raw_sections=parsed["raw_sections"],
        meta=meta
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _get_ext(filename: str) -> str:
    parts = filename.rsplit(".", 1)
    return f".{parts[1].lower()}" if len(parts) == 2 else ""

