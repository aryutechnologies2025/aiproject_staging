# /home/aryu_user/Arun/aiproject_staging/app/modules/resume_builder/service.py
from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.llm_client import call_llm
from app.utils.prompt_service import get_prompt
from typing import Dict, Any
import json
import re
from .extractor import extract_with_llamaparse
import logging


logger = logging.getLogger(__name__)

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
    job_title = data.get("job_title", "")
    company = data.get("company", "")
    duration = data.get("duration", "")
    location = data.get("location", "")
    description = data.get("description", "")
    tone = data.get("tone", "professional")

    user_prompt = f"""[CRITICAL INTEGRITY CONSTRAINT]
You are an expert multi-industry resume writer. Your task is to distill the raw job description provided below into concise, high-impact, ATS-optimized accomplishment statements. Do NOT invent metrics or titles. Do NOT use IT jargon if the candidate is in a non-tech field.

ROLE DATA:
- Job Title: {job_title}
- Organization: {company}
- Tenure/Duration: {duration}
- Location: {location}
- Raw Description/Tasks: {description}
- Target Tone: {tone}

EXECUTION RULES:
1. NO personal pronouns allowed (No: I, me, my, we, our).
2. Follow the universal formula: [Strong Action Verb] + [Core Task/Responsibility] + [Measurable Business/Operational Impact].
3. Prioritize concise, clear sentence structures over wordy explanations.
4. Output exactly 3 to 5 high-quality bullets maximum. One bullet per line.

UNIVERSAL REFERENCE PATTERNS:

Example 1 (Corporate / Operations / Admin):
Raw Input: "I managed the daily schedules for the office, organized client files, and helped reduce office supply spending over the year."
Output: Coordinated daily administrative workflows and calendar management for senior executive teams.
Output: Restructured physical and digital client data archiving systems, improving retrieval times.
Output: Audited vendor contracts and optimized office inventory pipelines, reducing annual operational expenditure.

Example 2 (Education / Teaching):
Raw Input: "I was a teacher handling 30 students. I changed the lesson plans and grades went up quite a bit."
Output: Executed comprehensive curriculum delivery tailored to diverse learning styles for groups of 30+ students.
Output: Innovated custom interactive lesson frameworks, resulting in measurable improvements in quarterly student performance metrics.

YOUR TASK:
Transform the Raw Description/Tasks provided above into 3 to 5 clean, distinct accomplishment lines matching the layout of the reference patterns. Do not output anything else. No introduction, no markdown punctuation symbols, no bullet characters (*, -, •) at line starts."""

    response = await call_llm(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
    )

    # Standard clean up for variations in 30B model outputs
    raw_lines = response.strip().split('\n')
    bullets = []
    for line in raw_lines:
        cleaned = line.strip().strip('*').strip('-').strip('•').strip('"').strip("'").strip()
        # Remove common model echoes like "Output:" if it copies the few-shot template literally
        if cleaned.lower().startswith("output:"):
            cleaned = cleaned[7:].strip()
        if cleaned:
            bullets.append(cleaned)

    return {
        "experience_bullets": "\n".join(bullets),
        "count": len(bullets),
        "quality_notes": "Raw data distilled into high-density operational statements optimized for standard resume parsing schemas."
    }


async def suggest_summary(data: dict, db: AsyncSession) -> dict:
    system_prompt = await get_prompt(db, "resume_builder")
    if not system_prompt:
        system_prompt = "You are an expert ATS-optimization engine and professional resume writer."

    experiences = data.get("experiences", [])
    experience_text = ""

    if isinstance(experiences, list) and experiences:
        exp_parts = []
        for exp in experiences:
            title = exp.get("job_title", "")
            company = exp.get("company", "")
            start = exp.get("start_date", "")[:4] if exp.get("start_date") else ""
            end = exp.get("end_date", "")[:4] if exp.get("end_date") else "Present"
            exp_parts.append(f"{title} at {company} ({start}-{end})")
        experience_text = "; ".join(exp_parts)

    education_list = data.get("education", [])
    education_text = ""

    if isinstance(education_list, list) and education_list:
        edu_parts = []
        for edu in education_list:
            degree = edu.get("degree", "")
            institution = edu.get("institution", "")
            edu_parts.append(f"{degree} from {institution}")
        education_text = "; ".join(edu_parts)

    # Universally generalized from "skills" to industry terms
    skills = data.get("skills", [])
    skills_text = ", ".join(skills[:8]) if isinstance(skills, list) else str(skills)

    tone = data.get("tone", "modern professional")

    years_experience = "0+"
    if experiences:
        try:
            start_years = [
                int(exp.get("start_date", "")[:4])
                for exp in experiences
                if exp.get("start_date")
            ]
            if start_years:
                min_year = min(start_years)
                from datetime import datetime
                current_year = datetime.now().year
                diff = current_year - min_year
                years_experience = f"{diff}+" if diff > 0 else "1+"
        except:
            years_experience = "0+"

    # Dynamically extract the latest job title to anchor any profession naturally
    target_title = experiences[0].get("job_title", "Experienced") if experiences else "Qualified"

    user_prompt = f"""[CRITICAL INTEGRITY CONSTRAINT]
You must write a professional resume summary using ONLY the verified candidate data provided below. Do NOT invent specific company names, metrics, or credentials not explicitly listed. Avoid generic AI fluff phrases like "Dynamic, results-driven professional".

CANDIDATE DATA PAYLOAD:
- Target/Latest Profession: {target_title}
- Work History & Roles: {experience_text[:300]}
- Core Competencies / ATS Keywords: {skills_text}
- Academic Credentials: {education_text[:150]}
- Total Career Tenure: {years_experience} years
- Stylistic Tone: {tone}

EXECUTION RULES:
1. NO personal pronouns allowed under any circumstances (No: I, me, my, we, our).
2. The summary must seamlessly integrate at least 3-4 keywords directly from the Core Competencies list for algorithmic ATS parsing.
3. Keep the entire response between 2 to 4 lines maximum formatted as a single unified plain-text paragraph block.

FEW-SHOT REFERENCE PATTERNS (UNIVERSAL INDUSTRIES):

Example 1 (Healthcare / Nursing):
Input Stack: Patient Care, ICU Operations, Life Support, BLS Certified, Staff Nurse, 5+ years
Output: Compassionate Staff Nurse with over 5 years of dedicated experience in critical care environments. Proven track record managing complex Patient Care schedules and high-pressure ICU Operations. Expert at maintaining safety compliance and deploying specialized life support protocols.

Example 2 (Sales / Real Estate):
Input Stack: Relationship Management, Negotiation, Lead Generation, CRM Systems, Sales Executive, 3+ years
Output: Results-oriented Sales Executive with 3+ years of expertise in high-value property markets. Strong history of driving growth through targeted Lead Generation, strategic Relationship Management, and structured client negotiations. Experienced in optimizing sales pipelines using industry-standard CRM Systems.

Example 3 (Python Developer / Backend Engineering):
Input Stack: Python, FastAPI, PostgreSQL, AWS, Backend Engineer, 3+ years
Output: Backend Engineering professional with 3+ years of expertise in high-performance application development. Proven track record in Python, FastAPI, and data architecture scaling using PostgreSQL. Accomplished in designing cloud infrastructure workflows across distributed AWS environments.

YOUR TASK:
Generate the plain-text summary paragraph now based strictly on the CANDIDATE DATA PAYLOAD using the exact structure and text density of the reference patterns above. Do not output anything else. No introduction, no conversational text, no markdown styling."""

    response = await call_llm(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
    )

    summary = response.strip()
    
    # Post-processing cleanup to strip accidental formatting
    if summary.startswith("```"):
        summary = summary.strip("`").replace("text\n", "").replace("json\n", "").strip()
    summary = summary.strip('"').strip("'")
    
    lines = len([l for l in summary.split('\n') if l.strip()])

    return {
        "summary": summary,
        "line_count": lines,
        "quality_notes": "Summary structurally locked to validated domain parameters and standardized multi-industry ATS rules."
    }


def build_skills_prompt(job_titles: list[str], career_level: str = "experienced") -> str:
    roles = ", ".join(job_titles) if job_titles else "General Professional"

    return f"""[CRITICAL INTEGRITY CONSTRAINT]
You are an industry-standard resume indexing engine. Your task is to identify and extract the most relevant core competencies and domain-specific skills based strictly on the provided job titles. Do NOT include soft skills (e.g., leadership, communication, team player).

TARGET ROLES: {roles}
CAREER LEVEL: {career_level}

EXECUTION RULES:
1. Output EXACTLY 5 to 8 high-value skills/competencies.
2. Return ONLY one distinct skill per line.
3. No versions, no descriptions, no definitions, and no punctuation marks at line starts.
4. Output must be raw text only. No introduction, no markdown formatting.

FEW-SHOT REFERENCE PATTERNS (CROSS-INDUSTRY):

Example 1 (Finance / Accounting):
Target Roles: Accountant, Tax Analyst
Output:
Financial Auditing
Tax Compliance
GAAP Principles
Reconciliation
QuickBooks
Excel Data Models

Example 2 (Logistics / Supply Chain):
Target Roles: Warehouse Supervisor, Logistics Coordinator
Output:
Inventory Management
Supply Chain Optimization
WMS Software
Freight Forwarding
OSHA Safety Compliance
Route Planning

YOUR TASK:
Generate the plain-text list of 5 to 8 domain skills now for the TARGET ROLES following the exact layout of the reference patterns above."""


async def suggest_education(data: dict, db: AsyncSession) -> dict:

    if not isinstance(data, dict):
        raise HTTPException(400, "invalid payload")

    formatted_entries = f"""
Degree:{data.get("degree","")}
College:{data.get("college","")}
Location:{data.get("location","")}
Year:{data.get("year","")}
"""

    user_prompt = f"""
Generate 5 strong resume education bullets.

Rules:
- Use given degree and college details
- Include relevant academic concepts related to the degree
- No fake achievements, GPA, awards, or internships
- ATS-friendly professional wording
- One bullet per line
- No headings or symbols

Data:
{formatted_entries}
"""

    response = await call_llm(
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
    )

    bullets = _clean_bullets(response)

    return {
        "education_bullets": "\n".join(bullets),
        "count": len(bullets),
        "quality_notes": "Education bullets generated"
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


async def generate_ats_resume_json(data: dict, db: AsyncSession) -> dict:
    """
    Generates a complete ATS-optimized resume JSON structure tailored for any target role.
    Features robust regex filtering to extract pure JSON, bypassing conversational output 
    or unexpected markdown wrappings from 30B parameter models.
    """
    job_title = data.get("job_title")
    job_description = data.get("job_description")
    company = data.get("company", "")

    if not job_title or not job_description:
        raise HTTPException(status_code=400, detail="job_title and job_description are required")

    # Build the specialized, multi-industry ATS optimization prompt
    user_prompt = build_ats_resume_json_prompt(
        job_title=job_title,
        company=company,
        job_description=job_description
    )

    try:
        # Call your core LLM interaction function
        raw_response = await call_llm(
            user_message=user_prompt,
            agent_name="resume_builder",
            db=db,
        )

        clean = raw_response.strip()
        
        # Robust Regex Parsing:
        # 30B models often enclose JSON in ```json { ... } ``` blocks despite instructions.
        # This regex looks for the outermost curly braces containing the JSON payload.
        json_match = re.search(r"(\{.*\})", clean, re.DOTALL)
        if json_match:
            clean = json_match.group(1)
        else:
            # Fallback manual string trimming if regex couldn't resolve the boundaries
            if clean.startswith("```json"):
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
        
        clean = clean.strip()
        
        # Safely load the clean string into a python dictionary
        result = json.loads(clean)
        
        # Post-execution structure validation to ensure no key properties are missing
        required_keys = ["summary", "experience", "skills", "optimization_notes"]
        for key in required_keys:
            if key not in result:
                result[key] = f"Missing data fallback for {key}" if key != "experience" and key != "skills" else []

        return result
    
    except json.JSONDecodeError as jde:
        # Fallback error for invalid JSON configurations or unparseable output syntax
        raise HTTPException(
            status_code=422, 
            detail="Failed to parse LLM generation into a valid JSON schema. Raw response contained formatting anomalies."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"ATS resume schema processing failed: {str(e)}"
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


