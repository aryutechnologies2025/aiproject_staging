from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Form
from sqlalchemy.ext.asyncio import AsyncSession
import os
import tempfile
from pathlib import Path
from fastapi.responses import JSONResponse
import logging
from app.modules.resume_builder.resume_parser_helper import parsing_resume
from app.core.database import get_db
from pydantic import BaseModel
from app.modules.resume_builder.service import (
    suggest_experience,
    suggest_summary,
    build_skills_prompt,
    suggest_education,
    generate_ats_resume_json,
    refine_resume_section,
    generate_cv_and_cover_letter_production,
    generate_cv_from_parsed_resume,
    generate_professional_cv_production,
    generate_targeted_cv_production
)
from app.modules.resume_builder.service import process_resume
from app.modules.resume_builder.linkedin.schemas import (
    ExtractionRequest,
    ExtractionStatus,
    LinkedInResponse,
)
from app.modules.resume_builder.linkedin.service import linkedin_service
from typing import Any, Dict
from app.modules.resume_builder.resume_parser_helper import extract_text_from_docx, extract_text_from_pdf
from app.modules.ats_scanner.utils.text_extraction import extract_text
from app.modules.resume_builder.resume_parser_service import parse_resume_to_schema, split_into_sections
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)
router = APIRouter()


# =====================================================
# EXISTING ENDPOINTS (Keep as-is)
# =====================================================

@router.post("/experience")
async def generate_experience(
    data: dict,
    db: AsyncSession = Depends(get_db),
):
    """Generate 15-20 impact-focused experience bullets with metrics"""
    try:
        if not data.get("job_title") or not data.get("company"):
            raise HTTPException(status_code=400, detail="job_title and company are required")
        
        logger.info(f"Generating experience bullets for {data.get('job_title')}")
        return await suggest_experience(data, db)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Experience generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate experience bullets")


@router.post("/summary")
async def generate_summary(
    data: dict,
    db: AsyncSession = Depends(get_db),
):
    """Generate 2-4 line professional summary with value proposition"""
    try:
        if not data.get("job_title"):
            raise HTTPException(status_code=400, detail="job_title is required")
        
        logger.info(f"Generating summary for {data.get('job_title')}")
        return await suggest_summary(data, db)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate summary")


@router.post("/skills")
async def generate_skills(
    data: dict,
    db: AsyncSession = Depends(get_db),
):
    """Generate 5-8 hard technical skills for target roles"""
    try:
        job_titles = data.get("job_titles", [])
        career_level = data.get("career_level", "experienced")

        if not job_titles:
            raise HTTPException(status_code=400, detail="job_titles list is required")
        
        if not isinstance(job_titles, list):
            raise HTTPException(status_code=400, detail="job_titles must be a list")

        logger.info(f"Generating skills for {len(job_titles)} roles")
        
        user_prompt = build_skills_prompt(
            job_titles=job_titles,
            career_level=career_level
        )

        response = await call_llm(
            user_message=user_prompt,
            agent_name="resume_builder",
            db=db,
        )

        skills = [line.strip() for line in response.splitlines() if line.strip()]
        
        if not skills:
            raise HTTPException(status_code=500, detail="Failed to generate skills")
        
        logger.info(f"Generated {len(skills)} skills")
        
        return {
            "skills": skills,
            "count": len(skills),
            "quality_notes": "Skills prioritized by market demand and role relevance"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Skills generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate skills")


@router.post("/education")
async def generate_education(
    data: dict,
    db: AsyncSession = Depends(get_db),
):
    """Generate education section with achievement-focused bullets"""
    try:
        logger.info("Generating education bullets")
        return await suggest_education(data, db)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Education generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate education bullets")


@router.post("/generate-resume")
async def generate_resume_json(
    data: dict,
    db: AsyncSession = Depends(get_db),
):
    """Generate complete ATS-optimized resume for specific job posting"""
    try:
        job_title = data.get("job_title")
        job_description = data.get("job_description")
        
        if not job_title:
            raise HTTPException(status_code=400, detail="job_title is required")
        if not job_description:
            raise HTTPException(status_code=400, detail="job_description is required")
        
        logger.info(f"Generating ATS-optimized resume for {job_title}")
        result = await generate_ats_resume_json(data, db)
        
        logger.info(f"Resume generated successfully for {job_title}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Resume generation failed")


@router.post("/refine")
async def refine_resume(
    payload: dict,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Refine resume section based on user feedback"""
    try:
        section = payload.get("section")
        existing = payload.get("existing_content")
        instruction = payload.get("instruction")
        
        if not section:
            raise HTTPException(status_code=400, detail="section is required")
        if not existing:
            raise HTTPException(status_code=400, detail="existing_content is required")
        if not instruction:
            raise HTTPException(status_code=400, detail="instruction is required")
        
        logger.info(f"Refining {section} section")
        
        return await refine_resume_section(
            section_name=section,
            existing_content=existing,
            user_instruction=instruction,
            experience_level=payload.get("experience_level", "experienced"),
            db=db,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refinement error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to refine section")




@router.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    return await process_resume(file)

# =====================================================
# NEW CV GENERATION ENDPOINTS - PRODUCTION GRADE
# =====================================================

@router.post("/cv/generate")
async def generate_cv(
    payload: dict,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate professional CV from JSON resume data.
    
    Input:
    {
        "resume_data": {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1234567890",
            "location": "New York, NY",
            "summary": "...",
            "experience": [...],
            "education": [...],
            "skills": [...]
        }
    }
    """
    try:
        resume_data = payload.get("resume_data")
        
        if not resume_data:
            raise HTTPException(400, "resume_data is required")
        
        if not isinstance(resume_data, dict):
            raise HTTPException(400, "resume_data must be a dictionary")
        
        logger.info(f"CV generation request for {resume_data.get('name', 'Unknown')}")
        
        result = await generate_professional_cv_production(resume_data, db)
        
        logger.info("CV generated successfully via JSON payload")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CV generation error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"CV generation failed: {str(e)}")
 
 
@router.post("/cv/generate-from-file")
async def generate_cv_from_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):

    filename = file.filename.lower()
    
    try:
        # Validate file type
        if not filename.endswith((".pdf", ".docx")):
            raise HTTPException(400, "Only PDF and DOCX files are supported")
        
        logger.info(f"Processing file upload: {file.filename}")
        
        # Step 1: Read file ONCE + create temp file + extract text
        try:
            file_type = filename.split(".")[-1]

            content = await file.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
                tmp.write(content)
                temp_path = tmp.name

            # Extract text using your helper
            text = extract_text_from_pdf(temp_path) if file_type == "pdf" else extract_text_from_docx(temp_path)

            if not text or len(text.strip()) < 50:
                raise HTTPException(400, "Unable to extract meaningful text from file.")

            logger.info(f"Text extracted from {file.filename} ({len(text)} characters)")

        except Exception as e:
            logger.error(f"Text extraction error: {str(e)}", exc_info=True)
            raise HTTPException(400, f"Failed to extract text: {str(e)}")
        
        # Step 2: Parse resume to schema
        try:
            file_type = filename.split(".")[-1]
            # Extract structured sections first
            parsed_data = parsing_resume(temp_path, f".{file_type}")

            sections_dict = {}

            # Convert list → dict
            for sec in parsed_data["sections"]:
                key = sec["heading"].lower()
                sections_dict[key] = sec["content"]

            parsed_resume = parse_resume_to_schema(text, file_type, sections_dict)
            
            logger.info(f"Resume parsed for {parsed_resume.name}")
        
        except Exception as e:
            logger.error(f"Resume parsing error: {str(e)}", exc_info=True)
            raise HTTPException(400, f"Failed to parse resume: {str(e)}")
        
        # Step 3: Generate CV from parsed resume
        try:
            result = await generate_cv_from_parsed_resume(parsed_resume, db)
            
            logger.info(f"CV generated from file upload: {file.filename}")
            
            return {
                **result,
                "source": "file_upload",
                "source_file": file.filename
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"CV generation from file error: {str(e)}", exc_info=True)
            raise HTTPException(500, f"CV generation failed: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload processing error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"File processing failed: {str(e)}")
 
 
@router.post("/cv/targeted")
async def generate_targeted_cv(
    payload: dict,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate CV specifically tailored to a job posting.
    
    Input:
    {
        "resume_data": {...},
        "job_title": "Senior Backend Developer",
        "job_description": "..."
    }
    """
    try:
        resume_data = payload.get("resume_data")
        job_title = payload.get("job_title")
        job_description = payload.get("job_description")
        
        if not resume_data:
            raise HTTPException(400, "resume_data is required")
        if not job_title:
            raise HTTPException(400, "job_title is required")
        if not job_description:
            raise HTTPException(400, "job_description is required")
        
        logger.info(f"Targeted CV generation for {job_title}")
        
        result = await generate_targeted_cv_production(
            resume_data,
            job_title,
            job_description,
            db
        )
        
        logger.info(f"Targeted CV generated for {job_title}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Targeted CV generation error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Targeted CV generation failed: {str(e)}")
 
 
@router.post("/cv/with-cover-letter")
async def generate_cv_and_cover_letter(
    payload: dict,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate coordinated CV and cover letter package.
    
    Input:
    {
        "resume_data": {...},
        "job_title": "Engineering Manager",
        "company_name": "TechCorp",
        "job_description": "..."
    }
    """
    try:
        resume_data = payload.get("resume_data")
        job_title = payload.get("job_title")
        company_name = payload.get("company_name")
        job_description = payload.get("job_description")
        
        if not resume_data:
            raise HTTPException(400, "resume_data is required")
        if not job_title:
            raise HTTPException(400, "job_title is required")
        if not company_name:
            raise HTTPException(400, "company_name is required")
        if not job_description:
            raise HTTPException(400, "job_description is required")
        
        logger.info(f"Application package generation for {job_title} at {company_name}")
        
        result = await generate_cv_and_cover_letter_production(
            resume_data,
            job_title,
            company_name,
            job_description,
            db
        )
        
        logger.info(f"Application package generated successfully")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Application package generation error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Application package generation failed: {str(e)}")
 
 
@router.post("/cv/from-file-targeted")
async def generate_targeted_cv_from_file(
    file: UploadFile = File(...),
    job_title: str = None,
    job_description: str = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Upload resume file + generate tailored CV for job posting.
    
    Query params:
    - job_title: Target job title
    - job_description: Job posting details
    """
    filename = file.filename.lower()
    
    try:
        # Validate inputs
        if not filename.endswith((".pdf", ".docx")):
            raise HTTPException(400, "Only PDF and DOCX files are supported")
        
        if not job_title or not job_description:
            raise HTTPException(400, "job_title and job_description are required as query parameters")
        
        logger.info(f"Processing file for targeted CV: {file.filename} → {job_title}")
        
        # Extract and parse
        try:
            text = await extract_text(file)
            
            if not text or len(text.strip()) < 50:
                raise HTTPException(400, "Unable to extract meaningful text from file")
            
            file_type = filename.split(".")[-1]
            parsed_resume = parse_resume_to_schema(text, file_type)
            
            logger.info(f"Resume parsed from file")
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File processing error: {str(e)}", exc_info=True)
            raise HTTPException(400, f"Failed to process file: {str(e)}")
        
        # Convert to dict and generate targeted CV
        try:
            from app.modules.resume_builder.service import _convert_parsed_schema_to_dict
            
            resume_data = _convert_parsed_schema_to_dict(parsed_resume)
            
            result = await generate_targeted_cv_production(
                resume_data,
                job_title,
                job_description,
                db
            )
            
            logger.info(f"Targeted CV generated from file")
            
            return {
                **result,
                "source": "file_upload_targeted",
                "source_file": file.filename,
                "targeted_job": job_title
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Targeted CV generation error: {str(e)}", exc_info=True)
            raise HTTPException(500, f"Targeted CV generation failed: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File-based targeted CV error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Processing failed: {str(e)}")
 
 
# =====================================================
# UTILITY ENDPOINTS
# =====================================================
 
@router.get("/cv/help")
async def cv_help():
    """Get detailed help on CV generation endpoints"""
    return {
        "service": "Production-Grade CV Generation",
        "version": "2.0",
        "endpoints": {
            "/cv/generate": {
                "method": "POST",
                "description": "Generate professional CV from JSON resume data",
                "input": {
                    "resume_data": {
                        "name": "string",
                        "email": "string",
                        "phone": "string",
                        "location": "string",
                        "summary": "string",
                        "experience": [{"title": "string", "company": "string", "bullets": ["string"]}],
                        "education": [{"degree": "string", "college": "string", "year": "string"}],
                        "skills": ["string"]
                    }
                },
                "output": "Comprehensive professional CV",
                "quality": "Letter-quality, substantial content"
            },
            "/cv/generate-from-file": {
                "method": "POST",
                "description": "Upload PDF/DOCX resume and generate CV automatically",
                "input": "multipart/form-data (file upload)",
                "supported_formats": ["PDF", "DOCX"],
                "output": "Professional CV with parsed resume data",
                "quality": "Letter-quality, complete document"
            },
            "/cv/targeted": {
                "method": "POST",
                "description": "Generate CV tailored to specific job posting",
                "input": {
                    "resume_data": "dict",
                    "job_title": "string",
                    "job_description": "string"
                },
                "output": "Job-specific CV emphasizing relevant experience",
                "quality": "Professionally tailored, impressive positioning"
            },
            "/cv/with-cover-letter": {
                "method": "POST",
                "description": "Generate coordinated CV + cover letter package",
                "input": {
                    "resume_data": "dict",
                    "job_title": "string",
                    "company_name": "string",
                    "job_description": "string"
                },
                "output": "Both CV and cover letter ready to submit",
                "quality": "Cohesive, professional application package"
            },
            "/cv/from-file-targeted": {
                "method": "POST",
                "description": "Upload file + generate tailored CV in one call",
                "input": "file upload + query params (job_title, job_description)",
                "output": "Tailored CV from uploaded resume",
                "quality": "Job-specific, substantial content"
            }
        },
        "features": {
            "quality": "Letter-quality, professional documents",
            "substantial": "Comprehensive content (not skeleton CVs)",
            "authentic": "Real professional writing (no clichés)",
            "file_support": "PDF and DOCX uploads with automatic parsing",
            "optimization": "Token-optimized for Llama 3.1 8B",
            "error_handling": "Comprehensive error messages and logging"
        },
        "output_characteristics": {
            "cv_length": "1.5-3 pages depending on experience level",
            "word_count": "800-2500 words for substantial content",
            "structure": "Professional sections with clear formatting",
            "tone": "Adapts to experience level (entry/mid/senior/executive)",
            "content": "Specific metrics, achievements, and professional narrative"
        }
    }
 
 
@router.post("/cv/parse-only")
async def parse_resume_only(
    file: UploadFile = File(...),
):
    """
    Parse resume file without generating CV.
    Useful for debugging or getting parsed data structure.
    """
    filename = file.filename.lower()
    
    try:
        if not filename.endswith((".pdf", ".docx")):
            raise HTTPException(400, "Only PDF and DOCX files are supported")
        
        logger.info(f"Parsing resume file: {file.filename}")
        
        text = await extract_text(file)
        
        if not text or len(text.strip()) < 50:
            raise HTTPException(400, "Unable to extract text from file")
        
        file_type = filename.split(".")[-1]
        parsed_resume = parse_resume_to_schema(text, file_type)
        
        logger.info(f"Resume parsed successfully")
        
        return {
            "status": "success",
            "source_file": file.filename,
            "parsed_data": parsed_resume.dict() if hasattr(parsed_resume, 'dict') else dict(parsed_resume),
            "extraction_stats": {
                "text_length": len(text),
                "name_detected": bool(parsed_resume.name),
                "email_detected": bool(parsed_resume.email),
                "phone_detected": bool(parsed_resume.phone),
                "skills_found": len(parsed_resume.skills),
                "experiences_found": len(parsed_resume.experience),
                "educations_found": len(parsed_resume.education)
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parse-only error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Parsing failed: {str(e)}")
    

class LoginStartResponse(BaseModel):
    message: str
    success: bool
    session_active: bool
 
 
class SessionStatusResponse(BaseModel):
    session_active: bool
    message: str
 
 
class CacheInvalidateRequest(BaseModel):
    linkedin_url: str
 
 
class LogoutResponse(BaseModel):
    message: str
 
 
# ─────────────────────────── Endpoints ───────────────────────────────────────
 
@router.get(
    "/session",
    response_model=SessionStatusResponse,
    summary="Check if LinkedIn session is active",
    description=(
        "Returns whether a saved LinkedIn session exists. "
        "If false, the frontend should show the consent + login flow."
    ),
)
async def check_session() -> SessionStatusResponse:
    active = linkedin_service.has_active_session()
    return SessionStatusResponse(
        session_active=active,
        message=(
            "LinkedIn session is active. You can extract profiles."
            if active
            else "No active session. Please connect your LinkedIn account."
        ),
    )
 
 
@router.post(
    "/login",
    response_model=LoginStartResponse,
    summary="Start LinkedIn consent login flow",
    description=(
        "Opens a visible Chrome browser window so the user can log in to LinkedIn. "
        "The window closes automatically once login is detected. "
        "This endpoint blocks until login completes or times out (~2 min)."
    ),
)
async def start_login() -> LoginStartResponse:
    logger.info("[API] Starting LinkedIn login flow")
    try:
        result = await linkedin_service.start_login_flow()
        success = result.get("success", False)
        return LoginStartResponse(
            message=(
                "✅ LinkedIn connected successfully! You can now extract your profile."
                if success
                else "⏱ Login window timed out. Please try again and complete login within 2 minutes."
            ),
            success=success,
            session_active=linkedin_service.has_active_session(),
        )
    except Exception as exc:
        logger.error(f"[API] Login error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login flow failed: {str(exc)}",
        )
 
 
@router.post(
    "/linkedin/import",
    response_model=LinkedInResponse,
    summary="Extract LinkedIn profile",
    description=(
        "Extracts a LinkedIn profile and returns structured JSON for resume generation. "
        "Checks cache first, then scrapes if needed. "
        "Returns LOGIN_NEEDED status if no active session exists."
    ),
)
async def extract_profile(request: ExtractionRequest) -> LinkedInResponse:
    logger.info(f"[API] Extract request for: {request.linkedin_url}")
 
    try:
        response = await linkedin_service.extract_profile(request)
    except Exception as exc:
        logger.error(f"[API] Extraction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(exc)}",
        )
 
    # Return 401 hint if login is needed (but still return body for frontend to handle)
    if response.meta.status == ExtractionStatus.LOGIN_NEEDED:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=response.dict(),
        )
 
    return response
 
 
@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Disconnect LinkedIn session",
    description="Clears saved LinkedIn session cookies. The user will need to log in again for the next extraction.",
)
async def logout() -> LogoutResponse:
    linkedin_service.logout()
    return LogoutResponse(message="LinkedIn session disconnected successfully.")
 
 
@router.post(
    "/cache/invalidate",
    summary="Invalidate cached profile",
    description="Force a fresh scrape on the next extraction request for the given LinkedIn URL.",
)
async def invalidate_cache(body: CacheInvalidateRequest) -> Dict[str, Any]:
    removed = linkedin_service.invalidate_cache(body.linkedin_url)
    return {
        "success": removed,
        "message": (
            "Cache cleared. Next extraction will fetch fresh data."
            if removed
            else "No cached data found for this URL."
        ),
    }
 
 
# ─────────────────────────── Health ──────────────────────────────────────────
 
@router.get(
    "/health",
    summary="Module health check",
    include_in_schema=False,
)
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "module": "linkedin_extraction",
        "session_active": linkedin_service.has_active_session(),
        "cache_stats": linkedin_service.cache.stats(),
    }
