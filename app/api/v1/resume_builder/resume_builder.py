# /home/aryu_user/Arun/aiproject_staging/app/api/v1/resume_builder.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.database import get_db
from app.services.resume_builder_services.resume_services import (
    suggest_experience,
    suggest_summary,
    build_skills_prompt,
    suggest_education,
    generate_ats_resume_json,
    refine_resume_section,
)
from app.utils.ats_scanner.text_extraction import extract_text
from app.services.resume_builder_services.resume_parser_service import parse_resume_to_schema
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)
router = APIRouter()


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
    
@router.post("/parse")
async def parse_resume(file: UploadFile = File(...)):

    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Only PDF and DOCX supported")

    try:

        text = await extract_text(file)

        resume_json = parse_resume_to_schema(
            text=text,
            file_type=file.filename.split(".")[-1]
        )

        return {
            "status": "success",
            "resume": resume_json
        }

    except Exception as e:
        raise HTTPException(500, f"Resume parsing failed: {str(e)}")