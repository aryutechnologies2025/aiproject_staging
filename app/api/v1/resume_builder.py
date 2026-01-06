from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.resume_services import (
    suggest_experience,
    suggest_summary,
    build_skills_prompt,
    suggest_education,
    generate_ats_resume_json,
)
from app.services.llm_client import call_llm

router = APIRouter()


@router.post("/experience")
async def generate_experience(
    data: dict,
    db: AsyncSession = Depends(get_db),
):
    return await suggest_experience(data, db)

@router.post("/summary")
async def generate_summary(
    data: dict,
    db: AsyncSession = Depends(get_db),
):
    return await suggest_summary(data, db)

@router.post("/skills")
async def generate_skills(
    data: dict,
    db: AsyncSession = Depends(get_db),
):
    job_titles = data.get("job_titles", [])
    career_level = data.get("career_level", "experienced")

    if not isinstance(job_titles, list):
        raise HTTPException(400, "job_titles must be a list")

    user_prompt = build_skills_prompt(
        job_titles=job_titles,
        career_level=career_level
    )

    response = await call_llm(
        model="groq",
        user_message=user_prompt,
        agent_name="resume_builder",
        db=db,
    )

    skills = [line.strip() for line in response.splitlines() if line.strip()]
    return {"skills": skills}

@router.post("/education")
async def generate_education(
    data: dict,
    db: AsyncSession = Depends(get_db),
):
    return await suggest_education(data, db)

@router.post("/generate-resume")
async def generate_resume_json(
    data: dict,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await generate_ats_resume_json(data, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

