from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services.llm_client import call_llm
from app.services.prompt_service import get_prompt

router = APIRouter()


@router.post("/generate")
async def generate_resume_suggestion(
    data: dict,
    type: str = Query(..., description="exp, summary, skills, edu"),
    db: AsyncSession = Depends(get_db),
):
    system_prompt = await get_prompt(db, "resume_builder")

    # Build minimal prompt based on type → EASY + FAST
    if type == "exp":
        user_msg = f"""
Generate 5-8 resume bullet points.

Job Title: {data.get("job_title")}
Company: {data.get("company")}
Description: {data.get("description")}

Rules:
- Short bullets.
- ATS-friendly.
- No fake details.
"""
    elif type == "summary":
        user_msg = f"""
Write a 5-8 line resume summary.

Skills: {data.get("skills")}
Experience: {data.get("experience")}
"""
    elif type == "skills":
        user_msg = f"""
Suggest 10–15 resume skills based on:

Job Title: {data.get("job_title")}
Existing Skills: {data.get("skills")}
"""
    elif type == "edu":
        user_msg = f"""
Write 5-8 resume bullet points for education.

Degree: {data.get("degree")}
College: {data.get("college")}
"""
    else:
        raise HTTPException(400, "Invalid type. Use exp, summary, skills, edu.")

    # ONE FAST LLM CALL
    response = await call_llm(
        model="llama",
        user_message=user_msg,
        agent_name="resume_builder",
        db=db,
    )

    return {"result": response}
