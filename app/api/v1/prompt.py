from fastapi import APIRouter, Depends, Form, Body, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

from app.services.prompt_service import (
    get_prompt, create_prompt, update_prompt, list_prompts
)

router = APIRouter()


@router.get("/")
async def get_all_prompts(db: AsyncSession = Depends(get_db)):
    prompts = await list_prompts(db)
    return {"success": True, "data": prompts}

@router.post("/")
async def create_new_prompt(
    agent_name: str = Form(None),
    description: str = Form(None),
    system_prompt: str = Form(None),
    body: dict = Body(None),
    db: AsyncSession = Depends(get_db)
):
    # Allow JSON OR Form data
    if body:
        agent_name = body.get("agent_name")
        description = body.get("description")
        system_prompt = body.get("system_prompt")

    if not agent_name or not system_prompt:
        raise HTTPException(400, "agent_name and system_prompt are required.")

    prompt = await create_prompt(db, agent_name, description, system_prompt)
    return {"success": True, "prompt": prompt}


@router.put("/{agent_name}")
async def update_existing_prompt(
    agent_name: str,
    description: str = Form(None),
    system_prompt: str = Form(None),
    body: dict = Body(None),
    db: AsyncSession = Depends(get_db)
):
    if body:
        description = body.get("description")
        system_prompt = body.get("system_prompt")

    if not system_prompt:
        raise HTTPException(400, "system_prompt is required.")

    prompt = await update_prompt(db, agent_name, description, system_prompt)
    return {"success": True, "prompt": prompt}
