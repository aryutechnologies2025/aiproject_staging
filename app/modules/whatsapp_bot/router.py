# router.py

import re
from fastapi import APIRouter, Depends
from app.utils.llm_client import call_llm
from app.utils.language_detect import detect_language
from app.api.v1.prompt import router as prompt_router
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

router = APIRouter()

router.include_router(prompt_router)

async def route_message(message: str, user_id: str):
    msg_lower = message.lower()
    lang = detect_language(message)

    # High-priority rules

    # 1. First-time welcome
    if msg_lower in ("hi", "hello", "hey", "hai", "hii"):
        return (
            "👋 Hello! Welcome to Aryu Academy.\n"
            "I’m YURA, your AI assistant. May I know your name?\n"
            "Are you here to speak with Mr. Y or enquire about a course?"
        )

    # 2. Document request → always Qwen
    if re.search(r"(pdf|notes|document|material|file|assignment|syllabus)", msg_lower):
        return await call_llm("qwen", message, user_id)

    # 3. Long or multilingual → Qwen
    if len(message.split()) > 30 or lang != "en":
        return await call_llm("qwen", message, user_id)

    # 4. General conversation → Llama
    return await call_llm("llama", message, user_id)

