from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.yura_chat_service import yura_chat

router = APIRouter(prefix="/api/yura", tags=["YURA Bot"])


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    success: bool
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat_with_yura(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    reply = await yura_chat(
        payload.message,
        payload.session_id,
        db
    )
    return ChatResponse(success=True, reply=reply)
