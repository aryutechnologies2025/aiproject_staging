# suggestion_api.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.llm_client import call_llm

router = APIRouter()


# --------------- Request Schema ---------------
class SuggestRequest(BaseModel):
    text: str
    tone: str | None = "neutral"


# --------------- Response Schema ---------------
class SuggestResponse(BaseModel):
    success: bool
    output: str

def build_tone_instruction(tone: str):
    tone = tone.lower()
    if tone == "professional":
        return "Write the output in a professional and formal tone."
    if tone == "friendly":
        return "Write the output in a friendly, warm, conversational tone."
    if tone == "simple":
        return "Write the output in simple and easy-to-understand language."
    if tone == "formal":
        return "Write the output in a very polite and formal tone."
    if tone == "casual":
        return "Write the output in a casual, natural tone."
    return ""  # neutral


# --------------- MAIN UNIVERSAL SUGGESTION API ---------------
@router.post("/", response_model=SuggestResponse)
async def suggest_text(payload: SuggestRequest, db: AsyncSession = Depends(get_db)):

    if not payload.text.strip():
        return SuggestResponse(success=False, output="Please provide some text.")

    # Build tone-specific instruction
    tone_instruction = build_tone_instruction(payload.tone)

    # Append tone instruction to the user message
    user_message = payload.text + f"\n\nTONE_INSTRUCTION: {tone_instruction}"

    # Call the universal agent with Qwen
    ai_result = await call_llm(
        model="gemma",
        user_message=user_message,
        agent_name="universal",
        db=db
    )

    return SuggestResponse(success=True, output=ai_result)

