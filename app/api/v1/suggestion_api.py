from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.llm_client import call_llm
from pydantic import BaseModel

router = APIRouter()


# -------------------- REQUEST SCHEMA --------------------
class SuggestRequest(BaseModel):
    text: str
    tone: str | None = "neutral"


# -------------------- RESPONSE SCHEMA --------------------
class SuggestResponse(BaseModel):
    success: bool
    output: str


# -------------------- TONE HANDLER --------------------
def build_tone_instruction(tone: str) -> str:
    tone = tone.lower()

    tone_map = {
        "professional": "Write in a professional and formal tone.",
        "friendly": "Write in a friendly and conversational tone.",
        "simple": "Write in simple, easy-to-understand language.",
        "formal": "Write in a polite and formal tone.",
        "casual": "Write in a casual and natural tone.",
    }

    return tone_map.get(tone, "")


# -------------------- MAIN API --------------------
@router.post("/", response_model=SuggestResponse)
async def suggest_text(
    payload: SuggestRequest,
    db: AsyncSession = Depends(get_db),
):
    if not payload.text.strip():
        return SuggestResponse(
            success=False,
            output="Input text is required."
        )

    tone_instruction = build_tone_instruction(payload.tone)

    user_prompt = payload.text
    if tone_instruction:
        user_prompt += f"\n\nInstruction: {tone_instruction}"

    ai_output = await call_llm(
        user_message=user_prompt,
        agent_name="universal",
        db=db,
    )

    return SuggestResponse(
        success=True,
        output=ai_output.strip()
    )
