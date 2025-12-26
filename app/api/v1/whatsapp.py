from fastapi import APIRouter, Depends, Request, Query
from app.services.whatsapp_service import process_incoming_message
from dotenv import load_dotenv
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.prompt_service import get_prompt


load_dotenv()

router = APIRouter()

VERIFY_TOKEN = "akzworld"

from fastapi.responses import PlainTextResponse

@router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(content=hub_challenge, status_code=200)

    return PlainTextResponse(content="Unauthorized", status_code=401)

@router.post("/webhook")
async def receive_message(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    await process_incoming_message(data, db)
    return {"status": "received"}
