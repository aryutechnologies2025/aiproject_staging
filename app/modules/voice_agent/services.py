import os
import json
from urllib import response
import httpx
import logging
from datetime import datetime, timezone
from typing import Optional
from dateutil import parser
from app.modules.voice_agent.schemas import LeadAnalysisResult, VapiWebhookPayload
from app.modules.voice_agent.models import Lead, LeadScoreEnum, CallStatusEnum
from app.core.database import AsyncSessionLocal # Ensure this matches your async session maker
from sqlalchemy import select
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
VAPI_API_KEY = os.getenv("VAPI_API_KEY", "")
# Using 1.5-flash for better stability in staging
GEMINI_MODEL = "gemini-2.5-flash-lite" 

async def analyze_transcript_with_gemini(transcript: str) -> Optional[LeadAnalysisResult]:
    """
    Uses Gemini 1.5 Flash via REST to analyze the call transcript.
    """
    if not transcript or not GEMINI_API_KEY:
        logger.warning("Missing transcript or Gemini API Key.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    system_prompt = """
    You are an expert Voice AI Lead Qualification agent for Aryu Academy.
    Analyze the provided call transcript and extract:
    1. lead_score: 'Hot' (ready to enroll), 'Warm' (needs follow-up), or 'Cold' (not interested).
    2. summary: A brief 2-sentence summary of the prospect's needs.
    3. follow_up_date: ISO 8601 UTC datetime if they asked for a callback, else null.
    """

    payload = {
        "contents": [{"parts": [{"text": f"System: {system_prompt}\n\nTranscript:\n{transcript}"}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "lead_score": {"type": "STRING", "enum": ["Hot", "Warm", "Cold"]},
                    "summary": {"type": "STRING"},
                    "follow_up_date": {"type": "STRING", "nullable": True}
                },
                "required": ["lead_score", "summary"]
            }
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=20.0)
            response.raise_for_status()
            data = response.json()
            
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            result_json = json.loads(raw_text)
            
            return LeadAnalysisResult(**result_json)
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return None

async def handle_call_end_logic(payload: VapiWebhookPayload):
    """
    Core business logic for post-call processing.
    Moved from router to service.
    """
    message = payload.message
    transcript = message.transcript
    call_data = message.call or {}
    customer = call_data.get("customer", {})
    phone_number = customer.get("number")

    if not phone_number or not transcript:
        logger.warning("Incomplete payload received.")
        return

    # 1. AI Analysis
    analysis = await analyze_transcript_with_gemini(transcript)
    
    # 2. Database Operation with Fresh Session
    async with AsyncSessionLocal() as session:
        try:
            lead = Lead(
                phone_number=phone_number,
                transcript=transcript,
                status=CallStatusEnum.COMPLETED
            )

            if analysis:
                lead.lead_score = LeadScoreEnum(analysis.lead_score)
                lead.summary = analysis.summary
                
                if analysis.follow_up_date:
                    try:
                        lead.recall_timestamp = parser.parse(analysis.follow_up_date)
                        lead.status = CallStatusEnum.PENDING_RECALL
                    except Exception as e:
                        logger.error(f"Date parse error: {e}")

            session.add(lead)
            await session.commit()
            logger.info(f"Successfully processed lead for {phone_number}")
        except Exception as e:
            logger.error(f"Database error in background task: {e}")
            await session.rollback()

async def trigger_outbound_call(phone_number: str) -> bool:
    """
    Triggers an asynchronous outbound call via Vapi.ai API.
    """
    if not VAPI_API_KEY:
        logger.error("VAPI_API_KEY not configured.")
        return False

    url = "https://api.vapi.ai/call"
    headers = {"Authorization": f"Bearer {VAPI_API_KEY}", "Content-Type": "application/json"}
    
    payload = {
        "assistantId": os.getenv("VAPI_ASSISTANT_ID"),
        "customer": {"number": phone_number}
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            if response.status_code != 200:
                logger.info(f"Vapi status: {response.status_code}")
                logger.info(f"Vapi response: {response.text}")
                return False
            
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Outbound call failed: {e}")
            return False
        