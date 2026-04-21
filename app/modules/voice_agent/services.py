"""
services.py — Core AI services for Voice Agent

STT: Sarvam AI  (saarika:v2 — Tamil-first ASR, same vendor as TTS)
TTS: Sarvam AI  (bulbul:v2 — Tamil voice synthesis)
LLM: Groq (llama-3.3-70b-versatile) or Google Gemini

Why Sarvam STT over Deepgram for Tamil:
  • saarika:v2 is trained on 22 Indian languages with deep Tamil coverage
  • Same API key as TTS — one vendor, one billing, simpler ops
  • Native handling of code-mixed Tamil-English (common in job calls)
  • Significantly lower WER on Tamil vs Deepgram nova-3 for Indian accents
  • Supports streaming WebSocket (mulaw 8kHz) — same format as Vobiz
"""

import asyncio
import json
import base64
import httpx
import redis.asyncio as aioredis
from datetime import datetime, timedelta
from typing import Optional, List
import uuid
import logging

from app.modules.voice_agent import config
from app.modules.voice_agent.models import (
    CallSessionData, CallState, LeadStatus,
    LeadData, CompanyScriptData, CompanyData,
)
from app.modules.voice_agent.schemas import CallSessionRedis
from app.modules.voice_agent.script import build_system_prompt
from app.modules.voice_agent.tamil_normalizer import normalize

logger = logging.getLogger("voice_agent.services")

_redis: Optional[aioredis.Redis] = None


# ─────────────────────────────────────────────────────────────────────────────
# Redis Session Store
# ─────────────────────────────────────────────────────────────────────────────

async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_keepalive=True,
        )
    return _redis


async def session_save(session: CallSessionData) -> None:
    r = await get_redis()
    data = CallSessionRedis(
        call_id=session.call_id,
        lead_id=session.lead_id,
        lead_phone=session.lead_phone,
        lead_name=session.lead_name,
        company_id=session.company_id,
        script_id=session.script_id,
        state=session.state.value,
        script_pos=session.script_pos,
        history=session.history,
        lead_score=session.lead_score,
        score_confidence=session.score_confidence,
        intent_flags=session.intent_flags,
        tts_playing=session.tts_playing,
        started_at=session.started_at.isoformat(),
        transcript_full=session.transcript_full,
        proposed_slots=session.proposed_slots,
    )
    await r.setex(
        f"call:{session.call_id}",
        config.REDIS_SESSION_TTL,
        data.model_dump_json(),
    )


async def session_load(call_id: str) -> Optional[CallSessionData]:
    r = await get_redis()
    raw = await r.get(f"call:{call_id}")
    if not raw:
        return None
    data = CallSessionRedis.model_validate_json(raw)
    return CallSessionData(
        call_id=data.call_id,
        lead_id=data.lead_id,
        lead_phone=data.lead_phone,
        lead_name=data.lead_name,
        company_id=data.company_id,
        script_id=data.script_id,
        state=CallState(data.state),
        script_pos=data.script_pos,
        history=data.history,
        lead_score=data.lead_score,
        score_confidence=data.score_confidence,
        intent_flags=data.intent_flags,
        tts_playing=data.tts_playing,
        transcript_full=data.transcript_full,
        proposed_slots=data.proposed_slots,
    )


async def session_delete(call_id: str) -> None:
    r = await get_redis()
    await r.delete(f"call:{call_id}")


# ─────────────────────────────────────────────────────────────────────────────
# LLM — Groq / Gemini
# ─────────────────────────────────────────────────────────────────────────────

async def llm_respond(
    session: CallSessionData,
    script: CompanyScriptData,
    company: CompanyData,
) -> dict:
    steps = script.steps
    step = steps[min(session.script_pos, len(steps) - 1)]
    current_question = step["question"].replace("{name}", session.lead_name)

    system_prompt = build_system_prompt(
        company_name=company.name,
        agent_name=company.agent_name,
        extra_instructions=script.system_prompt_extra,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"current_question: {current_question}"},
        {"role": "system", "content": f"script_pos: {session.script_pos}/{len(steps) - 1}"},
        {"role": "system", "content": f"current lead_score: {session.lead_score}"},
    ]
    # Keep last 8 turns to fit context window without truncation issues
    messages += session.history[-8:]

    raw = ""

    try:
        if config.LLM_PROVIDER == "groq":
            import groq
            client = groq.AsyncGroq(api_key=config.GROQ_API_KEY)
            resp = await client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=350,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""

        else:  # gemini
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            model = genai.GenerativeModel(config.GEMINI_MODEL)
            prompt_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
            resp = await asyncio.to_thread(
                model.generate_content,
                prompt_text + "\nRespond ONLY with valid JSON. No markdown.",
            )
            raw = resp.text or ""

        result = json.loads(raw)

    except json.JSONDecodeError:
        logger.warning(f"[llm_respond] JSON parse failed, using fallback. raw={raw[:200]}")
        result = {
            "speech": step.get("fallback", current_question),
            "lead_score": session.lead_score,
            "score_confidence": session.score_confidence,
            "intent_flags": [],
            "advance_script": False,
            "should_end_call": False,
        }
    except Exception as e:
        logger.error(f"[llm_respond] LLM error: {e}")
        result = {
            "speech": step.get("fallback", current_question),
            "lead_score": session.lead_score,
            "score_confidence": session.score_confidence,
            "intent_flags": [],
            "advance_script": False,
            "should_end_call": False,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TTS — Sarvam AI (bulbul:v2)
# ─────────────────────────────────────────────────────────────────────────────

async def sarvam_tts(text: str) -> bytes:
    """
    Convert text → Tamil speech (mulaw 8kHz PCM16).
    Normalizes abbreviations and numbers to Tamil words before synthesis.
    Returns raw PCM16 bytes ready to base64-encode and send to Vobiz.
    """
    normalized = normalize(text)
    if not normalized.strip():
        return b""

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            config.SARVAM_TTS_URL,
            headers={"API-Subscription-Key": config.SARVAM_API_KEY},
            json={
                "inputs": [normalized],
                "target_language_code": config.STT_LANGUAGE_CODE,  # ta-IN
                "speaker": config.TTS_SPEAKER,
                "model": config.TTS_MODEL,
                "enable_preprocessing": True,
                "speech_sample_rate": config.TTS_SAMPLE_RATE,
                "encoding": "linear16",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return base64.b64decode(data["audios"][0])


# ─────────────────────────────────────────────────────────────────────────────
# STT — Sarvam AI Streaming (saarika:v2)
#
# Sarvam's streaming STT WebSocket protocol:
#   1. Connect to wss://api.sarvam.ai/speech-to-text-streaming
#      with header: API-Subscription-Key: <key>
#   2. Send config frame (JSON) first:
#      {"language_code": "ta-IN", "model": "saarika:v2",
#       "encoding": "mulaw", "sample_rate": 8000}
#   3. Send raw audio chunks as binary frames
#   4. Receive JSON transcript events:
#      {"type": "partial", "transcript": "..."} — interim
#      {"type": "final",   "transcript": "..."} — sentence complete
#   5. Send {"type": "end_of_stream"} to flush last utterance
#
# For REST (non-streaming) fallback, POST audio to
#   https://api.sarvam.ai/speech-to-text
#   with multipart/form-data: file=<wav>, language_code=ta-IN, model=saarika:v2
# ─────────────────────────────────────────────────────────────────────────────

SARVAM_STT_CONFIG_FRAME = {
    "language_code": "ta-IN",
    "model": "saarika:v2",
    "encoding": "mulaw",
    "sample_rate": 8000,
    # endpointing_silence_ms: how long of silence triggers a final transcript
    "endpointing_silence_ms": 600,
}


async def sarvam_stt_rest(audio_bytes: bytes, sample_rate: int = 8000) -> str:
    """
    REST fallback STT — send a WAV buffer, get transcript back.
    Used for very short utterances or when WS fails.
    """
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    buf.seek(0)

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            config.SARVAM_STT_REST_URL,
            headers={"API-Subscription-Key": config.SARVAM_API_KEY},
            files={"file": ("audio.wav", buf, "audio/wav")},
            data={
                "language_code": "ta-IN",
                "model": "saarika:v2",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("transcript", "")


# ─────────────────────────────────────────────────────────────────────────────
# Telephony — Vobiz
# ─────────────────────────────────────────────────────────────────────────────

def _vobiz_headers() -> dict:
    return {
        "X-Auth-ID": config.VOBIZ_AUTH_ID,
        "X-Auth-Token": config.VOBIZ_AUTH_TOKEN,
        "Content-Type": "application/json",
    }
 
 
async def vobiz_initiate_call(lead: LeadData, stream_url: str) -> str:
    """
    Initiate outbound call via Vobiz.
 
    stream_url is the WSS URL Vobiz will connect to AFTER the lead answers.
    It is embedded in the VoiceXML returned by answer_url.
 
    Correct Vobiz endpoint: POST /account/{auth_id}/calls/
    """
    # The answer_url is an HTTP endpoint on YOUR server that returns VoiceXML.
    # When the lead answers, Vobiz fetches this URL and follows the XML instructions.
    # We encode lead_id as a query param so our answer endpoint knows which lead.
    answer_url = (
        f"{config.PUBLIC_BASE_URL}/api/v1/voice/answer"
        f"?lead_id={lead.id}"
        f"&stream_url={stream_url}"
    )
    status_url = f"{config.PUBLIC_BASE_URL}/api/v1/voice/call-status"
 
    logger.info(
        f"[vobiz] Initiating outbound call → {lead.phone} ({lead.name})\n"
        f"  answer_url : {answer_url}\n"
        f"  status_url : {status_url}"
    )
 
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{config.VOBIZ_API_URL}/Account/{config.VOBIZ_AUTH_ID}/Call/",
            headers=_vobiz_headers(),
            json={
                "from": config.VOBIZ_CALLER_ID,
                "to": lead.phone,
                "answer_url": answer_url,
                "answer_method": "POST",
                "status_url": status_url,
                "status_method": "POST",
                "ring_timeout": config.CALL_TIMEOUT_SECONDS,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        call_id = data.get("call_uuid") or data.get("CallUUID") or data.get("call_id", "")
        logger.info(f"[vobiz] Call placed: CallUUID={call_id}")
        return call_id
 
 
async def vobiz_hangup(call_id: str) -> None:
    """Hang up a live call by CallUUID."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.delete(
                f"{config.VOBIZ_API_URL}/Account/{config.VOBIZ_AUTH_ID}/Call/{call_id}/",
                headers=_vobiz_headers(),
            )
            logger.info(f"[vobiz] Hung up call {call_id}: {resp.status_code}")
        except Exception as e:
            logger.warning(f"[vobiz] Hangup failed for {call_id}: {e}")


async def simulate_call(lead: LeadData, stream_url: str) -> str:
    """
    Simulation mode — does NOT place a real call.
    Returns a fake call_id for use with the browser simulator.
    """
    call_id = str(uuid.uuid4())
    logger.info(
        f"[SIMULATION] Lead={lead.name} ({lead.phone}) | "
        f"Lead ID={lead.id} | "
        f"Open call_simulator.html and paste Lead ID to test"
    )
    return call_id

def build_stream_xml(stream_wss_url: str) -> str:
    """
    Return the VoiceXML that tells Vobiz to stream audio to our WebSocket server.
 
    Vobiz WebSocket message types (per docs):
      connected → start → media (audio) → dtmf → stop
 
    The <Stream> directive starts bidirectional mulaw-8kHz audio streaming.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_wss_url}" />
    </Connect>
</Response>"""


# ─────────────────────────────────────────────────────────────────────────────
# SMS — MSG91
# ─────────────────────────────────────────────────────────────────────────────

async def send_sms(phone: str, message: str) -> bool:
    if not config.MSG91_AUTH_KEY:
        logger.warning("[sms] MSG91_AUTH_KEY not set — skipping SMS")
        return False
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(
                "https://api.msg91.com/api/v5/flow/",
                headers={
                    "authkey": config.MSG91_AUTH_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "template_id": config.MSG91_TEMPLATE_ID,
                    "recipients": [
                        {"mobiles": phone.lstrip("+"), "message": message}
                    ],
                    "sender": config.MSG91_SENDER_ID,
                },
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"[sms] Failed to send to {phone}: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Google Calendar
# ─────────────────────────────────────────────────────────────────────────────

async def get_calendar_slots(lookahead_days: int = 3) -> List[datetime]:
    if not config.GOOGLE_CALENDAR_CREDENTIALS or not config.GOOGLE_CALENDAR_ID:
        # Return synthetic slots if calendar not configured
        logger.warning("[calendar] Credentials not set — returning synthetic slots")
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        return [now + timedelta(hours=h) for h in [24, 48, 72]]

    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_authorized_user_file(config.GOOGLE_CALENDAR_CREDENTIALS)
        service = build("calendar", "v3", credentials=creds)
        now = datetime.utcnow()
        end = now + timedelta(days=lookahead_days)

        events = service.events().list(
            calendarId=config.GOOGLE_CALENDAR_ID,
            timeMin=now.isoformat() + "Z",
            timeMax=end.isoformat() + "Z",
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        busy = set()
        for event in events.get("items", []):
            start = event["start"].get("dateTime", "")
            if start:
                busy.add(datetime.fromisoformat(start.rstrip("Z")))

        available = []
        cursor = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        while cursor < end and len(available) < 3:
            if cursor.weekday() < 6 and 9 <= cursor.hour < 18 and cursor not in busy:
                available.append(cursor)
            cursor += timedelta(hours=1)
        return available

    except Exception as e:
        logger.error(f"[calendar] get_calendar_slots error: {e}")
        # Fallback synthetic
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        return [now + timedelta(hours=h) for h in [24, 48, 72]]


async def create_calendar_event(
    lead: LeadData, company: CompanyData, slot: datetime, call_id: str
) -> str:
    if not config.GOOGLE_CALENDAR_CREDENTIALS:
        logger.warning("[calendar] Credentials not set — skipping calendar event")
        return ""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_authorized_user_file(config.GOOGLE_CALENDAR_CREDENTIALS)
        service = build("calendar", "v3", credentials=creds)

        event = {
            "summary": f"Interview — {lead.name} ({company.name})",
            "description": (
                f"Lead: {lead.name}\n"
                f"Phone: {lead.phone}\n"
                f"Company: {company.name}\n"
                f"Call ID: {call_id}"
            ),
            "start": {
                "dateTime": slot.isoformat() + "+05:30",
                "timeZone": "Asia/Kolkata",
            },
            "end": {
                "dateTime": (
                    slot + timedelta(minutes=config.INTERVIEW_DURATION_MINUTES)
                ).isoformat() + "+05:30",
                "timeZone": "Asia/Kolkata",
            },
            "reminders": {"useDefault": True},
        }
        created = service.events().insert(
            calendarId=config.GOOGLE_CALENDAR_ID, body=event
        ).execute()
        return created.get("id", "")
    except Exception as e:
        logger.error(f"[calendar] create_calendar_event error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def score_to_status(score: str) -> LeadStatus:
    return {
        "hot": LeadStatus.HOT,
        "warm": LeadStatus.WARM,
        "cold": LeadStatus.COLD,
    }.get(score.lower(), LeadStatus.COLD)


def build_interview_sms(lead_name: str, company_name: str, slot: datetime) -> str:
    ist = slot + timedelta(hours=5, minutes=30)
    return (
        f"வணக்கம் {lead_name}, உங்கள் நேர்காணல் "
        f"{ist.strftime('%d/%m/%Y')} அன்று "
        f"{ist.strftime('%I:%M %p')} மணிக்கு நிர்ணயிக்கப்பட்டுள்ளது. "
        f"{company_name} நிறுவனம். தொடர்புக்கு: {config.VOBIZ_CALLER_ID}"
    )


def build_recall_sms(lead_name: str, company_name: str, recall_at: datetime) -> str:
    ist = recall_at + timedelta(hours=5, minutes=30)
    return (
        f"வணக்கம் {lead_name}, {company_name} நிறுவனத்திலிருந்து "
        f"{ist.strftime('%I:%M %p')} மணிக்கு திரும்ப அழைக்கிறோம்."
    )