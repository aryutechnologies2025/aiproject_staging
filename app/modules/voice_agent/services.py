"""
services.py — Core AI services (FIXED)

Key fix in sarvam_tts():
  Before: returned base64-decoded bytes but the encoding param was "linear16"
          which is correct PCM16 — but call_handler was sending it raw to Vobiz
          without converting to mulaw first.
  After:  sarvam_tts() still returns raw PCM16 bytes.
          call_handler._speak() now does: PCM16 → audioop.lin2ulaw() → mulaw → Vobiz.

The encoding field is explicitly "linear16" here so Sarvam gives us PCM16.
The mulaw conversion happens in call_handler._pcm16_to_mulaw().
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
    step  = steps[min(session.script_pos, len(steps) - 1)]
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
    messages += session.history[-8:]

    raw = ""
    fallback = {
        "speech": step.get("fallback", current_question),
        "lead_score": session.lead_score,
        "score_confidence": session.score_confidence,
        "intent_flags": [],
        "advance_script": False,
        "should_end_call": False,
    }

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
        else:
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
        return result

    except json.JSONDecodeError:
        logger.warning(f"[llm_respond] JSON parse fail. raw={raw[:200]}")
        return fallback
    except Exception as e:
        logger.error(f"[llm_respond] LLM error: {e}")
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# TTS — Sarvam AI (bulbul:v2)
# Returns raw LINEAR16 PCM bytes at 8kHz.
# Caller (call_handler._speak) is responsible for converting to µ-law for Vobiz.
# ─────────────────────────────────────────────────────────────────────────────

async def sarvam_tts(text: str) -> bytes:
    """
    Convert text → LINEAR16 PCM audio at 8kHz via Sarvam TTS.
    Returns raw PCM16 bytes.

    NOTE: call_handler._speak() converts this to µ-law before sending to Vobiz.
    Do NOT change encoding to "mulaw" here — Sarvam's mulaw output quality is
    lower than converting from linear16 ourselves.
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
                "target_language_code": "ta-IN",
                "speaker": config.TTS_SPEAKER,
                "model": config.TTS_MODEL,
                "enable_preprocessing": True,
                "speech_sample_rate": 8000,
                "encoding": "linear16",   # PCM16 — we convert to mulaw ourselves
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # Sarvam returns base64-encoded audio in data["audios"][0]
        audio_b64 = data.get("audios", [None])[0]
        if not audio_b64:
            logger.error(f"[sarvam_tts] No audio in response: {data}")
            return b""

        return base64.b64decode(audio_b64)


# ─────────────────────────────────────────────────────────────────────────────
# STT config frame — sent to Sarvam STT WS on connect
# ─────────────────────────────────────────────────────────────────────────────

SARVAM_STT_CONFIG_FRAME = {
    "language_code": "ta-IN",
    "model": "saarika:v2",
    "encoding": "mulaw",          # Vobiz sends mulaw audio to us
    "sample_rate": 8000,
    "endpointing_silence_ms": 600,
}


async def sarvam_stt_rest(audio_bytes: bytes, sample_rate: int = 8000) -> str:
    """REST fallback STT — used when streaming WS is unavailable."""
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    buf.seek(0)

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            config.SARVAM_STT_REST_URL,
            headers={"API-Subscription-Key": config.SARVAM_API_KEY},
            files={"file": ("audio.wav", buf, "audio/wav")},
            data={"language_code": "ta-IN", "model": "saarika:v2"},
        )
        resp.raise_for_status()
        return resp.json().get("transcript", "")


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
    Correct endpoint: POST /Account/{auth_id}/Call/
    """
    answer_url = (
        f"{config.PUBLIC_BASE_URL}/api/v1/voice/answer"
        f"?lead_id={lead.id}"
        f"&stream_url={stream_url}"
    )
    status_url = f"{config.PUBLIC_BASE_URL}/api/v1/voice/call-status"

    logger.info(
        f"[vobiz] Calling {lead.phone} ({lead.name})\n"
        f"  answer_url={answer_url}"
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
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.delete(
                f"{config.VOBIZ_API_URL}/Account/{config.VOBIZ_AUTH_ID}/Call/{call_id}/",
                headers=_vobiz_headers(),
            )
            logger.info(f"[vobiz] Hung up {call_id}: {resp.status_code}")
        except Exception as e:
            logger.warning(f"[vobiz] Hangup failed for {call_id}: {e}")


async def simulate_call(lead: LeadData, stream_url: str) -> str:
    call_id = str(uuid.uuid4())
    logger.info(
        f"[SIMULATION] Lead={lead.name} ({lead.phone}) | "
        f"ID={lead.id} | stream_url={stream_url}"
    )
    return call_id


def build_stream_xml(stream_wss_url: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream url="{stream_wss_url}" />
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
                headers={"authkey": config.MSG91_AUTH_KEY, "Content-Type": "application/json"},
                json={
                    "template_id": config.MSG91_TEMPLATE_ID,
                    "recipients": [{"mobiles": phone.lstrip("+"), "message": message}],
                    "sender": config.MSG91_SENDER_ID,
                },
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"[sms] Failed to {phone}: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Google Calendar
# ─────────────────────────────────────────────────────────────────────────────

async def get_calendar_slots(lookahead_days: int = 3) -> List[datetime]:
    if not config.GOOGLE_CALENDAR_CREDENTIALS or not config.GOOGLE_CALENDAR_ID:
        logger.warning("[calendar] Not configured — returning synthetic slots")
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
        for ev in events.get("items", []):
            start = ev["start"].get("dateTime", "")
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
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        return [now + timedelta(hours=h) for h in [24, 48, 72]]


async def create_calendar_event(
    lead: LeadData, company: CompanyData, slot: datetime, call_id: str
) -> str:
    if not config.GOOGLE_CALENDAR_CREDENTIALS:
        return ""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        creds = Credentials.from_authorized_user_file(config.GOOGLE_CALENDAR_CREDENTIALS)
        service = build("calendar", "v3", credentials=creds)
        event = {
            "summary": f"Interview — {lead.name} ({company.name})",
            "description": f"Lead: {lead.name}\nPhone: {lead.phone}\nCall ID: {call_id}",
            "start": {"dateTime": slot.isoformat() + "+05:30", "timeZone": "Asia/Kolkata"},
            "end": {
                "dateTime": (slot + timedelta(minutes=config.INTERVIEW_DURATION_MINUTES)).isoformat() + "+05:30",
                "timeZone": "Asia/Kolkata",
            },
            "reminders": {"useDefault": True},
        }
        created = service.events().insert(calendarId=config.GOOGLE_CALENDAR_ID, body=event).execute()
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
        f"{ist.strftime('%d/%m/%Y')} அன்று {ist.strftime('%I:%M %p')} மணிக்கு. "
        f"{company_name}. தொடர்புக்கு: {config.VOBIZ_CALLER_ID}"
    )


def build_recall_sms(lead_name: str, company_name: str, recall_at: datetime) -> str:
    ist = recall_at + timedelta(hours=5, minutes=30)
    return (
        f"வணக்கம் {lead_name}, {company_name} இல் இருந்து "
        f"{ist.strftime('%I:%M %p')} மணிக்கு திரும்ப அழைக்கிறோம்."
    )