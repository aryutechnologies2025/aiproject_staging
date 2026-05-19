"""
services.py — Core AI services

Audio contract:
  sarvam_tts()  → returns raw LINEAR16 PCM bytes at 8kHz (NOT mulaw, NOT base64)
  sarvam_stt_rest() → REST fallback, wraps PCM16 bytes in WAV and posts to Sarvam

  call_handler._speak() converts TTS output: PCM16 → audioop.lin2ulaw() → mulaw → Vobiz.
  Do NOT change sarvam_tts() encoding to "mulaw" — Sarvam's mulaw output quality
  is lower than converting ourselves via audioop.lin2ulaw().

STT event contract (saaras:v3 with vad_signals=True):
  type="speech_start"  → VAD detected onset of speech
  type="events"        → keepalive heartbeat (text may be empty — this is normal)
  type="speech_end"    → VAD detected end of utterance / silence
  type="transcript"    → final recognised text for the utterance

  SARVAM_STT_TRANSCRIPT_TYPES and SARVAM_STT_PARTIAL_TYPES are imported by
  call_handler so both modules stay in sync on event-type strings.
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
# Sarvam STT event-type constants
#
# Imported by call_handler._sarvam_stt_reader() — keep in sync with Sarvam docs.
#
# saaras:v3 final event:   {"type": "transcript", "transcript": "..."}
# legacy saarika:v2 final: {"type": "final",      "text": "..."}
# We handle both so a model downgrade doesn't break the pipeline.
#
# "speech_start" appears in PARTIAL_TYPES because it sometimes carries
# a preliminary text field in some SDK versions — handled defensively.
# ─────────────────────────────────────────────────────────────────────────────

SARVAM_STT_TRANSCRIPT_TYPES = frozenset({"transcript", "final"})
SARVAM_STT_PARTIAL_TYPES    = frozenset({"partial", "speech_start"})


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
#
# Returns raw LINEAR16 PCM bytes at 8kHz.
# Caller (call_handler._speak) converts to µ-law before sending to Vobiz.
#
# Why linear16 and not mulaw?
#   Sarvam's mulaw output is encoded server-side from a lower-quality path.
#   Doing the conversion ourselves with audioop.lin2ulaw gives better audio.
# ─────────────────────────────────────────────────────────────────────────────

async def sarvam_tts(text: str) -> bytes:
    """
    Convert text → LINEAR16 PCM audio at 8kHz via Sarvam TTS.
    Returns raw PCM16 bytes (NOT base64, NOT mulaw).
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
# STT helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_sarvam_stt_url() -> str:
    """
    Build the Sarvam STT WebSocket URL with connection params as query params.

    This is used if you ever want to connect via raw websockets instead of the
    SDK. Normally call_handler uses the AsyncSarvamAI SDK client directly.

    Audio flow before calling this endpoint:
      Vobiz → mulaw 8kHz → audioop.ulaw2lin() → pcm_s16le 8kHz
      → audioop.ratecv() → pcm_s16le 16kHz → send here

    Note: sample_rate=16000 because call_handler upsamples to 16kHz.
    """
    from urllib.parse import urlencode
    params = urlencode({
        "model":                "saaras:v3",
        "language_code":        "ta-IN",
        "mode":                 "transcribe",
        "sample_rate":          "16000",      # upsampled in call_handler
        "input_audio_codec":    "pcm_s16le",  # Sarvam does NOT support mulaw
        "high_vad_sensitivity": "true",
        "vad_signals":          "true",
        "api_subscription_key": config.SARVAM_API_KEY,
    })
    return f"{config.SARVAM_STT_WS_URL}?{params}"


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """
    Convert G.711 µ-law bytes (from Vobiz) to LINEAR16 PCM (for Sarvam STT).
    Sarvam STT only accepts PCM codecs — mulaw is NOT a valid input_audio_codec.
    """
    import audioop
    return audioop.ulaw2lin(mulaw_bytes, 2)  # 2 = 16-bit output samples


async def sarvam_stt_rest(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    REST fallback STT — wraps raw PCM16 bytes in a WAV container and posts
    to the Sarvam REST endpoint. Used when the streaming WS is unavailable.

    audio_bytes: raw PCM16 (signed 16-bit little-endian, mono)
    sample_rate: must match the actual audio (default 16000 after upsampling)
    """
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)      # mono
        wf.setsampwidth(2)      # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)

    wav_bytes = buf.getvalue()

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            config.SARVAM_STT_REST_URL,
            headers={"API-Subscription-Key": config.SARVAM_API_KEY},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={
                "language_code": "ta-IN",
                "model": "saaras:v3",
                "mode": "transcribe",
            },
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
    Endpoint: POST /Account/{auth_id}/Call/
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
    """
    Build the Vobiz Stream XML response for bidirectional audio.

    Vobiz XML rules (confirmed from docs.vobiz.ai):
      - <Stream> text content = WSS URL (NOT an attribute)
      - bidirectional="true"  — required to receive inbound (caller) audio
      - keepCallAlive="true"  — prevents call from hanging up while streaming
      - contentType="audio/x-mulaw;rate=8000" — tells Vobiz the codec we want
      - Must be wss:// not https://
      - NO <Connect> wrapper

    If bidirectional="true" is missing or ignored by Vobiz, you will receive
    ONLY outbound (TTS) audio looped back, not the caller's voice.
    Symptom: all µ-law bytes = 0x7F (silence) with unique_bytes ≈ 1-2.
    """
    wss_url = stream_wss_url.replace("https://", "wss://").replace("http://", "ws://")

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream
        bidirectional="true"
        keepCallAlive="true"
        contentType="audio/x-mulaw;rate=8000"
    >{wss_url}</Stream>
</Response>'''


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
    lead: LeadData,
    company: CompanyData,
    slot: datetime,
    call_id: str,
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
            "description": (
                f"Lead: {lead.name}\nPhone: {lead.phone}\nCall ID: {call_id}"
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
        "hot":  LeadStatus.HOT,
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