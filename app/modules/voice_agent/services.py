import asyncio
import json
import base64
import httpx
import redis.asyncio as aioredis
from datetime import datetime, timedelta
from typing import Optional, List

import groq
import google.generativeai as genai

from app.modules.voice_agent import config
from app.modules.voice_agent.models import (
    CallSessionData, CallState, LeadStatus,
    LeadData, CompanyScriptData, CompanyData,
)
from app.modules.voice_agent.schemas import CallSessionRedis
from app.modules.voice_agent.script import build_system_prompt
from app.modules.voice_agent.tamil_normalizer import normalize


_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(config.REDIS_URL, decode_responses=True)
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
    await r.setex(f"call:{session.call_id}", config.REDIS_SESSION_TTL, data.model_dump_json())


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
        {"role": "system", "content": f"script_pos: {session.script_pos}/{len(steps)-1}"},
        {"role": "system", "content": f"current lead_score: {session.lead_score}"},
    ]
    messages += session.history[-6:]

    raw = ""
    if config.LLM_PROVIDER == "groq":
        client = groq.AsyncGroq(api_key=config.GROQ_API_KEY)
        resp = await client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
    else:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        resp = await asyncio.to_thread(
            model.generate_content,
            prompt_text + "\nRespond only with valid JSON.",
        )
        raw = resp.text

    try:
        result = json.loads(raw)
    except Exception:
        result = {
            "speech": step.get("fallback", current_question),
            "lead_score": session.lead_score,
            "score_confidence": session.score_confidence,
            "intent_flags": [],
            "advance_script": False,
            "should_end_call": False,
        }
    return result


async def sarvam_tts(text: str) -> bytes:
    normalized = normalize(text)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"API-Subscription-Key": config.SARVAM_API_KEY},
            json={
                "inputs": [normalized],
                "target_language_code": "ta-IN",
                "speaker": "pavithra",
                "model": "bulbul:v2",
                "enable_preprocessing": True,
                "speech_sample_rate": 8000,
                "encoding": "linear16",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return base64.b64decode(data["audios"][0])


async def vobiz_initiate_call(lead: LeadData, stream_url: str) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{config.VOBIZ_API_URL}/calls/outbound",
            headers={"Authorization": f"Bearer {config.VOBIZ_API_KEY}"},
            json={
                "from": config.VOBIZ_CALLER_ID,
                "to": lead.phone,
                "stream_url": stream_url,
                "stream_events": ["start", "media", "stop"],
                "timeout": config.CALL_TIMEOUT_SECONDS,
                "custom_parameters": {"lead_id": lead.id, "company_id": lead.company_id},
            },
        )
        resp.raise_for_status()
        return resp.json()["call_id"]


async def vobiz_hangup(call_id: str) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        await client.delete(
            f"{config.VOBIZ_API_URL}/calls/{call_id}",
            headers={"Authorization": f"Bearer {config.VOBIZ_API_KEY}"},
        )


async def send_sms(phone: str, message: str) -> bool:
    async with httpx.AsyncClient(timeout=10) as client:
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


async def get_calendar_slots(lookahead_days: int = 3) -> List[datetime]:
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


async def create_calendar_event(
    lead: LeadData, company: CompanyData, slot: datetime, call_id: str
) -> str:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(config.GOOGLE_CALENDAR_CREDENTIALS)
    service = build("calendar", "v3", credentials=creds)

    event = {
        "summary": f"Interview - {lead.name} ({company.name})",
        "description": f"Lead: {lead.name}\nPhone: {lead.phone}\nCompany: {company.name}\nCall ID: {call_id}",
        "start": {
            "dateTime": slot.isoformat() + "+05:30",
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": (slot + timedelta(minutes=config.INTERVIEW_DURATION_MINUTES)).isoformat() + "+05:30",
            "timeZone": "Asia/Kolkata",
        },
        "reminders": {"useDefault": True},
    }
    created = service.events().insert(calendarId=config.GOOGLE_CALENDAR_ID, body=event).execute()
    return created.get("id", "")


def score_to_status(score: str) -> LeadStatus:
    return {"hot": LeadStatus.HOT, "warm": LeadStatus.WARM, "cold": LeadStatus.COLD}.get(
        score.lower(), LeadStatus.COLD
    )


def build_interview_sms(lead_name: str, company_name: str, slot: datetime) -> str:
    ist = slot + timedelta(hours=5, minutes=30)
    return (
        f"வணக்கம் {lead_name}, உங்கள் நேர்காணல் {ist.strftime('%d/%m/%Y')} அன்று "
        f"{ist.strftime('%I:%M %p')} மணிக்கு நிர்ணயிக்கப்பட்டுள்ளது. "
        f"{company_name} நிறுவனம். தொடர்புக்கு: {config.VOBIZ_CALLER_ID}"
    )


def build_recall_sms(lead_name: str, company_name: str, recall_at: datetime) -> str:
    ist = recall_at + timedelta(hours=5, minutes=30)
    return (
        f"வணக்கம் {lead_name}, {company_name} நிறுவனத்திலிருந்து "
        f"{ist.strftime('%I:%M %p')} மணிக்கு திரும்ப அழைக்கிறோம்."
    )
