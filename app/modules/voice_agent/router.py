import os
import uuid
import json
import base64
import httpx
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, WebSocket, UploadFile, File, HTTPException, Query, Form, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app.modules.voice_agent import config
from app.modules.voice_agent import database as db
from app.modules.voice_agent.schemas import (
    CompanyCreate, CompanyResponse,
    ScriptCreateManual, ScriptResponse,
    LeadCreate, LeadUpdate, LeadResponse,
)
from app.modules.voice_agent.services import (
    send_sms, build_recall_sms, build_stream_xml
)
from app.modules.voice_agent.csv_handler import process_csv_upload
from app.modules.voice_agent.call_handler import CallHandler
from app.modules.voice_agent.scheduler import trigger_immediate_call
from app.modules.voice_agent.script import (
    parse_uploaded_script, get_script_json_template,
    DEFAULT_OBJECTIONS, DEFAULT_CLOSING_HOT, DEFAULT_CLOSING_WARM, DEFAULT_CLOSING_COLD,
)

import logging
logger = logging.getLogger("voice_agent.router")

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — Vobiz streams audio here
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/call")
async def call_websocket(websocket: WebSocket, lead_id: Optional[str] = Query(None)):
    if not lead_id:
        await websocket.close(code=4000)
        return

    lead = await db.get_lead_by_id(lead_id)
    if not lead:
        logger.error(f"[ws/call] lead_id={lead_id} not found")
        await websocket.close(code=4004)
        return

    company = await db.get_company_by_id(lead.company_id)
    if not company:
        logger.error(f"[ws/call] company not found for lead={lead_id}")
        await websocket.close(code=4004)
        return

    script = await db.get_active_script_for_company(lead.company_id)
    if not script:
        logger.error(f"[ws/call] no active script for company={lead.company_id}")
        await websocket.close(code=4005)
        return

    logger.info(f"[ws/call] Starting handler | lead={lead.name} company={company.name} script={script.id}")
    handler = CallHandler(websocket, lead, company, script)
    await handler.start()


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER URL — Vobiz fetches this when lead picks up, we return VoiceXML
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/answer", include_in_schema=False)
@router.get("/answer", include_in_schema=False)
async def vobiz_answer(
    request: Request,
    lead_id: Optional[str] = Query(None),
):
    """
    Vobiz calls this (GET or POST) when the outbound call is answered.
    Returns VoiceXML with <Connect><Stream url="wss://..."/> telling
    Vobiz to open a WebSocket to our /ws/call endpoint.

    RULES that prevent "Invalid serviceUrl":
      1. ALWAYS build the WSS URL here from PUBLIC_BASE_URL — never
         trust an incoming stream_url query param (it arrives corrupted
         because '?' and '&' inside it break the outer query string)
      2. Use wss:// not https://
      3. Return raw bytes with explicit Content-Type — do NOT let
         FastAPI touch the response body (it would escape the slashes)
    """
    # Vobiz POSTs form data on hangup — lead_id may be in the form body
    if not lead_id:
        try:
            form = await request.form()
            lead_id = form.get("lead_id") or ""
        except Exception:
            pass

    # Build the WSS URL cleanly — this is the ONLY place it's constructed
    base = config.PUBLIC_BASE_URL.rstrip("/")
    wss_base = base.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{wss_base}/api/v1/voice/ws/call?lead_id={lead_id}"

    # Build XML using string concatenation — f-string multiline with triple
    # quotes can introduce whitespace that some XML parsers reject
    xml = build_stream_xml(ws_url)

    logger.info(f"[answer] lead_id={lead_id} → WSS={ws_url}")
    logger.info(f"[answer] XML: {xml}")

    # Return as raw bytes with explicit content-type
    # This bypasses any FastAPI response serialization that could escape /
    from starlette.responses import Response
    return Response(
        content=xml.encode("utf-8"),
        media_type="application/xml; charset=utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# STATUS CALLBACK — Vobiz POSTs call lifecycle events here
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/call-status", include_in_schema=False)
async def vobiz_call_status(request: Request):
    try:
        form = await request.form()
        event     = form.get("Event", "")
        call_uuid = form.get("CallUUID", "")
        status    = form.get("Status", "")
        duration  = form.get("Duration", "0")
        to_number = form.get("To", "")
        logger.info(
            f"[call-status] Event={event} UUID={call_uuid} "
            f"Status={status} Duration={duration}s To={to_number}"
        )
    except Exception as e:
        logger.warning(f"[call-status] Could not parse form: {e}")

    return JSONResponse({"status": "received"})


# ─────────────────────────────────────────────────────────────────────────────
# CALL NOW — trigger immediate outbound call
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/leads/{lead_id}/call-now", response_model=dict)
async def call_lead_now(lead_id: str):
    lead = await db.get_lead_by_id(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    if lead.status.value == "calling":
        raise HTTPException(status_code=409, detail="Lead is already being called")

    script = await db.get_active_script_for_company(lead.company_id)
    if not script:
        raise HTTPException(
            status_code=422,
            detail="No active script for this company. Upload and activate a script first.",
        )

    queued = await trigger_immediate_call(lead)
    if not queued:
        raise HTTPException(status_code=429, detail="Max concurrent calls reached")

    return {"message": "Call queued", "lead_id": lead_id}


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG — test every single step of the call chain in one shot
# Hit this endpoint and you'll see EXACTLY where it breaks
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/debug/call-test/{lead_id}")
async def debug_call_test(lead_id: str):
    """
    Diagnostic endpoint. Checks every component end-to-end.
    Call this before placing a real call to find exactly what's broken.

    Returns a JSON report with pass/fail for each step.
    """
    results = {}

    # ── 1. Config sanity check ───────────────────────────────────────────────
    results["config"] = {
        "VOBIZ_API_URL":    config.VOBIZ_API_URL or "❌ EMPTY",
        "VOBIZ_AUTH_ID":    config.VOBIZ_AUTH_ID[:8] + "..." if config.VOBIZ_AUTH_ID else "❌ EMPTY",
        "VOBIZ_AUTH_TOKEN": "✓ set" if config.VOBIZ_AUTH_TOKEN else "❌ EMPTY",
        "VOBIZ_CALLER_ID":  config.VOBIZ_CALLER_ID or "❌ EMPTY",
        "SARVAM_API_KEY":   "✓ set" if config.SARVAM_API_KEY else "❌ EMPTY",
        "GROQ_API_KEY":     "✓ set" if config.GROQ_API_KEY else "❌ EMPTY",
        "PUBLIC_BASE_URL":  config.PUBLIC_BASE_URL or "❌ EMPTY",
        "SIMULATION_MODE":  config.SIMULATION_MODE,
    }

    # ── 2. Lead + company + script in DB ────────────────────────────────────
    lead = await db.get_lead_by_id(lead_id)
    if not lead:
        results["db_lead"] = f"❌ Lead {lead_id} not found"
        return results
    results["db_lead"] = f"✓ {lead.name} ({lead.phone}) status={lead.status.value}"

    company = await db.get_company_by_id(lead.company_id)
    if not company:
        results["db_company"] = "❌ Company not found"
        return results
    results["db_company"] = f"✓ {company.name} (id={company.id[:8]}...)"

    script = await db.get_active_script_for_company(lead.company_id)
    if not script:
        results["db_script"] = "❌ No active script — upload & activate a script first"
        return results
    results["db_script"] = f"✓ script id={script.id[:8]}... steps={len(script.steps)}"

    # ── 3. Test answer_url is reachable ─────────────────────────────────────
    answer_url = f"{config.PUBLIC_BASE_URL}/api/v1/voice/answer?lead_id={lead_id}&stream_url=wss://test"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(answer_url)
            if r.status_code == 200 and "<Stream" in r.text:
                results["answer_url"] = f"✓ returns VoiceXML (HTTP {r.status_code})"
            else:
                results["answer_url"] = f"❌ HTTP {r.status_code} — body: {r.text[:200]}"
    except Exception as e:
        results["answer_url"] = f"❌ Cannot reach answer_url: {e}. Is ngrok running? Is PUBLIC_BASE_URL correct?"

    # ── 4. Test ws/call endpoint is reachable (HTTP upgrade check) ──────────
    ws_http_url = f"{config.PUBLIC_BASE_URL}/api/v1/voice/ws/call?lead_id={lead_id}"
    ws_http_url = ws_http_url.replace("wss://", "https://").replace("ws://", "http://")
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(ws_http_url, headers={"Upgrade": "websocket"})
            # FastAPI returns 403 or 426 for non-WS requests — both mean the route exists
            results["ws_endpoint"] = f"✓ endpoint reachable (HTTP {r.status_code})"
    except Exception as e:
        results["ws_endpoint"] = f"❌ Cannot reach ws/call: {e}"

    # ── 5. Test Sarvam TTS ───────────────────────────────────────────────────
    if config.SARVAM_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    config.SARVAM_TTS_URL,
                    headers={"API-Subscription-Key": config.SARVAM_API_KEY},
                    json={
                        "inputs": ["வணக்கம்"],
                        "target_language_code": "ta-IN",
                        "speaker": config.TTS_SPEAKER,
                        "model": config.TTS_MODEL,
                        "speech_sample_rate": 8000,
                        "encoding": "linear16",
                    },
                )
                if r.status_code == 200:
                    data = r.json()
                    audio_b64 = data.get("audios", [None])[0]
                    audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
                    results["sarvam_tts"] = f"✓ {len(audio_bytes)} PCM bytes returned"
                else:
                    results["sarvam_tts"] = f"❌ HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            results["sarvam_tts"] = f"❌ {e}"
    else:
        results["sarvam_tts"] = "❌ SARVAM_API_KEY not set"

    # ── 6. Test Vobiz API auth ───────────────────────────────────────────────
    if config.VOBIZ_AUTH_ID and config.VOBIZ_AUTH_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Hit the account balance endpoint — safe read-only check
                r = await client.get(
                    f"{config.VOBIZ_API_URL}/Account/{config.VOBIZ_AUTH_ID}/",
                    headers={
                        "X-Auth-ID": config.VOBIZ_AUTH_ID,
                        "X-Auth-Token": config.VOBIZ_AUTH_TOKEN,
                    },
                )
                if r.status_code == 200:
                    results["vobiz_auth"] = f"✓ Authenticated (HTTP 200)"
                elif r.status_code == 401:
                    results["vobiz_auth"] = f"❌ 401 Unauthorized — check VOBIZ_AUTH_ID and VOBIZ_AUTH_TOKEN in .env"
                else:
                    results["vobiz_auth"] = f"⚠ HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            results["vobiz_auth"] = f"❌ {e}"
    else:
        results["vobiz_auth"] = "❌ VOBIZ_AUTH_ID or VOBIZ_AUTH_TOKEN not set in .env"

    # ── 7. Build the actual call payload (dry-run, no actual call placed) ────
    stream_url = f"{config.PUBLIC_BASE_URL}/api/v1/voice/ws/call?lead_id={lead_id}"
    answer_url_real = (
        f"{config.PUBLIC_BASE_URL}/api/v1/voice/answer"
        f"?lead_id={lead_id}&stream_url={stream_url}"
    )
    results["call_payload_dry_run"] = {
        "endpoint": f"{config.VOBIZ_API_URL}/Account/{config.VOBIZ_AUTH_ID}/Call/",
        "from":       config.VOBIZ_CALLER_ID or "❌ EMPTY — set VOBIZ_CALLER_ID",
        "to":         lead.phone,
        "answer_url": answer_url_real,
        "status_url": f"{config.PUBLIC_BASE_URL}/api/v1/voice/call-status",
    }

    # ── Summary ──────────────────────────────────────────────────────────────
    failures = [k for k, v in results.items() if isinstance(v, str) and "❌" in v]
    results["_summary"] = (
        "✅ ALL CHECKS PASSED — safe to place real call"
        if not failures
        else f"❌ FIX THESE FIRST: {failures}"
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Company endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/companies", response_model=CompanyResponse, status_code=201)
async def create_company(data: CompanyCreate):
    existing = await db.get_company_by_slug(data.slug)
    if existing:
        raise HTTPException(status_code=409, detail="Company with this slug already exists")
    company_id = await db.create_company({
        "name": data.name, "slug": data.slug, "industry": data.industry,
        "language": data.language, "agent_name": data.agent_name,
    })
    company = await db.get_company_by_id(company_id)
    return CompanyResponse(
        id=company.id, name=company.name, slug=company.slug,
        industry=company.industry, language=company.language,
        agent_name=company.agent_name, active_script_id=company.active_script_id,
        created_at=company.created_at,
    )


@router.get("/companies", response_model=List[CompanyResponse])
async def list_companies():
    companies = await db.list_companies()
    return [
        CompanyResponse(
            id=c.id, name=c.name, slug=c.slug, industry=c.industry,
            language=c.language, agent_name=c.agent_name,
            active_script_id=c.active_script_id, created_at=c.created_at,
        )
        for c in companies
    ]


@router.get("/companies/{company_id}", response_model=CompanyResponse)
async def get_company(company_id: str):
    company = await db.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return CompanyResponse(
        id=company.id, name=company.name, slug=company.slug,
        industry=company.industry, language=company.language,
        agent_name=company.agent_name, active_script_id=company.active_script_id,
        created_at=company.created_at,
    )


@router.post("/companies/{company_id}/scripts/upload", response_model=ScriptResponse, status_code=201)
async def upload_script(
    company_id: str,
    file: UploadFile = File(...),
    version: str = Form("1.0"),
    activate: bool = Form(False),
):
    company = await db.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    allowed = {".json", ".pdf", ".txt", ".md"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Allowed: {allowed}")
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    safe_name = f"{company_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    save_path = os.path.join(config.UPLOADS_DIR, safe_name)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    try:
        parsed = parse_uploaded_script(save_path, file.filename or "")
    except ValueError as e:
        os.unlink(save_path)
        raise HTTPException(status_code=422, detail=str(e))
    script_id = await db.create_script({
        "company_id": company_id, "version": version,
        "status": "active" if activate else "draft",
        "steps": parsed["steps"],
        "objection_responses": parsed["objection_responses"],
        "closing_hot": parsed["closing_hot"],
        "closing_warm": parsed["closing_warm"],
        "closing_cold": parsed["closing_cold"],
        "system_prompt_extra": parsed["system_prompt_extra"],
        "uploaded_filename": safe_name,
    })
    if activate:
        await db.activate_script(company_id, script_id)
    script = await db.get_script_by_id(script_id)
    return ScriptResponse(
        id=script.id, company_id=script.company_id, version=script.version,
        status=script.status, uploaded_filename=script.uploaded_filename,
        steps_count=len(script.steps), created_at=script.created_at,
    )


@router.post("/companies/{company_id}/scripts", response_model=ScriptResponse, status_code=201)
async def create_script_manual(company_id: str, data: ScriptCreateManual):
    company = await db.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    script_id = await db.create_script({
        "company_id": company_id, "version": data.version, "status": "draft",
        "steps": data.steps,
        "objection_responses": data.objection_responses or DEFAULT_OBJECTIONS,
        "closing_hot": data.closing_hot or DEFAULT_CLOSING_HOT,
        "closing_warm": data.closing_warm or DEFAULT_CLOSING_WARM,
        "closing_cold": data.closing_cold or DEFAULT_CLOSING_COLD,
        "system_prompt_extra": data.system_prompt_extra,
        "uploaded_filename": None,
    })
    script = await db.get_script_by_id(script_id)
    return ScriptResponse(
        id=script.id, company_id=script.company_id, version=script.version,
        status=script.status, uploaded_filename=script.uploaded_filename,
        steps_count=len(script.steps), created_at=script.created_at,
    )


@router.get("/companies/{company_id}/scripts", response_model=List[ScriptResponse])
async def list_scripts(company_id: str):
    scripts = await db.list_scripts_for_company(company_id)
    return [
        ScriptResponse(
            id=s.id, company_id=s.company_id, version=s.version,
            status=s.status, uploaded_filename=s.uploaded_filename,
            steps_count=len(s.steps), created_at=s.created_at,
        )
        for s in scripts
    ]


@router.get("/companies/{company_id}/scripts/{script_id}")
async def get_script_detail(company_id: str, script_id: str):
    script = await db.get_script_by_id(script_id)
    if not script or script.company_id != company_id:
        raise HTTPException(status_code=404, detail="Script not found")
    return {
        "id": script.id, "company_id": script.company_id,
        "version": script.version, "status": script.status,
        "steps": script.steps, "objection_responses": script.objection_responses,
        "closing_hot": script.closing_hot, "closing_warm": script.closing_warm,
        "closing_cold": script.closing_cold,
        "system_prompt_extra": script.system_prompt_extra,
        "uploaded_filename": script.uploaded_filename,
        "created_at": script.created_at.isoformat(),
    }


@router.post("/companies/{company_id}/scripts/{script_id}/activate")
async def activate_script(company_id: str, script_id: str):
    company = await db.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    script = await db.get_script_by_id(script_id)
    if not script or script.company_id != company_id:
        raise HTTPException(status_code=404, detail="Script not found")
    await db.activate_script(company_id, script_id)
    return {"message": "Script activated", "script_id": script_id}


@router.get("/companies/{company_id}/scripts/template/json")
async def get_script_template(company_id: str):
    company = await db.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return get_script_json_template(company_name=company.name, agent_name=company.agent_name)


@router.post("/companies/{company_id}/leads", response_model=dict, status_code=201)
async def create_lead(company_id: str, data: LeadCreate):
    company = await db.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    existing = await db.get_lead_by_phone_and_company(data.phone, company_id)
    if existing:
        raise HTTPException(status_code=409, detail="Lead already exists for this company")
    lead_id = await db.create_lead({
        "name": data.name, "phone": data.phone, "company_id": company_id,
        "qualification": data.qualification, "experience_years": data.experience_years,
        "language_preference": data.language_preference,
        "source_file": data.source_file, "status": "pending",
    })
    return {"lead_id": lead_id, "message": "Lead created"}


@router.get("/companies/{company_id}/leads", response_model=List[LeadResponse])
async def list_leads(
    company_id: str,
    status: Optional[str] = None,
    limit: int = Query(50, le=200),
):
    leads = await db.get_pending_leads(company_id=company_id, limit=limit)
    if status:
        leads = [l for l in leads if l.status.value == status]
    return [
        LeadResponse(
            id=l.id, name=l.name, phone=l.phone, company_id=l.company_id,
            status=l.status, call_attempts=l.call_attempts, score=l.score,
            scheduled_interview_at=l.scheduled_interview_at,
            next_call_at=l.next_call_at, notes=l.notes,
        )
        for l in leads
    ]


@router.patch("/leads/{lead_id}", response_model=dict)
async def update_lead(lead_id: str, data: LeadUpdate):
    lead = await db.get_lead_by_id(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    await db.update_lead(lead_id, data.model_dump(exclude_none=True))
    return {"message": "Lead updated"}


@router.post("/companies/{company_id}/leads/upload-csv", response_model=dict)
async def upload_csv(company_id: str, file: UploadFile = File(...)):
    company = await db.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    if not (file.filename or "").endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")
    try:
        created, skipped, errors = await process_csv_upload(file, company_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "created": created, "skipped_duplicates": skipped,
        "errors": errors[:20],
        "message": f"Imported {created} leads for {company.name}",
    }


@router.get("/companies/{company_id}/stats")
async def get_company_stats(company_id: str):
    company = await db.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    stats = await db.get_lead_stats_by_company(company_id)
    return {"company": company.name, "lead_counts": stats, "total": sum(stats.values())}


@router.get("/stats")
async def get_all_stats():
    stats = await db.get_all_stats()
    return {"companies": stats}