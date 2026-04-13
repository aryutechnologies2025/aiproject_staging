import os
import uuid
import json
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, WebSocket, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse

from app.modules.voice_agent import config
from app.modules.voice_agent import database as db
from app.modules.voice_agent.schemas import (
    CompanyCreate, CompanyResponse,
    ScriptCreateManual, ScriptResponse,
    LeadCreate, LeadUpdate, LeadResponse,
)
from app.modules.voice_agent.csv_handler import process_csv_upload
from app.modules.voice_agent.call_handler import CallHandler
from app.modules.voice_agent.scheduler import trigger_immediate_call
from app.modules.voice_agent.script import parse_uploaded_script, get_script_json_template, DEFAULT_OBJECTIONS, DEFAULT_CLOSING_HOT, DEFAULT_CLOSING_WARM, DEFAULT_CLOSING_COLD

router = APIRouter()


@router.websocket("/ws/call")
async def call_websocket(websocket: WebSocket, lead_id: Optional[str] = Query(None)):
    if not lead_id:
        await websocket.close(code=4000)
        return

    lead = await db.get_lead_by_id(lead_id)
    if not lead:
        await websocket.close(code=4004)
        return

    company = await db.get_company_by_id(lead.company_id)
    if not company:
        await websocket.close(code=4004)
        return

    script = await db.get_active_script_for_company(lead.company_id)
    if not script:
        await websocket.close(code=4005)
        return

    handler = CallHandler(websocket, lead, company, script)
    await handler.start()


@router.post("/companies", response_model=CompanyResponse, status_code=201)
async def create_company(data: CompanyCreate):
    existing = await db.get_company_by_slug(data.slug)
    if existing:
        raise HTTPException(status_code=409, detail="Company with this slug already exists")

    company_id = await db.create_company({
        "name": data.name,
        "slug": data.slug,
        "industry": data.industry,
        "language": data.language,
        "agent_name": data.agent_name,
    })
    company = await db.get_company_by_id(company_id)
    return CompanyResponse(
        id=company.id,
        name=company.name,
        slug=company.slug,
        industry=company.industry,
        language=company.language,
        agent_name=company.agent_name,
        active_script_id=company.active_script_id,
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
        raise HTTPException(status_code=400, detail=f"Allowed file types: {allowed}")

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
        "company_id": company_id,
        "version": version,
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
        id=script.id,
        company_id=script.company_id,
        version=script.version,
        status=script.status,
        uploaded_filename=script.uploaded_filename,
        steps_count=len(script.steps),
        created_at=script.created_at,
    )


@router.post("/companies/{company_id}/scripts", response_model=ScriptResponse, status_code=201)
async def create_script_manual(company_id: str, data: ScriptCreateManual):
    company = await db.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    script_id = await db.create_script({
        "company_id": company_id,
        "version": data.version,
        "status": "draft",
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
        id=script.id,
        company_id=script.company_id,
        version=script.version,
        status=script.status,
        uploaded_filename=script.uploaded_filename,
        steps_count=len(script.steps),
        created_at=script.created_at,
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
        "id": script.id,
        "company_id": script.company_id,
        "version": script.version,
        "status": script.status,
        "steps": script.steps,
        "objection_responses": script.objection_responses,
        "closing_hot": script.closing_hot,
        "closing_warm": script.closing_warm,
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
        raise HTTPException(status_code=409, detail="Lead with this phone already exists for this company")

    lead_id = await db.create_lead({
        "name": data.name,
        "phone": data.phone,
        "company_id": company_id,
        "qualification": data.qualification,
        "experience_years": data.experience_years,
        "language_preference": data.language_preference,
        "source_file": data.source_file,
        "status": "pending",
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


@router.post("/leads/{lead_id}/call-now", response_model=dict)
async def call_lead_now(lead_id: str):
    lead = await db.get_lead_by_id(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    if lead.status == "calling":
        raise HTTPException(status_code=409, detail="Lead is already being called")
    script = await db.get_active_script_for_company(lead.company_id)
    if not script:
        raise HTTPException(status_code=422, detail="No active script for this company. Upload and activate a script first.")
    queued = await trigger_immediate_call(lead)
    if not queued:
        raise HTTPException(status_code=429, detail="Max concurrent calls reached")
    return {"message": "Call queued", "lead_id": lead_id}


@router.post("/companies/{company_id}/leads/upload-csv", response_model=dict)
async def upload_csv(company_id: str, file: UploadFile = File(...)):
    company = await db.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    if not (file.filename or "").endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    try:
        created, skipped, errors = await process_csv_upload(file, company_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "created": created,
        "skipped_duplicates": skipped,
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