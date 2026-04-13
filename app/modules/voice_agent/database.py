import uuid
import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select, update, func

from app.modules.voice_agent import config
from app.modules.voice_agent.models import (
    Base,
    Company, CompanyScript, Lead, InterviewSlot,
    CompanyData, CompanyScriptData, LeadData,
    LeadStatus, ScriptStatus,
)

engine = create_async_engine(config.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


def _orm_to_company(row: Company) -> CompanyData:
    return CompanyData(
        id=row.id,
        name=row.name,
        slug=row.slug,
        industry=row.industry,
        language=row.language,
        agent_name=row.agent_name,
        active_script_id=row.active_script_id,
        created_at=row.created_at,
    )


def _orm_to_script(row: CompanyScript) -> CompanyScriptData:
    steps = row.steps if isinstance(row.steps, list) else json.loads(row.steps)
    objections = (
        row.objection_responses
        if isinstance(row.objection_responses, dict)
        else json.loads(row.objection_responses)
    )
    return CompanyScriptData(
        id=row.id,
        company_id=row.company_id,
        version=row.version,
        status=ScriptStatus(row.status),
        steps=steps,
        objection_responses=objections,
        closing_hot=row.closing_hot,
        closing_warm=row.closing_warm,
        closing_cold=row.closing_cold,
        system_prompt_extra=row.system_prompt_extra or "",
        uploaded_filename=row.uploaded_filename,
        created_at=row.created_at,
    )


def _orm_to_lead(row: Lead) -> LeadData:
    return LeadData(
        id=row.id,
        name=row.name,
        phone=row.phone,
        company_id=row.company_id,
        status=LeadStatus(row.status),
        qualification=row.qualification,
        experience_years=row.experience_years,
        language_preference=row.language_preference,
        call_attempts=row.call_attempts,
        max_attempts=row.max_attempts,
        last_called_at=row.last_called_at,
        next_call_at=row.next_call_at,
        scheduled_interview_at=row.scheduled_interview_at,
        notes=row.notes,
        source_file=row.source_file,
        score=row.score,
    )


async def get_company_by_id(company_id: str) -> Optional[CompanyData]:
    async with AsyncSessionLocal() as s:
        row = await s.get(Company, company_id)
        return _orm_to_company(row) if row else None


async def get_company_by_slug(slug: str) -> Optional[CompanyData]:
    async with AsyncSessionLocal() as s:
        result = await s.execute(select(Company).where(Company.slug == slug))
        row = result.scalar_one_or_none()
        return _orm_to_company(row) if row else None


async def list_companies() -> List[CompanyData]:
    async with AsyncSessionLocal() as s:
        result = await s.execute(select(Company).order_by(Company.name))
        return [_orm_to_company(r) for r in result.scalars().all()]


async def create_company(data: dict) -> str:
    company_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as s:
        s.add(Company(id=company_id, **data))
        await s.commit()
    return company_id


async def update_company(company_id: str, updates: dict) -> None:
    async with AsyncSessionLocal() as s:
        await s.execute(
            update(Company).where(Company.id == company_id).values(**updates)
        )
        await s.commit()


async def create_script(data: dict) -> str:
    script_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as s:
        s.add(CompanyScript(id=script_id, **data))
        await s.commit()
    return script_id


async def get_script_by_id(script_id: str) -> Optional[CompanyScriptData]:
    async with AsyncSessionLocal() as s:
        row = await s.get(CompanyScript, script_id)
        return _orm_to_script(row) if row else None


async def get_active_script_for_company(company_id: str) -> Optional[CompanyScriptData]:
    async with AsyncSessionLocal() as s:
        company = await s.get(Company, company_id)
        if not company or not company.active_script_id:
            return None
        row = await s.get(CompanyScript, company.active_script_id)
        return _orm_to_script(row) if row else None


async def list_scripts_for_company(company_id: str) -> List[CompanyScriptData]:
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(CompanyScript)
            .where(CompanyScript.company_id == company_id)
            .order_by(CompanyScript.created_at.desc())
        )
        return [_orm_to_script(r) for r in result.scalars().all()]


async def activate_script(company_id: str, script_id: str) -> None:
    async with AsyncSessionLocal() as s:
        await s.execute(
            update(Company).where(Company.id == company_id)
            .values(active_script_id=script_id)
        )
        await s.execute(
            update(CompanyScript).where(CompanyScript.id == script_id)
            .values(status="active")
        )
        await s.commit()


async def get_lead_by_id(lead_id: str) -> Optional[LeadData]:
    async with AsyncSessionLocal() as s:
        row = await s.get(Lead, lead_id)
        return _orm_to_lead(row) if row else None


async def get_lead_by_phone_and_company(phone: str, company_id: str) -> Optional[LeadData]:
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(Lead).where(Lead.phone == phone, Lead.company_id == company_id)
        )
        row = result.scalar_one_or_none()
        return _orm_to_lead(row) if row else None


async def get_pending_leads(company_id: str, limit: int = 50) -> List[LeadData]:
    async with AsyncSessionLocal() as s:
        now = datetime.utcnow()
        result = await s.execute(
            select(Lead)
            .where(
                Lead.company_id == company_id,
                Lead.status.in_(["pending", "recall"]),
                Lead.call_attempts < Lead.max_attempts,
            )
            .where(
                (Lead.next_call_at == None) | (Lead.next_call_at <= now)
            )
            .order_by(Lead.next_call_at.asc().nullsfirst())
            .limit(limit)
        )
        return [_orm_to_lead(r) for r in result.scalars().all()]


async def create_lead(lead_data: dict) -> str:
    lead_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as s:
        s.add(Lead(id=lead_id, **lead_data))
        await s.commit()
    return lead_id


async def update_lead(lead_id: str, updates: dict) -> None:
    async with AsyncSessionLocal() as s:
        await s.execute(
            update(Lead).where(Lead.id == lead_id).values(**updates)
        )
        await s.commit()


async def increment_call_attempts(lead_id: str) -> None:
    async with AsyncSessionLocal() as s:
        await s.execute(
            update(Lead)
            .where(Lead.id == lead_id)
            .values(
                call_attempts=Lead.call_attempts + 1,
                last_called_at=datetime.utcnow(),
            )
        )
        await s.commit()


async def bulk_create_leads(leads: List[dict]) -> int:
    created = 0
    async with AsyncSessionLocal() as s:
        for ld in leads:
            existing = await s.execute(
                select(Lead).where(
                    Lead.phone == ld["phone"],
                    Lead.company_id == ld["company_id"],
                )
            )
            if existing.scalar_one_or_none() is None:
                s.add(Lead(id=str(uuid.uuid4()), **ld))
                created += 1
        await s.commit()
    return created


async def create_interview_slot(
    lead_id: str,
    call_id: str,
    company_id: str,
    scheduled_at: datetime,
    calendar_event_id: str = "",
    sms_sent: bool = False,
) -> str:
    slot_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as s:
        s.add(InterviewSlot(
            id=slot_id,
            lead_id=lead_id,
            call_id=call_id,
            company_id=company_id,
            scheduled_at=scheduled_at,
            calendar_event_id=calendar_event_id,
            sms_sent=sms_sent,
        ))
        await s.commit()
    return slot_id


async def get_lead_stats_by_company(company_id: str) -> dict:
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(Lead.status, func.count(Lead.id))
            .where(Lead.company_id == company_id)
            .group_by(Lead.status)
        )
        return {row[0]: row[1] for row in result.all()}


async def get_all_stats() -> dict:
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(Company.name, Lead.status, func.count(Lead.id))
            .join(Lead, Lead.company_id == Company.id)
            .group_by(Company.name, Lead.status)
        )
    stats: dict = {}
    for company_name, status, count in result.all():
        if company_name not in stats:
            stats[company_name] = {}
        stats[company_name][status] = count
    return stats
