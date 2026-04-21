import asyncio
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.modules.voice_agent import config
from app.modules.voice_agent import database as db
from app.modules.voice_agent.models import LeadData, LeadStatus
from app.modules.voice_agent.services import vobiz_initiate_call, simulate_call

logger = logging.getLogger("scheduler")
scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")

_active_calls: dict = {}
MAX_CONCURRENT_CALLS = int(getattr(config, "MAX_CONCURRENT_CALLS", 5))


async def _place_single_call(lead: LeadData) -> None:
    if lead.id in _active_calls:
        return
    if len(_active_calls) >= MAX_CONCURRENT_CALLS:
        return

    try:
        _active_calls[lead.id] = True
        await db.update_lead(lead.id, {"status": LeadStatus.CALLING})

        stream_url = f"{config.PUBLIC_BASE_URL}/api/v1/voice/ws/call"

        if getattr(config, "SIMULATION_MODE", False):
            from app.modules.voice_agent.services import simulate_call
            call_id = await simulate_call(lead, stream_url)
        else:
            from app.modules.voice_agent.services import vobiz_initiate_call
            call_id = await vobiz_initiate_call(lead, stream_url)

        logger.info(f"[{lead.company_id}] Call initiated: {call_id} -> {lead.phone} ({lead.name})")
    except Exception as e:
        logger.error(f"Failed to call {lead.phone}: {e}")
        await db.update_lead(lead.id, {"status": LeadStatus.FAILED})
    finally:
        _active_calls.pop(lead.id, None)


async def run_call_batch_for_company(company_id: str) -> None:
    if len(_active_calls) >= MAX_CONCURRENT_CALLS:
        return
    slots = MAX_CONCURRENT_CALLS - len(_active_calls)
    leads = await db.get_pending_leads(company_id=company_id, limit=slots)
    if not leads:
        return
    logger.info(f"[{company_id}] Batch: {len(leads)} calls")
    await asyncio.gather(*[_place_single_call(lead) for lead in leads], return_exceptions=True)


async def run_all_companies_batch() -> None:
    companies = await db.list_companies()
    for company in companies:
        await run_call_batch_for_company(company.id)


async def run_recall_check() -> None:
    companies = await db.list_companies()
    now = datetime.utcnow()
    for company in companies:
        leads = await db.get_pending_leads(company_id=company.id, limit=10)
        for lead in leads:
            if lead.status == LeadStatus.RECALL:
                if lead.next_call_at and lead.next_call_at <= now:
                    await _place_single_call(lead)


def setup_scheduler() -> AsyncIOScheduler:
    scheduler.add_job(
        run_all_companies_batch,
        CronTrigger(hour="9-18", minute="*/15", timezone="Asia/Kolkata"),
        id="call_batch",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        run_recall_check,
        IntervalTrigger(minutes=10),
        id="recall_check",
        replace_existing=True,
        max_instances=1,
    )
    return scheduler


async def trigger_immediate_call(lead: LeadData) -> bool:
    if len(_active_calls) >= MAX_CONCURRENT_CALLS:
        return False
    asyncio.create_task(_place_single_call(lead))
    return True
