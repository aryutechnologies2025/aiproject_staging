import asyncio
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.modules.voice_agent import config
from app.modules.voice_agent import database as db
from app.modules.voice_agent.models import LeadData, LeadStatus

logger = logging.getLogger("voice_agent.scheduler")
scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")

_active_calls: dict = {}
MAX_CONCURRENT_CALLS = int(getattr(config, "MAX_CONCURRENT_CALLS", 5))


async def _place_single_call(lead: LeadData) -> None:
    if lead.id in _active_calls:
        logger.warning(f"[scheduler] Lead {lead.id} already in active calls, skipping")
        return
    if len(_active_calls) >= MAX_CONCURRENT_CALLS:
        logger.warning(f"[scheduler] Max concurrent calls reached ({MAX_CONCURRENT_CALLS})")
        return

    _active_calls[lead.id] = True
    logger.info(
        f"[scheduler] Placing call → {lead.name} ({lead.phone}) | "
        f"simulation={config.SIMULATION_MODE}"
    )

    try:
        await db.update_lead(lead.id, {"status": LeadStatus.CALLING})

        # Build the WSS stream URL that Vobiz will connect to
        # stream_url is for logging only — /answer endpoint builds it for Vobiz
        base = config.PUBLIC_BASE_URL.rstrip("/")
        wss_base = base.replace("https://", "wss://").replace("http://", "ws://")
        stream_url = f"{wss_base}/api/v1/voice/ws/call?lead_id={lead.id}"

        if config.SIMULATION_MODE:
            from app.modules.voice_agent.services import simulate_call
            call_id = await simulate_call(lead, stream_url)
            logger.info(f"[scheduler] SIMULATION call_id={call_id} for {lead.phone}")
        else:
            from app.modules.voice_agent.services import vobiz_initiate_call
            call_id = await vobiz_initiate_call(lead, stream_url)
            logger.info(
                f"[scheduler] ✓ Call placed | CallUUID={call_id} | "
                f"{lead.name} ({lead.phone})"
            )

    except Exception as e:
        # Log the FULL exception — this is why calls were silently failing
        logger.error(
            f"[scheduler] ❌ CALL FAILED for {lead.name} ({lead.phone}): {e}",
            exc_info=True,   # prints full traceback including HTTP response body
        )
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
    logger.info(f"[scheduler] Batch: {len(leads)} calls for company={company_id}")
    await asyncio.gather(
        *[_place_single_call(lead) for lead in leads],
        return_exceptions=True,
    )


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
    if config.ENABLE_CAMPAIGN_SCHEDULER:
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
        logger.warning(f"[scheduler] trigger_immediate_call blocked — max concurrent reached")
        return False
    logger.info(f"[scheduler] trigger_immediate_call → {lead.name} ({lead.phone})")
    asyncio.create_task(_place_single_call(lead))
    return True