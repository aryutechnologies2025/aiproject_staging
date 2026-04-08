from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select
from datetime import datetime, timezone
import logging

# 1. Import your REAL session and models
from app.core.database import AsyncSessionLocal  # Ensure this is your actual async_sessionmaker
from app.modules.voice_agent.models import Lead, CallStatusEnum
from app.modules.voice_agent.services import trigger_outbound_call

logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()

async def process_pending_recalls():
    """
    Polls the database for leads that are pending recall.
    """
    logger.info("Checking for pending recalls...")
    
    # 2. Use 'async with' for your AsyncSession
    async with AsyncSessionLocal() as session:
        async with session.begin():
            now = datetime.now(timezone.utc)
            
            # 3. Use SQLAlchemy 2.0 select syntax for Async
            stmt = select(Lead).where(
                Lead.status == CallStatusEnum.PENDING_RECALL,
                Lead.recall_timestamp <= now
            )
            
            result = await session.execute(stmt)
            leads_to_recall = result.scalars().all()

            if not leads_to_recall:
                logger.info("No recalls pending at this time.")
                return

            for lead in leads_to_recall:
                logger.info(f"Triggering scheduled recall for {lead.phone_number}")
                
                # 4. Trigger the outbound call
                success = await trigger_outbound_call(lead.phone_number)
                
                if success:
                    lead.status = CallStatusEnum.RECALLED
                else:
                    # You might want to implement a retry count here
                    lead.status = CallStatusEnum.FAILED
            
            # Commit changes to the database
            await session.commit()

def start_scheduler():
    """Starts the APScheduler background tasks."""
    # We use 'cron' or 'interval'. 5 minutes is good for production.
    scheduler.add_job(
        process_pending_recalls, 
        'interval', 
        minutes=5, 
        id="recall_job", 
        replace_existing=True
    )
    if not scheduler.running:
        scheduler.start()
        logger.info("--- Voice Bot Scheduler Started ---")

def stop_scheduler():
    """Gracefully shuts down the scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("--- Voice Bot Scheduler Stopped ---")
        