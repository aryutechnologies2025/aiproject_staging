from fastapi import APIRouter, BackgroundTasks
import logging

from app.modules.voice_agent.schemas import VapiWebhookPayload, WebhookResponse
from app.modules.voice_agent.services import handle_call_end_logic

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/vapi-webhook", response_model=WebhookResponse)
async def vapi_webhook_handler(
    payload: VapiWebhookPayload,
    background_tasks: BackgroundTasks
):
    """
    Receives Call events from Vapi/Retell.
    Delegates processing to background task.
    """

    if payload.message.type == "end-of-call-report":
        # No DB passed
        background_tasks.add_task(handle_call_end_logic, payload)

        return WebhookResponse(
            status="success",
            message="Call processing initiated."
        )

    return WebhookResponse(
        status="ignored",
        message="Event type not handled."
    )
