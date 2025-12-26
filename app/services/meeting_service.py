from sqlalchemy.ext.asyncio import AsyncSession
from app.models.meeting import Meeting
from app.services.notification_email_service import send_meeting_email
import re
from app.services.notification_whatsapp_service import send_meeting_whatsapp

async def save_meeting(data: dict, session_id: str, db: AsyncSession):
    meeting = Meeting(
        session_id=session_id,
        name=data["name"],
        phone=data["phone"],
        email=data["email"],
        preferred_datetime=data["datetime"],
        purpose=data.get("purpose"),
    )

    db.add(meeting)
    await db.commit()

    payload = {
        "session_id": session_id,
        "name": data["name"],
        "phone": data["phone"],
        "email": data["email"],
        "datetime": data["datetime"],
        "purpose": data.get("purpose"),
    }

    # 🔔 Notifications
    send_meeting_email(payload)
    await send_meeting_whatsapp(payload)


MEETING_KEYWORDS = [
    "meeting",
    "schedule",
    "appointment",
    "call",
    "discussion",
    "demo",
    "connect",
]

def detect_meeting_intent(message: str) -> bool:
    msg = message.lower()
    return any(word in msg for word in MEETING_KEYWORDS)