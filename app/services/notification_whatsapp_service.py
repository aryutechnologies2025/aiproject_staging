import os
import httpx
from dotenv import load_dotenv

load_dotenv()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL")
ADMIN_PHONE = os.getenv("ADMIN_PHONE")

async def send_meeting_whatsapp(meeting_data: dict):
    message = (
        "📅 *New Meeting Booked*\n\n"
        f"👤 Name: {meeting_data['name']}\n"
        f"📞 Phone: {meeting_data['phone']}\n"
        f"📧 Email: {meeting_data['email']}\n"
        f"🕒 Time: {meeting_data['datetime']}\n"
        f"📝 Purpose: {meeting_data.get('purpose', '-')}"
    )

    payload = {
        "messaging_product": "whatsapp",
        "to": ADMIN_PHONE,
        "type": "text",
        "text": {"body": message}
    }

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{WHATSAPP_API_URL}/{WHATSAPP_PHONE_ID}/messages",
            headers=headers,
            json=payload
        )
