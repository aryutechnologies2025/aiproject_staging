from app.services.lead_service import detect_lead_intent
from app.services.enterprise_lead_service import detect_contact_intent, detect_specific_contact
from app.services.meeting_service import detect_meeting_intent
from app.services.meeting_state import (
    get_meeting_state,
    save_meeting_state,
    next_meeting_step
)
from app.services.chat_memory_service import (
    get_chat_history,
    save_chat_message
)
from app.services.meeting_service import save_meeting
from app.services.llm_client import call_llm
from app.core.config import WHATSAPP_NUMBER, CONTACT_EMAIL

async def yura_chat(message: str, session_id: str, db):
    if not message.strip():
        return "Hi 👋 How can I help you today?"

    # Save user message to Redis memory
    save_chat_message(session_id, "user", message)

    # 1️⃣ CONTACT HANDLING
    # CONTACT HANDLING (HIGHEST PRIORITY)
    if detect_contact_intent(message):
        contact_type = detect_specific_contact(message)

        wa_link = f"https://wa.me/{WHATSAPP_NUMBER.replace('+','')}" if WHATSAPP_NUMBER else ""

        if contact_type == "whatsapp":
            reply = f"You can contact us on WhatsApp: {wa_link}"

        elif contact_type == "email":
            reply = f"You can email us at {CONTACT_EMAIL}"

        else:
            reply = (
                f"WhatsApp: {wa_link}\n"
                f"Email: {CONTACT_EMAIL}"
            )

        save_chat_message(session_id, "assistant", reply)
        return reply

    # 2️⃣ MEETING FLOW
    meeting_state = get_meeting_state(session_id)

    if meeting_state["mode"]:
        data = meeting_state["data"]
        step = next_meeting_step(data)

        if step == "name":
            data["name"] = message
            reply = "Thanks. Please share your phone or WhatsApp number."
        elif step == "phone":
            data["phone"] = message
            reply = "Got it. Please share your email address."
        elif step == "email":
            data["email"] = message
            reply = "When would you like to schedule the meeting?"
        elif step == "datetime":
            data["datetime"] = message
            reply = "Briefly tell us the purpose of the meeting."
        elif step == "purpose":
            data["purpose"] = message
            await save_meeting(data, session_id, db)
            reply = (
                "Thank you. Your meeting request has been recorded. "
                "Our team will contact you shortly to confirm."
            )
            meeting_state = {"mode": False, "data": {}}
            save_chat_message(session_id, "assistant", reply)
            return reply

        meeting_state["data"] = data
        save_meeting_state(session_id, meeting_state)
        save_chat_message(session_id, "assistant", reply)
        return reply

    if detect_meeting_intent(message):
        meeting_state = {"mode": True, "data": {}}
        save_meeting_state(session_id, meeting_state)
        reply = "Sure. To arrange a meeting, may I know your name?"
        save_chat_message(session_id, "assistant", reply)
        return reply

    # 3️⃣ NORMAL CHAT WITH MEMORY
    history = get_chat_history(session_id)

    context = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-6:]
    )

    ai_reply = await call_llm(
        model="gemma",
        user_message=f"{context}\nUser: {message}",
        agent_name="yura_website_bot",
        db=db
    )

    # 4️⃣ LEAD CTA
    if detect_lead_intent(message):
        ai_reply += "\n\nI can help you with batch details or connect you with our team."

    save_chat_message(session_id, "assistant", ai_reply)
    return ai_reply.strip()
