#whatsapp_service.py
import os
import httpx
from app.services.llm_client import call_llm
from app.services.prompt_service import get_prompt
from app.services.router import route_message
from dotenv import load_dotenv
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.chat_state_service import (
    get_chat_state,
    update_chat_state,
)
from app.services.chat_memory_service import (
    get_chat_history,
    save_chat_message,
)

from app.utils.courses import COURSES

load_dotenv()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
ADMIN_PHONE = os.getenv("ADMIN_PHONE")
BASE_URL = "https://graph.facebook.com/v19.0"

async def send_whatsapp_message(to: str, message: str):
    url = f"{BASE_URL}/{PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, json=payload, headers=headers)

    print("\nWHATSAPP SEND STATUS:", response.status_code)
    print("WHATSAPP RESPONSE:", response.text)

async def notify_admin_lead(name, phone, time, user_number):
    admin_number = ADMIN_PHONE

    if not admin_number:
        print("ERROR: ADMIN_PHONE missing in .env")
        return

    # format +91
    if not admin_number.startswith("+"):
        admin_number = f"+91{admin_number[-10:]}"

    message = (
        f"📌 *New Lead Request – Speak With Mr. Y*\n\n"
        f"👤 Name: {name}\n"
        f"📞 Phone: {phone}\n"
        f"⏰ Preferred Time: {time}\n"
        f"📲 WhatsApp: {user_number}\n\n"
        f"Please contact them as soon as possible."
    )

    print("ADMIN_PHONE USED:", admin_number)
    await send_whatsapp_message(admin_number, message)

async def send_interactive_buttons(to: str):
    print("🚀 send_interactive_buttons() CALLED for", to)
    url = f"{BASE_URL}/{PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": "👋 Welcome to Aryu Academy!\nHow can I assist you today?"
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {"id": "speak_mr_y", "title": "Speak with Mr. Y"},
                    },
                    {
                        "type": "reply",
                        "reply": {"id": "about_courses", "title": "About Courses"},
                    },
                ]
            },
        },
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(url, headers=headers, json=payload)

    print("\nBUTTON SEND STATUS:", r.status_code)
    print("BUTTON RESPONSE:", r.text)

async def send_document(to: str, file_url: str, file_name: str):
    url = f"{BASE_URL}/{PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "document",
        "document": {
            "link": file_url,
            "filename": file_name
        }
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
    print("DOCUMENT SEND STATUS:", r.status_code, r.text)

async def send_course_list(to: str):

    rows = [{"id": key, "title": COURSES[key]["name"]} for key in COURSES]
    rows.append({"id": "others", "title": "Others (Type your course)"})

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": """"🎓 *Explore Our Career-Focused Programs*\n\n
                    Select a course below to view:\n
                    • What you’ll learn\n
                    • Duration\n
                    • Job roles\n
                    • Fee details\n\n
                    👇 Choose a program to continue:
                     """},
            "footer": {"text": "Aryu Academy"},
            "action": {
                "button": "Select Course",
                "sections": [
                    {
                        "title": "Available Courses",
                        "rows": rows
                    }
                ],
            },
        },
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(
            f"{BASE_URL}/{PHONE_ID}/messages",
            json=payload,
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
        )

    print("COURSE LIST STATUS:", r.status_code)
    print("COURSE LIST RESPONSE:", r.text)

def is_admin(number: str):
    if not ADMIN_PHONE:
        return False

    return number[-10:] == ADMIN_PHONE[-10:]

async def process_incoming_message(payload: dict, db: AsyncSession):

    try:
        entry = payload["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        messages = value.get("messages")

        if not messages:
            return

        msg = messages[0]
        from_number = msg.get("from")
        msg_type = msg.get("type")

        # Ignore Admin
        if is_admin(from_number):
            print("Admin message ignored.")
            return

        state = get_chat_state(from_number)

        if msg_type == "interactive" and "list_reply" in msg["interactive"]:
            list_selection = msg["interactive"]["list_reply"]["id"]

            # Others → user types course manually
            if list_selection == "others":
                update_chat_state(from_number, mode="other_course")
                await send_whatsapp_message(from_number, "Please type the course name 😊")
                return

            # Valid course
            if list_selection in COURSES:
                info = COURSES[list_selection]

                update_chat_state(from_number, mode="after_course")
                update_chat_state(from_number, selected_course=list_selection) # ← FIX

                await send_whatsapp_message(
                    from_number,
                    f"📘 *{info['name']}*\n\n{info['details']}\n\n"
                    "Would you like the *full syllabus* or a *free demo session*? 🙂"
                )
                return

            # backup
            await send_whatsapp_message(
                from_number,
                "❌ Something went wrong while selecting the course. Please type the course name."
            )
            return

        if msg_type == "interactive" and "button_reply" in msg["interactive"]:
            btn = msg["interactive"]["button_reply"]["id"]

            if btn == "speak_mr_y":
                update_chat_state(from_number, mode="name")
                await send_whatsapp_message(
                    from_number,
                    "😊 Sure! To connect you with Mr. Y, may I know your good name?"
                )
                return

            if btn == "about_courses":
                update_chat_state(from_number, mode="select_course")

                url = f"{BASE_URL}/{PHONE_ID}/messages"
                headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

                payload = {
                    "messaging_product": "whatsapp",
                    "to": from_number,
                    "type": "interactive",
                    "interactive": {
                        "type": "list",
                        "body": {"text": "📚 Please choose a course:"},
                        "footer": {"text": "Aryu Academy"},
                        "action": {
                            "button": "Select Course",
                            "sections": [
                                {
                                    "title": "Available Courses",
                                    "rows": [
                                        {"id": "python", "title": "Python Programming"},
                                        {"id": "fullstack", "title": "Full Stack Dev"},
                                        {"id": "uiux", "title": "UI/UX Design"},
                                        {"id": "reactjs", "title": "React JS"},
                                        {"id": "mern", "title": "MERN Stack"},
                                        {"id": "others", "title": "Other Course"},
                                    ],
                                }
                            ],
                        },
                    },
                }

                async with httpx.AsyncClient(timeout=20.0) as client:
                    r = await client.post(url, json=payload, headers=headers)

                print("COURSE LIST STATUS:", r.status_code)
                print("COURSE LIST RESPONSE:", r.text)
                return

        if msg_type == "text":
            user_text = msg["text"]["body"].strip()
        else:
            user_text = ""
        
        if not user_text:
            return

        # Greetings
        print("✅ BEFORE GREETING CHECK:", user_text, state)
        if user_text.lower() in ["hi", "hello", "hey", "hai", "hii", "hi!", "hello!", "hey!"]:
            if not state.get("greeted"):
                update_chat_state(from_number, greeted=True)
                print("🟢 GREETING TRIGGERED, SENDING BUTTONS")
                await send_interactive_buttons(from_number)
            return

        if state.get("mode") == "name":
            update_chat_state(from_number, name=user_text)
            update_chat_state(from_number, mode="phone")

            await send_whatsapp_message(
                from_number,
                f"Thank you, {user_text} 😊\nMay I have your mobile number?"
            )
            return

        if state.get("mode") == "phone":
            update_chat_state(from_number, phone=user_text)
            update_chat_state(from_number, mode="time")

            await send_whatsapp_message(
                from_number,
                "Perfect! At what time should Mr. Y contact you? 😊"
            )
            return

        if state.get("mode") == "time":
            update_chat_state(from_number, time=user_text)
            state = get_chat_state(from_number)
            name = state.get("name")
            phone = state.get("phone")

            update_chat_state(from_number, mode=None)

            await send_whatsapp_message(
                from_number,
                f"Great! Your details are noted:\n\n"
                f"👤 Name: {name}\n"
                f"📞 Phone: {phone}\n"
                f"⏰ Time: {user_text}\n\n"
                "Mr. Y will call you 😊"
            )

            await notify_admin_lead(name, phone, user_text, from_number)
            return

        if state.get("mode") == "other_course":
            textkey = user_text.lower().replace(" ", "")

            for key, info in COURSES.items():
                if key in textkey or info["name"].lower().replace(" ", "") in textkey:
                    update_chat_state(from_number, mode=None)
                    await send_whatsapp_message(
                        from_number,
                        f"📘 *{info['name']}*\n\n{info['details']}\n\n"
                        "Would you like the full syllabus or a free demo session? 🙂"
                    )
                    return

            # No match
            all_courses = "\n".join(f"• {c['name']}" for c in COURSES.values())

            await send_whatsapp_message(
                from_number,
                "❌ Sorry, we don’t offer that course.\n\n"
                "Here are our available courses:\n\n"
                f"{all_courses}\n\n"
                "Please choose one from the list 😊"
            )
            update_chat_state(from_number, mode=None)
            return



        # ============================================================
        # COURSE SELECTION MODE (old logic kept)
        # ============================================================
        if state.get("mode") == "select_course":
            textkey = user_text.lower().replace(" ", "")

            if textkey in COURSES:
                info = COURSES[textkey]
                update_chat_state(from_number, mode=None)

                await send_whatsapp_message(
                    from_number,
                    f"📘 *{info['name']}*\n\n{info['details']}\n\n"
                    "Would you like the full syllabus or a free demo session? 🙂"
                )
                return

            for k, info in COURSES.items():
                if k in textkey or info["name"].lower().replace(" ", "") in textkey:
                    update_chat_state(from_number, mode="after_course")   # ← FIX
                    update_chat_state(from_number, selected_course=k)
                    await send_whatsapp_message(
                        from_number,
                        f"📘 *{info['name']}*\n\n{info['details']}\n\n"
                        "Would you like the full syllabus or a free demo session? 🙂"
                    )
                    return

            all_courses = "\n".join(f"• {c['name']}" for c in COURSES.values())

            await send_whatsapp_message(
                from_number,
                "❌ Sorry, we don't offer that course.\n\n"
                "Here are the available courses:\n\n"
                f"{all_courses}\n\n"
                "Please type the course correctly 😊"
            )
            return

        # ============================================================
        # AFTER COURSE SELECTION: SYLLABUS or DEMO HANDLING
        # ============================================================
        if state.get("mode") == "after_course":
            text = user_text.lower()

            state = get_chat_state(from_number)
            selected_course = state.get("selected_course")
            course_data = COURSES.get(selected_course)

            # User asks for syllabus
            if "syllabus" in text or "pdf" in text or "notes" in text:
                file_url = course_data["syllabus"]
                file_name = f"{course_data['name']} - Syllabus.pdf"

                await send_document(from_number, file_url, file_name)

                update_chat_state(from_number, mode=None)
                update_chat_state(from_number, selected_course=None)
                return

            # User asks for demo
            if "demo" in text:
                update_chat_state(from_number, mode=None)
                update_chat_state(from_number, selected_course=None)
                await send_whatsapp_message(
                    from_number,
                    "Awesome! 😊\nPlease share your *name* *phone no* and *preferred time* so we can arrange a demo session for you."
                )
                return

            # User says something else
            await send_whatsapp_message(
                from_number,
                "I can send you the *full syllabus PDF* or arrange a *free demo session* 😊\nWhich one would you like?"
            )
            return
        
        # ============================================================
        # FALLBACK → AI
        # ============================================================

        try:
            # Save user message
            save_chat_message(from_number, "user", user_text)

            # Load chat history
            history = get_chat_history(from_number)
            context = "\n".join(
                f"{m['role']}: {m['content']}" for m in history
            )

            course_keywords = ["course", "training", "class", "learn", "coaching"]

            textkey = user_text.lower()

            if any(word in textkey for word in course_keywords):
                for key, info in COURSES.items():
                    if key in textkey or info["name"].lower() in textkey:
                        break
                else:
                    # No matching course found
                    all_courses = "\n".join(f"• {c['name']}" for c in COURSES.values())
                    await send_whatsapp_message(
                        from_number,
                        "I understand what you’re looking for 😊\n\n"
                        "Currently, we offer the following courses:\n\n"
                        f"{all_courses}\n\n"
                        "If you’re interested in any of these, I’ll be happy to help 👍"
                    )
                    return

            # Call AI with memory
            llm_reply = await call_llm(
                user_message=user_text,
                agent_name="whatsapp_bot",
                db=db
            )

            if not llm_reply or not isinstance(llm_reply, str):
                llm_reply = "I'm here to help 😊 Could you repeat that?"

            # Save AI reply
            save_chat_message(from_number, "assistant", llm_reply)

            # Send to WhatsApp
            await send_whatsapp_message(from_number, llm_reply)

        except Exception as e:
            print("LLM ERROR:", e)
            await send_whatsapp_message(
                from_number,
                "Sorry, my system took too long to respond. Please try again 😊"
            )

    except Exception as e:
        print("Error:", repr(e))
