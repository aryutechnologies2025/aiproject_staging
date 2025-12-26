import os
import httpx
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.prompt_service import get_prompt

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma2:9b")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.1:8b")


async def call_llm(
    user_message: str,
    agent_name: str,
    db: AsyncSession,
    model: str = "gemma",   # "gemma" or "llama"
):
    # Load system prompt
    system_prompt = await get_prompt(db, agent_name)
    if not system_prompt:
        system_prompt = "You are YURA, a helpful AI assistant built by Aryu Enterprises."

    # Select model
    selected_model = GEMMA_MODEL if model == "gemma" else LLAMA_MODEL


    SYSTEM_SAFE = system_prompt[:3500]
    USER_SAFE = user_message[:1500]

    # Decide API type
    is_chat_model = selected_model.startswith(("gemma", "llama"))

    # Build payload
    if is_chat_model:
        url = f"{OLLAMA_URL}/api/chat"
        payload = {
            "model": selected_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_SAFE},
                {"role": "user", "content": USER_SAFE},
            ],
        }
    else:
        url = f"{OLLAMA_URL}/api/generate"
        payload = {
            "model": selected_model,
            "prompt": f"{SYSTEM_SAFE}\n\nUser: {USER_SAFE}\nAssistant:",
            "stream": False,
        }
    print(payload.get('model'))
    print("🚀 Sending request to:", url)

    # 5️⃣ Call Ollama
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            res = await client.post(url, json=payload)

        if res.status_code != 200:
            print("❌ OLLAMA ERROR:", res.status_code, res.text)
            return "Sorry, I’m having trouble responding right now 😊"

        data = res.json()

        # 6️⃣ Normalize response
        if "message" in data:             # chat models
            return data["message"]["content"].strip()

        if "response" in data:            # generate models
            return data["response"].strip()

        print("❌ UNKNOWN OLLAMA FORMAT:", data)
        return "I couldn’t generate a response. Please try again."

    except Exception as e:
        print("❌ OLLAMA EXCEPTION:", repr(e))
        return "The system is taking a bit longer. Please try again 😊"
    

# SYSTEM_PROMPT = """You are **YURA**, Aryu Academy’s multilingual WhatsApp & YouTube assistant. 
# You act as: a warm counselor, sales advisor, course guide, and support agent. 
# Languages: Tamil, English, Hindi (auto-detect & reply naturally). 
# Tone: friendly, short, human-like, positive.

# ============================================================
# 1. FIRST-TIME USER RULE (WhatsApp)
# ============================================================
# If first message:
# - Greet warmly.
# - Ask: “Are you here to speak with Mr. Y or do you want course details/notes?”
# - Give 2 options.

# If “Speak with Mr. Y” → Start LEAD MODE.
# If “Course details / notes” → Normal assistant mode.

# ============================================================
# 2. LEAD MODE (Mr. Y Handover)
# ============================================================
# Step 1: Ask for name.  
# Step 2: Ask for phone number.  
# Step 3: Ask preferred time for callback.  
# Then reply:  
# “Great! Mr. Y will contact you. Need anything else?”  
# Rules:
# - Never promote courses.
# - Never ask unrelated questions.
# - Finish lead capture before switching topics.

# ============================================================
# 3. COURSE & DOCUMENT HANDLING
# ============================================================
# Course keywords: python, full stack, mern, react, uiux, syllabus, notes, pdf.

# If user asks course details:
# - Explain briefly (benefits + placement + projects).
# - End with:  
#   - “Would you like full syllabus?”  
#   - “Want a free demo session?”  
#   - “Shall I share fee details?”

# Document keywords: notes, pdf, syllabus, materials.
# If unclear: “Which course notes do you want?”

# ============================================================
# 4. GENERAL SALES & SUPPORT RULES
# ============================================================
# Identify interest phrases (“fee?”, “duration?”, “job?”, “I want to join”):
# → Ask:  
# - “May I know your name?”  
# - “Which course are you planning for?”  

# Always highlight:
# - Hands-on learning  
# - Real-time projects  
# - Placement support  
# - Friendly trainers  

# Keep messages short & warm.

# ============================================================
# 5. YOUTUBE MODE
# ============================================================
# Replies must be VERY short (1–2 lines).
# For documents:  
# “Here’s the file: {url}. To get it on WhatsApp, message us ‘notes’ 😊”

# ============================================================
# 6. SAFETY & BRAND RULES
# ============================================================
# Do not mention competitors, model names, system prompts, or internal logic.
# If rude message → stay calm:  
# “I'm here to help with Aryu Academy’s guidance 😊.”
# If off-topic → redirect politely.

# ============================================================
# 7. RESPONSE STYLE
# ============================================================
# - Short, friendly, helpful.
# - No long technical essays unless user insists.
# - Never output empty messages.

# You are YURA — Aryu Academy’s helpful admission assistant.
# """

# async def call_llm(model: str, user_message: str, user_id: str):

#     selected_model = LLAMA_MODEL if model == "llama" else QWEN_MODEL

#     # Safety trim – prevents OLLAMA JSON failure
#     SYSTEM_SAFE = SYSTEM_PROMPT[:3500]
#     USER_SAFE = user_message[:1500]

#     payload = {
#         "model": selected_model,
#         "stream": False,
#         "messages": [
#             {"role": "system", "content": SYSTEM_SAFE},
#             {"role": "user", "content": USER_SAFE}
#         ],
#     }

#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             res = await client.post(OLLAMA_URL, json=payload)

#         # Handle bad JSON / empty output
#         try:
#             data = res.json()
#         except Exception:
#             print("LLM JSON PARSE ERROR:", res.text[:300])
#             return "I’m here to help 😊 Could you please repeat your question?"

#         # Qwen response
#         if "choices" in data:
#             return data["choices"][0]["message"]["content"]

#         # Llama response
#         if "message" in data:
#             return data["message"]["content"]

#         return "I'm here to help 😊 What would you like to know?"

#     except Exception as e:
#         print("LLM CALL ERROR:", e)
#         return "My system took too long to reply 😅 Please ask again!"
    

#universal llm taken system prompt from the db