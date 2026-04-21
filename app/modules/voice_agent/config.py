import os
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

VOBIZ_API_KEY = os.getenv("VOBIZ_API_KEY", "vKvmHsM8gfHuO3ZRqdINpsZX5RV6OnGUE68F1qBWjzsQ36mybjInVaJzweGJzfUV")
VOBIZ_API_URL = os.getenv("VOBIZ_API_URL", "https://api.vobiz.com/v1")
VOBIZ_CALLER_ID = os.getenv("VOBIZ_CALLER_ID", "+911171366938")
VOBIZ_WEBSOCKET_URL = os.getenv("VOBIZ_WEBSOCKET_URL", "wss://media.vobiz.com/ws")
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "false").lower() == "true"

REDIS_URL = os.getenv("REDIS_URL")
DATABASE_URL = os.getenv("POSTGRES_URL")

MSG91_AUTH_KEY = os.getenv("MSG91_AUTH_KEY", "")
MSG91_SENDER_ID = os.getenv("MSG91_SENDER_ID", "VAGENT")
MSG91_TEMPLATE_ID = os.getenv("MSG91_TEMPLATE_ID", "")

GOOGLE_CALENDAR_CREDENTIALS = os.getenv("GOOGLE_CALENDAR_CREDENTIALS", "credentials.json")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_CALL_ATTEMPTS = int(os.getenv("MAX_CALL_ATTEMPTS", "3"))
RECALL_AFTER_HOURS = int(os.getenv("RECALL_AFTER_HOURS", "4"))
CALL_TIMEOUT_SECONDS = int(os.getenv("CALL_TIMEOUT_SECONDS", "180"))
DEEPGRAM_ENDPOINTING_MS = int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "400"))
REDIS_SESSION_TTL = int(os.getenv("REDIS_SESSION_TTL", "1800"))

INTERVIEW_SLOTS_LOOKAHEAD_DAYS = int(os.getenv("INTERVIEW_SLOTS_LOOKAHEAD_DAYS", "3"))
INTERVIEW_DURATION_MINUTES = int(os.getenv("INTERVIEW_DURATION_MINUTES", "30"))

UPLOADS_DIR = os.getenv("UPLOADS_DIR", "./uploads")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://telophasic-aliza-numerous.ngrok-free.dev")