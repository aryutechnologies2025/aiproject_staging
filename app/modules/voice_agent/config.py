"""
config.py — Voice Agent Configuration
All values come from environment variables. No secrets are hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── STT (Sarvam AI — Tamil-first) ────────────────────────────────────────────
SARVAM_API_KEY        = os.getenv("SARVAM_API_KEY", "")
# Sarvam STT WebSocket endpoint (streaming, mulaw-8k compatible)
SARVAM_STT_WS_URL     = os.getenv(
    "SARVAM_STT_WS_URL",
    "wss://api.sarvam.ai/speech-to-text-streaming"
)
# Sarvam STT REST endpoint (for short utterance fallback)
SARVAM_STT_REST_URL   = os.getenv(
    "SARVAM_STT_REST_URL",
    "https://api.sarvam.ai/speech-to-text"
)
STT_LANGUAGE_CODE     = os.getenv("STT_LANGUAGE_CODE", "ta-IN")
# ms of silence before utterance is considered complete
STT_ENDPOINTING_MS    = int(os.getenv("STT_ENDPOINTING_MS", "600"))

# ── TTS (Sarvam AI) ───────────────────────────────────────────────────────────
SARVAM_TTS_URL        = os.getenv(
    "SARVAM_TTS_URL",
    "https://api.sarvam.ai/text-to-speech"
)
TTS_SPEAKER           = os.getenv("TTS_SPEAKER", "manisha")
TTS_MODEL             = os.getenv("TTS_MODEL", "bulbul:v2")
TTS_SAMPLE_RATE       = int(os.getenv("TTS_SAMPLE_RATE", "8000"))

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER          = os.getenv("LLM_PROVIDER", "groq")      # groq | gemini
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL            = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL          = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Telephony (Vobiz) ─────────────────────────────────────────────────────────
VOBIZ_API_KEY         = os.getenv("VOBIZ_API_KEY", "")
VOBIZ_AUTH_ID         = os.getenv("VOBIZ_AUTH_ID", "SA_1CQLY4CU")
VOBIZ_API_URL         = os.getenv("VOBIZ_API_URL", "https://api.vobiz.ai/api/v1")
VOBIZ_AUTH_TOKEN      = os.getenv("VOBIZ_AUTH_TOKEN", "")
VOBIZ_CALLER_ID       = os.getenv("VOBIZ_CALLER_ID", "")
SIMULATION_MODE       = os.getenv("SIMULATION_MODE", "false").lower() == "true"

# ── Infrastructure ────────────────────────────────────────────────────────────
REDIS_URL             = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL          = os.getenv("POSTGRES_URL", "")
REDIS_SESSION_TTL     = int(os.getenv("REDIS_SESSION_TTL", "1800"))

# ── SMS (MSG91) ───────────────────────────────────────────────────────────────
MSG91_AUTH_KEY        = os.getenv("MSG91_AUTH_KEY", "")
MSG91_SENDER_ID       = os.getenv("MSG91_SENDER_ID", "VAGENT")
MSG91_TEMPLATE_ID     = os.getenv("MSG91_TEMPLATE_ID", "")

# ── Google Calendar ───────────────────────────────────────────────────────────
GOOGLE_CALENDAR_CREDENTIALS = os.getenv("GOOGLE_CALENDAR_CREDENTIALS", "credentials.json")
GOOGLE_CALENDAR_ID    = os.getenv("GOOGLE_CALENDAR_ID", "359568931258-oe92pujqf9mff808morht2spv3aer05a.apps.googleusercontent.com")

# ── Call Behaviour ────────────────────────────────────────────────────────────
MAX_CALL_ATTEMPTS         = int(os.getenv("MAX_CALL_ATTEMPTS", "3"))
RECALL_AFTER_HOURS        = int(os.getenv("RECALL_AFTER_HOURS", "4"))
CALL_TIMEOUT_SECONDS      = int(os.getenv("CALL_TIMEOUT_SECONDS", "180"))
INTERVIEW_SLOTS_LOOKAHEAD_DAYS = int(os.getenv("INTERVIEW_SLOTS_LOOKAHEAD_DAYS", "3"))
INTERVIEW_DURATION_MINUTES     = int(os.getenv("INTERVIEW_DURATION_MINUTES", "30"))
MAX_CONCURRENT_CALLS      = int(os.getenv("MAX_CONCURRENT_CALLS", "5"))

# ── Paths ─────────────────────────────────────────────────────────────────────
UPLOADS_DIR           = os.getenv("UPLOADS_DIR", "./uploads")
PUBLIC_BASE_URL       = os.getenv("PUBLIC_BASE_URL", "https://landscape-scrubber-unsaved.ngrok-free.dev")
