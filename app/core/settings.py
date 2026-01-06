from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # =========================
    # CORE APP SETTINGS
    # =========================
    ENV: str = "production"

    # =========================
    # DATABASE
    # =========================
    POSTGRES_URL: str

    # =========================
    # LLM PROVIDER
    # =========================
    LLM_PROVIDER: str = "groq"  # groq | ollama

    # ---- GROQ ----
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # ---- OLLAMA (optional fallback) ----
    OLLAMA_API_URL: Optional[str] = None
    OLLAMA_MODEL: str = "llama3.1:8b"
    QWEN_MODEL: str = "qwen2.5:14b"

    # =========================
    # REDIS
    # =========================
    REDIS_URL: Optional[str] = None

    # =========================
    # WHATSAPP
    # =========================
    WHATSAPP_TOKEN: Optional[str] = None
    WHATSAPP_PHONE_ID: Optional[str] = None
    WHATSAPP_API_URL: Optional[str] = None

    # =========================
    # ADMIN
    # =========================
    ADMIN_EMAIL: Optional[str] = None
    ADMIN_PHONE: Optional[str] = None

    # =========================
    # Pydantic v2 config
    # =========================
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
        case_sensitive=False
    )


settings = Settings()
