from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    POSTGRES_URL: str
    OLLAMA_API_URL: str
    LLAMA_MODEL: str = "llama3.1:8b"
    QWEN_MODEL: str = "qwen2.5:14b"

    # OPTIONAL FIELDS
    whatsapp_token: str | None = None
    whatsapp_phone_id: str | None = None
    whatsapp_api_url: str | None = None
    redis_url: str | None = None
    admin_email: str | None = None
    admin_phone: str | None = None

    class Config:
        env_file = ".env"
        extra = "allow"   # ALLOW extra keys from .env


settings = Settings()
