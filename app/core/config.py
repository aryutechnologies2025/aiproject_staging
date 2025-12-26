import redis
import os
from dotenv import load_dotenv

load_dotenv()

# Redis
REDIS = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    db=int(os.getenv("REDIS_DB", 2)),
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5,
)

# Contact details (from .env)
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "")

