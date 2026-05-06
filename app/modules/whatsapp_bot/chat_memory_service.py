import json
from app.core.config import REDIS

CHAT_TTL = 60 * 30  # 30 minutes


def get_chat_history(session_id: str) -> list:
    data = REDIS.get(f"chat:{session_id}")
    return json.loads(data) if data else []


def save_chat_message(session_id: str, role: str, content: str):
    key = f"chat:{session_id}"

    data = REDIS.get(key)
    history = json.loads(data) if data else []

    history.append({"role": role, "content": content})

    # keep last 6 messages only (PERFORMANCE)
    history = history[-6:]

    REDIS.setex(key, CHAT_TTL, json.dumps(history))
