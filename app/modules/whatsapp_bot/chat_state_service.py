import json
from app.core.config import REDIS

STATE_TTL = 60 * 60 * 24  # 24 hours


def get_chat_state(session_id: str) -> dict:
    data = REDIS.get(f"chat:state:{session_id}")
    return json.loads(data) if data else {}


def update_chat_state(session_id: str, **updates):
    key = f"chat:state:{session_id}"

    data = REDIS.get(key)
    state = json.loads(data) if data else {}

    state.update(updates)

    REDIS.setex(key, STATE_TTL, json.dumps(state))
