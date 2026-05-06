import json
from app.core.config import REDIS

MEETING_TTL = 60 * 30  # 30 minutes

MEETING_FIELDS = ["name", "phone", "email", "datetime", "purpose"]

def get_meeting_state(session_id: str):
    data = REDIS.get(f"meeting:{session_id}")
    return json.loads(data) if data else {"mode": False, "data": {}}

def save_meeting_state(session_id: str, state: dict):
    REDIS.setex(
        f"meeting:{session_id}",
        MEETING_TTL,
        json.dumps(state)
    )

def next_meeting_step(data: dict):
    for field in MEETING_FIELDS:
        if field not in data:
            return field
    return None
