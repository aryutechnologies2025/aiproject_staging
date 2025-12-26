from core.config import REDIS
import json

def set_state(user, key, value, ttl=900):  # ttl = 15 minutes
    data = get_state(user)
    data[key] = value
    REDIS.setex(f"user:{user}", ttl, json.dumps(data))

def get_state(user):
    raw = REDIS.get(f"user:{user}")
    return json.loads(raw) if raw else {}

def clear_state(user):
    REDIS.delete(f"user:{user}")
