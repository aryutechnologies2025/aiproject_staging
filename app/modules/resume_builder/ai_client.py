import os
import logging
import httpx
import asyncio
import time
import hashlib
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

MAX_RETRIES = 3
INITIAL_BACKOFF = 2
MAX_CONCURRENT_CALLS = 2

ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

_token_usage = {"prompt": 0, "completion": 0, "calls": 0}


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# ── Development-mode controls ───────────────────────────────────────────────
# AI_DEV_MODE / MINIMIZE_AI_CALLS: enables BOTH response caching AND
# fail-fast-on-rate-limit behaviour (skip same-provider retries so one 429
# doesn't burn 3 extra Gemini requests before falling back to Groq).
# USE_AI_CACHE: enables ONLY the response cache, independent of retries.
# With no env vars set, behaviour is identical to the original implementation.
AI_DEV_MODE  = _env_flag("AI_DEV_MODE") or _env_flag("MINIMIZE_AI_CALLS")
USE_AI_CACHE = _env_flag("USE_AI_CACHE") or AI_DEV_MODE

# In-memory response cache: cache_key -> response text.
# Process-lifetime only — sufficient for local dev / repeated test uploads.
# Never consulted unless one of the flags above is enabled.
_AI_RESPONSE_CACHE: Dict[str, str] = {}
_cache_stats = {"hits": 0, "misses": 0}


def _make_cache_key(prompt: str, system_prompt: str, explicit_key: Optional[str]) -> str:
    if explicit_key:
        return f"k:{explicit_key}"
    raw = f"{system_prompt}\n---\n{prompt}"
    return f"h:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def clear_ai_cache() -> int:
    """Dev helper — wipe the in-memory AI response cache. Returns entries cleared."""
    n = len(_AI_RESPONSE_CACHE)
    _AI_RESPONSE_CACHE.clear()
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0
    return n


class RateLimitError(Exception):
    pass


class PayloadTooLargeError(Exception):
    pass


class BaseAIClient:
    def __init__(self, api_key: str, min_request_interval: float = 0.5):
        self.api_key = api_key
        self.timeout = httpx.Timeout(120.0)
        self.max_retries = MAX_RETRIES
        self.min_request_interval = min_request_interval
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def _enforce_rate_limit(self):
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
            self.last_request_time = time.time()

    async def _request_with_retry(self, func, *args, **kwargs):
        attempt = 0
        while attempt <= self.max_retries:
            try:
                async with ai_semaphore:
                    await self._enforce_rate_limit()
                    return await func(*args, **kwargs)
            except RateLimitError:
                if AI_DEV_MODE:
                    # Dev mode: never burn extra quota retrying a 429 against
                    # the same provider — fail fast so the manager can fall
                    # back to the secondary provider immediately.
                    logger.warning(
                        "Rate limit hit — AI_DEV_MODE active, skipping same-provider "
                        "retries to conserve free-tier quota"
                    )
                    raise
                if attempt == self.max_retries:
                    raise
                wait_time = min((INITIAL_BACKOFF ** (attempt + 2)) + (time.time() % 1), 60.0)
                logger.warning(f"Rate limit hit attempt {attempt + 1}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                attempt += 1
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                wait_time = INITIAL_BACKOFF ** (attempt + 1)
                logger.warning(f"Request failed attempt {attempt + 1}, retrying in {wait_time}s: {str(e)[:80]}")
                await asyncio.sleep(wait_time)
                attempt += 1


class GroqClient(BaseAIClient):
    def __init__(self, api_key: str):
        super().__init__(api_key, min_request_interval=4.0)

    async def call(self, prompt: str, system_prompt: str = "", max_output_tokens: int = 1024) -> str:
        async def _execute():
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt or "You are a resume parser. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": max_output_tokens
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(GROQ_ENDPOINT, json=payload, headers=headers)
                if response.status_code == 413:
                    raise PayloadTooLargeError("Groq Payload Too Large")
                if response.status_code == 429:
                    raise RateLimitError("Groq Rate Limit")
                if response.status_code != 200:
                    raise Exception(f"Groq Error: {response.status_code} - {response.text[:200]}")
                data = response.json()
                usage = data.get("usage", {})
                _token_usage["prompt"] += usage.get("prompt_tokens", 0)
                _token_usage["completion"] += usage.get("completion_tokens", 0)
                _token_usage["calls"] += 1
                return data["choices"][0]["message"]["content"]
        return await self._request_with_retry(_execute)


class GeminiClient(BaseAIClient):
    def __init__(self, api_key: str):
        super().__init__(api_key, min_request_interval=1.0)

    async def call(self, prompt: str, system_prompt: str = "", max_output_tokens: int = 1024) -> str:
        async def _execute():
            full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {prompt}" if system_prompt else prompt
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": max_output_tokens
                }
            }
            url = f"{GEMINI_ENDPOINT}?key={self.api_key}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 413:
                    raise PayloadTooLargeError("Gemini Payload Too Large")
                if response.status_code == 429:
                    raise RateLimitError("Gemini Rate Limit")
                if response.status_code != 200:
                    raise Exception(f"Gemini Error: {response.status_code} - {response.text[:200]}")
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return await self._request_with_retry(_execute)


class ImprovedAIClientManager:
    def __init__(self):
        self.gemini = GeminiClient(GEMINI_API_KEY) if GEMINI_API_KEY else None
        self.groq = GroqClient(GROQ_API_KEY) if GROQ_API_KEY else None
        self.call_count = {"gemini": 0, "groq": 0, "cache": 0}
        self.error_count = {"gemini": 0, "groq": 0}

    async def call(self, prompt: str, system_prompt: str = "", max_output_tokens: int = 1024, **kwargs) -> str:
        cache_key = None
        if USE_AI_CACHE:
            cache_key = _make_cache_key(prompt, system_prompt, kwargs.get("cache_key"))
            cached = _AI_RESPONSE_CACHE.get(cache_key)
            if cached is not None:
                _cache_stats["hits"] += 1
                self.call_count["cache"] += 1
                logger.info(f"[AI cache] hit — skipping provider call ({'dev' if AI_DEV_MODE else 'cache'} mode)")
                return cached
            _cache_stats["misses"] += 1

        use_gemini_first = kwargs.get("use_gemini_first", True)
        if use_gemini_first and self.gemini:
            providers = [("gemini", self.gemini), ("groq", self.groq)]
        elif self.groq:
            providers = [("groq", self.groq), ("gemini", self.gemini)]
        else:
            providers = [("gemini", self.gemini)]

        last_error = None
        for provider_name, client in providers:
            if not client:
                continue
            try:
                result = await client.call(prompt, system_prompt, max_output_tokens)
                self.call_count[provider_name] += 1
                if USE_AI_CACHE and cache_key:
                    _AI_RESPONSE_CACHE[cache_key] = result
                return result
            except RateLimitError as e:
                self.error_count[provider_name] += 1
                logger.warning(f"{provider_name} rate limited, trying fallback")
                last_error = e
            except Exception as e:
                self.error_count[provider_name] += 1
                logger.warning(f"{provider_name} failed: {str(e)[:80]}")
                last_error = e

        raise last_error or Exception("No AI clients available")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "token_usage": _token_usage,
            "ai_dev_mode": AI_DEV_MODE,
            "use_ai_cache": USE_AI_CACHE,
            "cache_size": len(_AI_RESPONSE_CACHE),
            "cache_stats": dict(_cache_stats),
        }


_manager = None


def get_improved_ai_client():
    global _manager
    if _manager is None:
        _manager = ImprovedAIClientManager()
    return _manager


async def call_ai(
    prompt: str,
    system_prompt: str = "",
    max_output_tokens: int = 1024,
    cache_key: Optional[str] = None,
    **kwargs,
) -> str:
    if cache_key is not None:
        kwargs["cache_key"] = cache_key
    return await get_improved_ai_client().call(prompt, system_prompt, max_output_tokens, **kwargs)