import os
import logging
import httpx
import asyncio
import time
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"

MAX_RETRIES = 3
INITIAL_BACKOFF = 2
MAX_CONCURRENT_CALLS = 2

ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

_token_usage = {"prompt": 0, "completion": 0, "calls": 0}


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
        self.call_count = {"gemini": 0, "groq": 0}
        self.error_count = {"gemini": 0, "groq": 0}

    async def call(self, prompt: str, system_prompt: str = "", max_output_tokens: int = 1024, **kwargs) -> str:
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
            "token_usage": _token_usage
        }


_manager = None


def get_improved_ai_client():
    global _manager
    if _manager is None:
        _manager = ImprovedAIClientManager()
    return _manager


async def call_ai(prompt: str, system_prompt: str = "", max_output_tokens: int = 1024, **kwargs) -> str:
    return await get_improved_ai_client().call(prompt, system_prompt, max_output_tokens, **kwargs)
