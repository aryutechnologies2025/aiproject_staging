"""
ai_client.py — Unified AI Client for resume_builder powered exclusively by Google Gemini.

Groq has been completely removed in Phase 2 in compliance with the architecture directive.
All resume parsing, extraction, and ATS semantic evaluations route through Gemini.
"""

import asyncio
import hashlib
import logging
import os
import time
from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel
from dotenv import load_dotenv

from app.modules.resume_builder.gemini_client import get_gemini_client, GeminiServiceError
from app.modules.resume_builder.telemetry import log_ai_usage

load_dotenv()
logger = logging.getLogger("resume_builder.ai_client")

# Optional in-memory response cache (Process lifetime only, NO database dependency)
_AI_RESPONSE_CACHE: Dict[str, str] = {}
_cache_stats = {"hits": 0, "misses": 0}


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


USE_AI_CACHE = _env_flag("USE_AI_CACHE") or _env_flag("AI_DEV_MODE")


def _make_cache_key(prompt: str, system_prompt: str, explicit_key: Optional[str]) -> str:
    if explicit_key:
        return f"k:{explicit_key}"
    raw = f"{system_prompt}\n---\n{prompt}"
    return f"h:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def clear_ai_cache() -> int:
    """Wipe the in-memory AI response cache. Returns entries cleared."""
    n = len(_AI_RESPONSE_CACHE)
    _AI_RESPONSE_CACHE.clear()
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0
    return n


class ImprovedAIClientManager:
    """
    Unified AI client manager backed by Google Gemini SDK.
    """

    def __init__(self):
        self.gemini = get_gemini_client()
        self.call_count = {"gemini": 0, "cache": 0}
        self.error_count = {"gemini": 0}

    async def call(
        self,
        prompt: Union[str, list],
        system_prompt: str = "",
        max_output_tokens: int = 2048,
        response_schema: Optional[Type[BaseModel]] = None,
        operation: str = "ai_generation",
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        document_bytes: Optional[bytes] = None,
        document_mime_type: Optional[str] = None,
        **kwargs,
    ) -> str:
        cache_key = None
        if USE_AI_CACHE and isinstance(prompt, str) and not document_bytes:
            cache_key = _make_cache_key(prompt, system_prompt, kwargs.get("cache_key"))
            cached = _AI_RESPONSE_CACHE.get(cache_key)
            if cached is not None:
                _cache_stats["hits"] += 1
                self.call_count["cache"] += 1
                logger.info("[AI cache] hit — serving cached response")
                return cached
            _cache_stats["misses"] += 1

        try:
            result = await self.gemini.generate(
                prompt=prompt,
                system_instruction=system_prompt if system_prompt else None,
                max_output_tokens=max_output_tokens,
                response_schema=response_schema,
                operation=operation,
                user_id=user_id,
                request_id=request_id,
                document_bytes=document_bytes,
                document_mime_type=document_mime_type,
            )
            self.call_count["gemini"] += 1

            if USE_AI_CACHE and cache_key and result:
                _AI_RESPONSE_CACHE[cache_key] = result

            return result

        except Exception as e:
            self.error_count["gemini"] += 1
            logger.error(f"[AIClient] Gemini call error: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "provider": "gemini",
            "use_ai_cache": USE_AI_CACHE,
            "cache_size": len(_AI_RESPONSE_CACHE),
            "cache_stats": dict(_cache_stats),
        }


_manager: Optional[ImprovedAIClientManager] = None


def get_improved_ai_client() -> ImprovedAIClientManager:
    global _manager
    if _manager is None:
        _manager = ImprovedAIClientManager()
    return _manager


async def call_ai(
    prompt: Union[str, list],
    system_prompt: str = "",
    max_output_tokens: int = 2048,
    cache_key: Optional[str] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    operation: str = "ai_generation",
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    document_bytes: Optional[bytes] = None,
    document_mime_type: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Universal call_ai entry point for all resume_builder and ATS modules.
    Exclusively powered by Google Gemini.
    """
    if cache_key is not None:
        kwargs["cache_key"] = cache_key

    return await get_improved_ai_client().call(
        prompt=prompt,
        system_prompt=system_prompt,
        max_output_tokens=max_output_tokens,
        response_schema=response_schema,
        operation=operation,
        user_id=user_id,
        request_id=request_id,
        document_bytes=document_bytes,
        document_mime_type=document_mime_type,
        **kwargs,
    )