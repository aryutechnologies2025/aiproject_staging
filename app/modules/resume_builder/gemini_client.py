"""
gemini_client.py — Official Google GenAI Python SDK integration for resume_builder.

Features:
- Uses official `google.genai` SDK (`from google import genai`).
- Direct document understanding (PDF bytes, text) and structured output via Pydantic/JSON Schema.
- Resilient async calls with exponential backoff for 429/503.
- Extracts usage_metadata (input, output, cached, total tokens).
- Emits structured telemetry on every call via telemetry.log_ai_usage.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types

from app.modules.resume_builder.telemetry import log_ai_usage

load_dotenv()
logger = logging.getLogger("resume_builder.gemini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_RESUME_MODEL = os.getenv("GEMINI_RESUME_MODEL", "gemini-3.1-flash-lite")
DEFAULT_ATS_MODEL = os.getenv("GEMINI_ATS_MODEL", "gemini-3.1-flash-lite")

MAX_RETRIES = 3
INITIAL_BACKOFF_SEC = 2.0


class GeminiServiceError(Exception):
    """Base exception for Gemini service errors."""
    pass


class GeminiRateLimitError(GeminiServiceError):
    """Raised when Gemini returns HTTP 429 / RESOURCE_EXHAUSTED."""
    pass


class GeminiClient:
    """
    Asynchronous client wrapper around the official Google GenAI SDK.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini calls will fail if invoked.")
            self._client = None
        else:
            self._client = genai.Client(api_key=self.api_key)

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    @property
    def default_model(self) -> str:
        return DEFAULT_RESUME_MODEL

    async def generate(
        self,
        *,
        prompt: Union[str, List[Any]],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        operation: str = "general_inference",
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        document_bytes: Optional[bytes] = None,
        document_mime_type: Optional[str] = None,
    ) -> str:
        """
        Execute an asynchronous generation request against Gemini with telemetry and retries.
        """
        if not self.is_configured:
            error_msg = "Gemini API key is not configured."
            log_ai_usage(
                provider="gemini",
                model=model or DEFAULT_RESUME_MODEL,
                operation=operation,
                status="failed",
                error_type="MISSING_API_KEY",
                user_id=user_id,
                request_id=request_id,
            )
            raise GeminiServiceError(error_msg)

        target_model = model or DEFAULT_RESUME_MODEL
        start_time = time.time()

        # Build contents list
        contents: List[Any] = []
        if document_bytes and document_mime_type:
            contents.append(
                types.Part.from_bytes(
                    data=document_bytes,
                    mime_type=document_mime_type,
                )
            )

        if isinstance(prompt, str):
            contents.append(prompt)
        elif isinstance(prompt, list):
            contents.extend(prompt)

        # Build GenerateContentConfig
        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if max_output_tokens:
            config_kwargs["max_output_tokens"] = max_output_tokens
        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_kwargs)

        last_exception = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config,
                )

                latency_ms = (time.time() - start_time) * 1000

                # Extract usage metadata
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0
                cached_tokens = 0

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                    total_tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0
                    cached_tokens = getattr(response.usage_metadata, "cached_content_token_count", 0) or 0

                # Telemetry logging (NO DB, NO PII)
                log_ai_usage(
                    provider="gemini",
                    model=target_model,
                    operation=operation,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    latency_ms=latency_ms,
                    status="success",
                    retry_count=attempt,
                    user_id=user_id,
                    request_id=request_id,
                )

                return response.text or ""

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str
                last_exception = e

                if attempt < MAX_RETRIES and is_rate_limit:
                    backoff = INITIAL_BACKOFF_SEC * (2 ** attempt)
                    logger.warning(
                        f"[Gemini] Rate limit hit on {operation} (attempt {attempt+1}), retrying in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                elif attempt < MAX_RETRIES and ("503" in err_str or "unavailable" in err_str or "timeout" in err_str):
                    backoff = INITIAL_BACKOFF_SEC * (2 ** attempt)
                    logger.warning(
                        f"[Gemini] Service error on {operation} (attempt {attempt+1}), retrying in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                else:
                    break

        latency_ms = (time.time() - start_time) * 1000
        error_type = "RATE_LIMIT" if "429" in str(last_exception) else "API_ERROR"

        log_ai_usage(
            provider="gemini",
            model=target_model,
            operation=operation,
            latency_ms=latency_ms,
            status="failed",
            error_type=error_type,
            retry_count=MAX_RETRIES,
            user_id=user_id,
            request_id=request_id,
        )

        logger.error(f"[Gemini] Operation '{operation}' failed after retries: {last_exception}")
        raise GeminiServiceError(f"Gemini generation failed: {str(last_exception)}")


# Singleton instance
_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
