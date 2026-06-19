# /home/aryu_user/Arun/aiproject_staging/app/services/llm_client.py
import os
import logging
import httpx
import json
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.prompt_service import get_prompt

logger = logging.getLogger(__name__)

# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://172.17.0.1:11434")
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds

_OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"


def _build_payload(
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    return {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": max_tokens,
        },
    }


async def call_llm(
    *,
    user_message: str,
    agent_name: str,
    db: AsyncSession,
    model: str = "ollama",  # Kept for compatibility
) -> str:

    try:
        # Load system prompt
        system_prompt = await get_prompt(db, agent_name)
        if not system_prompt:
            system_prompt = "You are YURA, a helpful AI assistant built by Aryu Enterprises. Provide clear, professional responses."

        # Optimize for token limits
        system_safe = system_prompt[:3500]   # System prompt limit
        user_safe   = user_message[:2000]    # User message limit for better quality

        payload = _build_payload(
            system_prompt=system_safe,
            user_message=user_safe,
            temperature=0.7,    # Balanced creativity and consistency
            max_tokens=4096,
        )

        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(_OLLAMA_CHAT_URL, json=payload)
            resp.raise_for_status()

        data = resp.json()
        response = data["message"]["content"].strip()

        if not response:
            logger.warning(f"Empty response from LLM for agent {agent_name}")
            return "Unable to generate response. Please try again."

        logger.info(f"LLM response received for {agent_name} ({len(response)} chars)")
        return response

    except Exception as e:
        logger.error(f"LLM Error ({agent_name}): {repr(e)}")
        return "The system is taking a bit longer. Please try again 😊"


async def call_llm_json(
    *,
    user_message: str,
    agent_name: str,
    db: AsyncSession,
) -> str:
    """
    Call LLM with JSON output guarantee.
    Includes JSON-specific formatting and validation.
    """

    try:
        # Load system prompt
        system_prompt = await get_prompt(db, agent_name)
        if not system_prompt:
            system_prompt = "You are a JSON-generating assistant. Output valid JSON only."

        # Add JSON instruction to user message
        json_instruction = "\n\nIMPORTANT: Output ONLY valid JSON. No markdown. No explanations."
        if json_instruction not in user_message:
            user_message = user_message + json_instruction

        system_safe = system_prompt[:3500]
        user_safe   = user_message[:2000]

        payload = _build_payload(
            system_prompt=system_safe,
            user_message=user_safe,
            temperature=0.3,    # Lower temp for consistent JSON
            max_tokens=1024,
        )
        # Ask Ollama to enforce JSON format at the sampler level
        payload["format"] = "json"

        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(_OLLAMA_CHAT_URL, json=payload)
            resp.raise_for_status()

        data = resp.json()
        response = data["message"]["content"].strip()

        # Clean markdown artifacts if present (defensive)
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        logger.info(f"JSON response received for {agent_name}")
        return response.strip()

    except Exception as e:
        logger.error(f"JSON LLM Error ({agent_name}): {repr(e)}")
        return "{}"