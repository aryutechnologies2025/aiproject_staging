import os
import logging
import httpx
import json
import asyncio

from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite-001:generateContent"

MAX_RETRIES = 1
INITIAL_BACKOFF = 2

def estimate_tokens(text: str) -> int:
    # rough approximation: 1 token ≈ 4 chars OR 0.75 words
    return int(len(text.split()) * 1.3)

class RateLimitError(Exception):
    pass


class PayloadTooLargeError(Exception):
    pass


class GroqClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.timeout = httpx.Timeout(45.0)
        self.max_retries = MAX_RETRIES
    
    async def call(self, prompt: str, system_prompt: str = "", max_output_tokens: int = 1024) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_output_tokens
        }
        
        attempt = 0
        while attempt <= self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(GROQ_ENDPOINT, json=payload, headers=headers)
                    
                    if response.status_code == 413:
                        logger.error(f"Groq: Request too large (413)")
                        raise PayloadTooLargeError("Payload too large for Groq")
                    
                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after", str(INITIAL_BACKOFF ** (attempt + 1)))
                        try:
                            wait_time = int(retry_after)
                        except:
                            wait_time = INITIAL_BACKOFF ** (attempt + 1)
                        
                        logger.warning(f"Groq rate limited, waiting {wait_time}s")
                        raise RateLimitError(f"Rate limited, retry after {wait_time}s")
                    
                    if response.status_code != 200:
                        logger.error(f"Groq API error: {response.status_code} - {response.text[:200]}")
                        raise Exception(f"Groq API failed: {response.status_code}")
                    
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
            
            except (RateLimitError, PayloadTooLargeError):
                raise
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = INITIAL_BACKOFF ** (attempt + 1)
                    logger.warning(f"Groq attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)[:100]}")
                    await asyncio.sleep(wait_time)
                    attempt += 1
                else:
                    raise


class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.timeout = httpx.Timeout(45.0)
        self.max_retries = MAX_RETRIES
    
    async def call(self, prompt: str, system_prompt: str = "", max_output_tokens: int = 1024) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        input_tokens = estimate_tokens(full_prompt)
        logger.warning(f"[TOKEN DEBUG] Input tokens: {input_tokens}")

        
        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_output_tokens
            }
        }
        logger.warning(f"[TOKEN DEBUG] Output tokens: {max_output_tokens}")
        url = f"{GEMINI_ENDPOINT}?key={self.api_key}"
        
        attempt = 0
        while attempt <= self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    
                    if response.status_code == 413:
                        logger.error(f"Gemini: Request too large (413)")
                        raise PayloadTooLargeError("Payload too large for Gemini")
                    
                    if response.status_code == 429:
                        error_data = response.json()
                        retry_after = error_data.get("error", {}).get("details", [{}])[0].get("retryDelay", {}).get("seconds", INITIAL_BACKOFF ** (attempt + 1))
                        
                        wait_time = int(retry_after) if isinstance(retry_after, (int, float)) else INITIAL_BACKOFF ** (attempt + 1)
                        logger.warning(f"Gemini rate limited, waiting {wait_time}s")
                        raise RateLimitError(f"Rate limited, retry after {wait_time}s")
                    
                    if response.status_code != 200:
                        logger.error(f"Gemini API error: {response.status_code} - {response.text[:200]}")
                        raise Exception(f"Gemini API failed: {response.status_code}")
                    
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            
            except (RateLimitError, PayloadTooLargeError):
                raise
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = INITIAL_BACKOFF ** (attempt + 1)
                    logger.warning(f"Gemini attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)[:100]}")
                    await asyncio.sleep(wait_time)
                    attempt += 1
                else:
                    raise


class AIClientManager:
    def __init__(self):
        self.gemini = GeminiClient(GEMINI_API_KEY) if GEMINI_API_KEY else None
        self.groq = GroqClient(GROQ_API_KEY) if GROQ_API_KEY else None
    
    async def call(
        self,
        prompt: str,
        system_prompt: str = "",
        max_output_tokens: int = 1024,
        use_gemini_first: bool = False
    ) -> str:
        
        if use_gemini_first and self.gemini:
            try:
                logger.info("Calling Gemini API (primary)")
                return await self.gemini.call(prompt, system_prompt, max_output_tokens)
            except (RateLimitError, PayloadTooLargeError) as e:
                logger.warning(f"Gemini failed ({type(e).__name__}), trying Groq fallback")
                
            except Exception as e:
                logger.warning(f"Gemini call failed: {str(e)[:100]}, trying Groq")
        
        if self.groq:
            try:
                logger.info("Calling Groq API (fallback)")
                return await self.groq.call(prompt, system_prompt, max_output_tokens)
            except Exception as e:
                logger.error(f"Groq call failed: {str(e)[:100]}")
                raise
        
        raise Exception("No AI client available")


_manager = None


def get_ai_client() -> AIClientManager:
    global _manager
    if _manager is None:
        _manager = AIClientManager()
    return _manager


async def call_ai(
    prompt: str,
    system_prompt: str = "",
    max_output_tokens: int = 1024,
    use_gemini_first: bool = True
) -> str:
    manager = get_ai_client()
    return await manager.call(prompt, system_prompt, max_output_tokens, use_gemini_first)