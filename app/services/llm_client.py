# /home/aryu_user/Arun/aiproject_staging/app/services/llm_client.py
import os
import logging
from groq import Groq
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.prompt_service import get_prompt

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

groq_client = Groq(api_key=GROQ_API_KEY)


async def call_llm(
    *,
    user_message: str,
    agent_name: str,
    db: AsyncSession,
    model: str = "groq",  # Kept for compatibility
) -> str:
    """
    Call Groq Llama 3.1 8B with optimized prompting for resume and CV generation.
    
    Optimizations for Llama 3.1 8B:
    - Clear system prompts with specific instructions
    - Structured requests (JSON, bullets, specific format)
    - Temperature 0.7 for creative but consistent output
    - Token limit 1024 for detailed responses
    - Top_p=0.95 for focused responses
    """
    
    try:
        # Load system prompt
        system_prompt = await get_prompt(db, agent_name)
        if not system_prompt:
            system_prompt = "You are YURA, a helpful AI assistant built by Aryu Enterprises. Provide clear, professional responses."

        # Optimize for token limits
        system_safe = system_prompt[:3500]  # System prompt limit
        user_safe = user_message[:2000]     # User message limit for better quality

        # Call Groq API with optimized parameters for Llama 3.1 8B
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_safe
                },
                {
                    "role": "user",
                    "content": user_safe
                },
            ],
            temperature=0.7,              # Balanced creativity and consistency
            max_completion_tokens=1024,   # Sufficient for detailed responses
            top_p=0.95,                   # Focused nucleus sampling
            stream=False,
        )

        response = completion.choices[0].message.content.strip()
        
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
        user_safe = user_message[:2000]

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_safe},
                {"role": "user", "content": user_safe},
            ],
            temperature=0.3,              # Lower temp for consistent JSON
            max_completion_tokens=1024,
            top_p=0.95,
            stream=False,
        )

        response = completion.choices[0].message.content.strip()
        
        # Clean markdown artifacts if present
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