import os
from groq import Groq
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.prompt_service import get_prompt

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

groq_client = Groq(api_key=GROQ_API_KEY)


async def call_llm(
    *,
    user_message: str,
    agent_name: str,
    db: AsyncSession,
    model: str = "groq",  # kept for compatibility
):
    # load system prompt
    system_prompt = await get_prompt(db, agent_name)
    if not system_prompt:
        system_prompt = "You are YURA, a helpful AI assistant built by Aryu Enterprises."

    SYSTEM_SAFE = system_prompt[:3500]
    USER_SAFE = user_message[:1500]

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,   # model is fixed here
            messages=[
                {"role": "system", "content": SYSTEM_SAFE},
                {"role": "user", "content": USER_SAFE},
            ],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        print("❌ GROQ ERROR:", repr(e))
        return "The system is taking a bit longer. Please try again 😊"


