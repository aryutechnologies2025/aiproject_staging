from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.prompt import Prompt


async def get_prompt(db: AsyncSession, agent_name: str) -> str:
    result = await db.execute(select(Prompt).where(Prompt.agent_name == agent_name))
    prompt = result.scalars().first()
    return prompt.system_prompt if prompt else ""


async def create_prompt(db: AsyncSession, agent_name: str, description: str, system_prompt: str):
    prompt = Prompt(
        agent_name=agent_name,
        description=description,
        system_prompt=(system_prompt)
    )
    db.add(prompt)
    await db.commit()
    await db.refresh(prompt)
    return prompt


async def update_prompt(db: AsyncSession, agent_name: str, description: str, system_prompt: str):
    result = await db.execute(select(Prompt).where(Prompt.agent_name == agent_name))
    prompt = result.scalars().first()

    if not prompt:
        return None

    prompt.description = description
    prompt.system_prompt = system_prompt

    await db.commit()
    await db.refresh(prompt)
    return prompt


async def list_prompts(db: AsyncSession):
    result = await db.execute(select(Prompt))
    return result.scalars().all()
