from app.models.ai_interaction import AIInteraction


async def log_ai_interaction(
    db,
    *,
    agent_name: str,
    mode: str,
    project_name: str,
    input_payload: dict,
    ai_raw_response: str,
    ai_parsed_response: dict | None,
    created_by: str | None = None
):
    obj = AIInteraction(
        agent_name=agent_name,
        mode=mode,
        project_name=project_name,
        input_payload=input_payload,
        ai_raw_response=ai_raw_response,
        ai_parsed_response=ai_parsed_response,
        created_by=created_by
    )

    db.add(obj)
    await db.commit()
