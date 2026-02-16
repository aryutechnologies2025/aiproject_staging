# hrms_ai/llm/inference.py

import json
import re
from app.services.llm_client import call_llm


def extract_json(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except:
        return None


async def run_llm_analysis(prompt: str, db=None):
    raw = await call_llm(
        user_message=prompt,
        agent_name="hrms_management",
        db=db
    )

    parsed = extract_json(raw)

    if parsed:
        return parsed

    return {
        "risk_score": 0.5,
        "reason": "LLM response parsing failed"
    }
