# hrms_ai/llm/prompt_templates.py

def build_risk_analysis_prompt(data: dict) -> str:
    return f"""
You are an HRMS AI risk evaluator.

Analyze the following HRMS state and:
1. Return risk_score (0 to 1)
2. Short reasoning (1-2 sentences)

Respond strictly in JSON:

{{
  "risk_score": 0.0,
  "reason": ""
}}

INPUT:
{data}
""".strip()


def build_project_summary_prompt(data: dict) -> str:
    return f"""
You are an HRMS AI manager.

Summarize project progress concisely.
No generic phrases.

INPUT:
{data}
""".strip()
