# hrms_ai/engine/context_builder.py

def build_hrms_context(data: dict) -> str:
    return f"""
Analyze HRMS data and provide risk score (0-1).
INPUT:
{data}
""".strip()
