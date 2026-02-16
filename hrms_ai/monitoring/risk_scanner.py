# hrms_ai/monitoring/risk_scanner.py

async def compute_risk_score(data: dict) -> float:
    tasks = data.get("tasks", [])
    overdue = sum(1 for t in tasks if t.get("status") == "pending")
    total = len(tasks) or 1
    return min(overdue / total, 1.0)
