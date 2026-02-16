# hrms_ai/monitoring/daily_monitor.py

from hrms_ai.engine.manager_ai import HRMSManagerAI


async def run_daily_monitor(tasks: list):
    ai = HRMSManagerAI()
    alerts = []

    for task in tasks:
        decision = ai.process_task(task)
        if decision.auto_execute:
            alerts.append({
                "task": task.get("title"),
                "action": decision.decision_type,
                "reason": decision.reason
            })

    return alerts
