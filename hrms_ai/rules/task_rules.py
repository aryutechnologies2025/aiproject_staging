# hrms_ai/rules/task_rules.py

from datetime import datetime
from hrms_ai.schema.decision_schema import Decision
from hrms_ai.schema.policy_schema import POLICY_CONFIG


def evaluate_task(task: dict) -> Decision:
    due = task.get("dueDate")
    status = task.get("status")

    if status in ["completed", "done"]:
        return Decision(
            decision_type="monitor",
            confidence=1.0,
            reason="Task already completed"
        )

    due_date = datetime.fromisoformat(due)
    pending_days = (datetime.utcnow() - due_date).days

    threshold = POLICY_CONFIG["task"]["overdue_days_threshold"]

    if pending_days > threshold:
        return Decision(
            decision_type="notify",
            confidence=0.9,
            reason="Task overdue",
            auto_execute=True,
            metadata={"pending_days": pending_days}
        )

    return Decision(
        decision_type="monitor",
        confidence=0.7,
        reason="Within allowed time"
    )
