# hrms_ai/rules/leave_rules.py

from hrms_ai.schema.decision_schema import Decision
from hrms_ai.schema.policy_schema import POLICY_CONFIG


def evaluate_leave(request: dict) -> Decision:
    leave_type = request.get("leaveType")
    requested_days = request.get("requestedDays")
    available = request.get("availableLeaves")

    policy = POLICY_CONFIG["leave"].get(leave_type)

    if not policy:
        return Decision(
            decision_type="reject",
            confidence=0.9,
            reason="Invalid leave type"
        )

    if not policy.get("allow_negative_balance") and requested_days > available:
        return Decision(
            decision_type="reject",
            confidence=0.95,
            reason="Requested leave exceeds available balance"
        )

    return Decision(
        decision_type="escalate",
        confidence=0.85,
        reason="Requires manager approval",
        auto_execute=False
    )
