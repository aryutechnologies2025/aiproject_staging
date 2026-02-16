# hrms_ai/rules/expense_rules.py

from hrms_ai.schema.decision_schema import Decision
from hrms_ai.schema.policy_schema import POLICY_CONFIG


def evaluate_expense(claim: dict) -> Decision:
    amount = claim.get("amount", 0)
    receipts = claim.get("receipts", False)

    policy = POLICY_CONFIG["expense"]

    if policy["requires_receipt"] and not receipts:
        return Decision(
            decision_type="reject",
            confidence=0.95,
            reason="Receipt missing"
        )

    if amount <= policy["max_auto_approval_limit"]:
        return Decision(
            decision_type="approve",
            confidence=0.9,
            reason="Within auto-approval limit",
            auto_execute=True
        )

    return Decision(
        decision_type="escalate",
        confidence=0.8,
        reason="Exceeds auto approval limit"
    )
