# hrms_ai/rules/compliance_rules.py

from hrms_ai.schema.decision_schema import Decision


def evaluate_compliance(record: dict) -> Decision:
    if record.get("status") == "expired":
        return Decision(
            decision_type="escalate",
            confidence=0.95,
            reason="Compliance expired",
            auto_execute=True
        )

    return Decision(
        decision_type="monitor",
        confidence=0.8,
        reason="Compliance valid"
    )
