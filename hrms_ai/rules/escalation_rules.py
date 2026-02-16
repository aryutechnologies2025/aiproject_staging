# hrms_ai/rules/escalation_rules.py

from hrms_ai.schema.decision_schema import Decision


def escalate_if_critical(risk_score: float) -> Decision:
    if risk_score > 0.8:
        return Decision(
            decision_type="escalate",
            confidence=0.95,
            reason="High risk detected",
            auto_execute=True
        )

    return Decision(
        decision_type="monitor",
        confidence=0.7,
        reason="Risk within tolerance"
    )
