# hrms_ai/engine/decision_fusion.py

from hrms_ai.schema.decision_schema import Decision


def fuse(rule_decision: Decision, llm_insight: dict | None) -> Decision:
    if not llm_insight:
        return rule_decision

    llm_risk = llm_insight.get("risk_score", 0)

    if llm_risk > 0.85:
        return Decision(
            decision_type="escalate",
            confidence=0.95,
            reason="LLM detected critical risk",
            auto_execute=True
        )

    return rule_decision
