# hrms_ai/engine/manager_ai.py

from hrms_ai.rules.leave_rules import evaluate_leave
from hrms_ai.rules.expense_rules import evaluate_expense
from hrms_ai.rules.task_rules import evaluate_task
from hrms_ai.rules.compliance_rules import evaluate_compliance
from hrms_ai.engine.decision_fusion import fuse
from hrms_ai.schema.decision_schema import Decision
from hrms_ai.llm.prompt_templates import build_risk_analysis_prompt
from hrms_ai.llm.inference import run_llm_analysis


class HRMSManagerAI:

    def process_leave(self, data: dict) -> Decision:
        return evaluate_leave(data)

    def process_expense(self, data: dict) -> Decision:
        return evaluate_expense(data)

    async def process_task(self, data: dict, db=None) -> Decision:
        rule_decision = evaluate_task(data)

        prompt = build_risk_analysis_prompt(data)
        llm_output = await run_llm_analysis(prompt, db=db)

        final_decision = fuse(rule_decision, llm_output)

        return final_decision

    def process_compliance(self, data: dict) -> Decision:
        return evaluate_compliance(data)

    def fuse_with_llm(self, rule_decision: Decision, llm_output: dict):
        return fuse(rule_decision, llm_output)
