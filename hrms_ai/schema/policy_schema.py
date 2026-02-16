# hrms_ai/schema/policy_schema.py

POLICY_CONFIG = {
    "leave": {
        "annual": {
            "requires_manager_approval": True,
            "allow_negative_balance": False
        },
        "sick": {
            "requires_manager_approval": True,
            "medical_certificate_required_after_days": 2
        }
    },
    "expense": {
        "requires_receipt": True,
        "max_auto_approval_limit": 10000
    },
    "task": {
        "overdue_days_threshold": 5,
        "auto_followup_after_days": 3
    },
    "escalation": {
        "high_risk_overdue_days": 10
    }
}
