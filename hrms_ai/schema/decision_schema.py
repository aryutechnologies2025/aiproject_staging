# hrms_ai/schema/decision_schema.py

from pydantic import BaseModel
from typing import Optional, Dict, Any


class Decision(BaseModel):
    decision_type: str  # approve | reject | escalate | notify | monitor
    confidence: float
    reason: str
    auto_execute: bool = False
    metadata: Optional[Dict[str, Any]] = None
