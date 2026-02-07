from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class AIInteractionBase(BaseModel):
    agent_name: str
    mode: str
    project_name: Optional[str]
    input_payload: Dict
    ai_raw_response: str
    ai_parsed_response: Optional[Dict]
    created_by: Optional[str]

class AIInteractionOut(AIInteractionBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
