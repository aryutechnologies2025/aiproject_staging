from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.modules.voice_agent.models import LeadStatus, ScriptStatus


class CompanyCreate(BaseModel):
    name: str
    slug: str
    industry: str = ""
    language: str = "ta"
    agent_name: str = "பிரியா"


class CompanyResponse(BaseModel):
    id: str
    name: str
    slug: str
    industry: str
    language: str
    agent_name: str
    active_script_id: Optional[str]
    created_at: datetime


class ScriptStepSchema(BaseModel):
    id: int
    state: str
    question: str
    fallback: str
    intent_trigger: Optional[str] = None


class ScriptCreateManual(BaseModel):
    company_id: str
    version: str = "1.0"
    steps: List[Dict[str, Any]]
    objection_responses: Optional[Dict[str, str]] = None
    closing_hot: Optional[str] = None
    closing_warm: Optional[str] = None
    closing_cold: Optional[str] = None
    system_prompt_extra: str = ""


class ScriptResponse(BaseModel):
    id: str
    company_id: str
    version: str
    status: ScriptStatus
    uploaded_filename: Optional[str]
    steps_count: int
    created_at: datetime


class LeadCreate(BaseModel):
    name: str
    phone: str
    company_id: str
    qualification: Optional[str] = None
    experience_years: Optional[int] = None
    language_preference: str = "ta"
    source_file: Optional[str] = None


class LeadUpdate(BaseModel):
    status: Optional[LeadStatus] = None
    qualification: Optional[str] = None
    experience_years: Optional[int] = None
    notes: Optional[str] = None
    next_call_at: Optional[datetime] = None
    scheduled_interview_at: Optional[datetime] = None
    score: Optional[int] = None


class LeadResponse(BaseModel):
    id: str
    name: str
    phone: str
    company_id: str
    status: LeadStatus
    call_attempts: int
    score: int
    scheduled_interview_at: Optional[datetime]
    next_call_at: Optional[datetime]
    notes: str


class CallSessionRedis(BaseModel):
    call_id: str
    lead_id: str
    lead_phone: str
    lead_name: str
    company_id: str
    script_id: str
    state: str = "greeting"
    script_pos: int = 0
    history: List[dict] = []
    lead_score: str = "cold"
    score_confidence: int = 0
    intent_flags: List[str] = []
    tts_playing: bool = False
    started_at: str = ""
    transcript_full: str = ""
    proposed_slots: List[str] = []


class CSVLeadRow(BaseModel):
    name: str
    phone: str
    qualification: Optional[str] = None
    experience_years: Optional[str] = None
    language_preference: Optional[str] = "ta"

    @field_validator("phone")
    @classmethod
    def clean_phone(cls, v):
        digits = "".join(filter(str.isdigit, str(v)))
        if len(digits) == 10:
            return "+91" + digits
        if len(digits) == 12 and digits.startswith("91"):
            return "+" + digits
        return v

    @field_validator("experience_years")
    @classmethod
    def parse_exp(cls, v):
        if v is None:
            return None
        try:
            return str(int(float(str(v))))
        except Exception:
            return None
        