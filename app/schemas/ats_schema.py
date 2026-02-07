from pydantic import BaseModel
from typing import List, Optional, Dict


class Experience(BaseModel):
    title: str
    company: str
    bullets: List[str]

class Education(BaseModel):
    degree: str
    educationDescription: List[str]

class ATSScanRequest(BaseModel):
    name: str
    email: str
    phone: str
    summary: Optional[str]
    skills: List[str]
    experience: List[Experience]
    education: List[Education]

    # editor metadata
    font: str
    uses_table: bool
    uses_columns: bool
    file_type: str  # pdf | docx

    job_description: Optional[str] = None

class ATSSection(BaseModel):
    issues_count: int
    issues: List[str]

class ATSScanResponse(BaseModel):
    ats_score: int
    ai_issues: List[str] = []
    recommendations: List[str] = []
