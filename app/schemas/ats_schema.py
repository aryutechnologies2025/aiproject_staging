from pydantic import BaseModel
from typing import List, Optional


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


class ATSScanResponse(BaseModel):
    ats_score: int
    keyword_match_percentage: int
    issues: List[str]
    suggestions: List[str]
