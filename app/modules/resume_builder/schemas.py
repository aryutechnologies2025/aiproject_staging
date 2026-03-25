# /home/aryu_user/Arun/aiproject_staging/app/modules/resume_builder/schemas.py

from pydantic import BaseModel
from typing import Dict, Any, List


class FileInfo(BaseModel):
    filename: str
    file_type: str
    size: int


class ExtractedData(BaseModel):
    raw_text: str
    sections: Dict[str, str]
    metadata: Dict[str, Any]


class ParsedData(BaseModel):
    skills: List[str] = []
    education: List[Dict] = []
    projects: List[Dict] = []


class ResumeExtractionResponse(BaseModel):
    success: bool
    file_info: FileInfo
    extracted: ExtractedData
    parsed: ParsedData