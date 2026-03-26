# /home/aryu_user/Arun/aiproject_staging/app/modules/resume_builder/schemas.py

from pydantic import BaseModel, Field, RootModel
from typing import Dict, Any, List, Optional


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

class ContactInfo(BaseModel):
    phone: str = ""
    email: str = ""
    location: str = ""

class LinkEntry(BaseModel):
    name: str = ""
    url: Optional[str] = None

class PersonalInfo(BaseModel):
    name: str = ""
    title: str = ""
    contact: ContactInfo = Field(default_factory=ContactInfo)
    links: List[LinkEntry] = Field(default_factory=list)

class ExperienceEntry(BaseModel):
    title: str = ""
    company: str = ""
    duration: str = ""
    location: str = ""
    details: List[str] = Field(default_factory=list)

class EducationEntry(BaseModel):
    degree: str = ""
    institution: str = ""
    duration: str = ""
    location: str = ""
    cgpa: str = ""
    description: str = ""

class DSADetail(BaseModel):
    linear: List[str] = Field(default_factory=list)
    non_linear: List[str] = Field(default_factory=list)

class TechnicalStack(BaseModel):
    frontend: List[str] = Field(default_factory=list)
    backend_apis: List[str] = Field(default_factory=list)
    tools_devops: List[str] = Field(default_factory=list)
    engineering_practices: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    data_structures_algorithms: DSADetail = Field(default_factory=DSADetail)
    other: Dict[str, List[str]] = Field(default_factory=dict)

class ProjectEntry(BaseModel):
    name: str = ""
    details: List[str] = Field(default_factory=list)
    tech_stack: List[str] = Field(default_factory=list)

class LanguageEntry(BaseModel):
    language: str = ""
    proficiency: str = ""

class CertificationEntry(BaseModel):
    name: str = ""
    issuer: str = ""
    year: str = ""

class ResumeResponse(BaseModel):
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    summary: List[str] = Field(default_factory=list)
    professional_experience: List[ExperienceEntry] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    technical_stack: TechnicalStack = Field(default_factory=TechnicalStack)
    projects: List[ProjectEntry] = Field(default_factory=list)
    languages: List[LanguageEntry] = Field(default_factory=list)
    certifications: List[CertificationEntry] = Field(default_factory=list)
    custom_sections: Dict[str, Any] = Field(default_factory=dict)
    raw_sections: Dict[str, str] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
