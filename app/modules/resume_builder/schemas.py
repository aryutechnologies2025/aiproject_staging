"""
schemas.py — Canonical Pydantic schemas and legacy response adapters for resume_builder.

Ensures:
- Robust structured extraction schemas for Gemini.
- 100% backward compatibility with all legacy frontend response contracts.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL STRUCTURED RESUME MODELS
# ─────────────────────────────────────────────────────────────────────────────

class PersonalInformation(BaseModel):
    name: str = Field(default="", description="Candidate's full legal or professional name")
    title: str = Field(default="", description="Current or target professional job title/designation")
    email: str = Field(default="", description="Email address")
    phone: str = Field(default="", description="Contact telephone/mobile number with country code if present")
    location: str = Field(default="", description="City, State, Country or Location")
    link: str = Field(default="", description="Comma-separated URLs (LinkedIn, GitHub, portfolio, website)")


class ExperienceItem(BaseModel):
    position: str = Field(default="", description="Job title / role position")
    company: str = Field(default="", description="Company or organization name")
    location: str = Field(default="", description="Job location or Remote")
    fromYear: str = Field(default="", description="Start year or start date (e.g. '2021' or 'Jan 2021')")
    toYear: str = Field(default="", description="End year or 'Present'")
    isOngoing: bool = Field(default=False, description="True if this is the current active role")
    bullets: List[str] = Field(default_factory=list, description="Key accomplishments, responsibilities, and metrics")


class EducationItem(BaseModel):
    degree: str = Field(default="", description="Degree, diploma, or certification level and major")
    institution: str = Field(default="", description="University, college, or school name")
    location: str = Field(default="", description="Institution location")
    fromYear: str = Field(default="", description="Start year (e.g. '2017')")
    toYear: str = Field(default="", description="Graduation/completion year (e.g. '2021')")


class ProjectItem(BaseModel):
    title: str = Field(default="", description="Project name / title")
    description: str = Field(default="", description="High-level project summary or purpose")
    technologies: List[str] = Field(default_factory=list, description="Tools, frameworks, and programming languages used")
    fromYear: str = Field(default="", description="Start year")
    toYear: str = Field(default="", description="End year")
    bullets: List[str] = Field(default_factory=list, description="Specific project achievements and features")


class CertificationItem(BaseModel):
    title: str = Field(default="", description="Certification or license name")
    issuer: str = Field(default="", description="Issuing organization or authority")
    year: str = Field(default="", description="Year obtained or validity year")


class CanonicalResume(BaseModel):
    """
    Unified canonical resume schema representing the entire structured resume.
    """
    personal_information: PersonalInformation = Field(default_factory=PersonalInformation)
    summary: str = Field(default="", description="Executive summary, profile, or objective statement")
    skills: List[str] = Field(default_factory=list, description="List of technical skills, competencies, and tools")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Chronological work experience history")
    education: List[EducationItem] = Field(default_factory=list, description="Academic credentials and educational background")
    projects: List[ProjectItem] = Field(default_factory=list, description="Key personal, academic, or professional projects")
    certifications: List[CertificationItem] = Field(default_factory=list, description="Certificates, credentials, or licenses")
    languages: List[str] = Field(default_factory=list, description="Spoken/written human languages (e.g. English, Tamil)")
    other: List[str] = Field(default_factory=list, description="Additional sections like awards, volunteer work, publications")


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD-COMPATIBILITY RESPONSE MAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def map_to_legacy_parse_dict(canonical: CanonicalResume) -> Dict[str, Any]:
    """
    Maps CanonicalResume into the exact legacy dictionary schema expected by
    the `/parse-resume` endpoint and frontend builder components.
    """
    p_info = canonical.personal_information
    return {
        "header": {
            "name": p_info.name,
            "title": p_info.title,
            "email": p_info.email,
            "phone": p_info.phone,
            "location": p_info.location,
            "link": p_info.link,
        },
        "summary": {
            "summary": canonical.summary or ""
        },
        "experience": [
            {
                "position": exp.position,
                "company": exp.company,
                "location": exp.location,
                "fromYear": exp.fromYear,
                "toYear": exp.toYear,
                "isOngoing": exp.isOngoing,
                "bullets": exp.bullets,
            }
            for exp in canonical.experience
        ],
        "education": [
            {
                "degree": edu.degree,
                "institution": edu.institution,
                "location": edu.location,
                "fromYear": edu.fromYear,
                "toYear": edu.toYear,
            }
            for edu in canonical.education
        ],
        "skills": canonical.skills or [],
        "projects": [
            {
                "title": proj.title,
                "description": proj.description,
                "technologies": proj.technologies,
                "fromYear": proj.fromYear,
                "toYear": proj.toYear,
                "bullets": proj.bullets,
            }
            for proj in canonical.projects
        ],
        "certifications": [
            {
                "title": cert.title,
                "issuer": cert.issuer,
                "year": cert.year,
            }
            for cert in canonical.certifications
        ],
        "languages": canonical.languages or [],
        "other": canonical.other or [],
    }


def map_to_legacy_flat_dict(canonical: CanonicalResume) -> Dict[str, Any]:
    """
    Flat dictionary representation for CV generators, ATS scanners, and exporters.
    """
    p_info = canonical.personal_information
    return {
        "name": p_info.name,
        "title": p_info.title,
        "email": p_info.email,
        "phone": p_info.phone,
        "location": p_info.location,
        "link": p_info.link,
        "linkedin": p_info.link,
        "summary": canonical.summary,
        "skills": canonical.skills,
        "experience": [
            {
                "title": exp.position,
                "position": exp.position,
                "company": exp.company,
                "location": exp.location,
                "fromYear": exp.fromYear,
                "toYear": exp.toYear,
                "bullets": exp.bullets,
            }
            for exp in canonical.experience
        ],
        "education": [
            {
                "degree": edu.degree,
                "institution": edu.institution,
                "college": edu.institution,
                "year": edu.toYear,
                "fromYear": edu.fromYear,
                "toYear": edu.toYear,
            }
            for edu in canonical.education
        ],
        "projects": [proj.dict() for proj in canonical.projects],
        "certifications": [cert.dict() for cert in canonical.certifications],
        "languages": canonical.languages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY FILE AND EXTRACTION SCHEMAS (Preserved for existing imports)
# ─────────────────────────────────────────────────────────────────────────────

class FileInfo(BaseModel):
    filename: str
    file_type: str
    size: int


class ExtractedData(BaseModel):
    raw_text: str
    sections: Dict[str, str]
    metadata: Dict[str, Any]


class ParsedData(BaseModel):
    skills: List[str] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    projects: List[Dict[str, Any]] = Field(default_factory=list)


class ResumeExtractionResponse(BaseModel):
    success: bool
    file_info: FileInfo
    extracted: ExtractedData
    parsed: ParsedData


class ResumeResponse(BaseModel):
    raw_text: str
    sections: Dict[str, Any]
    metadata: Dict[str, Any]
