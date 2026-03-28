"""
schemas.py — Pydantic models for LinkedIn profile extraction.
Covers ALL job domains: tech, non-tech, creative, healthcare, legal, finance, education, etc.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, validator


# ─────────────────────────── Enums ────────────────────────────────────────────

class ExtractionStatus(str, Enum):
    SUCCESS      = "success"
    PARTIAL      = "partial"       # Got some data but not everything
    CACHED       = "cached"
    LOGIN_NEEDED = "login_needed"
    FAILED       = "failed"


class LanguageProficiency(str, Enum):
    ELEMENTARY      = "Elementary"
    LIMITED         = "Limited Working"
    PROFESSIONAL    = "Professional Working"
    FULL_PROFESSIONAL = "Full Professional"
    NATIVE          = "Native or Bilingual"


# ─────────────────────────── Sub-models ───────────────────────────────────────

class DateRange(BaseModel):
    start_month: Optional[int]   = Field(None, ge=1, le=12)
    start_year:  Optional[int]   = Field(None, ge=1950, le=2100)
    end_month:   Optional[int]   = Field(None, ge=1, le=12)
    end_year:    Optional[int]   = Field(None, ge=1950, le=2100)
    is_current:  bool            = False

    @property
    def duration_months(self) -> Optional[int]:
        """Calculate duration in months if both start and end are known."""
        if not (self.start_year and self.end_year):
            return None
        sm = self.start_month or 1
        em = self.end_month or 12
        return (self.end_year - self.start_year) * 12 + (em - sm)

    @property
    def formatted(self) -> str:
        months_map = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        start = f"{months_map.get(self.start_month, '')} {self.start_year or ''}".strip()
        if self.is_current:
            end = "Present"
        elif self.end_year:
            end = f"{months_map.get(self.end_month, '')} {self.end_year}".strip()
        else:
            end = "Present"
        return f"{start} – {end}"


class Location(BaseModel):
    city:       Optional[str] = None
    state:      Optional[str] = None
    country:    Optional[str] = None
    country_code: Optional[str] = None   # ISO 3166-1 alpha-2

    @property
    def display(self) -> str:
        parts = [p for p in [self.city, self.state, self.country] if p]
        return ", ".join(parts)


class ContactInfo(BaseModel):
    email:       Optional[str]     = None
    phone:       Optional[str]     = None
    website:     Optional[HttpUrl] = None
    linkedin_url: Optional[str]    = None
    twitter:     Optional[str]     = None
    portfolio:   Optional[str]     = None     # For designers, developers, writers


class Experience(BaseModel):
    company:         str
    title:           str
    employment_type: Optional[str] = None   # Full-time, Part-time, Contract, etc.
    location:        Optional[Location] = None
    location_type:   Optional[str] = None   # On-site, Remote, Hybrid
    date_range:      Optional[DateRange] = None
    description:     Optional[str] = None
    skills_used:     List[str] = Field(default_factory=list)
    media:           List[str] = Field(default_factory=list)   # attached links/docs


class Education(BaseModel):
    institution:  str
    degree:       Optional[str] = None    # Bachelor's, Master's, Ph.D., Diploma, etc.
    field_of_study: Optional[str] = None
    grade:        Optional[str] = None    # GPA, percentage, distinction, etc.
    date_range:   Optional[DateRange] = None
    activities:   Optional[str] = None   # clubs, sports, societies
    description:  Optional[str] = None


class Certification(BaseModel):
    name:            str
    issuing_org:     Optional[str] = None
    issue_date:      Optional[DateRange] = None
    expiry_date:     Optional[DateRange] = None
    credential_id:   Optional[str] = None
    credential_url:  Optional[str] = None


class Skill(BaseModel):
    name:        str
    endorsements: int = 0
    category:    Optional[str] = None  # Technical, Soft, Domain-specific, etc.


class Project(BaseModel):
    name:        str
    description: Optional[str] = None
    url:         Optional[str] = None
    date_range:  Optional[DateRange] = None
    skills_used: List[str] = Field(default_factory=list)
    contributors: List[str] = Field(default_factory=list)


class Publication(BaseModel):
    title:       str
    publisher:   Optional[str] = None
    date:        Optional[DateRange] = None
    description: Optional[str] = None
    url:         Optional[str] = None
    authors:     List[str] = Field(default_factory=list)


class Award(BaseModel):
    title:      str
    issuer:     Optional[str] = None
    date:       Optional[DateRange] = None
    description: Optional[str] = None


class VolunteerExperience(BaseModel):
    organization: str
    role:         str
    cause:        Optional[str] = None
    date_range:   Optional[DateRange] = None
    description:  Optional[str] = None


class Language(BaseModel):
    name:        str
    proficiency: Optional[LanguageProficiency] = None


class Recommendation(BaseModel):
    recommender_name:  str
    recommender_title: Optional[str] = None
    relationship:      Optional[str] = None
    text:              Optional[str] = None
    date:              Optional[DateRange] = None


class TestScore(BaseModel):
    name:        str
    score:       Optional[str] = None
    date:        Optional[DateRange] = None
    description: Optional[str] = None


class Patent(BaseModel):
    title:       str
    patent_number: Optional[str] = None
    status:      Optional[str] = None
    date:        Optional[DateRange] = None
    description: Optional[str] = None


# ─────────────────────────── Root Model ───────────────────────────────────────

class LinkedInProfile(BaseModel):
    """
    Complete LinkedIn profile — universal across all job domains.
    """
    # Identity
    full_name:        str
    first_name:       Optional[str] = None
    last_name:        Optional[str] = None
    headline:         Optional[str] = None   # "Software Engineer at XYZ" or "Cardiologist | Author"
    about:            Optional[str] = None
    profile_url:      Optional[str] = None
    profile_picture:  Optional[str] = None   # base64 or URL

    # Location & Contact
    location:         Optional[Location] = None
    contact:          Optional[ContactInfo] = None

    # Professional sections
    experiences:      List[Experience]           = Field(default_factory=list)
    educations:       List[Education]            = Field(default_factory=list)
    certifications:   List[Certification]        = Field(default_factory=list)
    skills:           List[Skill]                = Field(default_factory=list)
    projects:         List[Project]              = Field(default_factory=list)

    # Academic / creative sections
    publications:     List[Publication]          = Field(default_factory=list)
    patents:          List[Patent]               = Field(default_factory=list)
    test_scores:      List[TestScore]            = Field(default_factory=list)
    awards:           List[Award]                = Field(default_factory=list)

    # Personal / community
    volunteer:        List[VolunteerExperience]  = Field(default_factory=list)
    languages:        List[Language]             = Field(default_factory=list)
    recommendations:  List[Recommendation]       = Field(default_factory=list)

    # Meta
    connections:      Optional[int]  = None
    followers:        Optional[int]  = None
    open_to_work:     bool           = False
    industry:         Optional[str]  = None

    class Config:
        json_encoders = {date: lambda v: v.isoformat()}


# ─────────────────────────── API Response Wrappers ────────────────────────────

class ExtractionMeta(BaseModel):
    status:          ExtractionStatus
    extracted_at:    Optional[str] = None   # ISO datetime string
    cache_hit:       bool          = False
    cache_age_hours: Optional[float] = None
    profile_hash:    Optional[str] = None
    sections_found:  List[str]     = Field(default_factory=list)
    warnings:        List[str]     = Field(default_factory=list)


class LinkedInResponse(BaseModel):
    meta:    ExtractionMeta
    profile: Optional[LinkedInProfile] = None


class ExtractionRequest(BaseModel):
    linkedin_url:   Optional[str] = Field(None, description="Public LinkedIn profile URL")
    use_cache:      bool          = Field(True,  description="Return cached data if fresh")
    cache_ttl_hours: int          = Field(24,    description="Cache freshness window in hours")
    include_recommendations: bool = Field(True)
    include_media:  bool          = Field(False, description="Download profile picture")

