import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field

from sqlalchemy import (
    String, Integer, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, text, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class LeadStatus(str, Enum):
    PENDING = "pending"
    CALLING = "calling"
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    SCHEDULED = "scheduled"
    RECALL = "recall"
    DNC = "dnc"
    FAILED = "failed"


class ScriptStatus(str, Enum):
    ACTIVE = "active"
    DRAFT = "draft"
    ARCHIVED = "archived"


class CallState(str, Enum):
    GREETING = "greeting"
    QUALIFYING = "qualifying"
    OBJECTION = "objection"
    SCHEDULING = "scheduling"
    CLOSING = "closing"
    ENDED = "ended"


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    industry: Mapped[str] = mapped_column(
        String(100), nullable=False, server_default=text("''")
    )
    language: Mapped[str] = mapped_column(
        String(10), nullable=False, server_default=text("'ta'")
    )
    agent_name: Mapped[str] = mapped_column(
        String(100), nullable=False, server_default=text("'Agent'")
    )
    active_script_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey(
            "company_scripts.id",
            use_alter=True,
            name="fk_companies_active_script",
        ),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
    )

    scripts: Mapped[List["CompanyScript"]] = relationship(
        "CompanyScript",
        back_populates="company",
        foreign_keys="CompanyScript.company_id",
        lazy="select",
    )
    leads: Mapped[List["Lead"]] = relationship(
        "Lead", back_populates="company", lazy="select"
    )

    __table_args__ = (
        Index("ix_companies_slug", "slug"),
    )


class CompanyScript(Base):
    __tablename__ = "company_scripts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    company_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("companies.id", ondelete="CASCADE", name="fk_company_scripts_company"),
        nullable=False,
    )
    version: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default=text("'1.0'")
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'draft'")
    )
    steps: Mapped[dict] = mapped_column(JSON, nullable=False)
    objection_responses: Mapped[dict] = mapped_column(JSON, nullable=False)
    closing_hot: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("''")
    )
    closing_warm: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("''")
    )
    closing_cold: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("''")
    )
    system_prompt_extra: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("''")
    )
    uploaded_filename: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
    )

    company: Mapped["Company"] = relationship(
        "Company",
        back_populates="scripts",
        foreign_keys=[company_id],
    )

    __table_args__ = (
        Index("ix_company_scripts_company_id", "company_id"),
        Index("ix_company_scripts_status", "status"),
    )


class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    phone: Mapped[str] = mapped_column(String(20), nullable=False)
    company_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("companies.id", ondelete="CASCADE", name="fk_leads_company"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'pending'")
    )
    qualification: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    experience_years: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    language_preference: Mapped[str] = mapped_column(
        String(10), nullable=False, server_default=text("'ta'")
    )
    call_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("3")
    )
    last_called_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    next_call_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    scheduled_interview_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    notes: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("''")
    )
    source_file: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    score: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
    )

    company: Mapped["Company"] = relationship(
        "Company", back_populates="leads"
    )
    interview_slots: Mapped[List["InterviewSlot"]] = relationship(
        "InterviewSlot", back_populates="lead", lazy="select"
    )

    __table_args__ = (
        Index("ix_leads_phone_company", "phone", "company_id", unique=True),
        Index("ix_leads_company_id", "company_id"),
        Index("ix_leads_status", "status"),
        Index("ix_leads_next_call_at", "next_call_at"),
    )


class InterviewSlot(Base):
    __tablename__ = "interview_slots"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    lead_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("leads.id", ondelete="CASCADE", name="fk_interview_slots_lead"),
        nullable=False,
    )
    call_id: Mapped[str] = mapped_column(String(36), nullable=False)
    company_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "companies.id", ondelete="CASCADE", name="fk_interview_slots_company"
        ),
        nullable=False,
    )
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False
    )
    confirmed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    calendar_event_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    sms_sent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
    )

    lead: Mapped["Lead"] = relationship(
        "Lead", back_populates="interview_slots"
    )

    __table_args__ = (
        Index("ix_interview_slots_lead_id", "lead_id"),
        Index("ix_interview_slots_company_id", "company_id"),
    )


@dataclass
class CompanyData:
    id: str
    name: str
    slug: str
    industry: str = ""
    language: str = "ta"
    agent_name: str = "Agent"
    active_script_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CompanyScriptData:
    id: str
    company_id: str
    version: str
    status: ScriptStatus
    steps: list
    objection_responses: dict
    closing_hot: str
    closing_warm: str
    closing_cold: str
    system_prompt_extra: str = ""
    uploaded_filename: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LeadData:
    id: str
    name: str
    phone: str
    company_id: str
    status: LeadStatus = LeadStatus.PENDING
    qualification: Optional[str] = None
    experience_years: Optional[int] = None
    language_preference: str = "ta"
    call_attempts: int = 0
    max_attempts: int = 3
    last_called_at: Optional[datetime] = None
    next_call_at: Optional[datetime] = None
    scheduled_interview_at: Optional[datetime] = None
    notes: str = ""
    source_file: Optional[str] = None
    score: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CallSessionData:
    call_id: str
    lead_id: str
    lead_phone: str
    lead_name: str
    company_id: str
    script_id: str
    state: CallState = CallState.GREETING
    script_pos: int = 0
    history: list = field(default_factory=list)
    lead_score: str = "cold"
    score_confidence: int = 0
    intent_flags: list = field(default_factory=list)
    tts_playing: bool = False
    started_at: datetime = field(default_factory=datetime.utcnow)
    transcript_full: str = ""
    proposed_slots: list = field(default_factory=list)


@dataclass
class InterviewSlotData:
    slot_id: str
    lead_id: str
    call_id: str
    scheduled_at: datetime
    confirmed: bool = False
    calendar_event_id: Optional[str] = None
    sms_sent: bool = False
    