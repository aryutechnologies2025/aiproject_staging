from sqlalchemy import Column, Integer, String, Text, DateTime, Enum
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class LeadScoreEnum(str, enum.Enum):
    HOT = "Hot"
    WARM = "Warm"
    COLD = "Cold"
    UNSCORED = "Unscored"

class CallStatusEnum(str, enum.Enum):
    COMPLETED = "completed"
    PENDING_RECALL = "pending_recall"
    RECALLED = "recalled"
    FAILED = "failed"

class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), index=True, nullable=False)
    
    transcript = Column(Text, nullable=True)
    lead_score = Column(Enum(LeadScoreEnum), default=LeadScoreEnum.UNSCORED)
    summary = Column(Text, nullable=True)
    
    status = Column(Enum(CallStatusEnum), default=CallStatusEnum.COMPLETED)
    
    recall_timestamp = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())