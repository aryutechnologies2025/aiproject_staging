from sqlalchemy import Column, Integer, String, Text, DateTime, func, Index
from app.core.database import Base

class Prompt(Base):
    __tablename__ = "prompts"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(50), nullable=False, unique=True, index=True)
    description = Column(String(255))
    system_prompt = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Extra index optimized for lookup (faster SELECT on agent_name)
    __table_args__ = (
        Index("idx_agent_name", "agent_name"),
    )
