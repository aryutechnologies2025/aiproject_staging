from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from app.core.database import Base  # your declarative base

class AIInteraction(Base):
    __tablename__ = "ai_interactions"

    id = Column(Integer, primary_key=True, index=True)

    agent_name = Column(String(100), nullable=False)
    mode = Column(String(50), nullable=False)

    project_name = Column(String(255), nullable=True)

    input_payload = Column(JSON, nullable=False)
    ai_raw_response = Column(Text, nullable=False)
    ai_parsed_response = Column(JSON, nullable=True)

    created_by = Column(String(100), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
