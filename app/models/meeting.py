from sqlalchemy import Column, String, DateTime, Text
from app.core.database import Base
import uuid
from datetime import datetime

class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, index=True)

    name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    email = Column(String, nullable=False)
    preferred_datetime = Column(String, nullable=False)
    purpose = Column(Text)

    source = Column(String, default="website")
    status = Column(String, default="pending")

    created_at = Column(DateTime, default=datetime.utcnow)
