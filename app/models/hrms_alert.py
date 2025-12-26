from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.core.database import Base



class AlertRequest(Base):
    __tablename__ = "alert_requests"

    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(String, nullable=False)

    project_name = Column(String, nullable=True)
    min_days = Column(Integer, default=3)
    max_days = Column(Integer, default=5)

    total_employees = Column(Integer, default=0)
    status = Column(String, default="sent")  # sent | closed

    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AlertMessage(Base):
    __tablename__ = "alert_messages"

    id = Column(Integer, primary_key=True, index=True)

    alert_request_id = Column(
        Integer,
        ForeignKey("alert_requests.id", ondelete="CASCADE"),
        nullable=False
    )

    employee_id = Column(String, nullable=False)
    employee_name = Column(String, nullable=False)

    message = Column(Text, nullable=False)
    sent_at = Column(DateTime(timezone=True), server_default=func.now())

class AlertResponse(Base):
    __tablename__ = "alert_responses"

    id = Column(Integer, primary_key=True, index=True)

    employee_id = Column(String, nullable=False)
    employee_name = Column(String, nullable=False)

    response = Column(Text, nullable=False)
    received_at = Column(DateTime(timezone=True), server_default=func.now())
