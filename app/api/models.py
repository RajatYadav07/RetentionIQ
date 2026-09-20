from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.api.database import Base

class ActionQueue(Base):
    __tablename__ = "action_queue"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    priority_score = Column(Float)
    priority = Column(String)  # Critical, High, Medium, Low
    action = Column(String)
    reason = Column(String)
    urgency = Column(String)
    objective = Column(String)
    status = Column(String, default="New") # New, In Progress, Contacted, Resolved, Dismissed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
