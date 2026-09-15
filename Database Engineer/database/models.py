"""
SQLAlchemy ORM models: Users, Calls, DetectionResults, Alerts.
"""

import enum
import uuid

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Enum,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from .connection import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    analyst = "analyst"
    viewer = "viewer"


class CallStatus(str, enum.Enum):
    uploaded = "uploaded"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Verdict(str, enum.Enum):
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"


class AlertSeverity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AlertStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    role = Column(Enum(UserRole), default=UserRole.viewer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    calls = relationship("Call", back_populates="user", cascade="all, delete-orphan")


class Call(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    caller_number = Column(String(32), nullable=True)
    audio_filename = Column(String(255), nullable=False)
    audio_path = Column(String(512), nullable=False)
    duration_seconds = Column(Float, nullable=True)
    status = Column(Enum(CallStatus), default=CallStatus.uploaded, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="calls")
    detection_result = relationship(
        "DetectionResult", back_populates="call", uselist=False, cascade="all, delete-orphan"
    )
    alerts = relationship("Alert", back_populates="call", cascade="all, delete-orphan")


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, unique=True)
    deepfake_score = Column(Float, nullable=False)
    speaker_score = Column(Float, nullable=False)
    prosody_score = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    verdict = Column(Enum(Verdict), nullable=False)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    call = relationship("Call", back_populates="detection_result")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    alert_type = Column(String(64), nullable=False)
    severity = Column(Enum(AlertSeverity), nullable=False)
    message = Column(String(512), nullable=False)
    status = Column(Enum(AlertStatus), default=AlertStatus.ACTIVE, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    call = relationship("Call", back_populates="alerts")


def generate_upload_filename(original_filename: str) -> str:
    """Generate a collision-safe filename for storing an uploaded audio file."""
    ext = "." + original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
    return f"{uuid.uuid4().hex}{ext}"
