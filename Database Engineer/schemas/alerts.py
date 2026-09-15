from datetime import datetime

from pydantic import BaseModel, ConfigDict

from database.models import AlertSeverity, AlertStatus


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    call_id: int
    alert_type: str
    severity: AlertSeverity
    message: str
    status: AlertStatus
    created_at: datetime


class AlertStatusUpdate(BaseModel):
    status: AlertStatus
