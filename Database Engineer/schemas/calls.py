from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from database.models import CallStatus


class CallCreate(BaseModel):
    user_id: int
    caller_number: Optional[str] = None


class CallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    caller_number: Optional[str]
    audio_filename: str
    audio_path: str
    duration_seconds: Optional[float]
    status: CallStatus
    created_at: datetime
