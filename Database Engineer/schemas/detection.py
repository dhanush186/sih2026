from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from database.models import Verdict


class DetectionInput(BaseModel):
    """
    Raw signal scores coming from the ML / Audio-Speaker components.
    The backend does not compute these — it forwards to the Risk Engine
    and persists the combined result.
    """
    deepfake_score: float = Field(..., ge=0, le=100)
    speaker_score: float = Field(..., ge=0, le=100)
    prosody_score: float = Field(..., ge=0, le=100)


class DetectionResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    call_id: int
    deepfake_score: float
    speaker_score: float
    prosody_score: float
    risk_score: float
    verdict: Verdict
    explanation: Optional[str]
    created_at: datetime
