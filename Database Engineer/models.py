from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any


@dataclass
class AnalysisResult:
    id: Optional[int] = None
    filename: str = ""
    timestamp: Optional[datetime] = None
    label: str = ""
    confidence: float = 0.0
    fake_probability: float = 0.0
    risk_score: float = 0.0
    risk_level: str = ""
    audio_duration: Optional[float] = None
    sample_rate: Optional[int] = None
    alerts: Optional[Any] = None
    recommendation: Optional[str] = None
