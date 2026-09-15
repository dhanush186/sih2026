"""
Risk Engine.

Combines the three ML signal scores into a single risk score and verdict.
This is intentionally simple for the MVP — the Risk/Security engineer on
the team owns tuning these weights and thresholds.
"""

from database.config import settings
from database.models import Verdict

# Signal weights — must sum to 1.0
WEIGHTS = {
    "deepfake_score": 0.5,
    "speaker_score": 0.3,
    "prosody_score": 0.2,
}


def calculate_risk_score(deepfake_score: float, speaker_score: float, prosody_score: float) -> float:
    """Weighted combination of the three signals into a 0-100 risk score."""
    raw = (
        deepfake_score * WEIGHTS["deepfake_score"]
        + speaker_score * WEIGHTS["speaker_score"]
        + prosody_score * WEIGHTS["prosody_score"]
    )
    return round(min(100.0, max(0.0, raw)), 2)


def classify_risk(risk_score: float) -> Verdict:
    if risk_score >= settings.RISK_THRESHOLD_HIGH:
        return Verdict.HIGH_RISK
    if risk_score >= settings.RISK_THRESHOLD_SUSPICIOUS:
        return Verdict.SUSPICIOUS
    return Verdict.SAFE


def build_explanation(deepfake_score: float, speaker_score: float, prosody_score: float, verdict: Verdict) -> str:
    parts = [
        f"Deepfake probability {deepfake_score:.0f}%",
        f"speaker similarity {speaker_score:.0f}%",
        f"prosody anomaly score {prosody_score:.0f}%",
    ]
    summary = ", ".join(parts)

    if verdict == Verdict.HIGH_RISK:
        recommendation = (
            "Verify the caller using an independent communication channel. "
            "Do not approve financial transactions or disclose sensitive information "
            "until the caller has been independently verified."
        )
    elif verdict == Verdict.SUSPICIOUS:
        recommendation = (
            "Treat this call with caution. Confirm the caller's identity through "
            "a secondary channel before acting on any request."
        )
    else:
        recommendation = "No strong indicators of voice manipulation detected. Normal caution still applies."

    return f"{summary}. {recommendation}"
