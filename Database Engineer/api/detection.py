from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import Call, CallStatus, DetectionResult, Alert, AlertSeverity, Verdict
from schemas.detection import DetectionInput, DetectionResultOut
from services.risk_engine import calculate_risk_score, classify_risk, build_explanation

router = APIRouter(prefix="/api/detection", tags=["Detection"])

_SEVERITY_BY_VERDICT = {
    Verdict.HIGH_RISK: AlertSeverity.HIGH,
    Verdict.SUSPICIOUS: AlertSeverity.MEDIUM,
    Verdict.SAFE: AlertSeverity.LOW,
}


@router.post("/{call_id}", response_model=DetectionResultOut, status_code=status.HTTP_201_CREATED)
def run_detection(call_id: int, payload: DetectionInput, db: Session = Depends(get_db)):
    """
    Accepts the three raw signal scores (from the ML / Audio-Speaker pipeline),
    runs the Risk Engine, and persists a DetectionResult plus an Alert.

    NOTE: for the MVP this endpoint takes pre-computed scores directly so the
    backend can be built and tested independently of the ML component. Once
    the ML service is live, wire it in here (or upstream, before calling this
    endpoint) instead of accepting raw scores from the client.
    """
    call = db.get(Call, call_id)
    if not call:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Call not found.")

    if call.detection_result is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Detection already exists for this call.")

    risk_score = calculate_risk_score(
        payload.deepfake_score, payload.speaker_score, payload.prosody_score
    )
    verdict = classify_risk(risk_score)
    explanation = build_explanation(
        payload.deepfake_score, payload.speaker_score, payload.prosody_score, verdict
    )

    result = DetectionResult(
        call_id=call_id,
        deepfake_score=payload.deepfake_score,
        speaker_score=payload.speaker_score,
        prosody_score=payload.prosody_score,
        risk_score=risk_score,
        verdict=verdict,
        explanation=explanation,
    )
    db.add(result)

    call.status = CallStatus.completed

    if verdict in (Verdict.SUSPICIOUS, Verdict.HIGH_RISK):
        alert = Alert(
            call_id=call_id,
            alert_type="DEEPFAKE_DETECTED",
            severity=_SEVERITY_BY_VERDICT[verdict],
            message=explanation,
        )
        db.add(alert)

    db.commit()
    db.refresh(result)
    return result


@router.get("/{call_id}", response_model=DetectionResultOut)
def get_detection(call_id: int, db: Session = Depends(get_db)):
    call = db.get(Call, call_id)
    if not call:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Call not found.")
    if not call.detection_result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No detection result for this call yet.")
    return call.detection_result
