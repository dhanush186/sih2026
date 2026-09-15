"""
Calls API — audio upload, listing, retrieval, deletion.

Upload flow:
  1. Validate + save audio file to disk (synchronous, always happens).
  2. If ML_SERVICE_URL is configured in .env, fire a background task that
     sends the file to that URL for analysis, then runs the Risk Engine on
     the returned scores and persists a DetectionResult (+ Alert if needed).
     This runs AFTER the HTTP response is returned so the upload stays fast.
  3. If ML_SERVICE_URL is NOT configured (default), the call is simply
     created with status=uploaded. Detection scores can then be submitted
     manually via POST /api/detection/{call_id} — this is the primary way
     to exercise the API until the ML team's service is ready.

INTEGRATION CONTRACT for the ML team's service (once ML_SERVICE_URL is set):
    POST {ML_SERVICE_URL}/analyze-by-path
    form fields: file_path (str), original_filename (str)
    expected JSON response: {"deepfake_score": 0-100, "speaker_score": 0-100,
                              "prosody_score": 0-100}
If the ML team's actual endpoint differs, update ANALYZE_PATH below or the
request body in _run_auto_detection() to match — nothing else needs to change.
"""

from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from database.config import settings
from database.connection import get_db
from database.models import Call, CallStatus, User, DetectionResult, Alert, AlertSeverity, Verdict
from schemas.calls import CallOut
from services.risk_engine import calculate_risk_score, classify_risk, build_explanation
from services.storage import save_audio_file

router = APIRouter(prefix="/api/calls", tags=["Calls"])

_SEVERITY_BY_VERDICT = {
    Verdict.HIGH_RISK: AlertSeverity.HIGH,
    Verdict.SUSPICIOUS: AlertSeverity.MEDIUM,
    Verdict.SAFE: AlertSeverity.LOW,
}

# Path appended to ML_SERVICE_URL when calling out for analysis.
# Update this if the ML team's service uses a different route.
ANALYZE_PATH = "/analyze-by-path"


# ---------------------------------------------------------------------------
# Background task: call the ML service → risk engine → persist result
# ---------------------------------------------------------------------------

def _run_auto_detection(call_id: int, audio_path: str, audio_filename: str) -> None:
    """
    Background task — fires AFTER the upload HTTP response is sent.
    Only runs when ML_SERVICE_URL is configured in .env.

    Uses a fresh DB session (background tasks run outside the request lifecycle).
    """
    from database.connection import SessionLocal  # local import avoids circular deps

    if not settings.ML_SERVICE_URL:
        return  # ML service not configured — skip auto-detection

    # --- 1. Call the ML service ---
    try:
        resp = httpx.post(
            f"{settings.ML_SERVICE_URL}{ANALYZE_PATH}",
            data={
                "file_path": audio_path,
                "original_filename": audio_filename,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        ml_result = resp.json()
    except Exception as exc:  # noqa: BLE001
        # ML call failed — don't fail silently. Mark the call as failed and
        # log a visible alert so this shows up in the alerts feed instead of
        # the call just sitting at 'uploaded' with no explanation.
        # Manual detection via POST /api/detection/{call_id} still works as
        # a fallback (e.g. once the ML service is back up).
        print(f"[auto-detect] ML service call failed for call_id={call_id}: {exc}")
        db = SessionLocal()
        try:
            call = db.get(Call, call_id)
            if call and call.status == CallStatus.uploaded:
                call.status = CallStatus.failed
                db.add(
                    Alert(
                        call_id=call_id,
                        alert_type="ML_SERVICE_UNAVAILABLE",
                        severity=AlertSeverity.MEDIUM,
                        message=(
                            "Automatic analysis failed because the ML service could not "
                            f"be reached or errored ({exc}). Submit scores manually via "
                            "POST /api/detection/{call_id} or retry once the service is up."
                        ),
                    )
                )
                db.commit()
        except Exception as inner_exc:  # noqa: BLE001
            db.rollback()
            print(f"[auto-detect] Failed to record ML failure for call_id={call_id}: {inner_exc}")
        finally:
            db.close()
        return

    deepfake_score = float(ml_result["deepfake_score"])
    speaker_score = float(ml_result["speaker_score"])
    prosody_score = float(ml_result["prosody_score"])

    # --- 2. Run Risk Engine ---
    risk_score = calculate_risk_score(deepfake_score, speaker_score, prosody_score)
    verdict = classify_risk(risk_score)
    explanation = build_explanation(deepfake_score, speaker_score, prosody_score, verdict)

    # --- 3. Persist DetectionResult + Alert ---
    db = SessionLocal()
    try:
        call = db.get(Call, call_id)
        if not call:
            print(f"[auto-detect] Call {call_id} not found in DB — skipping.")
            return
        if call.detection_result is not None:
            print(f"[auto-detect] Call {call_id} already has a detection result — skipping.")
            return

        result = DetectionResult(
            call_id=call_id,
            deepfake_score=deepfake_score,
            speaker_score=speaker_score,
            prosody_score=prosody_score,
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
        print(
            f"[auto-detect] call_id={call_id} → verdict={verdict} "
            f"risk_score={risk_score}"
        )
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        print(f"[auto-detect] DB error for call_id={call_id}: {exc}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=CallOut, status_code=status.HTTP_201_CREATED)
async def upload_call_audio(
    background_tasks: BackgroundTasks,
    user_id: int = Form(...),
    caller_number: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Accepts an audio file plus metadata, validates it, stores it on disk,
    and creates a Call record with status=uploaded.

    If ML_SERVICE_URL is configured (see .env), auto-detection runs in the
    background after this response returns — results appear in
    GET /api/detection/{call_id} once the ML service responds.

    Manual detection is always available via: POST /api/detection/{call_id}
    """
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")

    stored_filename, stored_path = await save_audio_file(audio_file)

    call = Call(
        user_id=user_id,
        caller_number=caller_number,
        audio_filename=stored_filename,
        audio_path=stored_path,
        status=CallStatus.uploaded,
    )
    db.add(call)
    db.commit()
    db.refresh(call)

    # Fire-and-forget: auto-detect via the ML service (if configured)
    if settings.ML_SERVICE_URL:
        background_tasks.add_task(
            _run_auto_detection,
            call.id,
            stored_path,
            stored_filename,
        )

    return call


@router.get("/", response_model=list[CallOut])
def list_calls(
    skip: int = 0,
    limit: int = 50,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Call)
    if user_id is not None:
        query = query.filter(Call.user_id == user_id)
    return query.order_by(Call.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/{call_id}", response_model=CallOut)
def get_call(call_id: int, db: Session = Depends(get_db)):
    call = db.get(Call, call_id)
    if not call:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Call not found.")
    return call


@router.delete("/{call_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_call(call_id: int, db: Session = Depends(get_db)):
    call = db.get(Call, call_id)
    if not call:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Call not found.")
    db.delete(call)
    db.commit()
