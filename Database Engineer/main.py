from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from database.config import settings
from database.connection import Base, engine, SessionLocal
from api import users, calls, detection, alerts
from api.auth import verify_api_key
from analysis_database import insert_analysis_result


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MVP: create tables directly. Swap for Alembic migrations once the schema stabilizes.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="AI Voice Deepfake Detection API",
    description="Backend for the AI Voice Deepfake Detection & Fraud Prevention System.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All API routers require the X-API-Key header.
# The exempted public routes (GET /, GET /health) are defined below at app level.
_auth = [Depends(verify_api_key)]

app.include_router(users.router, dependencies=_auth)
app.include_router(calls.router, dependencies=_auth)
app.include_router(detection.router, dependencies=_auth)
app.include_router(alerts.router, dependencies=_auth)


@app.get("/")
def root():
    """Public — no auth required."""
    return {
        "service": settings.APP_NAME,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Public — no auth required."""
    return {"status": "healthy", "service": settings.APP_NAME}


@app.get("/api/database/test")
def database_test():
    """Verifies the FastAPI <-> database connection. No auth required (low-sensitivity probe)."""
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return {"database": "connected"}
    except Exception as exc:  # noqa: BLE001
        return {"database": "error", "detail": str(exc)}
    finally:
        db.close()

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    MVP analyze endpoint.

    Receives an audio file, builds the current MVP analysis result,
    saves it to MySQL, and returns the result to the frontend.

    Audio/ML values are currently mock values until the ML/audio
    services are connected.
    """
    filename = file.filename or "unknown_audio"

    audio_analysis = {
        "audio_path": f"uploads/{filename}",
        "sample_rate": 44100,
        "duration_seconds": 5.94,
        "mfcc_shape": [39, 157],
        "spectrogram_shape": [128, 157],
        "prosody": {
            "pitch_mean": 201.268,
            "pitch_std": 76.433,
            "pitch_min": 123.254,
            "pitch_max": 391.306,
            "energy_mean": 0.401,
            "energy_std": 0.143,
            "speaking_rate": 10.016,
            "pause_ratio": 0.0,
        },
        "speaker_verification": None,
    }

    ml_prediction = {
        "prediction": 1,
        "label": "FAKE",
        "real_probability": 0.0218,
        "fake_probability": 0.9782,
        "risk_level": "HIGH",
    }

    risk_assessment = {
        "risk_score": 39.13,
        "risk_level": "MEDIUM",
        "confidence": 20.0,
        "risk_factors": [
            "High AI-generated voice probability"
        ],
        "recommendation": [
            "Perform additional caller verification",
            "Monitor the request carefully",
        ],
        "alert": "MEDIUM RISK: Suspicious call characteristics detected",
        "security_decision": "VERIFY",
    }

    insert_analysis_result(
        filename=filename,
        audio_analysis=audio_analysis,
        ml_prediction=ml_prediction,
        risk_assessment=risk_assessment,
    )

    return {
        "filename": filename,
        "audio_analysis": audio_analysis,
        "ml_prediction": ml_prediction,
        "risk_assessment": risk_assessment,
    }
