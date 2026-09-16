 
from fastapi import FastAPI, UploadFile, File
from typing import Dict, Any
import os

from backend.audio.pipeline import run_pipeline
from backend.risk_engine.risk_engine import calculate_risk
from importlib.machinery import SourceFileLoader

ml_model = SourceFileLoader(
    "predict",
    "Data Engineer/ml/predict.py"
).load_module()

processor, encoder, classifier = ml_model.load_models()

app = FastAPI(
    title="AI Voice Deepfake Detection API",
    description="API for detecting AI-generated voice and assessing risk.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "AI Voice Deepfake Detection API running"
    }


# -----------------------------
# MOCK AUDIO ANALYSIS
# -----------------------------
def mock_audio_analysis(audio_file: str) -> Dict[str, Any]:
    return {
        "audio_quality": 0.92,
        "mfcc_score": 0.85,
        "spectral_score": 0.78,
        "prosody_score": 0.88
    }


# -----------------------------
# MOCK ML PREDICTION
# -----------------------------
def mock_ml_prediction(audio_file: str) -> Dict[str, Any]:
    return {
        "label": "FAKE",
        "confidence": 0.94,
        "deepfake_probability": 0.94
    }


# -----------------------------
# MOCK RISK ENGINE
# -----------------------------
def mock_risk_engine(
    audio_result: Dict[str, Any],
    ml_result: Dict[str, Any]
) -> Dict[str, Any]:

    risk_score = 91

    return {
        "risk_score": risk_score,
        "risk_level": "HIGH",
        "alerts": [
            "Possible AI-generated voice",
            "Suspicious audio characteristics"
        ],
        "recommendation": "Verify caller using an independent channel"
    }


# -----------------------------
# ANALYZE AUDIO
# -----------------------------
@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    # Save uploaded audio
    upload_dir = "backend/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # REAL audio analysis
    audio_result = run_pipeline(
        file_path,
        save_features=True,
    )

    # REAL ML prediction
    ml_result = ml_model.predict_audio(
        file_path,
        processor,
        encoder,
        classifier,
    )

    risk_result = calculate_risk(
        ai_probability=ml_result["fake_probability"] * 100,
        speaker_similarity=100,
        prosody_anomaly=0,
        caller_anomaly=0,
        transaction_risk=0,
    )

    return {
        "filename": file.filename,
        "audio_analysis": audio_result,
        "ml_prediction": ml_result,
        "risk_assessment": risk_result,
    }
    
   
