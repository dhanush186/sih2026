from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict import load_models, predict_audio


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Deepfake Voice Detection API",
    description="Wav2Vec2-based deepfake voice detection service",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================
# Allows the frontend running on another local port
# to communicate with this API during the hackathon.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD ML MODELS ONCE
# ============================================================

print("Loading ML models...")

processor, encoder, classifier = load_models()

print("ML models ready.")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Deepfake Voice Detection API",
        "status": "running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Wav2Vec2 + classifier"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    allowed_extensions = {
        ".wav",
        ".mp3"
    }

    extension = (
        Path(file.filename)
        .suffix
        .lower()
        .strip()
    )

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only WAV and MP3 files are supported."
            )
        )

    # --------------------------------------------------------
    # Temporary file
    # --------------------------------------------------------

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:

            temp_path = Path(
                temp_file.name
            )

            shutil.copyfileobj(
                file.file,
                temp_file
            )

        # ----------------------------------------------------
        # Run ML prediction
        # ----------------------------------------------------

        result = predict_audio(
            temp_path,
            processor,
            encoder,
            classifier
        )

        fake_probability = (
            result["fake_probability"]
        )

        real_probability = (
            result["real_probability"]
        )

        prediction = result["label"]

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------
        # Confidence = probability of the predicted class.

        if prediction == "FAKE":

            confidence = fake_probability

        else:

            confidence = real_probability

        # ----------------------------------------------------
        # Verification requirement
        # ----------------------------------------------------
        # This is a safety/business rule for the prototype.
        # HIGH risk should trigger independent verification.

        requires_verification = (
            result["risk_level"] == "HIGH"
        )

        # ----------------------------------------------------
        # API response
        # ----------------------------------------------------

        return {
            "filename": file.filename,

            "prediction": prediction,

            "real_probability": round(
                real_probability,
                4
            ),

            "fake_probability": round(
                fake_probability,
                4
            ),

            "confidence": round(
                confidence,
                4
            ),

            "risk_level": result["risk_level"],

            "requires_verification": (
                requires_verification
            )
        }

    # --------------------------------------------------------
    # Client/input errors
    # --------------------------------------------------------

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    # --------------------------------------------------------
    # Server/ML errors
    # --------------------------------------------------------

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        )

    # --------------------------------------------------------
    # Always remove temporary audio
    # --------------------------------------------------------

    finally:

        if temp_path is not None:

            try:

                temp_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass