from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict import load_models
from realtime_predict import (
    load_full_audio,
    create_chunks,
    analyze_chunk,
    SAMPLE_RATE,
    CHUNK_DURATION,
)
from risk_engine import RiskEngine
from speaker_verifier import SpeakerVerifier


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Trusted speaker reference
#
# Set to None to disable speaker verification.
# ------------------------------------------------------------

REFERENCE_AUDIO = Path(
    r"D:\kaggle_cache\datasets"
    r"\mohammedabdeldayem"
    r"\the-fake-or-real-dataset"
    r"\versions\2"
    r"\for-original"
    r"\for-original"
    r"\testing"
    r"\real"
    r"\file1.wav"
)

# Example:
# REFERENCE_AUDIO = None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Real-Time Deepfake Voice Detection API",
    description=(
        "Chunk-based Wav2Vec2 deepfake detection "
        "with optional ECAPA speaker verification"
    ),
    version="1.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DEEPFAKE MODELS ONCE
# ============================================================

print("Loading ML models...")

processor, encoder, classifier = load_models()

print("ML models ready.")


# ============================================================
# LOAD SPEAKER MODEL ONCE
# ============================================================

speaker_verifier = None

if REFERENCE_AUDIO is not None:

    if not REFERENCE_AUDIO.exists():

        print(
            "WARNING: Trusted speaker reference does not exist:"
        )
        print(REFERENCE_AUDIO)

    else:

        print("Loading speaker verification model...")

        speaker_verifier = SpeakerVerifier()

        print(
            "Speaker verification model ready."
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": (
            "Real-Time Deepfake Voice Detection API"
        ),
        "status": "running",
        "speaker_verification": (
            speaker_verifier is not None
        ),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models": {
            "deepfake": "Wav2Vec2",
            "speaker_verification": (
                speaker_verifier is not None
            ),
            "risk_engine": "enabled",
        },
    }


# ============================================================
# REAL-TIME PREDICTION
# ============================================================

@app.post("/predict/realtime")
async def predict_realtime(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    allowed_extensions = {
        ".wav",
        ".mp3",
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
            ),
        )

    temp_path = None

    try:

        # ====================================================
        # 1. SAVE UPLOADED AUDIO
        # ====================================================

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:

            temp_path = Path(
                temp_file.name
            )

            shutil.copyfileobj(
                file.file,
                temp_file
            )

        # ====================================================
        # 2. LOAD AUDIO
        # ====================================================

        audio = load_full_audio(
            temp_path
        )

        duration = (
            len(audio) / SAMPLE_RATE
        )

        # ====================================================
        # 3. SPEAKER VERIFICATION
        # ====================================================

        speaker_similarity = None
        speaker_same = None

        if (
            speaker_verifier is not None
            and REFERENCE_AUDIO is not None
        ):

            speaker_result = (
                speaker_verifier.compare(
                    REFERENCE_AUDIO,
                    temp_path
                )
            )

            speaker_similarity = float(
                speaker_result[
                    "similarity_score"
                ]
            )

            speaker_same = bool(
                speaker_result[
                    "same_speaker"
                ]
            )

        # ====================================================
        # 4. CREATE AUDIO CHUNKS
        # ====================================================

        chunks = create_chunks(
            audio
        )

        # ====================================================
        # 5. CREATE RISK ENGINE
        # ====================================================

        risk_engine = RiskEngine()

        chunk_results = []

        # ====================================================
        # 6. ANALYZE CHUNKS
        # ====================================================

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            start_time = (
                (index - 1)
                * CHUNK_DURATION
            )

            end_time = (
                start_time
                + CHUNK_DURATION
            )

            # ------------------------------------------------
            # Wav2Vec2 prediction
            # ------------------------------------------------

            (
                real_probability,
                fake_probability
            ) = analyze_chunk(
                chunk,
                processor,
                encoder,
                classifier
            )

            # ------------------------------------------------
            # Risk calculation
            # ------------------------------------------------

            risk_result = (
                risk_engine.add_result(
                    fake_probability=(
                        fake_probability
                    ),
                    speaker_similarity=(
                        speaker_similarity
                    ),
                )
            )

            chunk_results.append(
                {
                    "chunk": index,

                    "start_seconds": round(
                        start_time,
                        2
                    ),

                    "end_seconds": round(
                        min(
                            end_time,
                            duration
                        ),
                        2
                    ),

                    "real_probability": round(
                        real_probability,
                        4
                    ),

                    "fake_probability": round(
                        fake_probability,
                        4
                    ),

                    "risk_level": (
                        risk_result[
                            "risk_level"
                        ]
                    ),

                    "risk_score": round(
                        risk_result[
                            "risk_score"
                        ],
                        2
                    ),

                    "high_evidence": (
                        risk_result[
                            "high_evidence"
                        ]
                    ),

                    "speaker_similarity": (
                        None
                        if speaker_similarity is None
                        else round(
                            speaker_similarity,
                            4
                        )
                    ),

                    "speaker_mismatch_score": (
                        None
                        if risk_result[
                            "speaker_mismatch_score"
                        ] is None
                        else round(
                            risk_result[
                                "speaker_mismatch_score"
                            ],
                            2
                        )
                    ),

                    "speaker_reference_used": (
                        risk_result[
                            "speaker_reference_used"
                        ]
                    ),
                }
            )

        # ====================================================
        # 7. FINAL RISK
        # ====================================================

        final_result = (
            risk_engine.get_risk(
                speaker_similarity=(
                    speaker_similarity
                )
            )
        )

        overall_risk = (
            final_result[
                "risk_level"
            ]
        )

        overall_risk_score = (
            final_result[
                "risk_score"
            ]
        )

        rolling_fake_probability = (
            final_result[
                "average_fake_probability"
            ]
        )

        session_alert = (
            final_result[
                "session_alert"
            ]
        )

        requires_verification = (
            overall_risk
            in {"MEDIUM", "HIGH"}
        )

        # ====================================================
        # 8. RETURN RESPONSE
        # ====================================================

        return {

            "filename": file.filename,

            "duration_seconds": round(
                duration,
                2
            ),

            "chunk_duration_seconds": (
                CHUNK_DURATION
            ),

            "total_chunks": len(
                chunk_results
            ),

            "chunks": chunk_results,

            # ----------------------------------------------
            # Overall risk
            # ----------------------------------------------

            "overall_risk": overall_risk,

            "overall_risk_score": round(
                overall_risk_score,
                2
            ),

            "rolling_fake_probability": round(
                rolling_fake_probability,
                4
            ),

            "session_alert": (
                session_alert
            ),

            "requires_verification": (
                requires_verification
            ),

            # ----------------------------------------------
            # Speaker evidence
            # ----------------------------------------------

            "speaker_verification": {
                "enabled": (
                    speaker_verifier is not None
                ),

                "reference_file": (
                    None
                    if REFERENCE_AUDIO is None
                    else REFERENCE_AUDIO.name
                ),

                "similarity_score": (
                    None
                    if speaker_similarity is None
                    else round(
                        speaker_similarity,
                        4
                    )
                ),

                "same_speaker": (
                    speaker_same
                ),

                "mismatch_score": (
                    None
                    if final_result[
                        "speaker_mismatch_score"
                    ] is None
                    else round(
                        final_result[
                            "speaker_mismatch_score"
                        ],
                        2
                    )
                ),
            },
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Real-time prediction failed: "
                f"{error}"
            ),
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded audio
        # ----------------------------------------------------

        if temp_path is not None:

            try:

                temp_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass