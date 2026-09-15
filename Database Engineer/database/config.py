"""
Configuration loader.

Reads settings from environment variables (via a .env file in development).
Never hardcode secrets here — everything sensitive comes from the environment.
"""

import os
from dotenv import load_dotenv

# Load variables from a .env file if present (no-op in prod if the file is absent)
load_dotenv()


class Settings:
    # --- Database ---
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:password@localhost:3306/voice_security",
    )

    # --- App ---
    APP_NAME: str = os.getenv("APP_NAME", "voice-deepfake-detection-api")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # --- CORS ---
    ALLOWED_ORIGINS: list[str] = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000"
    ).split(",")

    # --- File uploads ---
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    ALLOWED_AUDIO_EXTENSIONS: set[str] = {".wav", ".mp3", ".m4a", ".flac"}

    # --- API Key Authentication ---
    # All protected routes require header: X-API-Key: <API_KEY>
    API_KEY: str = os.getenv("API_KEY", "")

    # --- Optional external ML service (set once your ML teammate exposes an endpoint) ---
    ML_SERVICE_URL: str | None = os.getenv("ML_SERVICE_URL") or None
    AI_API_KEY: str | None = os.getenv("AI_API_KEY")

    # --- Risk thresholds (MVP defaults — Risk/Security engineer owns final values) ---
    RISK_THRESHOLD_SUSPICIOUS: int = int(os.getenv("RISK_THRESHOLD_SUSPICIOUS", "40"))
    RISK_THRESHOLD_HIGH: int = int(os.getenv("RISK_THRESHOLD_HIGH", "70"))


settings = Settings()
