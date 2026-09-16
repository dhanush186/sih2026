import json

from analysis_database import insert_analysis_result, get_analysis


sample_result = {
    "filename": "sample.wav",

    "audio_analysis": {
        "audio_path": "backend/uploads/sample.wav",
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
            "pause_ratio": 0.0
        },
        "speaker_verification": None
    },

    "ml_prediction": {
        "prediction": 1,
        "label": "FAKE",
        "real_probability": 0.0218,
        "fake_probability": 0.9782,
        "risk_level": "HIGH"
    },

    "risk_assessment": {
        "risk_score": 39.13,
        "risk_level": "MEDIUM",
        "confidence": 20.0,
        "risk_factors": [
            "High AI-generated voice probability"
        ],
        "recommendation": [
            "Perform additional caller verification",
            "Monitor the request carefully"
        ],
        "alert": "MEDIUM RISK: Suspicious call characteristics detected",
        "security_decision": "VERIFY"
    }
}


analysis_id = insert_analysis_result(
    filename=sample_result["filename"],
    audio_analysis=sample_result["audio_analysis"],
    ml_prediction=sample_result["ml_prediction"],
    risk_assessment=sample_result["risk_assessment"],
)

print("Inserted analysis ID:", analysis_id)


result = get_analysis(analysis_id)

print("\nDatabase result:")
print(json.dumps(result, default=str, indent=2))