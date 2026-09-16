"""
test_pipeline.py
------------------
Basic sanity tests using synthetic (generated) audio, so they run
instantly with no sample files required. Not exhaustive — just enough
to confirm every module imports and runs without crashing.

Run with:
    python -m pytest tests/ -v
or simply:
    python tests/test_pipeline.py
"""

import sys
import os
import numpy as np

# Allow running this file directly (python tests/test_pipeline.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.preprocessing import preprocess_pipeline, segment_audio
from services.mfcc_extractor import extract_mfcc_with_deltas
from services.spectrogram_generator import generate_mel_spectrogram, convert_to_db
from services.prosody_analyzer import analyze_prosody
from services.speaker_verifier import compare_speakers


def _make_tone(freq=150, seconds=2, sr=22050):
    t = np.linspace(0, seconds, int(sr * seconds))
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr


def test_preprocessing():
    wave, sr = _make_tone()
    cleaned, new_sr = preprocess_pipeline(wave, sr)
    assert len(cleaned) > 0
    assert new_sr == 16000
    chunks = segment_audio(cleaned, new_sr, chunk_seconds=0.5)
    assert len(chunks) > 0


def test_mfcc_extraction():
    wave, sr = _make_tone()
    result = extract_mfcc_with_deltas(wave, sr)
    assert result["mfcc"].shape[0] == 13
    assert result["combined"].shape[0] == 39  # 13 * 3


def test_spectrogram_generation():
    wave, sr = _make_tone()
    mel = generate_mel_spectrogram(wave, sr)
    mel_db = convert_to_db(mel)
    assert mel.shape == mel_db.shape
    assert mel.shape[0] == 128  # default n_mels


def test_prosody_analysis():
    wave, sr = _make_tone()
    features = analyze_prosody(wave, sr)
    expected_keys = {
        "pitch_mean", "pitch_std", "pitch_min", "pitch_max",
        "energy_mean", "energy_std", "speaking_rate", "pause_ratio",
    }
    assert expected_keys.issubset(features.keys())


def test_speaker_verification():
    wave_a, sr = _make_tone(freq=150)
    wave_a2, _ = _make_tone(freq=150)
    wave_b, _ = _make_tone(freq=300)

    same_result = compare_speakers(wave_a, sr, wave_a2, sr)
    diff_result = compare_speakers(wave_a, sr, wave_b, sr)

    assert "similarity_score" in same_result
    assert "same_speaker" in same_result
    assert same_result["similarity_score"] >= diff_result["similarity_score"]


if __name__ == "__main__":
    # Allow running without pytest too — just call each test manually.
    test_preprocessing()
    print("test_preprocessing passed")
    test_mfcc_extraction()
    print("test_mfcc_extraction passed")
    test_spectrogram_generation()
    print("test_spectrogram_generation passed")
    test_prosody_analysis()
    print("test_prosody_analysis passed")
    test_speaker_verification()
    print("test_speaker_verification passed")
    print("\nAll tests passed!")
