"""
prosody_analyzer.py
----------------------
Prosody = the "rhythm and melody" of speech: pitch, loudness, pace,
and pauses. Cloned/synthetic voices often have subtly unnatural
prosody (too flat, too regular, wrong pause patterns), so these
features are valuable signals for deepfake detection.

Extracts:
- pitch / F0 (fundamental frequency) statistics
- energy (loudness) statistics
- speaking rate estimate (syllable-ish rate via onset detection)
- pause/silence ratio
"""

import numpy as np
import librosa


def extract_pitch(waveform: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 400.0):
    """
    Estimate the fundamental frequency (pitch/F0) over time using
    librosa's pyin algorithm (works well for speech).

    Returns
    -------
    f0 : np.ndarray
        Estimated pitch per frame (NaN where unvoiced/silent).
    voiced_flag : np.ndarray
        Boolean array — True where a pitch was confidently detected.
    """
    f0, voiced_flag, _voiced_prob = librosa.pyin(
        waveform, fmin=fmin, fmax=fmax, sr=sr
    )
    return f0, voiced_flag


def extract_energy(waveform: np.ndarray, frame_length: int = 2048, hop_length: int = 512):
    """
    Compute short-term energy (loudness) of the signal using RMS
    (root-mean-square) over sliding frames.
    """
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)
    return rms.flatten()


def estimate_speaking_rate(waveform: np.ndarray, sr: int) -> float:
    """
    Rough estimate of speaking rate using onset (syllable-like peak)
    detection: counts detected onsets per second.

    This is an approximation — good enough as a comparative feature,
    not a precise linguistic measurement.
    """
    onsets = librosa.onset.onset_detect(y=waveform, sr=sr, units="time")
    duration = len(waveform) / float(sr)
    if duration == 0:
        return 0.0
    return len(onsets) / duration  # onsets per second


def estimate_pause_ratio(waveform: np.ndarray, top_db: int = 25) -> float:
    """
    Estimate what fraction of the audio is silence/pauses.
    0.0 = no silence, 1.0 = entirely silence.
    """
    intervals = librosa.effects.split(waveform, top_db=top_db)
    voiced_samples = sum(end - start for start, end in intervals)
    total_samples = len(waveform)

    if total_samples == 0:
        return 0.0

    silent_samples = total_samples - voiced_samples
    return silent_samples / total_samples


def analyze_prosody(waveform: np.ndarray, sr: int) -> dict:
    """
    Run the full prosody analysis and return a clean dictionary of
    summary statistics (easy to log, store, or feed into a model).
    """
    f0, voiced_flag = extract_pitch(waveform, sr)
    energy = extract_energy(waveform)

    # Only look at frames where pitch was actually detected (voiced speech)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

    prosody_features = {
        "pitch_mean": float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0,
        "pitch_std": float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0,
        "pitch_min": float(np.min(voiced_f0)) if len(voiced_f0) > 0 else 0.0,
        "pitch_max": float(np.max(voiced_f0)) if len(voiced_f0) > 0 else 0.0,
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
        "speaking_rate": float(estimate_speaking_rate(waveform, sr)),
        "pause_ratio": float(estimate_pause_ratio(waveform)),
    }

    return prosody_features


if __name__ == "__main__":
    # Quick smoke test with a synthetic tone (won't have realistic
    # pitch/prosody, just verifies the code runs end-to-end).
    sr = 22050
    t = np.linspace(0, 2, int(sr * 2))
    fake_audio = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)

    features = analyze_prosody(fake_audio, sr)
    print("Prosody features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
