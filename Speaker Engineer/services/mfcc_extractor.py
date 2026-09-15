"""
mfcc_extractor.py
-------------------
Extracts MFCC (Mel-Frequency Cepstral Coefficients) features, which
describe the "shape" of the audio spectrum over time. This is one
of the most common feature types used in speech/voice ML models.

Also computes delta (velocity) and delta-delta (acceleration)
features, which capture how the MFCCs change over time — useful
for detecting unnatural transitions typical of cloned voices.
"""

import os
import numpy as np
import librosa

from utils.audio_utils import MFCC_DIR, get_filename_without_extension, ensure_dir


def extract_mfcc(waveform: np.ndarray, sr: int, n_mfcc: int = 13):
    """
    Extract MFCC features from a waveform.

    Returns
    -------
    np.ndarray of shape (n_mfcc, num_frames)
    """
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def extract_mfcc_with_deltas(waveform: np.ndarray, sr: int, n_mfcc: int = 13):
    """
    Extract MFCC + delta + delta-delta features and stack them
    together into one feature matrix.

    Returns
    -------
    dict with keys: 'mfcc', 'delta', 'delta2', 'combined'
        'combined' stacks all three vertically -> shape (n_mfcc*3, num_frames)
    """
    mfcc = extract_mfcc(waveform, sr, n_mfcc=n_mfcc)

    # Delta = rate of change of MFCCs (1st derivative)
    delta = librosa.feature.delta(mfcc)

    # Delta-delta = rate of change of the deltas (2nd derivative)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, delta, delta2])

    return {
        "mfcc": mfcc,
        "delta": delta,
        "delta2": delta2,
        "combined": combined,
    }


def save_mfcc_features(features: np.ndarray, file_name: str, output_dir: str = MFCC_DIR) -> str:
    """
    Save MFCC features to disk as a .npy file so they can be reused
    later without recomputing (e.g. by the CNN/deepfake model).

    Returns the full path to the saved file.
    """
    ensure_dir(output_dir)
    base_name = get_filename_without_extension(file_name)
    save_path = os.path.join(output_dir, f"{base_name}_mfcc.npy")
    np.save(save_path, features)
    return save_path


if __name__ == "__main__":
    # Quick smoke test with a synthetic tone.
    sr = 22050
    t = np.linspace(0, 2, int(sr * 2))
    fake_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    result = extract_mfcc_with_deltas(fake_audio, sr)
    print("MFCC shape:", result["mfcc"].shape)
    print("Delta shape:", result["delta"].shape)
    print("Delta2 shape:", result["delta2"].shape)
    print("Combined shape:", result["combined"].shape)

    saved_path = save_mfcc_features(result["combined"], "test_tone.wav")
    print("Saved MFCC features to:", saved_path)
