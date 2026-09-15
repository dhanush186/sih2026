"""
spectrogram_generator.py
--------------------------
Generates Mel spectrograms from audio. A spectrogram is a 2D "image"
of sound (time x frequency) and is exactly the kind of input a
CNN-based deepfake detector expects.

This module:
1. Computes the Mel spectrogram (linear power scale).
2. Converts it to decibels (log scale) — matches human hearing
   perception and is the standard input format for ML models.
3. Saves it as a .png image for visualization/debugging.
4. Returns the raw array for direct use in ML pipelines.
"""

import os
import numpy as np
import librosa
import matplotlib

# Use a non-interactive backend so this works on servers / no-display
# environments (important when teammates run this inside FastAPI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display

from utils.audio_utils import SPECTROGRAM_DIR, get_filename_without_extension, ensure_dir


def generate_mel_spectrogram(waveform: np.ndarray, sr: int, n_mels: int = 128):
    """
    Generate a Mel spectrogram in power scale.

    Returns
    -------
    np.ndarray of shape (n_mels, num_frames)
    """
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels)
    return mel_spec


def convert_to_db(mel_spectrogram: np.ndarray) -> np.ndarray:
    """
    Convert a power-scale Mel spectrogram to decibel (log) scale.
    This is the format almost every audio ML model expects as input.
    """
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def save_spectrogram_image(mel_spec_db: np.ndarray, sr: int, file_name: str,
                            output_dir: str = SPECTROGRAM_DIR) -> str:
    """
    Save the spectrogram as a .png image (for visual inspection /
    reports / demo screenshots).

    Returns the full path to the saved image.
    """
    ensure_dir(output_dir)
    base_name = get_filename_without_extension(file_name)
    save_path = os.path.join(output_dir, f"{base_name}_spectrogram.png")

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram: {base_name}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # important: free memory, especially in a server/API context

    return save_path


def generate_and_save(waveform: np.ndarray, sr: int, file_name: str,
                       n_mels: int = 128, output_dir: str = SPECTROGRAM_DIR):
    """
    Convenience function: run the full spectrogram pipeline in one call.

    Returns
    -------
    dict with keys: 'mel_spec' (power scale), 'mel_spec_db' (dB scale),
    'image_path' (where the PNG was saved).
    """
    mel_spec = generate_mel_spectrogram(waveform, sr, n_mels=n_mels)
    mel_spec_db = convert_to_db(mel_spec)
    image_path = save_spectrogram_image(mel_spec_db, sr, file_name, output_dir)

    return {
        "mel_spec": mel_spec,
        "mel_spec_db": mel_spec_db,
        "image_path": image_path,
    }


if __name__ == "__main__":
    # Quick smoke test with a synthetic tone.
    sr = 22050
    t = np.linspace(0, 2, int(sr * 2))
    fake_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    result = generate_and_save(fake_audio, sr, "test_tone.wav")
    print("Mel spectrogram shape:", result["mel_spec"].shape)
    print("Saved image to:", result["image_path"])
