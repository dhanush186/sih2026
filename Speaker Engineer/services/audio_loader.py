"""
audio_loader.py
-----------------
Responsible for ONE job: getting audio off disk and into memory
as a NumPy waveform + sample rate, so every other module can work
with a consistent format.

Supports WAV, MP3, FLAC, and most common audio formats because
librosa uses 'soundfile' / 'audioread' under the hood.
"""

import os
import numpy as np
import librosa

from utils.audio_utils import file_exists


def check_audio_file_exists(file_path: str) -> bool:
    """
    Check whether the given audio file actually exists on disk.
    Returns True/False instead of crashing, so callers can decide
    what to do (e.g. show a friendly error message).
    """
    return file_exists(file_path)


def load_audio(file_path: str, sample_rate: int = None):
    """
    Load an audio file and return (waveform, sample_rate).

    Parameters
    ----------
    file_path : str
        Path to a .wav / .mp3 / .flac file.
    sample_rate : int, optional
        If given, librosa will resample the audio to this rate
        while loading. If None, the file's native sample rate is kept.

    Returns
    -------
    waveform : np.ndarray
        1D NumPy array of audio samples (mono).
    sr : int
        The sample rate of the returned waveform.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file exists but librosa cannot read it (corrupted /
        unsupported format).
    """
    if not check_audio_file_exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        # sr=None keeps the file's original sample rate unless the
        # caller explicitly asked for one. mono=True merges channels.
        waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    except Exception as error:
        raise ValueError(f"Could not read audio file '{file_path}': {error}")

    if waveform is None or len(waveform) == 0:
        raise ValueError(f"Audio file '{file_path}' appears to be empty.")

    return waveform.astype(np.float32), sr


def get_audio_duration(waveform: np.ndarray, sr: int) -> float:
    """Return the duration of a waveform in seconds."""
    return len(waveform) / float(sr)


if __name__ == "__main__":
    # Quick manual test: python services/audio_loader.py <path_to_wav>
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_loader.py <path_to_audio_file>")
    else:
        test_path = sys.argv[1]
        wave, rate = load_audio(test_path)
        print(f"Loaded '{test_path}'")
        print(f"Sample rate: {rate} Hz")
        print(f"Duration: {get_audio_duration(wave, rate):.2f} seconds")
        print(f"Samples: {len(wave)}")
