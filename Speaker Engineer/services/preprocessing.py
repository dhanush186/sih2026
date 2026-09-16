"""
preprocessing.py
------------------
Cleans up raw audio before feature extraction:
- resample to a target sample rate
- convert stereo -> mono
- normalize volume
- trim/remove silence
- split long audio into fixed-length chunks

All functions take/return NumPy waveforms so they can be chained
together easily inside main.py.
"""

import numpy as np
import librosa


def resample_audio(waveform: np.ndarray, original_sr: int, target_sr: int = 16000):
    """
    Resample audio to a target sample rate (default 16kHz, which is
    the standard rate used by most speech models).
    """
    if original_sr == target_sr:
        return waveform, target_sr

    resampled = librosa.resample(waveform, orig_sr=original_sr, target_sr=target_sr)
    return resampled, target_sr


def convert_to_mono(waveform: np.ndarray) -> np.ndarray:
    """
    Convert stereo (2D) audio to mono (1D) by averaging channels.
    If the audio is already mono, it is returned unchanged.
    """
    if waveform.ndim > 1:
        return np.mean(waveform, axis=0)
    return waveform


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """
    Normalize audio so its peak amplitude is 1.0.
    This prevents very quiet or very loud recordings from throwing
    off feature extraction.
    """
    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude == 0:
        # Silent audio — nothing to normalize.
        return waveform
    return waveform / max_amplitude


def remove_silence(waveform: np.ndarray, top_db: int = 25) -> np.ndarray:
    """
    Trim leading/trailing silence and drop long silent gaps inside
    the audio using librosa's silence-split utility.

    top_db : how many decibels below peak volume counts as "silence".
             Lower value = more aggressive trimming.
    """
    # librosa.effects.split returns [start, end] sample intervals
    # that are considered "non-silent".
    intervals = librosa.effects.split(waveform, top_db=top_db)

    if len(intervals) == 0:
        # Entire clip was classified as silence — return it unchanged
        # so we don't hand back an empty array.
        return waveform

    non_silent_chunks = [waveform[start:end] for start, end in intervals]
    return np.concatenate(non_silent_chunks)


def segment_audio(waveform: np.ndarray, sr: int, chunk_seconds: float = 3.0):
    """
    Split a long audio waveform into a list of smaller, fixed-length
    chunks. The last chunk may be shorter than chunk_seconds.

    Returns
    -------
    List[np.ndarray] : list of waveform chunks.
    """
    chunk_size = int(chunk_seconds * sr)
    if chunk_size <= 0:
        raise ValueError("chunk_seconds * sr must be greater than 0")

    chunks = [
        waveform[i:i + chunk_size]
        for i in range(0, len(waveform), chunk_size)
    ]
    return chunks


def preprocess_pipeline(waveform: np.ndarray, sr: int, target_sr: int = 16000):
    """
    Convenience function that runs the full standard preprocessing
    chain in one call:
        mono -> resample -> normalize -> remove silence

    Returns
    -------
    waveform, sr : the cleaned waveform and its (possibly new) sample rate.
    """
    waveform = convert_to_mono(waveform)
    waveform, sr = resample_audio(waveform, sr, target_sr)
    waveform = normalize_audio(waveform)
    waveform = remove_silence(waveform)
    return waveform, sr


if __name__ == "__main__":
    # Quick smoke test using a synthetic sine wave (no file needed).
    sr = 22050
    t = np.linspace(0, 2, int(sr * 2))
    fake_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone

    cleaned, new_sr = preprocess_pipeline(fake_audio, sr)
    print(f"Original samples: {len(fake_audio)} @ {sr}Hz")
    print(f"Cleaned samples: {len(cleaned)} @ {new_sr}Hz")

    chunks = segment_audio(cleaned, new_sr, chunk_seconds=1.0)
    print(f"Split into {len(chunks)} chunk(s)")
