"""
Generate a minimal valid 1-second WAV file for integration testing.

A WAV file has a 44-byte header + PCM audio data.
This produces a valid 1-second 44100 Hz mono 16-bit silent WAV.
"""
import struct
import os

def make_wav(path: str, duration_s: float = 1.0, sample_rate: int = 44100):
    n_samples = int(sample_rate * duration_s)
    n_bytes = n_samples * 2  # 16-bit = 2 bytes per sample
    data_size = n_bytes
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,             # PCM subchunk size
        1,              # Audio format: PCM
        1,              # Channels: mono
        sample_rate,
        sample_rate * 2,  # Byte rate
        2,              # Block align
        16,             # Bits per sample
        b"data",
        data_size,
    )
    # Silent PCM data (all zeros = silence)
    pcm = b"\x00" * data_size

    with open(path, "wb") as f:
        f.write(header + pcm)

    print(f"Created: {path}  ({os.path.getsize(path)} bytes)")

if __name__ == "__main__":
    make_wav("test_audio.wav")
