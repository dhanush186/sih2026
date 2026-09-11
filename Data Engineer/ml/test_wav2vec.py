from pathlib import Path

import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"D:\kaggle_cache\datasets\mohammedabdeldayem"
    r"\the-fake-or-real-dataset\versions\2"
    r"\for-original\for-original"
)

SAMPLE_RATE = 16000
DURATION = 2


# ============================================================
# FIND ONE REAL AUDIO FILE
# ============================================================

real_dir = DATASET_ROOT / "training" / "real"

audio_files = [
    p for p in real_dir.iterdir()
    if p.is_file()
    and p.suffix.lower().strip() in {".wav", ".mp3"}
    and p.stat().st_size > 0
]

if not audio_files:
    raise FileNotFoundError(
        "No valid audio files found."
    )

audio_path = audio_files[0]

print("Audio file:", audio_path.name)


# ============================================================
# LOAD AUDIO
# ============================================================

audio, _ = librosa.load(
    audio_path,
    sr=SAMPLE_RATE,
    mono=True,
    duration=DURATION
)

print("Audio samples:", len(audio))


# ============================================================
# LOAD PRETRAINED WAV2VEC2
# ============================================================

print("Loading processor...")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

print("Loading model...")

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
)

model.eval()


# ============================================================
# PREPARE INPUT
# ============================================================

inputs = processor(
    audio,
    sampling_rate=SAMPLE_RATE,
    return_tensors="pt"
)


print(
    "Input shape:",
    inputs.input_values.shape
)


# ============================================================
# GENERATE SPEECH EMBEDDING
# ============================================================

with torch.no_grad():

    outputs = model(
        **inputs
    )


# ============================================================
# RESULTS
# ============================================================

print(
    "Wav2Vec2 output shape:",
    outputs.last_hidden_state.shape
)

print("Wav2Vec2 test successful!")