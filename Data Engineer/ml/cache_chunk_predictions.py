from pathlib import Path
import json

import librosa
import numpy as np
import torch
import torch.nn as nn

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"D:\kaggle_cache\datasets"
    r"\mohammedabdeldayem"
    r"\the-fake-or-real-dataset"
    r"\versions\2"
    r"\for-original"
    r"\for-original"
)

MODEL_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec2_deepfake_classifier_full.pth"
)

CACHE_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec_cache"
    r"\test_chunk_predictions.json"
)

MODEL_NAME = "facebook/wav2vec2-base"

SAMPLE_RATE = 16000

CHUNK_DURATION = 2

CHUNK_SAMPLES = (
    SAMPLE_RATE * CHUNK_DURATION
)

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASSIFIER
# ============================================================

class EmbeddingClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# AUDIO
# ============================================================

def load_audio(audio_path):

    waveform, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return np.asarray(
        waveform,
        dtype=np.float32
    )


def create_chunks(waveform):

    chunks = []

    total_samples = len(waveform)

    for start in range(
        0,
        total_samples,
        CHUNK_SAMPLES
    ):

        end = (
            start
            + CHUNK_SAMPLES
        )

        chunk = waveform[start:end]

        if len(chunk) < CHUNK_SAMPLES:

            chunk = np.pad(
                chunk,
                (
                    0,
                    CHUNK_SAMPLES
                    - len(chunk)
                )
            )

        chunks.append(
            chunk.astype(
                np.float32
            )
        )

    return chunks


# ============================================================
# FILE DISCOVERY
# ============================================================

def get_test_files():

    test_root = (
        DATASET_ROOT
        / "testing"
    )

    files = []

    for label_name, label in (
        ("real", 0),
        ("fake", 1)
    ):

        directory = (
            test_root
            / label_name
        )

        for path in sorted(
            directory.iterdir()
        ):

            if not path.is_file():
                continue

            extension = (
                path.suffix
                .lower()
                .strip()
            )

            if extension not in {
                ".wav",
                ".mp3"
            }:
                continue

            try:

                if path.stat().st_size == 0:
                    continue

            except OSError:

                continue

            files.append(
                (
                    path,
                    label
                )
            )

    return files


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():

    print(
        "Loading Wav2Vec2 processor..."
    )

    processor = (
        Wav2Vec2Processor
        .from_pretrained(
            MODEL_NAME
        )
    )

    print(
        "Loading Wav2Vec2 encoder..."
    )

    encoder = (
        Wav2Vec2Model
        .from_pretrained(
            MODEL_NAME
        )
        .to(DEVICE)
    )

    encoder.eval()

    for parameter in encoder.parameters():

        parameter.requires_grad = False

    print(
        "Loading full-data classifier..."
    )

    classifier = (
        EmbeddingClassifier()
        .to(DEVICE)
    )

    classifier.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    classifier.eval()

    return (
        processor,
        encoder,
        classifier
    )


# ============================================================
# CHUNK INFERENCE
# ============================================================

def predict_chunks(
    chunks,
    processor,
    encoder,
    classifier
):

    probabilities = []

    for start in range(
        0,
        len(chunks),
        BATCH_SIZE
    ):

        batch = chunks[
            start:
            start + BATCH_SIZE
        ]

        inputs = processor(
            batch,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        input_values = (
            inputs.input_values
            .to(DEVICE)
        )

        attention_mask = None

        if hasattr(
            inputs,
            "attention_mask"
        ):

            attention_mask = (
                inputs.attention_mask
                .to(DEVICE)
            )

        with torch.no_grad():

            outputs = encoder(
                input_values,
                attention_mask=(
                    attention_mask
                    if attention_mask is not None
                    else None
                )
            )

            embeddings = (
                outputs.last_hidden_state
                .mean(dim=1)
            )

            logits = classifier(
                embeddings
            )

            fake_probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )[:, 1]
                .cpu()
                .numpy()
            )

        probabilities.extend(
            fake_probabilities.tolist()
        )

    return probabilities


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CREATING TEST CHUNK PROBABILITY CACHE"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model:\n{MODEL_PATH}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    test_files = get_test_files()

    print(
        f"\nTest recordings: "
        f"{len(test_files)}"
    )

    processor, encoder, classifier = (
        load_models()
    )

    results = []

    total_chunks = 0

    for index, (
        audio_path,
        label
    ) in enumerate(
        test_files,
        start=1
    ):

        try:

            waveform = load_audio(
                audio_path
            )

            chunks = create_chunks(
                waveform
            )

            if not chunks:
                continue

            probabilities = predict_chunks(
                chunks,
                processor,
                encoder,
                classifier
            )

            record = {
                "path": str(audio_path),
                "label": int(label),
                "num_chunks": len(probabilities),
                "fake_probabilities": [
                    float(value)
                    for value in probabilities
                ]
            }

            results.append(record)

            total_chunks += len(
                probabilities
            )

            if (
                index % 100 == 0
                or index == len(test_files)
            ):

                print(
                    f"Processed "
                    f"{index}/"
                    f"{len(test_files)} | "
                    f"Chunks: "
                    f"{total_chunks}"
                )

        except Exception as error:

            print(
                f"\n[ERROR] "
                f"{audio_path}"
            )

            print(
                f"Reason: {error}"
            )

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CACHE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "CHUNK CACHE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Recordings cached: "
        f"{len(results)}"
    )

    print(
        f"Total chunks cached: "
        f"{total_chunks}"
    )

    print(
        f"Cache file:\n{CACHE_PATH}"
    )


if __name__ == "__main__":
    main()