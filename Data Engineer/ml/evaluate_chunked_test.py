from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

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

# NEW FULL-DATA MODEL
MODEL_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec2_deepfake_classifier_full.pth"
)

MODEL_NAME = "facebook/wav2vec2-base"

SAMPLE_RATE = 16000

CHUNK_DURATION = 2

CHUNK_SAMPLES = (
    SAMPLE_RATE * CHUNK_DURATION
)

BATCH_SIZE = 32

# Chunk-level threshold.
# Used for reporting individual chunk predictions.
CHUNK_THRESHOLD = 0.50

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

            nn.Linear(
                768,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                64,
                2
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(audio_path):

    waveform, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    waveform = np.asarray(
        waveform,
        dtype=np.float32
    )

    return waveform


# ============================================================
# CREATE 2-SECOND CHUNKS
# ============================================================

def create_chunks(waveform):

    if waveform.size == 0:

        return []

    chunks = []

    total_samples = len(
        waveform
    )

    start = 0

    while start < total_samples:

        end = (
            start
            + CHUNK_SAMPLES
        )

        chunk = waveform[
            start:end
        ]

        # Pad final chunk
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

        start += CHUNK_SAMPLES

    return chunks


# ============================================================
# RECORDING-LEVEL RISK AGGREGATION
# ============================================================

def calculate_recording_risk(
    fake_probabilities
):

    values = np.asarray(
        fake_probabilities,
        dtype=np.float32
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    values = np.clip(
        values,
        0.0,
        1.0
    )

    if len(values) == 0:

        return {
            "risk_level": "LOW",
            "recording_fake_probability": 0.0,
            "mean_fake_probability": 0.0,
            "max_fake_probability": 0.0,
            "weighted_fake_probability": 0.0,
            "high_chunk_count": 0,
            "extreme_chunk_count": 0,
        }

    mean_probability = float(
        np.mean(values)
    )

    max_probability = float(
        np.max(values)
    )

    # Newer chunks receive slightly greater weight.
    weights = np.arange(
        1,
        len(values) + 1,
        dtype=np.float32
    )

    weights = weights / np.sum(
        weights
    )

    weighted_probability = float(
        np.sum(
            values * weights
        )
    )

    high_chunk_count = int(
        np.sum(
            values >= 0.80
        )
    )

    extreme_chunk_count = int(
        np.sum(
            values >= 0.90
        )
    )

    # --------------------------------------------------------
    # Recording-level rules
    # --------------------------------------------------------

    if (
        max_probability >= 0.90
        or high_chunk_count >= 2
        or (
            max_probability >= 0.80
            and mean_probability >= 0.50
        )
    ):

        risk_level = "HIGH"

    elif (
        max_probability >= 0.50
        or mean_probability >= 0.50
    ):

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    return {
        "risk_level": risk_level,

        # Mean is used as the explanatory recording score.
        # This is NOT a calibrated probability.
        "recording_fake_probability": (
            mean_probability
        ),

        "mean_fake_probability": (
            mean_probability
        ),

        "max_fake_probability": (
            max_probability
        ),

        "weighted_fake_probability": (
            weighted_probability
        ),

        "high_chunk_count": (
            high_chunk_count
        ),

        "extreme_chunk_count": (
            extreme_chunk_count
        ),
    }


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print()
    print("Loading Wav2Vec2 processor...")

    processor = (
        Wav2Vec2Processor
        .from_pretrained(
            MODEL_NAME
        )
    )

    print("Loading Wav2Vec2 encoder...")

    encoder = (
        Wav2Vec2Model
        .from_pretrained(
            MODEL_NAME
        )
        .to(
            DEVICE
        )
    )

    encoder.eval()

    for parameter in (
        encoder.parameters()
    ):

        parameter.requires_grad = False

    print(
        "Wav2Vec2 encoder loaded."
    )

    print(
        "\nLoading full-data classifier..."
    )

    classifier = (
        EmbeddingClassifier()
        .to(
            DEVICE
        )
    )

    classifier.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    classifier.eval()

    print(
        "Full-data classifier loaded."
    )

    return (
        processor,
        encoder,
        classifier
    )


# ============================================================
# PREDICT CHUNKS
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

        batch_chunks = chunks[
            start:
            start + BATCH_SIZE
        ]

        inputs = processor(
            batch_chunks,
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
                )
                if attention_mask is not None
                else None
            )

            # Mean-pool the Wav2Vec2
            # hidden states across time.
            embeddings = (
                outputs.last_hidden_state
                .mean(dim=1)
            )

            logits = classifier(
                embeddings
            )

            batch_probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )[:, 1]
                .cpu()
                .numpy()
            )

        probabilities.extend(
            batch_probabilities
            .tolist()
        )

    return probabilities


# ============================================================
# COLLECT TEST FILES
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

                if (
                    path.stat().st_size
                    == 0
                ):

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
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "CHUNKED RECORDING-LEVEL "
        "WAV2VEC2 TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Chunk duration: "
        f"{CHUNK_DURATION} seconds"
    )

    print(
        f"Chunk threshold: "
        f"{CHUNK_THRESHOLD:.2f}"
    )

    print(
        "\nModel:"
    )

    print(
        MODEL_PATH
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    (
        processor,
        encoder,
        classifier
    ) = load_models()

    print(
        "\nFinding testing files..."
    )

    test_files = (
        get_test_files()
    )

    print(
        f"Test recordings found: "
        f"{len(test_files)}"
    )

    real_count = sum(
        1
        for _, label in test_files
        if label == 0
    )

    fake_count = sum(
        1
        for _, label in test_files
        if label == 1
    )

    print(
        f"Real recordings: {real_count}"
    )

    print(
        f"Fake recordings: {fake_count}"
    )

    # --------------------------------------------------------
    # Recording-level results
    # --------------------------------------------------------

    true_labels = []

    predicted_labels = []

    recording_scores = []

    total_chunks = 0

    print(
        "\nStarting chunked evaluation..."
    )

    print(
        "This is CPU-intensive and may take "
        "a while."
    )

    print()

    for index, (
        audio_path,
        true_label
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

            if len(chunks) == 0:

                print(
                    f"[SKIP] Empty: "
                    f"{audio_path.name}"
                )

                continue

            fake_probabilities = (
                predict_chunks(
                    chunks,
                    processor,
                    encoder,
                    classifier
                )
            )

            risk = (
                calculate_recording_risk(
                    fake_probabilities
                )
            )

            # ------------------------------------------------
            # Recording classification
            #
            # LOW  = REAL
            # MEDIUM/HIGH = FAKE
            # ------------------------------------------------

            predicted_label = (
                1
                if risk["risk_level"]
                in {
                    "MEDIUM",
                    "HIGH"
                }
                else 0
            )

            true_labels.append(
                true_label
            )

            predicted_labels.append(
                predicted_label
            )

            recording_scores.append(
                risk[
                    "recording_fake_probability"
                ]
            )

            total_chunks += len(
                chunks
            )

            # Progress every 100 files
            if (
                index % 100 == 0
                or index == len(
                    test_files
                )
            ):

                current_accuracy = (
                    accuracy_score(
                        true_labels,
                        predicted_labels
                    )
                )

                print(
                    f"Processed "
                    f"{index}/"
                    f"{len(test_files)} | "
                    f"Chunks: "
                    f"{total_chunks} | "
                    f"Current accuracy: "
                    f"{current_accuracy * 100:.2f}%"
                )

        except Exception as error:

            print(
                f"\n[ERROR] "
                f"{audio_path}"
            )

            print(
                f"Reason: {error}"
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CHUNKED RECORDING-LEVEL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Evaluated recordings: "
        f"{len(true_labels)}"
    )

    print(
        f"Total chunks processed: "
        f"{total_chunks}"
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    print(
        f"\nAccuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score : "
        f"{f1 * 100:.2f}%"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        true_labels,
        predicted_labels
    )

    print(
        "\nCONFUSION MATRIX"
    )

    print(
        "----------------------------------------"
    )

    print(
        "                 Predicted"
    )

    print(
        "                 REAL    FAKE"
    )

    print(
        f"Actual REAL     "
        f"{matrix[0][0]:6d}  "
        f"{matrix[0][1]:7d}"
    )

    print(
        f"Actual FAKE     "
        f"{matrix[1][0]:6d}  "
        f"{matrix[1][1]:7d}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nCLASSIFICATION REPORT"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=[
                "REAL",
                "FAKE"
            ],
            zero_division=0
        )
    )

    # ========================================================
    # RECORDING SCORE SUMMARY
    # ========================================================

    if recording_scores:

        scores = np.asarray(
            recording_scores,
            dtype=np.float32
        )

        print(
            "RECORDING SCORE SUMMARY"
        )

        print(
            "------------------------------------------------------------"
        )

        print(
            f"Minimum recording score: "
            f"{np.min(scores):.4f}"
        )

        print(
            f"Maximum recording score: "
            f"{np.max(scores):.4f}"
        )

        print(
            f"Mean recording score: "
            f"{np.mean(scores):.4f}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CHUNKED EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()