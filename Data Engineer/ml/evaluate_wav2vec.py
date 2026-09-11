from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import Wav2Vec2Processor, Wav2Vec2Model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from dataset import FakeRealDataset


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32

MODEL_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec2_deepfake_classifier.pth"
)

CACHE_DIR = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec_cache"
)

TEST_CACHE_PATH = (
    CACHE_DIR / "testing_embeddings.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
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
# LOAD AUDIO
# ============================================================

def load_audio(path):

    import librosa
    import numpy as np

    audio, _ = librosa.load(
        path,
        sr=16000,
        mono=True,
        duration=2
    )

    target_length = 32000

    if len(audio) < target_length:

        audio = np.pad(
            audio,
            (0, target_length - len(audio)),
            mode="constant"
        )

    else:

        audio = audio[:target_length]

    return audio.astype("float32")


# ============================================================
# EXTRACT TEST EMBEDDINGS
# ============================================================

def extract_test_embeddings(
    dataset,
    processor,
    encoder
):

    if TEST_CACHE_PATH.exists():

        print()
        print(
            "Loading cached testing embeddings..."
        )

        cached = torch.load(
            TEST_CACHE_PATH,
            map_location="cpu"
        )

        return (
            cached["embeddings"],
            cached["labels"]
        )

    print()
    print(
        "Extracting Wav2Vec2 embeddings "
        "for testing..."
    )

    embeddings = []
    labels = []

    encoder.eval()

    with torch.no_grad():

        for count, (path, label) in enumerate(
            dataset.files,
            start=1
        ):

            audio = load_audio(path)

            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )

            input_values = (
                inputs.input_values
                .to(DEVICE)
            )

            outputs = encoder(
                input_values=input_values
            )

            hidden_states = (
                outputs.last_hidden_state
            )

            embedding = (
                hidden_states
                .mean(dim=1)
                .squeeze(0)
                .cpu()
            )

            embeddings.append(
                embedding
            )

            labels.append(label)

            if count % 100 == 0 or count == len(dataset):

                print(
                    f"Processed "
                    f"{count}/{len(dataset)}"
                )

    embeddings = torch.stack(
        embeddings
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels
        },
        TEST_CACHE_PATH
    )

    print()
    print(
        "Cached testing embeddings saved to:"
    )
    print(TEST_CACHE_PATH)

    return embeddings, labels


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("WAV2VEC2 DEEPFAKE DETECTION - EVALUATION")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}\n\n"
            f"Run train_wav2vec.py first."
        )

    # --------------------------------------------------------
    # Load processor
    # --------------------------------------------------------

    print()
    print(
        "Loading Wav2Vec2 processor..."
    )

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base"
    )

    # --------------------------------------------------------
    # Load encoder
    # --------------------------------------------------------

    print(
        "Loading Wav2Vec2 encoder..."
    )

    encoder = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base"
    )

    encoder = encoder.to(DEVICE)

    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    print(
        "Wav2Vec2 encoder loaded."
    )

    # --------------------------------------------------------
    # Load testing dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading testing dataset..."
    )

    test_dataset = FakeRealDataset(
        "testing"
    )

    print()
    print(
        f"Testing samples: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Extract/load embeddings
    # --------------------------------------------------------

    test_embeddings, test_labels = (
        extract_test_embeddings(
            test_dataset,
            processor,
            encoder
        )
    )

    print()
    print(
        "Testing embedding shape:",
        test_embeddings.shape
    )

    # --------------------------------------------------------
    # Create classifier
    # --------------------------------------------------------

    print()
    print(
        "Creating classifier..."
    )

    model = EmbeddingClassifier()

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Load trained classifier
    # --------------------------------------------------------

    print(
        "Loading trained classifier..."
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    test_data = TensorDataset(
        test_embeddings,
        test_labels
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print()
    print(
        "Running predictions..."
    )

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch_index, (
            embeddings,
            labels
        ) in enumerate(
            test_loader,
            start=1
        ):

            embeddings = embeddings.to(
                DEVICE
            )

            outputs = model(
                embeddings
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            if batch_index % 20 == 0:

                print(
                    f"Processed batch "
                    f"{batch_index}/"
                    f"{len(test_loader)}"
                )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=["REAL", "FAKE"],
        zero_division=0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print()
    print("=" * 60)
    print("WAV2VEC2 EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Test samples : {len(all_labels)}"
    )

    print(
        f"Accuracy     : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision    : {precision * 100:.2f}%"
    )

    print(
        f"Recall       : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score     : {f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print(
        "CONFUSION MATRIX"
    )

    print("-" * 40)

    print(
        "                 Predicted"
    )

    print(
        "                 REAL    FAKE"
    )

    print(
        f"Actual REAL      "
        f"{matrix[0][0]:6d}"
        f"{matrix[0][1]:8d}"
    )

    print(
        f"Actual FAKE      "
        f"{matrix[1][0]:6d}"
        f"{matrix[1][1]:8d}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print(
        "CLASSIFICATION REPORT"
    )

    print("-" * 60)

    print(report)

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print("REAL = 0")
    print("FAKE = 1")
    print()

    print(
        "The metrics above are calculated on "
        "the separate testing split."
    )

    print(
        "Fake precision/recall/F1 use FAKE "
        "as the positive class."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()