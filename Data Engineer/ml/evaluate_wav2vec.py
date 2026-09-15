from pathlib import Path

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
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "facebook/wav2vec2-base"

MODEL_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec2_deepfake_classifier_full.pth"
)

CACHE_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec_cache\testing_embeddings.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 32

# Selected using the validation split
CLASSIFICATION_THRESHOLD = 0.55


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
# LOAD TEST EMBEDDINGS
# ============================================================

def load_testing_embeddings():

    if not CACHE_PATH.exists():

        raise FileNotFoundError(
            f"Testing embeddings not found:\n"
            f"{CACHE_PATH}\n\n"
            f"Run the embedding-generation step first."
        )

    print(
        "\nLoading cached testing embeddings..."
    )

    data = torch.load(
        CACHE_PATH,
        map_location="cpu"
    )

    embeddings = data["embeddings"]
    labels = data["labels"]

    print(
        f"Testing embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Testing label shape: "
        f"{labels.shape}"
    )

    return embeddings, labels


# ============================================================
# LOAD CLASSIFIER
# ============================================================

def load_classifier():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Classifier model not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        "\nCreating classifier..."
    )

    classifier = EmbeddingClassifier()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    classifier.load_state_dict(
        state_dict
    )

    classifier = classifier.to(
        DEVICE
    )

    classifier.eval()

    print(
        "Model loaded successfully."
    )

    return classifier


# ============================================================
# RUN PREDICTIONS
# ============================================================

def run_predictions(
    embeddings,
    labels,
    classifier
):

    dataset = TensorDataset(
        embeddings,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_predictions = []
    all_labels = []
    all_fake_probabilities = []

    print()
    print(
        "Running predictions..."
    )

    with torch.no_grad():

        for batch_index, (
            batch_embeddings,
            batch_labels
        ) in enumerate(loader):

            batch_embeddings = (
                batch_embeddings.to(DEVICE)
            )

            outputs = classifier(
                batch_embeddings
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            fake_probabilities = (
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Explicit 0.50 probability threshold
            # ------------------------------------------------

            predictions = (
                fake_probabilities
                >= CLASSIFICATION_THRESHOLD
            ).astype(int)

            all_predictions.extend(
                predictions.tolist()
            )

            all_fake_probabilities.extend(
                fake_probabilities.tolist()
            )

            all_labels.extend(
                batch_labels.numpy().tolist()
            )

            if batch_index % 20 == 0:

                print(
                    f"Processed batch "
                    f"{batch_index + 1}/"
                    f"{len(loader)}"
                )

    return (
        all_predictions,
        all_labels,
        all_fake_probabilities
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("WAV2VEC2 DEEPFAKE DETECTION - THRESHOLD EVALUATION")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Classification threshold: "
        f"{CLASSIFICATION_THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # Load testing embeddings
    # --------------------------------------------------------

    embeddings, labels = (
        load_testing_embeddings()
    )

    print()
    print(
        f"Test samples: "
        f"{len(labels)}"
    )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    classifier = load_classifier()

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    (
        all_predictions,
        all_labels,
        all_fake_probabilities
    ) = run_predictions(
        embeddings,
        labels,
        classifier
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
        target_names=[
            "REAL",
            "FAKE"
        ],
        zero_division=0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("WAV2VEC2 THRESHOLD EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Test samples : "
        f"{len(all_labels)}"
    )

    print(
        f"Threshold    : "
        f"{CLASSIFICATION_THRESHOLD:.2f}"
    )

    print(
        f"Accuracy     : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision    : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall       : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score     : "
        f"{f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("CONFUSION MATRIX")
    print("----------------------------------------")
    print(
        "                 Predicted"
    )
    print(
        "                 REAL    FAKE"
    )

    print(
        f"Actual REAL     "
        f"{matrix[0, 0]:6d}"
        f"{matrix[0, 1]:9d}"
    )

    print(
        f"Actual FAKE     "
        f"{matrix[1, 0]:6d}"
        f"{matrix[1, 1]:9d}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print(
        "CLASSIFICATION REPORT"
    )
    print(
        "------------------------------------------------------------"
    )

    print(
        report
    )

    # --------------------------------------------------------
    # Probability summary
    # --------------------------------------------------------

    fake_probabilities = torch.tensor(
        all_fake_probabilities
    )

    print(
        "FAKE PROBABILITY SUMMARY"
    )
    print(
        "------------------------------------------------------------"
    )

    print(
        f"Minimum fake probability: "
        f"{fake_probabilities.min().item():.4f}"
    )

    print(
        f"Maximum fake probability: "
        f"{fake_probabilities.max().item():.4f}"
    )

    print(
        f"Mean fake probability: "
        f"{fake_probabilities.mean().item():.4f}"
    )

    print()
    print(
        "REAL = 0"
    )

    print(
        "FAKE = 1"
    )

    print(
        "The threshold was selected using the validation split "
        "and evaluated here on the separate testing split."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()