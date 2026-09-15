from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from predict import load_models



# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_EMBEDDINGS = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec_cache\validation_full_embeddings.pt"
)

CLASSIFIER_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec2_deepfake_classifier_full.pth"
)

BATCH_SIZE = 64


# ============================================================
# LOAD VALIDATION EMBEDDINGS
# ============================================================

def load_validation_data():

    print("Loading validation embeddings...")

    data = torch.load(
        VALIDATION_EMBEDDINGS,
        map_location="cpu"
    )

    embeddings = data["embeddings"]
    labels = data["labels"]

    print(
        f"Embeddings: {embeddings.shape}"
    )

    print(
        f"Labels: {labels.shape}"
    )

    return embeddings, labels


# ============================================================
# BUILD CLASSIFIER
# ============================================================

def create_classifier():

    from predict import EmbeddingClassifier

    classifier = EmbeddingClassifier()

    state_dict = torch.load(
        CLASSIFIER_PATH,
        map_location="cpu"
    )

    classifier.load_state_dict(
        state_dict
    )

    classifier.eval()

    return classifier


# ============================================================
# GET FAKE PROBABILITIES
# ============================================================

def get_fake_probabilities(
    embeddings,
    classifier
):

    probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            len(embeddings),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(embeddings)
            )

            batch = embeddings[start:end]

            logits = classifier(
                batch
            )

            probs = torch.softmax(
                logits,
                dim=1
            )

            fake_probs = (
                probs[:, 1]
                .cpu()
            )

            probabilities.append(
                fake_probs
            )

    return torch.cat(
        probabilities
    ).numpy()


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    fake_probabilities,
    labels,
    threshold
):

    predictions = (
        fake_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("WAV2VEC2 THRESHOLD TUNING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load validation embeddings
    # --------------------------------------------------------

    embeddings, labels = (
        load_validation_data()
    )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    print(
        "\nLoading classifier..."
    )

    classifier = create_classifier()

    print(
        "Classifier loaded."
    )

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print(
        "\nCalculating fake probabilities..."
    )

    fake_probabilities = (
        get_fake_probabilities(
            embeddings,
            classifier
        )
    )

    # --------------------------------------------------------
    # Search thresholds
    # --------------------------------------------------------

    best_threshold = 0.50
    best_f1 = -1.0

    results = []

    print()
    print("=" * 60)
    print("THRESHOLD RESULTS")
    print("=" * 60)

    for threshold_int in range(
        30,
        81,
        5
    ):

        threshold = (
            threshold_int / 100
        )

        (
            accuracy,
            precision,
            recall,
            f1
        ) = evaluate_threshold(
            fake_probabilities,
            labels.numpy(),
            threshold
        )

        results.append(
            (
                threshold,
                accuracy,
                precision,
                recall,
                f1
            )
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"Accuracy {accuracy * 100:6.2f}% | "
            f"Precision {precision * 100:6.2f}% | "
            f"Recall {recall * 100:6.2f}% | "
            f"F1 {f1 * 100:6.2f}%"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    # --------------------------------------------------------
    # Final recommendation
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RECOMMENDED THRESHOLD")
    print("=" * 60)

    print(
        f"Best validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{best_f1 * 100:.2f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()