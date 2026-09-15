from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec2_deepfake_classifier_full.pth"
)

TEST_CACHE = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec_cache\testing_embeddings.pt"
)

THRESHOLD = 0.50
BATCH_SIZE = 64


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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FALSE NEGATIVE ANALYSIS")
    print("=" * 70)

    print("\nLoading test embeddings...")

    data = torch.load(
        TEST_CACHE,
        map_location="cpu"
    )

    embeddings = data["embeddings"]
    labels = data["labels"]

    print("Embeddings:", embeddings.shape)
    print("Labels:", labels.shape)

    print("\nLoading classifier...")

    model = EmbeddingClassifier()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    print("Model loaded.")

    dataset = TensorDataset(
        embeddings,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    false_negatives = []

    offset = 0

    print("\nCalculating predictions...")

    with torch.no_grad():

        for batch_embeddings, batch_labels in loader:

            logits = model(
                batch_embeddings
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            fake_probabilities = probabilities[:, 1]

            predictions = (
                fake_probabilities >= THRESHOLD
            ).long()

            for i in range(
                len(batch_labels)
            ):

                true_label = (
                    batch_labels[i].item()
                )

                prediction = (
                    predictions[i].item()
                )

                fake_probability = (
                    fake_probabilities[i].item()
                )

                original_index = (
                    offset + i
                )

                if (
                    true_label == 1
                    and prediction == 0
                ):

                    false_negatives.append(
                        (
                            original_index,
                            fake_probability
                        )
                    )

            offset += len(batch_labels)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"False-negative fake recordings: "
        f"{len(false_negatives)}"
    )

    print(
        "\nLowest fake probabilities among "
        "false negatives:"
    )

    false_negatives.sort(
        key=lambda item: item[1]
    )

    for index, probability in false_negatives[:30]:

        print(
            f"Test index {index:4d} | "
            f"Fake probability: "
            f"{probability:.6f}"
        )

    print(
        "\nHighest fake probabilities among "
        "false negatives:"
    )

    for index, probability in false_negatives[-30:]:

        print(
            f"Test index {index:4d} | "
            f"Fake probability: "
            f"{probability:.6f}"
        )


if __name__ == "__main__":
    main()