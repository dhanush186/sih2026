import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# IMPORT OUR ML FILES
# ============================================================

sys.path.append(str(Path(__file__).parent))

from dataset import FakeRealDataset
from model import DeepfakeCNN


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16

MODEL_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\deepfake_cnn_baseline.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("DEEPFAKE VOICE DETECTION - MODEL EVALUATION")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Check whether model exists
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}\n\n"
            f"Run train.py successfully before running "
            f"evaluate.py."
        )

    # --------------------------------------------------------
    # Load testing dataset
    # --------------------------------------------------------

    print("\nLoading testing dataset...")

    test_dataset = FakeRealDataset("testing")

    print(
        f"\nTesting samples: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating model...")

    model = DeepfakeCNN()

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    print("Loading trained weights...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []

    print("\nRunning predictions...")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_index, (mel, labels) in enumerate(
            test_loader
        ):

            # Dataset output:
            # [batch, 64, 126]

            # CNN expects:
            # [batch, 1, 64, 126]

            mel = mel.unsqueeze(1)

            mel = mel.to(DEVICE)

            # Forward pass
            outputs = model(mel)

            # Select class with highest score
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # Save results
            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            # Progress
            if (batch_index + 1) % 20 == 0:

                print(
                    f"Processed batch "
                    f"{batch_index + 1}/"
                    f"{len(test_loader)}"
                )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("EVALUATION RESULTS")
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

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\n")
    print("CONFUSION MATRIX")
    print("-" * 40)

    print(
        "                 Predicted"
    )

    print(
        "                 REAL    FAKE"
    )

    print(
        f"Actual REAL     "
        f"{matrix[0][0]:6d} "
        f"{matrix[0][1]:7d}"
    )

    print(
        f"Actual FAKE     "
        f"{matrix[1][0]:6d} "
        f"{matrix[1][1]:7d}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("CLASSIFICATION REPORT")
    print("-" * 60)

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "REAL",
            "FAKE"
        ],
        zero_division=0
    )

    print(report)

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print(
        "REAL = 0"
    )

    print(
        "FAKE = 1"
    )

    print(
        "\nThe metrics above are calculated on the "
        "separate testing split."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()