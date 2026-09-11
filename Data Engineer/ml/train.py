import sys
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# ============================================================
# LOCAL IMPORTS
# ============================================================

sys.path.append(str(Path(__file__).parent))

from dataset import FakeRealDataset
from model import DeepfakeCNN


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.001

TRAIN_SAMPLES = 5000
VALIDATION_SAMPLES = 1000

MODEL_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\deepfake_cnn_baseline.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# BALANCED SUBSET HELPER
# ============================================================

def create_balanced_subset(dataset, total_samples):
    """
    Create a balanced subset containing equal numbers
    of REAL (label 0) and FAKE (label 1) samples.
    """

    real_indices = [
        i
        for i, (_, label) in enumerate(dataset.files)
        if label == 0
    ]

    fake_indices = [
        i
        for i, (_, label) in enumerate(dataset.files)
        if label == 1
    ]

    samples_per_class = total_samples // 2

    samples_per_class = min(
        samples_per_class,
        len(real_indices),
        len(fake_indices)
    )

    selected_real = random.sample(
        real_indices,
        samples_per_class
    )

    selected_fake = random.sample(
        fake_indices,
        samples_per_class
    )

    selected_indices = (
        selected_real +
        selected_fake
    )

    random.shuffle(selected_indices)

    return Subset(
        dataset,
        selected_indices
    ), samples_per_class


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("DEEPFAKE VOICE DETECTION - CNN TRAINING")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print()

    # --------------------------------------------------------
    # LOAD TRAINING DATASET
    # --------------------------------------------------------

    print("Loading training dataset...")

    train_dataset = FakeRealDataset(
        "training"
    )

    train_subset, train_samples_per_class = (
        create_balanced_subset(
            train_dataset,
            TRAIN_SAMPLES
        )
    )

    print(
        f"Balanced training samples: "
        f"{len(train_subset)} "
        f"({train_samples_per_class} real + "
        f"{train_samples_per_class} fake)"
    )

    print()

    # --------------------------------------------------------
    # LOAD VALIDATION DATASET
    # --------------------------------------------------------

    print("Loading validation dataset...")

    validation_dataset = FakeRealDataset(
        "validation"
    )

    validation_subset, val_samples_per_class = (
        create_balanced_subset(
            validation_dataset,
            VALIDATION_SAMPLES
        )
    )

    print(
        f"Balanced validation samples: "
        f"{len(validation_subset)} "
        f"({val_samples_per_class} real + "
        f"{val_samples_per_class} fake)"
    )

    print()

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = DeepfakeCNN().to(DEVICE)

    print("Model created successfully.")
    print()

    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # BEST MODEL TRACKING
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        for batch_index, (mel, labels) in enumerate(
            train_loader,
            start=1
        ):

            # Mel shape:
            # [batch, 64, 126]
            #
            # CNN input:
            # [batch, 1, 64, 126]

            mel = mel.unsqueeze(1)

            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(mel)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            running_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total += labels.size(0)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if batch_index % 10 == 0:

                print(
                    f"Batch {batch_index}/"
                    f"{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        validation_loss = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():

            for mel, labels in validation_loader:

                mel = mel.unsqueeze(1)

                mel = mel.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(mel)

                loss = criterion(
                    outputs,
                    labels
                )

                validation_loss += (
                    loss.item()
                    * labels.size(0)
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                validation_correct += (
                    (predictions == labels)
                    .sum()
                    .item()
                )

                validation_total += (
                    labels.size(0)
                )

        val_loss = (
            validation_loss /
            validation_total
        )

        val_accuracy = (
            validation_correct /
            validation_total
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print()
        print("Results:")

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        print()

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            MODEL_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "Best model saved!"
            )

            print(
                f"Best Validation Accuracy: "
                f"{best_val_accuracy * 100:.2f}%"
            )

            print(
                "Model saved to:"
            )

            print(MODEL_PATH)

            print()

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print()

    print("Best model saved to:")
    print(MODEL_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()