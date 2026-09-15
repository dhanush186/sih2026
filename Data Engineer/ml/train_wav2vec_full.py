from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
)

from dataset import FakeRealDataset


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 64

EPOCHS = 5

LEARNING_RATE = 0.001

RANDOM_SEED = 42

SAMPLE_RATE = 16000

DURATION = 2

NUM_SAMPLES = (
    SAMPLE_RATE * DURATION
)

MODEL_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec2_deepfake_classifier_full.pth"
)

CACHE_DIR = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec_cache"
)


# New cache names so we never accidentally reuse the
# old 5,000/1,000 experiment.
TRAIN_CACHE = (
    CACHE_DIR
    / "train_full_embeddings.pt"
)

VALIDATION_CACHE = (
    CACHE_DIR
    / "validation_full_embeddings.pt"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
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
                0.30
            ),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                64,
                2
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)


# ============================================================
# CREATE BALANCED INDICES
# ============================================================

def create_balanced_indices(
    dataset
):

    real_indices = [
        index
        for index, (
            _,
            label
        ) in enumerate(
            dataset.files
        )
        if label == 0
    ]

    fake_indices = [
        index
        for index, (
            _,
            label
        ) in enumerate(
            dataset.files
        )
        if label == 1
    ]

    usable_per_class = min(
        len(real_indices),
        len(fake_indices)
    )

    print(
        f"Usable real samples: "
        f"{len(real_indices)}"
    )

    print(
        f"Usable fake samples: "
        f"{len(fake_indices)}"
    )

    print(
        f"Using balanced samples: "
        f"{usable_per_class} real + "
        f"{usable_per_class} fake"
    )

    # We use every sample from the smaller class.
    selected_real = random.sample(
        real_indices,
        usable_per_class
    )

    selected_fake = random.sample(
        fake_indices,
        usable_per_class
    )

    indices = (
        selected_real
        + selected_fake
    )

    random.shuffle(
        indices
    )

    return indices


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(
    path
):

    import librosa

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION
    )

    if len(audio) < NUM_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                NUM_SAMPLES - len(audio)
            ),
            mode="constant"
        )

    else:

        audio = (
            audio[:NUM_SAMPLES]
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

def extract_embeddings(
    dataset,
    indices,
    processor,
    encoder,
    cache_path,
    split_name
):

    # --------------------------------------------------------
    # Reuse cache if it already exists
    # --------------------------------------------------------

    if cache_path.exists():

        print()
        print(
            f"Loading cached {split_name} "
            f"embeddings..."
        )

        cached = torch.load(
            cache_path,
            map_location="cpu"
        )

        embeddings = (
            cached["embeddings"]
        )

        labels = (
            cached["labels"]
        )

        print(
            f"Loaded {len(labels)} "
            f"{split_name} embeddings."
        )

        return (
            embeddings,
            labels
        )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    print()
    print(
        f"Extracting Wav2Vec2 embeddings "
        f"for {split_name}..."
    )

    embeddings = []
    labels = []

    encoder.eval()

    with torch.no_grad():

        for count, index in enumerate(
            indices,
            start=1
        ):

            path, label = (
                dataset.files[index]
            )

            audio = load_audio(
                path
            )

            inputs = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
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

            labels.append(
                label
            )

            if (
                count % 500 == 0
                or count == len(indices)
            ):

                print(
                    f"{split_name}: "
                    f"{count}/"
                    f"{len(indices)}"
                )

    embeddings = torch.stack(
        embeddings
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels
        },
        cache_path
    )

    print()
    print(
        f"{split_name} embeddings saved to:"
    )

    print(
        cache_path
    )

    return (
        embeddings,
        labels
    )


# ============================================================
# VALIDATION METRICS
# ============================================================

def calculate_metrics(
    outputs,
    labels
):

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )

    real_label = 0
    fake_label = 1

    total = len(
        labels
    )

    accuracy = (
        (
            predictions == labels
        )
        .sum()
        .item()
        / total
    )

    true_positive = (
        (
            (predictions == fake_label)
            & (labels == fake_label)
        )
        .sum()
        .item()
    )

    false_positive = (
        (
            (predictions == fake_label)
            & (labels == real_label)
        )
        .sum()
        .item()
    )

    false_negative = (
        (
            (predictions == real_label)
            & (labels == fake_label)
        )
        .sum()
        .item()
    )

    precision = (
        true_positive
        / max(
            true_positive
            + false_positive,
            1
        )
    )

    recall = (
        true_positive
        / max(
            true_positive
            + false_negative,
            1
        )
    )

    f1 = (
        2
        * precision
        * recall
        / max(
            precision + recall,
            1e-12
        )
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

    print("=" * 70)
    print(
        "WAV2VEC2 FULL-DATA DEEPFAKE CLASSIFIER TRAINING"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    # ========================================================
    # LOAD WAV2VEC2
    # ========================================================

    print()
    print(
        "Loading Wav2Vec2 processor..."
    )

    processor = (
        Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )
    )

    print(
        "Loading Wav2Vec2 encoder..."
    )

    encoder = (
        Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )
    )

    encoder = encoder.to(
        DEVICE
    )

    encoder.eval()

    # --------------------------------------------------------
    # Freeze pretrained encoder
    # --------------------------------------------------------

    for parameter in (
        encoder.parameters()
    ):

        parameter.requires_grad = False

    print(
        "Wav2Vec2 encoder loaded and frozen."
    )

    # ========================================================
    # TRAINING DATASET
    # ========================================================

    print()
    print(
        "Loading full training dataset..."
    )

    train_dataset = (
        FakeRealDataset(
            "training"
        )
    )

    train_indices = (
        create_balanced_indices(
            train_dataset
        )
    )

    print(
        f"Total training samples used: "
        f"{len(train_indices)}"
    )

    # ========================================================
    # VALIDATION DATASET
    # ========================================================

    print()
    print(
        "Loading full validation dataset..."
    )

    validation_dataset = (
        FakeRealDataset(
            "validation"
        )
    )

    validation_indices = (
        create_balanced_indices(
            validation_dataset
        )
    )

    print(
        f"Total validation samples used: "
        f"{len(validation_indices)}"
    )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    train_embeddings, train_labels = (
        extract_embeddings(
            train_dataset,
            train_indices,
            processor,
            encoder,
            TRAIN_CACHE,
            "training"
        )
    )

    validation_embeddings, validation_labels = (
        extract_embeddings(
            validation_dataset,
            validation_indices,
            processor,
            encoder,
            VALIDATION_CACHE,
            "validation"
        )
    )

    print()
    print(
        "Training embeddings:",
        train_embeddings.shape
    )

    print(
        "Validation embeddings:",
        validation_embeddings.shape
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_data = TensorDataset(
        train_embeddings,
        train_labels
    )

    validation_data = TensorDataset(
        validation_embeddings,
        validation_labels
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = EmbeddingClassifier()

    model = model.to(
        DEVICE
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_f1 = 0.0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "=" * 70
    )

    for epoch in range(
        EPOCHS
    ):

        model.train()

        running_loss = 0.0

        train_correct = 0

        train_total = 0

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for batch_index, (
            embeddings,
            labels
        ) in enumerate(
            train_loader,
            start=1
        ):

            embeddings = (
                embeddings.to(
                    DEVICE
                )
            )

            labels = (
                labels.to(
                    DEVICE
                )
            )

            optimizer.zero_grad()

            outputs = model(
                embeddings
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            train_correct += (
                (
                    predictions == labels
                )
                .sum()
                .item()
            )

            train_total += (
                labels.size(0)
            )

            if (
                batch_index % 100 == 0
                or batch_index == len(
                    train_loader
                )
            ):

                print(
                    f"Training batch "
                    f"{batch_index}/"
                    f"{len(train_loader)} | "
                    f"Loss: "
                    f"{loss.item():.4f}"
                )

        train_loss = (
            running_loss
            / train_total
        )

        train_accuracy = (
            train_correct
            / train_total
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        validation_loss = 0.0

        all_validation_outputs = []

        all_validation_labels = []

        with torch.no_grad():

            for (
                embeddings,
                labels
            ) in validation_loader:

                embeddings = (
                    embeddings.to(
                        DEVICE
                    )
                )

                labels = (
                    labels.to(
                        DEVICE
                    )
                )

                outputs = model(
                    embeddings
                )

                loss = criterion(
                    outputs,
                    labels
                )

                validation_loss += (
                    loss.item()
                    * labels.size(0)
                )

                all_validation_outputs.append(
                    outputs.cpu()
                )

                all_validation_labels.append(
                    labels.cpu()
                )

        validation_outputs = (
            torch.cat(
                all_validation_outputs
            )
        )

        validation_labels_full = (
            torch.cat(
                all_validation_labels
            )
        )

        val_loss = (
            validation_loss
            / len(
                validation_labels_full
            )
        )

        (
            val_accuracy,
            val_precision,
            val_recall,
            val_f1
        ) = calculate_metrics(
            validation_outputs,
            validation_labels_full
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print()
        print(
            "Results:"
        )

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

        print(
            f"Validation Precision: "
            f"{val_precision * 100:.2f}%"
        )

        print(
            f"Validation Recall: "
            f"{val_recall * 100:.2f}%"
        )

        print(
            f"Validation F1: "
            f"{val_f1 * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Save best by F1
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            MODEL_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print()
            print(
                "NEW BEST MODEL SAVED"
            )

            print(
                f"Best validation F1: "
                f"{best_val_f1 * 100:.2f}%"
            )

            print(
                "Model:"
            )

            print(
                MODEL_PATH
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FULL-DATA TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Best Validation F1: "
        f"{best_val_f1 * 100:.2f}%"
    )

    print()
    print(
        "Best model saved to:"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":

    main()