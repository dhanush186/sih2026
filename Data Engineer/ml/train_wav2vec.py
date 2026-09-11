from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import Wav2Vec2Processor, Wav2Vec2Model

from dataset import FakeRealDataset


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

TRAIN_SAMPLES = 5000
VALIDATION_SAMPLES = 1000

MODEL_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec2_deepfake_classifier.pth"
)

CACHE_DIR = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec_cache"
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
# BALANCED SAMPLING
# ============================================================

def create_balanced_indices(dataset, total_samples):

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

    indices = (
        selected_real +
        selected_fake
    )

    random.shuffle(indices)

    return indices, samples_per_class


# ============================================================
# LOAD ONE AUDIO FILE
# ============================================================

def load_audio(path):

    import librosa

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

    return audio.astype(np.float32)


# ============================================================
# EXTRACT WAV2VEC2 EMBEDDINGS
# ============================================================

def extract_embeddings(
    dataset,
    indices,
    processor,
    encoder,
    split_name
):

    cache_path = CACHE_DIR / f"{split_name}_embeddings.pt"

    # --------------------------------------------------------
    # Use cached embeddings when available
    # --------------------------------------------------------

    if cache_path.exists():

        print()
        print(
            f"Loading cached {split_name} embeddings..."
        )

        cached = torch.load(
            cache_path,
            map_location="cpu"
        )

        return (
            cached["embeddings"],
            cached["labels"]
        )

    # --------------------------------------------------------
    # Extract new embeddings
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

            path, label = dataset.files[index]

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

            # [1, time, 768]
            hidden_states = (
                outputs.last_hidden_state
            )

            # Mean pooling
            embedding = (
                hidden_states.mean(dim=1)
                .squeeze(0)
                .cpu()
            )

            embeddings.append(
                embedding
            )

            labels.append(label)

            if count % 100 == 0 or count == len(indices):

                print(
                    f"Processed "
                    f"{count}/{len(indices)}"
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
        f"Cached embeddings saved to:"
    )
    print(cache_path)

    return embeddings, labels


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

    print("=" * 60)
    print("WAV2VEC2 DEEPFAKE DETECTION TRAINING")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print()

    # --------------------------------------------------------
    # Load processor and encoder
    # --------------------------------------------------------

    print(
        "Loading Wav2Vec2 processor..."
    )

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base"
    )

    print(
        "Loading Wav2Vec2 encoder..."
    )

    encoder = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base"
    )

    encoder = encoder.to(DEVICE)

    encoder.eval()

    # Freeze encoder
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    print(
        "Wav2Vec2 encoder loaded."
    )

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading training dataset..."
    )

    train_dataset = FakeRealDataset(
        "training"
    )

    train_indices, train_per_class = (
        create_balanced_indices(
            train_dataset,
            TRAIN_SAMPLES
        )
    )

    print(
        f"Balanced training samples: "
        f"{len(train_indices)} "
        f"({train_per_class} real + "
        f"{train_per_class} fake)"
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading validation dataset..."
    )

    validation_dataset = FakeRealDataset(
        "validation"
    )

    validation_indices, val_per_class = (
        create_balanced_indices(
            validation_dataset,
            VALIDATION_SAMPLES
        )
    )

    print(
        f"Balanced validation samples: "
        f"{len(validation_indices)} "
        f"({val_per_class} real + "
        f"{val_per_class} fake)"
    )

    # --------------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------------

    train_embeddings, train_labels = (
        extract_embeddings(
            train_dataset,
            train_indices,
            processor,
            encoder,
            "train"
        )
    )

    validation_embeddings, validation_labels = (
        extract_embeddings(
            validation_dataset,
            validation_indices,
            processor,
            encoder,
            "validation"
        )
    )

    print()
    print(
        "Training embedding shape:",
        train_embeddings.shape
    )

    print(
        "Validation embedding shape:",
        validation_embeddings.shape
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create classifier
    # --------------------------------------------------------

    model = EmbeddingClassifier()

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print(
        "Starting classifier training..."
    )

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for batch_index, (
            embeddings,
            labels
        ) in enumerate(
            train_loader,
            start=1
        ):

            embeddings = embeddings.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
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

            if batch_index % 20 == 0:

                print(
                    f"Batch "
                    f"{batch_index}/"
                    f"{len(train_loader)} | "
                    f"Loss: "
                    f"{loss.item():.4f}"
                )

        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        validation_loss = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():

            for embeddings, labels in validation_loader:

                embeddings = embeddings.to(
                    DEVICE
                )

                labels = labels.to(
                    DEVICE
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
            validation_loss
            / validation_total
        )

        val_accuracy = (
            validation_correct
            / validation_total
        )

        # ----------------------------------------------------
        # Results
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

        # ----------------------------------------------------
        # Save best classifier
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

            print()
            print(
                "Best Wav2Vec2 classifier saved!"
            )

            print(
                f"Best Validation Accuracy: "
                f"{best_val_accuracy * 100:.2f}%"
            )

            print(
                "Model saved to:"
            )

            print(MODEL_PATH)

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("WAV2VEC2 TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print()
    print(
        "Best model saved to:"
    )

    print(MODEL_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()