from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    r"D:\sih2026\Data Engineer\models"
    r"\wav2vec2_deepfake_classifier.pth"
)

MODEL_NAME = "facebook/wav2vec2-base"

SAMPLE_RATE = 16000
DURATION = 2
NUM_SAMPLES = SAMPLE_RATE * DURATION

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

def load_audio(audio_path):

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) < NUM_SAMPLES:

        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio)),
            mode="constant"
        )

    else:

        audio = audio[:NUM_SAMPLES]

    return audio.astype(np.float32)


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n\n"
            f"Run train_wav2vec.py first."
        )

    print("Loading Wav2Vec2 processor...")

    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_NAME
    )

    print("Loading Wav2Vec2 encoder...")

    encoder = Wav2Vec2Model.from_pretrained(
        MODEL_NAME
    )

    encoder = encoder.to(DEVICE)
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    print("Loading deepfake classifier...")

    classifier = EmbeddingClassifier()

    classifier = classifier.to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    classifier.load_state_dict(
        state_dict
    )

    classifier.eval()

    print("Models loaded successfully.")

    return processor, encoder, classifier


# ============================================================
# PREDICT AUDIO
# ============================================================

def predict_audio(
    audio_path,
    processor,
    encoder,
    classifier
):

    audio_path = Path(audio_path)

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    if not audio_path.is_file():

        raise ValueError(
            f"Not a file:\n{audio_path}"
        )

    if audio_path.stat().st_size == 0:

        raise ValueError(
            f"Audio file is empty:\n{audio_path}"
        )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio = load_audio(
        audio_path
    )

    # --------------------------------------------------------
    # Prepare Wav2Vec2 input
    # --------------------------------------------------------

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    input_values = (
        inputs.input_values
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Generate embedding
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = encoder(
            input_values=input_values
        )

        hidden_states = (
            outputs.last_hidden_state
        )

        embedding = (
            hidden_states
            .mean(dim=1)
        )

        logits = classifier(
            embedding
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    real_probability = (
        probabilities[0, 0].item()
    )

    fake_probability = (
        probabilities[0, 1].item()
    )

    prediction = (
        1
        if fake_probability >= real_probability
        else 0
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if fake_probability >= 0.80:

        risk_level = "HIGH"

    elif fake_probability >= 0.50:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    return {
        "prediction": prediction,
        "label": (
            "FAKE"
            if prediction == 1
            else "REAL"
        ),
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "risk_level": risk_level
    }


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("DEEPFAKE VOICE DETECTION - INFERENCE")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    processor, encoder, classifier = (
        load_models()
    )

    # --------------------------------------------------------
    # Find a test audio file
    # --------------------------------------------------------

    test_dir = Path(
        r"D:\kaggle_cache\datasets\mohammedabdeldayem"
        r"\the-fake-or-real-dataset\versions\2"
        r"\for-original\for-original"
        r"\testing\fake"
    )

    audio_files = [
        p
        for p in test_dir.iterdir()
        if (
            p.is_file()
            and p.suffix.lower().strip()
            in {".wav", ".mp3"}
            and p.stat().st_size > 0
        )
    ]

    if not audio_files:

        raise FileNotFoundError(
            "No valid test audio file found."
        )

    test_audio = audio_files[0]

    print()
    print(
        f"Testing audio: {test_audio.name}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = predict_audio(
        test_audio,
        processor,
        encoder,
        classifier
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(
        f"Prediction         : {result['label']}"
    )

    print(
        f"Real Probability   : "
        f"{result['real_probability'] * 100:.2f}%"
    )

    print(
        f"Fake Probability   : "
        f"{result['fake_probability'] * 100:.2f}%"
    )

    print(
        f"Risk Level         : "
        f"{result['risk_level']}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()