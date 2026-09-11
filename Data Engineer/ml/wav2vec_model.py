import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


# ============================================================
# WAV2VEC2 DEEPFAKE CLASSIFIER
# ============================================================

class Wav2Vec2DeepfakeClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Pretrained Wav2Vec2 encoder
        # ----------------------------------------------------

        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # Freeze the pretrained encoder for the first experiment
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2)
        )

    def forward(self, input_values):

        # ----------------------------------------------------
        # Extract pretrained speech representations
        # ----------------------------------------------------

        outputs = self.encoder(
            input_values=input_values
        )

        # Shape:
        # [batch, time, 768]

        hidden_states = outputs.last_hidden_state

        # ----------------------------------------------------
        # Mean pooling across time
        # ----------------------------------------------------

        embeddings = hidden_states.mean(
            dim=1
        )

        # Shape:
        # [batch, 768]

        # ----------------------------------------------------
        # Classify
        # ----------------------------------------------------

        logits = self.classifier(
            embeddings
        )

        # Shape:
        # [batch, 2]

        return logits


# ============================================================
# SIMPLE MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("WAV2VEC2 DEEPFAKE CLASSIFIER TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print()
    print("Loading Wav2Vec2 classifier...")

    model = Wav2Vec2DeepfakeClassifier()

    model = model.to(device)

    model.eval()

    # Simulated 2-second audio
    dummy_audio = torch.randn(
        2,
        32000
    ).to(device)

    print(
        "Input shape:",
        dummy_audio.shape
    )

    with torch.no_grad():

        output = model(
            dummy_audio
        )

    print(
        "Output shape:",
        output.shape
    )

    print()
    print("Model test successful!")