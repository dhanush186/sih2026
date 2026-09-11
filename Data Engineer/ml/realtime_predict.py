from pathlib import Path

import librosa
import numpy as np
import torch

from predict import load_models
from risk_engine import RiskEngine
from speaker_verifier import SpeakerVerifier


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

CHUNK_DURATION = 2

CHUNK_SAMPLES = (
    SAMPLE_RATE * CHUNK_DURATION
)

# ------------------------------------------------------------
# Demo audio
# ------------------------------------------------------------

AUDIO_PATH = Path(
    r"D:\kaggle_cache\datasets"
    r"\mohammedabdeldayem"
    r"\the-fake-or-real-dataset"
    r"\versions\2"
    r"\for-original"
    r"\for-original"
    r"\testing"
    r"\fake"
    r"\file1269.wav"
)

# ------------------------------------------------------------
# Trusted speaker reference
#
# Set this to None to disable speaker verification.
# ------------------------------------------------------------

REFERENCE_AUDIO = Path(
    r"D:\kaggle_cache\datasets"
    r"\mohammedabdeldayem"
    r"\the-fake-or-real-dataset"
    r"\versions\2"
    r"\for-original"
    r"\for-original"
    r"\testing"
    r"\real"
    r"\file1.wav"
)

# Example:
# REFERENCE_AUDIO = None


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD FULL AUDIO
# ============================================================

def load_full_audio(audio_path):

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    if not audio_path.is_file():

        raise ValueError(
            f"Path is not a file:\n{audio_path}"
        )

    if audio_path.stat().st_size == 0:

        raise ValueError(
            f"Audio file is empty:\n{audio_path}"
        )

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio is None or len(audio) == 0:

        raise ValueError(
            f"Audio could not be decoded:\n{audio_path}"
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# CREATE 2-SECOND CHUNKS
# ============================================================

def create_chunks(audio):

    chunks = []

    total_samples = len(audio)

    start = 0

    while start < total_samples:

        end = start + CHUNK_SAMPLES

        chunk = audio[start:end]

        # ----------------------------------------------------
        # Pad final chunk to exactly 2 seconds
        # ----------------------------------------------------

        if len(chunk) < CHUNK_SAMPLES:

            chunk = np.pad(
                chunk,
                (
                    0,
                    CHUNK_SAMPLES - len(chunk)
                ),
                mode="constant"
            )

        chunks.append(
            chunk
        )

        start += CHUNK_SAMPLES

    return chunks


# ============================================================
# ANALYZE ONE CHUNK
# ============================================================

def analyze_chunk(
    chunk,
    processor,
    encoder,
    classifier
):

    inputs = processor(
        chunk,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    input_values = (
        inputs.input_values
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = encoder(
            input_values=input_values
        )

        # Shape:
        # [1, time, 768]

        hidden_states = (
            outputs.last_hidden_state
        )

        # ----------------------------------------------------
        # Mean pooling
        # ----------------------------------------------------

        embedding = (
            hidden_states.mean(dim=1)
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = classifier(
            embedding
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    real_probability = (
        probabilities[0, 0].item()
    )

    fake_probability = (
        probabilities[0, 1].item()
    )

    return (
        real_probability,
        fake_probability
    )


# ============================================================
# SPEAKER VERIFICATION
# ============================================================

def verify_trusted_speaker(
    reference_audio,
    test_audio
):

    # --------------------------------------------------------
    # Speaker verification disabled
    # --------------------------------------------------------

    if reference_audio is None:

        print(
            "Trusted speaker reference: DISABLED"
        )

        return None

    # --------------------------------------------------------
    # Check reference
    # --------------------------------------------------------

    if not reference_audio.exists():

        raise FileNotFoundError(
            "Trusted speaker reference not found:\n"
            f"{reference_audio}"
        )

    # --------------------------------------------------------
    # Load speaker model
    # --------------------------------------------------------

    print()
    print("Loading speaker verification model...")

    verifier = SpeakerVerifier()

    print(
        "Comparing trusted reference "
        "with call audio..."
    )

    # --------------------------------------------------------
    # Compare speakers
    # --------------------------------------------------------

    result = verifier.compare(
        reference_audio,
        test_audio
    )

    similarity = float(
        result["similarity_score"]
    )

    same_speaker = bool(
        result["same_speaker"]
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SPEAKER VERIFICATION")
    print("=" * 60)

    print(
        f"Reference: "
        f"{reference_audio.name}"
    )

    print(
        f"Call audio: "
        f"{test_audio.name}"
    )

    print(
        f"Similarity score: "
        f"{similarity:.4f}"
    )

    print(
        f"Same speaker: "
        f"{'YES' if same_speaker else 'NO'}"
    )

    print("=" * 60)

    return similarity


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("REAL-TIME DEEPFAKE VOICE DETECTION")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # 1. LOAD DEEPFAKE MODELS
    # ========================================================

    print()
    print("Loading deepfake detection models...")

    (
        processor,
        encoder,
        classifier
    ) = load_models()

    print(
        "Deepfake detection models ready."
    )

    # ========================================================
    # 2. LOAD AUDIO
    # ========================================================

    print()
    print(
        f"Audio file: {AUDIO_PATH.name}"
    )

    audio = load_full_audio(
        AUDIO_PATH
    )

    duration = (
        len(audio) / SAMPLE_RATE
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    # ========================================================
    # 3. SPEAKER VERIFICATION
    # ========================================================

    speaker_similarity = (
        verify_trusted_speaker(
            REFERENCE_AUDIO,
            AUDIO_PATH
        )
    )

    # ========================================================
    # 4. CREATE CHUNKS
    # ========================================================

    chunks = create_chunks(
        audio
    )

    print()
    print(
        f"2-second chunks: {len(chunks)}"
    )

    # ========================================================
    # 5. INITIALIZE RISK ENGINE
    # ========================================================

    risk_engine = RiskEngine()

    fake_probabilities = []

    # ========================================================
    # 6. ANALYZE CHUNKS
    # ========================================================

    print()
    print("=" * 60)
    print("CHUNK ANALYSIS")
    print("=" * 60)

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        start_time = (
            (index - 1)
            * CHUNK_DURATION
        )

        end_time = (
            start_time
            + CHUNK_DURATION
        )

        # ----------------------------------------------------
        # Wav2Vec2 prediction
        # ----------------------------------------------------

        (
            real_probability,
            fake_probability
        ) = analyze_chunk(
            chunk,
            processor,
            encoder,
            classifier
        )

        fake_probabilities.append(
            fake_probability
        )

        # ----------------------------------------------------
        # Risk engine
        # ----------------------------------------------------

        risk_result = (
            risk_engine.add_result(
                fake_probability=fake_probability,
                speaker_similarity=speaker_similarity
            )
        )

        risk_level = (
            risk_result["risk_level"]
        )

        risk_score = (
            risk_result["risk_score"]
        )

        rolling_fake_probability = (
            risk_result[
                "average_fake_probability"
            ]
        )

        high_evidence = (
            risk_result["high_evidence"]
        )

        # ----------------------------------------------------
        # Speaker evidence
        # ----------------------------------------------------

        speaker_info = ""

        if speaker_similarity is not None:

            speaker_info = (
                f" | Speaker: "
                f"{speaker_similarity:.4f}"
            )

        # ----------------------------------------------------
        # Display chunk result
        # ----------------------------------------------------

        print(
            f"{start_time:05.1f}s - "
            f"{end_time:05.1f}s | "
            f"Fake: "
            f"{fake_probability * 100:6.2f}% | "
            f"Real: "
            f"{real_probability * 100:6.2f}% | "
            f"Rolling: "
            f"{rolling_fake_probability * 100:6.2f}% | "
            f"Risk: "
            f"{risk_level:6s} | "
            f"Score: "
            f"{risk_score:6.2f} | "
            f"High evidence: "
            f"{high_evidence}"
            f"{speaker_info}"
        )

    # ========================================================
    # 7. FINAL SUMMARY
    # ========================================================

    average_fake_probability = (
        sum(fake_probabilities)
        / len(fake_probabilities)
    )

    maximum_fake_probability = max(
        fake_probabilities
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the same trusted speaker information for the final
    # risk calculation as we used for each chunk.
    # --------------------------------------------------------

    final_risk = (
        risk_engine.get_risk(
            speaker_similarity=speaker_similarity
        )
    )

    overall_risk = (
        final_risk["risk_level"]
    )

    overall_risk_score = (
        final_risk["risk_score"]
    )

    final_rolling_probability = (
        final_risk[
            "average_fake_probability"
        ]
    )

    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("REAL-TIME ANALYSIS SUMMARY")
    print("=" * 60)

    print(
        f"Average fake probability: "
        f"{average_fake_probability * 100:.2f}%"
    )

    print(
        f"Maximum fake probability: "
        f"{maximum_fake_probability * 100:.2f}%"
    )

    print(
        f"Final rolling fake probability: "
        f"{final_rolling_probability * 100:.2f}%"
    )

    if speaker_similarity is not None:

        print(
            f"Speaker similarity: "
            f"{speaker_similarity:.4f}"
        )

        print(
            f"Speaker mismatch score: "
            f"{final_risk['speaker_mismatch_score']:.2f}%"
        )

    else:

        print(
            "Speaker verification: NOT USED"
        )

    print(
        f"Overall risk score: "
        f"{overall_risk_score:.2f}/100"
    )

    print(
        f"Overall risk level: "
        f"{overall_risk}"
    )

    print(
        f"High-risk evidence chunks: "
        f"{final_risk['high_evidence']}"
    )

    print(
        f"Session alert: "
        f"{final_risk['session_alert']}"
    )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if overall_risk == "HIGH":

        print(
            "Independent verification recommended."
        )

    elif overall_risk == "MEDIUM":

        print(
            "Additional verification recommended "
            "before sensitive action."
        )

    else:

        print(
            "No elevated synthetic-voice risk "
            "detected by the current model."
        )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()