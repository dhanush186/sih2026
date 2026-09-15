from pathlib import Path
import sys

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# AUDIO PATH
# ============================================================

def get_audio_path():

    if len(sys.argv) < 2:

        raise ValueError(
            "Please provide an audio file path.\n\n"
            "Example:\n"
            r'python realtime_predict.py "D:\audio\call.wav"'
        )

    audio_path = Path(
        sys.argv[1]
    )

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    if not audio_path.is_file():

        raise ValueError(
            f"Path is not a file:\n{audio_path}"
        )

    allowed_extensions = {
        ".wav",
        ".mp3",
    }

    extension = (
        audio_path.suffix
        .lower()
        .strip()
    )

    if extension not in allowed_extensions:

        raise ValueError(
            "Only WAV and MP3 files are supported."
        )

    if audio_path.stat().st_size == 0:

        raise ValueError(
            f"Audio file is empty:\n{audio_path}"
        )

    return audio_path


# ============================================================
# LOAD FULL AUDIO
# ============================================================

def load_full_audio(audio_path):

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
        # Pad final chunk
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

        hidden_states = (
            outputs.last_hidden_state
        )

        embedding = (
            hidden_states.mean(dim=1)
        )

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
# OPTIONAL SPEAKER VERIFICATION
# ============================================================

def verify_trusted_speaker(
    reference_audio,
    test_audio
):

    if reference_audio is None:

        return None

    if not reference_audio.exists():

        raise FileNotFoundError(
            f"Trusted speaker reference not found:\n"
            f"{reference_audio}"
        )

    print()
    print("Loading speaker verification model...")

    verifier = SpeakerVerifier()

    print(
        "Comparing trusted speaker "
        "with audio..."
    )

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

    print()
    print("=" * 60)
    print("SPEAKER VERIFICATION")
    print("=" * 60)

    print(
        f"Reference: {reference_audio.name}"
    )

    print(
        f"Audio: {test_audio.name}"
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
    # 1. GET AUDIO FILE
    # ========================================================

    audio_path = get_audio_path()

    print()
    print(
        f"Audio file: {audio_path.name}"
    )

    print(
        f"Full path: {audio_path}"
    )

    # ========================================================
    # 2. LOAD MODELS
    # ========================================================

    print()
    print(
        "Loading deepfake detection models..."
    )

    (
        processor,
        encoder,
        classifier
    ) = load_models()

    print(
        "Deepfake detection models ready."
    )

    # ========================================================
    # 3. LOAD AUDIO
    # ========================================================

    audio = load_full_audio(
        audio_path
    )

    duration = (
        len(audio) / SAMPLE_RATE
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    # ========================================================
    # 4. SPEAKER VERIFICATION
    # ========================================================
    #
    # Disabled by default.
    #
    # Later we can accept a second command-line argument:
    #
    # python realtime_predict.py call.wav reference.wav
    #
    # For now:
    #
    speaker_similarity = None

    # ========================================================
    # 5. CREATE CHUNKS
    # ========================================================

    chunks = create_chunks(
        audio
    )

    print(
        f"2-second chunks: {len(chunks)}"
    )

    # ========================================================
    # 6. RISK ENGINE
    # ========================================================

    risk_engine = RiskEngine()

    fake_probabilities = []

    # ========================================================
    # 7. CHUNK ANALYSIS
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

        end_time = min(
            start_time + CHUNK_DURATION,
            duration
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
        # Risk calculation
        # ----------------------------------------------------

        risk_result = (
            risk_engine.add_result(
                fake_probability=fake_probability,
                speaker_similarity=speaker_similarity
            )
        )

        print(
            f"{start_time:05.1f}s - "
            f"{end_time:05.1f}s | "
            f"Fake: "
            f"{fake_probability * 100:6.2f}% | "
            f"Real: "
            f"{real_probability * 100:6.2f}% | "
            f"Rolling: "
            f"{risk_result['average_fake_probability'] * 100:6.2f}% | "
            f"Risk: "
            f"{risk_result['risk_level']:6s} | "
            f"Score: "
            f"{risk_result['risk_score']:6.2f} | "
            f"High evidence: "
            f"{risk_result['high_evidence']}"
        )

    # ========================================================
    # 8. FINAL RISK
    # ========================================================

    final_risk = (
        risk_engine.get_risk(
            speaker_similarity=speaker_similarity
        )
    )

    average_fake_probability = (
        sum(fake_probabilities)
        / len(fake_probabilities)
    )

    maximum_fake_probability = max(
        fake_probabilities
    )

    # ========================================================
    # 9. SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("REAL-TIME ANALYSIS SUMMARY")
    print("=" * 60)

    print(
        f"Audio file: "
        f"{audio_path.name}"
    )

    print(
        f"Duration: "
        f"{duration:.2f} seconds"
    )

    print(
        f"Chunks analyzed: "
        f"{len(chunks)}"
    )

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
        f"{final_risk['average_fake_probability'] * 100:.2f}%"
    )

    print(
        f"Overall risk score: "
        f"{final_risk['risk_score']:.2f}/100"
    )

    print(
        f"Overall risk level: "
        f"{final_risk['risk_level']}"
    )

    print(
        f"High-risk evidence chunks: "
        f"{final_risk['high_evidence']}"
    )

    print(
        f"Session alert: "
        f"{final_risk['session_alert']}"
    )

    if final_risk["risk_level"] == "HIGH":

        print(
            "Independent verification recommended."
        )

    elif final_risk["risk_level"] == "MEDIUM":

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

    try:

        main()

    except Exception as error:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)