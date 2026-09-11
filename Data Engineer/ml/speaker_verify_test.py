from pathlib import Path

import librosa
import torch
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


# ============================================================
# AUDIO FILES
# ============================================================

REFERENCE_AUDIO = Path(
    r"D:\kaggle_cache\datasets\mohammedabdeldayem\the-fake-or-real-dataset\versions\2\for-original\for-original\testing\real\file1.wav"
)

TEST_AUDIO = Path(
    r"D:\kaggle_cache\datasets\mohammedabdeldayem\the-fake-or-real-dataset\versions\2\for-original\for-original\testing\real\file1000.wav"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

MODEL_DIR = Path("models/speaker_verification")


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path: Path) -> torch.Tensor:
    """
    Load an audio file as a mono 16 kHz waveform tensor.

    Returns:
        Tensor with shape [1, time]
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Audio file not found:\n{path}"
        )

    print(f"Loading audio: {path}")

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio is None or len(audio) == 0:
        raise ValueError(
            f"Audio file is empty or could not be loaded:\n{path}"
        )

    waveform = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)

    print(f"Audio shape: {waveform.shape}")

    return waveform


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SPEAKER VERIFICATION TEST")
    print("=" * 60)

    print(f"\nReference:")
    print(REFERENCE_AUDIO)

    print(f"\nTest:")
    print(TEST_AUDIO)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not REFERENCE_AUDIO.is_file():
        print("\nERROR: Reference audio file does not exist.")
        return

    if not TEST_AUDIO.is_file():
        print("\nERROR: Test audio file does not exist.")
        return

    # --------------------------------------------------------
    # Load reference audio
    # --------------------------------------------------------

    print("\nLoading reference audio...")

    try:
        reference_waveform = load_audio(REFERENCE_AUDIO)
    except Exception as e:
        print(f"\nERROR loading reference audio: {e}")
        return

    # --------------------------------------------------------
    # Load test audio
    # --------------------------------------------------------

    print("\nLoading test audio...")

    try:
        test_waveform = load_audio(TEST_AUDIO)
    except Exception as e:
        print(f"\nERROR loading test audio: {e}")
        return

    # --------------------------------------------------------
    # Load ECAPA speaker verification model
    # --------------------------------------------------------

    print("\nLoading ECAPA-TDNN speaker model...")

    try:
        verification = SpeakerRecognition.from_hparams(
            source=SPEAKER_MODEL,
            savedir=str(MODEL_DIR),
            local_strategy=LocalStrategy.COPY
        )
    except Exception as e:
        print(f"\nERROR loading speaker model: {e}")
        return

    print("Speaker model loaded.")

    # --------------------------------------------------------
    # Compare speakers
    # --------------------------------------------------------

    print("\nComparing speakers...")

    try:
        with torch.no_grad():

            score, prediction = verification.verify_batch(
                reference_waveform,
                test_waveform
            )

    except Exception as e:
        print(f"\nERROR during speaker verification: {e}")
        return

    # --------------------------------------------------------
    # Convert results
    # --------------------------------------------------------

    score_value = float(
        score.squeeze().item()
    )

    prediction_value = int(
        prediction.squeeze().item()
    )

    same_speaker = prediction_value == 1

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SPEAKER VERIFICATION RESULT")
    print("=" * 60)

    print(f"Reference file : {REFERENCE_AUDIO.name}")
    print(f"Test file      : {TEST_AUDIO.name}")

    print(f"\nSimilarity score: {score_value:.4f}")

    if same_speaker:
        print("Same speaker: YES")
    else:
        print("Same speaker: NO")

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()