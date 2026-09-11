from pathlib import Path

from predict import load_models, predict_audio
from speaker_verifier import SpeakerVerifier


# ============================================================
# AUDIO
# ============================================================

REFERENCE_AUDIO = Path(
    r"D:\kaggle_cache\datasets\mohammedabdeldayem"
    r"\the-fake-or-real-dataset\versions\2"
    r"\for-original\for-original\testing\real\file1.wav"
)

TEST_AUDIO = Path(
    r"D:\kaggle_cache\datasets\mohammedabdeldayem"
    r"\the-fake-or-real-dataset\versions\2"
    r"\for-original\for-original\testing\real\file1000.wav"
)


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(
    fake_probability: float,
    speaker_similarity=None
):
    """
    Calculate risk.

    Without a trusted speaker reference:
        use only the deepfake probability.

    With a trusted speaker reference:
        combine deepfake probability and speaker mismatch.
    """

    deepfake_score = fake_probability * 100

    # --------------------------------------------------------
    # No trusted speaker reference
    # --------------------------------------------------------

    if speaker_similarity is None:

        combined_score = deepfake_score

        if combined_score >= 80:
            risk_level = "HIGH"

        elif combined_score >= 50:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        return combined_score, risk_level

    # --------------------------------------------------------
    # Trusted speaker reference available
    # --------------------------------------------------------

    speaker_mismatch_score = (
        (1.0 - speaker_similarity) * 100
    )

    combined_score = (
        0.70 * deepfake_score
        + 0.30 * speaker_mismatch_score
    )

    combined_score = max(
        0.0,
        min(100.0, combined_score)
    )

    if combined_score >= 80:
        risk_level = "HIGH"

    elif combined_score >= 50:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return combined_score, risk_level


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DEEPFAKE + SPEAKER VERIFICATION RISK TEST")
    print("=" * 60)

    # ========================================================
    # 1. Wav2Vec2
    # ========================================================

    print("\nLoading Wav2Vec2 models...")

    processor, encoder, classifier = load_models()

    print("\nRunning deepfake detector...")

    prediction = predict_audio(
        TEST_AUDIO,
        processor,
        encoder,
        classifier
    )

    fake_probability = float(
        prediction["fake_probability"]
    )

    print("\nWav2Vec2 result:")
    print(
        f"Prediction: {prediction['label']}"
    )

    print(
        f"Real probability: "
        f"{prediction['real_probability'] * 100:.2f}%"
    )

    print(
        f"Fake probability: "
        f"{fake_probability * 100:.2f}%"
    )

    print(
        f"Model risk: "
        f"{prediction['risk_level']}"
    )

    # ========================================================
    # 2. Scenario A — NO trusted speaker reference
    # ========================================================

    print("\n" + "-" * 60)
    print("SCENARIO A: NO TRUSTED SPEAKER REFERENCE")
    print("-" * 60)

    score_a, risk_a = calculate_risk(
        fake_probability,
        None
    )

    print(
        f"Risk score: {score_a:.2f}/100"
    )

    print(
        f"Risk level: {risk_a}"
    )

    print(
        "Speaker verification: NOT USED"
    )

    # ========================================================
    # 3. Scenario B — trusted speaker reference
    # ========================================================

    print("\n" + "-" * 60)
    print("SCENARIO B: TRUSTED SPEAKER REFERENCE")
    print("-" * 60)

    print("\nLoading speaker verifier...")

    verifier = SpeakerVerifier()

    speaker_result = verifier.compare(
        REFERENCE_AUDIO,
        TEST_AUDIO
    )

    speaker_similarity = float(
        speaker_result["similarity_score"]
    )

    print(
        f"\nSpeaker similarity: "
        f"{speaker_similarity:.4f}"
    )

    print(
        f"Same speaker: "
        f"{'YES' if speaker_result['same_speaker'] else 'NO'}"
    )

    score_b, risk_b = calculate_risk(
        fake_probability,
        speaker_similarity
    )

    print(
        f"\nRisk score: {score_b:.2f}/100"
    )

    print(
        f"Risk level: {risk_b}"
    )

    # ========================================================
    # 4. Final comparison
    # ========================================================

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(
        f"No speaker reference : "
        f"{score_a:.2f}/100 ({risk_a})"
    )

    print(
        f"With speaker reference: "
        f"{score_b:.2f}/100 ({risk_b})"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()