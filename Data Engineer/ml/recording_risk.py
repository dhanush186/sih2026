# ============================================================
# RECORDING-LEVEL RISK ANALYZER
# ============================================================

def analyze_recording(
    fake_probabilities,
    high_threshold=0.80,
    extreme_threshold=0.90,
    medium_threshold=0.50
):
    """
    Convert per-chunk fake probabilities into a
    recording-level risk assessment.

    The decision is evidence-based rather than relying
    on an arbitrary weighted probability formula.
    """

    if not fake_probabilities:

        return {
            "recording_fake_probability": 0.0,
            "maximum_fake_probability": 0.0,
            "mean_fake_probability": 0.0,
            "weighted_fake_probability": 0.0,
            "high_chunk_count": 0,
            "extreme_chunk_count": 0,
            "high_chunk_ratio": 0.0,
            "medium_or_high_chunk_ratio": 0.0,
            "total_chunks": 0,
            "risk_level": "LOW",
        }

    # --------------------------------------------------------
    # Sanitize probabilities
    # --------------------------------------------------------

    values = [
        max(
            0.0,
            min(
                1.0,
                float(value)
            )
        )
        for value in fake_probabilities
    ]

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    mean_probability = (
        sum(values) / len(values)
    )

    maximum_probability = max(values)

    # --------------------------------------------------------
    # Weighted probability
    #
    # Newer chunks receive slightly greater weight.
    # This is reported as supporting information, not the
    # primary risk decision.
    # --------------------------------------------------------

    weights = list(
        range(
            1,
            len(values) + 1
        )
    )

    weighted_sum = sum(
        value * weight
        for value, weight in zip(
            values,
            weights
        )
    )

    weighted_probability = (
        weighted_sum / sum(weights)
    )

    # --------------------------------------------------------
    # Evidence counts
    # --------------------------------------------------------

    high_chunk_count = sum(
        1
        for value in values
        if value >= high_threshold
    )

    extreme_chunk_count = sum(
        1
        for value in values
        if value >= extreme_threshold
    )

    medium_or_high_chunk_count = sum(
        1
        for value in values
        if value >= medium_threshold
    )

    high_chunk_ratio = (
        high_chunk_count / len(values)
    )

    medium_or_high_chunk_ratio = (
        medium_or_high_chunk_count
        / len(values)
    )

    # --------------------------------------------------------
    # Recording-level decision
    #
    # HIGH:
    #   1. Any extremely strong chunk >= 0.90
    #   2. At least 2 strong chunks >= 0.80
    #   3. Strong chunk >= 0.80 AND recording mean >= 0.50
    #
    # MEDIUM:
    #   1. Any chunk >= 0.50
    #   2. Recording mean >= 0.50
    #
    # LOW:
    #   Otherwise
    # --------------------------------------------------------

    if (
        maximum_probability >= extreme_threshold
    ):

        risk_level = "HIGH"

    elif (
        high_chunk_count >= 2
    ):

        risk_level = "HIGH"

    elif (
        maximum_probability >= high_threshold
        and mean_probability >= medium_threshold
    ):

        risk_level = "HIGH"

    elif (
        maximum_probability >= medium_threshold
        or mean_probability >= medium_threshold
    ):

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # --------------------------------------------------------
    # Recording score
    #
    # The score is primarily the mean probability and is
    # intended as an explanatory metric, not a probability
    # calibrated to represent certainty.
    # --------------------------------------------------------

    recording_score = (
        mean_probability * 100
    )

    return {
        "recording_fake_probability": (
            mean_probability
        ),

        "maximum_fake_probability": (
            maximum_probability
        ),

        "mean_fake_probability": (
            mean_probability
        ),

        "weighted_fake_probability": (
            weighted_probability
        ),

        "high_chunk_count": (
            high_chunk_count
        ),

        "extreme_chunk_count": (
            extreme_chunk_count
        ),

        "high_chunk_ratio": (
            high_chunk_ratio
        ),

        "medium_or_high_chunk_ratio": (
            medium_or_high_chunk_ratio
        ),

        "total_chunks": (
            len(values)
        ),

        "risk_score": (
            recording_score
        ),

        "risk_level": (
            risk_level
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RECORDING-LEVEL RISK ANALYZER TEST")
    print("=" * 60)

    test_cases = {

        "file1269-like": [
            0.6672,
            0.4143,
            0.9850,
            0.2689,
        ],

        "file1000-like": [
            0.1071,
            0.7909,
        ],

        "mostly-real": [
            0.10,
            0.15,
            0.20,
            0.18,
        ],

        "mostly-fake": [
            0.91,
            0.88,
            0.94,
            0.97,
        ],
    }

    for name, values in test_cases.items():

        result = analyze_recording(
            values
        )

        print()
        print(
            f"TEST: {name}"
        )

        print(
            f"Chunks: "
            f"{result['total_chunks']}"
        )

        print(
            f"Mean fake: "
            f"{result['mean_fake_probability'] * 100:.2f}%"
        )

        print(
            f"Maximum fake: "
            f"{result['maximum_fake_probability'] * 100:.2f}%"
        )

        print(
            f"Weighted fake: "
            f"{result['weighted_fake_probability'] * 100:.2f}%"
        )

        print(
            f"High chunks: "
            f"{result['high_chunk_count']}"
        )

        print(
            f"Extreme chunks: "
            f"{result['extreme_chunk_count']}"
        )

        print(
            f"High chunk ratio: "
            f"{result['high_chunk_ratio'] * 100:.2f}%"
        )

        print(
            f"Recording score: "
            f"{result['risk_score']:.2f}/100"
        )

        print(
            f"Risk level: "
            f"{result['risk_level']}"
        )

    print()
    print("=" * 60)