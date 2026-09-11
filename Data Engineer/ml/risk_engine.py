from collections import deque


# ============================================================
# RISK ENGINE CONFIGURATION
# ============================================================

HISTORY_SIZE = 3

IMMEDIATE_HIGH_THRESHOLD = 0.90
HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.50

HIGH_EVIDENCE_COUNT = 2

# Speaker evidence is supporting evidence.
DEEPFAKE_WEIGHT = 0.70
SPEAKER_WEIGHT = 0.30

# Minimum deepfake signal required before speaker evidence
# is allowed to influence the final risk level.
SPEAKER_SUPPORT_MINIMUM = 0.50


# ============================================================
# RISK ENGINE
# ============================================================

class RiskEngine:

    def __init__(
        self,
        history_size=HISTORY_SIZE
    ):

        self.history = deque(
            maxlen=history_size
        )

        self.session_alert = False

    # ========================================================
    # ADD NEW CHUNK RESULT
    # ========================================================

    def add_result(
        self,
        fake_probability,
        speaker_similarity=None
    ):

        fake_probability = max(
            0.0,
            min(
                1.0,
                float(fake_probability)
            )
        )

        if speaker_similarity is not None:

            speaker_similarity = max(
                0.0,
                min(
                    1.0,
                    float(speaker_similarity)
                )
            )

        self.history.append(
            fake_probability
        )

        result = self.get_risk(
            current_probability=fake_probability,
            speaker_similarity=speaker_similarity
        )

        # ----------------------------------------------------
        # Persistent HIGH alert
        # ----------------------------------------------------

        if result["risk_level"] == "HIGH":

            self.session_alert = True

        if self.session_alert:

            result["risk_level"] = "HIGH"
            result["session_alert"] = True

        else:

            result["session_alert"] = False

        return result

    # ========================================================
    # CALCULATE RISK
    # ========================================================

    def get_risk(
        self,
        current_probability=None,
        speaker_similarity=None
    ):

        if not self.history:

            return {
                "risk_level": "LOW",
                "risk_score": 0.0,
                "average_fake_probability": 0.0,
                "high_evidence": 0,
                "speaker_similarity": speaker_similarity,
                "speaker_mismatch_score": None,
                "speaker_reference_used": (
                    speaker_similarity is not None
                ),
                "session_alert": self.session_alert
            }

        values = list(
            self.history
        )

        # ----------------------------------------------------
        # Weighted rolling average
        # ----------------------------------------------------

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

        weight_total = sum(
            weights
        )

        weighted_average = (
            weighted_sum
            / weight_total
        )

        # ----------------------------------------------------
        # High-evidence chunks
        # ----------------------------------------------------

        high_evidence = sum(
            1
            for value in values
            if value >= HIGH_THRESHOLD
        )

        # ----------------------------------------------------
        # Immediate HIGH
        # ----------------------------------------------------

        immediate_high = (
            current_probability is not None
            and current_probability
            >= IMMEDIATE_HIGH_THRESHOLD
        )

        # ----------------------------------------------------
        # Deepfake score
        # ----------------------------------------------------

        deepfake_score = (
            weighted_average * 100
        )

        # ----------------------------------------------------
        # Speaker evidence
        # ----------------------------------------------------

        speaker_reference_used = (
            speaker_similarity is not None
        )

        if speaker_reference_used:

            speaker_mismatch_score = (
                (1.0 - speaker_similarity) * 100
            )

        else:

            speaker_mismatch_score = None

        # ----------------------------------------------------
        # Combined score
        #
        # Speaker evidence is only allowed to contribute
        # when there is at least a MEDIUM deepfake signal.
        # ----------------------------------------------------

        if (
            speaker_reference_used
            and weighted_average
            >= SPEAKER_SUPPORT_MINIMUM
        ):

            combined_score = (
                DEEPFAKE_WEIGHT * deepfake_score
                + SPEAKER_WEIGHT * speaker_mismatch_score
            )

        else:

            combined_score = deepfake_score

        combined_score = max(
            0.0,
            min(100.0, combined_score)
        )

        # ----------------------------------------------------
        # Determine risk level
        # ----------------------------------------------------
        #
        # Deepfake evidence remains the primary trigger.
        # ----------------------------------------------------

        if (
            immediate_high
            or high_evidence >= HIGH_EVIDENCE_COUNT
            or weighted_average >= HIGH_THRESHOLD
        ):

            risk_level = "HIGH"

        elif weighted_average >= MEDIUM_THRESHOLD:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        # ----------------------------------------------------
        # Persistent session alert
        # ----------------------------------------------------

        if self.session_alert:

            risk_level = "HIGH"

        return {
            "risk_level": risk_level,

            "risk_score": combined_score,

            "average_fake_probability": (
                weighted_average
            ),

            "high_evidence": high_evidence,

            "speaker_similarity": (
                speaker_similarity
            ),

            "speaker_mismatch_score": (
                speaker_mismatch_score
            ),

            "speaker_reference_used": (
                speaker_reference_used
            ),

            "session_alert": (
                self.session_alert
            )
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.history.clear()

        self.session_alert = False


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RISK ENGINE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Test 1: Existing deepfake-only behavior
    # --------------------------------------------------------

    print("\nTEST 1: DEEPFAKE ONLY")

    engine = RiskEngine()

    test_values = [
        0.66,
        0.41,
        0.985,
        0.27
    ]

    for index, value in enumerate(
        test_values,
        start=1
    ):

        result = engine.add_result(
            value
        )

        print(
            f"Chunk {index}: "
            f"{value * 100:.2f}% fake | "
            f"Risk: "
            f"{result['risk_level']:6s} | "
            f"Score: "
            f"{result['risk_score']:6.2f} | "
            f"Session alert: "
            f"{result['session_alert']}"
        )

    # --------------------------------------------------------
    # Test 2: Low deepfake + strong mismatch
    # --------------------------------------------------------

    print("\nTEST 2: LOW DEEPFAKE + SPEAKER MISMATCH")

    engine.reset()

    result = engine.add_result(
        fake_probability=0.1071,
        speaker_similarity=0.0069
    )

    print(
        f"Fake probability: "
        f"{result['average_fake_probability'] * 100:.2f}%"
    )

    print(
        f"Speaker similarity: "
        f"{result['speaker_similarity']:.4f}"
    )

    print(
        f"Combined score: "
        f"{result['risk_score']:.2f}/100"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    # --------------------------------------------------------
    # Test 3: Moderate deepfake + speaker mismatch
    # --------------------------------------------------------

    print("\nTEST 3: MODERATE DEEPFAKE + SPEAKER MISMATCH")

    engine.reset()

    result = engine.add_result(
        fake_probability=0.6672,
        speaker_similarity=0.0160
    )

    print(
        f"Fake probability: "
        f"{result['average_fake_probability'] * 100:.2f}%"
    )

    print(
        f"Speaker similarity: "
        f"{result['speaker_similarity']:.4f}"
    )

    print(
        f"Combined score: "
        f"{result['risk_score']:.2f}/100"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print("=" * 60)