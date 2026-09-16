# backend/services/risk_engine.py

"""
Advanced Adaptive Risk Engine for AI Voice Deepfake Detection

Purpose:
    Converts multiple AI/security signals into an explainable,
    adaptive security risk assessment.

Pipeline:

    AI Detection Signals
            |
            v
    Signal Normalization
            |
            v
    Weighted Risk Fusion
            |
            v
    Correlation Analysis
            |
            v
    Adaptive Escalation
            |
            v
    Explainable Risk Assessment
            |
            v
    Security Decision

Inputs:
    - AI-generated voice probability
    - Speaker similarity
    - Prosody anomaly
    - Caller anomaly
    - Transaction risk

Outputs:
    - Risk score (0-100)
    - Risk level
    - Confidence
    - Signal contributions
    - Risk factors
    - Correlation factors
    - Escalation bonus
    - Security recommendation
    - Security decision
    - Alert
"""


# =========================================================
# CONSTANTS
# =========================================================

RISK_WEIGHTS = {
    "ai_probability": 0.40,
    "speaker_anomaly": 0.20,
    "prosody_anomaly": 0.10,
    "caller_anomaly": 0.15,
    "transaction_risk": 0.15
}

RISK_THRESHOLDS = {
    "critical": 80,
    "high": 60,
    "medium": 30
}


# =========================================================
# VALIDATION
# =========================================================

def _validate_score(name: str, value: float):
    """
    Validate a numerical security signal.

    Every signal must be within the range 0-100.
    """

    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a number"
        )

    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a number"
        )

    if value < 0 or value > 100:
        raise ValueError(
            f"{name} must be between 0 and 100"
        )


# =========================================================
# RISK LEVEL CLASSIFICATION
# =========================================================

def _classify_risk(score: float) -> str:
    """
    Convert numerical risk score into a security level.
    """

    if score >= RISK_THRESHOLDS["critical"]:
        return "CRITICAL"

    if score >= RISK_THRESHOLDS["high"]:
        return "HIGH"

    if score >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM"

    return "LOW"


# =========================================================
# CONFIDENCE CALCULATION
# =========================================================

def _calculate_confidence(
    ai_probability: float,
    speaker_anomaly: float,
    prosody_anomaly: float,
    caller_anomaly: float,
    transaction_risk: float
) -> float:
    """
    Estimate confidence based on the number and strength
    of independently suspicious signals.

    This is an internal consistency indicator, not a
    machine-learning model probability.
    """

    signals = [
        ai_probability,
        speaker_anomaly,
        prosody_anomaly,
        caller_anomaly,
        transaction_risk
    ]

    active_signals = sum(
        signal >= 50
        for signal in signals
    )

    strong_signals = sum(
        signal >= 70
        for signal in signals
    )

    confidence = (
        (active_signals / 5) * 60
        + (strong_signals / 5) * 40
    )

    return round(
        min(100, confidence),
        2
    )


# =========================================================
# SIGNAL CONTRIBUTION ANALYSIS
# =========================================================

def _calculate_signal_contributions(
    ai_probability: float,
    speaker_anomaly: float,
    prosody_anomaly: float,
    caller_anomaly: float,
    transaction_risk: float
):
    """
    Calculate the contribution of each signal to the
    weighted base risk score.
    """

    contributions = {
        "ai_probability": round(
            ai_probability *
            RISK_WEIGHTS["ai_probability"],
            2
        ),

        "speaker_anomaly": round(
            speaker_anomaly *
            RISK_WEIGHTS["speaker_anomaly"],
            2
        ),

        "prosody_anomaly": round(
            prosody_anomaly *
            RISK_WEIGHTS["prosody_anomaly"],
            2
        ),

        "caller_anomaly": round(
            caller_anomaly *
            RISK_WEIGHTS["caller_anomaly"],
            2
        ),

        "transaction_risk": round(
            transaction_risk *
            RISK_WEIGHTS["transaction_risk"],
            2
        )
    }

    return contributions


# =========================================================
# MAIN RISK ENGINE
# =========================================================

def calculate_risk(
    ai_probability: float,
    speaker_similarity: float,
    prosody_anomaly: float,
    caller_anomaly: float,
    transaction_risk: float
):
    """
    Calculate adaptive and explainable security risk.

    All inputs must be between 0 and 100.
    """

    # =====================================================
    # 1. VALIDATE INPUTS
    # =====================================================

    values = {
        "ai_probability": ai_probability,
        "speaker_similarity": speaker_similarity,
        "prosody_anomaly": prosody_anomaly,
        "caller_anomaly": caller_anomaly,
        "transaction_risk": transaction_risk
    }

    for name, value in values.items():
        _validate_score(name, value)

    # =====================================================
    # 2. NORMALIZE SPEAKER SIGNAL
    # =====================================================

    # High speaker similarity = low risk.
    # Therefore convert similarity into anomaly.

    speaker_anomaly = round(
        100 - speaker_similarity,
        2
    )

    # =====================================================
    # 3. WEIGHTED SIGNAL FUSION
    # =====================================================

    signal_contributions = _calculate_signal_contributions(
        ai_probability=ai_probability,
        speaker_anomaly=speaker_anomaly,
        prosody_anomaly=prosody_anomaly,
        caller_anomaly=caller_anomaly,
        transaction_risk=transaction_risk
    )

    weighted_score = sum(
        signal_contributions.values()
    )

    # =====================================================
    # 4. INDIVIDUAL RISK FACTORS
    # =====================================================

    risk_factors = []

    if ai_probability >= 70:

        risk_factors.append(
            "High AI-generated voice probability"
        )

    elif ai_probability >= 50:

        risk_factors.append(
            "Moderate AI-generated voice probability"
        )

    if speaker_anomaly >= 50:

        risk_factors.append(
            "Low speaker similarity"
        )

    elif speaker_anomaly >= 40:

        risk_factors.append(
            "Reduced speaker similarity"
        )

    if prosody_anomaly >= 60:

        risk_factors.append(
            "Significant prosody anomaly"
        )

    elif prosody_anomaly >= 50:

        risk_factors.append(
            "Moderate prosody anomaly"
        )

    if caller_anomaly >= 60:

        risk_factors.append(
            "Suspicious caller behavior"
        )

    elif caller_anomaly >= 50:

        risk_factors.append(
            "Unusual caller behavior"
        )

    if transaction_risk >= 70:

        risk_factors.append(
            "High transaction risk"
        )

    elif transaction_risk >= 50:

        risk_factors.append(
            "Elevated transaction risk"
        )

    # =====================================================
    # 5. CORRELATION ANALYSIS
    # =====================================================

    escalation_bonus = 0

    correlation_factors = []

    high_ai = ai_probability >= 70

    suspicious_speaker = speaker_anomaly >= 40

    suspicious_prosody = prosody_anomaly >= 50

    suspicious_caller = caller_anomaly >= 60

    high_transaction = transaction_risk >= 70

    # -----------------------------------------------------
    # AI + Speaker mismatch
    # -----------------------------------------------------

    if high_ai and suspicious_speaker:

        escalation_bonus += 8

        correlation_factors.append(
            "AI voice and speaker mismatch detected together"
        )

    # -----------------------------------------------------
    # AI + Caller anomaly
    # -----------------------------------------------------

    if high_ai and suspicious_caller:

        escalation_bonus += 7

        correlation_factors.append(
            "AI voice probability combined with caller anomaly"
        )

    # -----------------------------------------------------
    # Caller + Transaction risk
    # -----------------------------------------------------

    if suspicious_caller and high_transaction:

        escalation_bonus += 7

        correlation_factors.append(
            "Suspicious caller behavior with high-risk transaction"
        )

    # -----------------------------------------------------
    # AI + Transaction risk
    # -----------------------------------------------------

    if high_ai and high_transaction:

        escalation_bonus += 8

        correlation_factors.append(
            "AI voice risk combined with high-value transaction"
        )

    # -----------------------------------------------------
    # Multi-signal attack pattern
    # -----------------------------------------------------

    active_high_risk_signals = sum([
        high_ai,
        suspicious_speaker,
        suspicious_prosody,
        suspicious_caller,
        high_transaction
    ])

    if active_high_risk_signals >= 4:

        escalation_bonus += 5

        correlation_factors.append(
            "Multiple independent high-risk indicators detected"
        )

    # -----------------------------------------------------
    # Strong AI + caller + transaction pattern
    # -----------------------------------------------------

    if (
        high_ai
        and suspicious_caller
        and high_transaction
    ):

        escalation_bonus += 5

        correlation_factors.append(
            "AI impersonation pattern combined with "
            "caller anomaly and transaction risk"
        )

    # =====================================================
    # 6. FINAL RISK SCORE
    # =====================================================

    raw_score = (
        weighted_score
        + escalation_bonus
    )

    risk_score = round(
        max(0, min(100, raw_score)),
        2
    )

    # =====================================================
    # 7. RISK LEVEL
    # =====================================================

    risk_level = _classify_risk(
        risk_score
    )

    # =====================================================
    # 8. CONFIDENCE
    # =====================================================

    confidence = _calculate_confidence(
        ai_probability=ai_probability,
        speaker_anomaly=speaker_anomaly,
        prosody_anomaly=prosody_anomaly,
        caller_anomaly=caller_anomaly,
        transaction_risk=transaction_risk
    )

    # =====================================================
    # 9. SECURITY DECISION
    # =====================================================

    if risk_level == "CRITICAL":

        recommendation = [
            "Do NOT authorize the requested transaction",
            "Perform an independent callback",
            "Require multi-factor authentication",
            "Escalate to a supervisor or security team"
        ]

        alert = (
            "CRITICAL ALERT: Potential AI voice "
            "impersonation detected"
        )

        decision = "BLOCK_AND_ESCALATE"

    elif risk_level == "HIGH":

        recommendation = [
            "Verify the caller independently",
            "Require additional authentication",
            "Do not approve high-value requests "
            "without verification"
        ]

        alert = (
            "HIGH RISK ALERT: Suspicious voice call detected"
        )

        decision = "STEP_UP_AUTHENTICATION"

    elif risk_level == "MEDIUM":

        recommendation = [
            "Perform additional caller verification",
            "Monitor the request carefully"
        ]

        alert = (
            "MEDIUM RISK: Suspicious call "
            "characteristics detected"
        )

        decision = "VERIFY"

    else:

        recommendation = [
            "Continue normal verification procedures"
        ]

        alert = (
            "LOW RISK: No significant security "
            "threat detected"
        )

        decision = "ALLOW_WITH_NORMAL_VERIFICATION"

    # =====================================================
    # 10. COMBINE EXPLANATION FACTORS
    # =====================================================

    all_factors = list(
        dict.fromkeys(
            risk_factors
            + correlation_factors
        )
    )

    # =====================================================
    # 11. RETURN COMPLETE RESULT
    # =====================================================

    return {

        # -------------------------------------------------
        # Core risk result
        # -------------------------------------------------

        "risk_score": risk_score,

        "risk_level": risk_level,

        "confidence": confidence,

        # -------------------------------------------------
        # Signal information
        # -------------------------------------------------

        "signal_values": {
            "ai_probability": ai_probability,
            "speaker_similarity": speaker_similarity,
            "speaker_anomaly": speaker_anomaly,
            "prosody_anomaly": prosody_anomaly,
            "caller_anomaly": caller_anomaly,
            "transaction_risk": transaction_risk
        },

        # -------------------------------------------------
        # Explainable contributions
        # -------------------------------------------------

        "signal_contributions": signal_contributions,

        # -------------------------------------------------
        # Risk explanation
        # -------------------------------------------------

        "risk_factors": all_factors,

        "individual_risk_factors": risk_factors,

        "correlation_factors": correlation_factors,

        # -------------------------------------------------
        # Escalation
        # -------------------------------------------------

        "escalation_bonus": escalation_bonus,

        # -------------------------------------------------
        # Security response
        # -------------------------------------------------

        "recommendation": recommendation,

        "alert": alert,

        "security_decision": decision
    }


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":

    result = calculate_risk(

        ai_probability=91,

        speaker_similarity=72,

        prosody_anomaly=60,

        caller_anomaly=80,

        transaction_risk=90
    )

    print()
    print("=" * 65)
    print("             ADVANCED ADAPTIVE RISK ENGINE")
    print("=" * 65)

    # -----------------------------------------------------
    # Risk summary
    # -----------------------------------------------------

    print("\nRISK ASSESSMENT")
    print("-" * 65)

    print(
        f"Risk Score       : "
        f"{result['risk_score']}/100"
    )

    print(
        f"Risk Level       : "
        f"{result['risk_level']}"
    )

    print(
        f"Confidence       : "
        f"{result['confidence']}%"
    )

    print(
        f"Escalation Bonus : "
        f"+{result['escalation_bonus']}"
    )

    # -----------------------------------------------------
    # Signal values
    # -----------------------------------------------------

    print("\nSIGNAL ANALYSIS")
    print("-" * 65)

    for name, value in result[
        "signal_values"
    ].items():

        print(
            f"{name:<25}: {value}"
        )

    # -----------------------------------------------------
    # Signal contributions
    # -----------------------------------------------------

    print("\nRISK CONTRIBUTIONS")
    print("-" * 65)

    for name, value in result[
        "signal_contributions"
    ].items():

        print(
            f"{name:<25}: +{value}"
        )

    # -----------------------------------------------------
    # Risk factors
    # -----------------------------------------------------

    print("\nRISK FACTORS")
    print("-" * 65)

    if result["risk_factors"]:

        for factor in result["risk_factors"]:

            print(
                f"- {factor}"
            )

    else:

        print(
            "- No significant risk factors"
        )

    # -----------------------------------------------------
    # Security decision
    # -----------------------------------------------------

    print("\nSECURITY DECISION")
    print("-" * 65)

    print(
        f"Decision : "
        f"{result['security_decision']}"
    )

    # -----------------------------------------------------
    # Alert
    # -----------------------------------------------------

    print("\nALERT")
    print("-" * 65)

    print(
        result["alert"]
    )

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    print("\nRECOMMENDATIONS")
    print("-" * 65)

    for recommendation in result[
        "recommendation"
    ]:

        print(
            f"- {recommendation}"
        )

    print()
    print("=" * 65)
    print("          ADVANCED RISK ENGINE COMPLETED")
    print("=" * 65)