# backend/services/alert_action.py

"""
Alert and Action Handler

Converts the final risk level into a security alert
and recommended action for the voice-cloning detection system.
"""


def generate_alert_action(
    risk_score: float,
    risk_level: str,
    risk_factors: list
):
    """
    Generate a security alert and recommended action
    based on the final contextual risk level.
    """

    # ---------------------------------------------------------
    # 1. Validate risk score
    # ---------------------------------------------------------

    if not isinstance(risk_score, (int, float)):
        raise TypeError("risk_score must be a number")

    if risk_score < 0 or risk_score > 100:
        raise ValueError(
            "risk_score must be between 0 and 100"
        )

    # ---------------------------------------------------------
    # 2. Validate risk level
    # ---------------------------------------------------------

    valid_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    }

    risk_level = risk_level.upper()

    if risk_level not in valid_levels:
        raise ValueError(
            "Invalid risk level"
        )

    # ---------------------------------------------------------
    # 3. Generate alert and action
    # ---------------------------------------------------------

    if risk_level == "CRITICAL":

        alert = (
            "CRITICAL ALERT: Potential AI voice "
            "impersonation detected."
        )

        action = (
            "Do NOT authorize the requested transaction. "
            "Require independent callback and additional "
            "authentication."
        )

        priority = "IMMEDIATE"

    elif risk_level == "HIGH":

        alert = (
            "HIGH RISK ALERT: Suspicious voice call detected."
        )

        action = (
            "Perform additional authentication and "
            "independently verify the caller."
        )

        priority = "HIGH"

    elif risk_level == "MEDIUM":

        alert = (
            "MEDIUM RISK: Suspicious call characteristics detected."
        )

        action = (
            "Proceed with caution and perform additional "
            "caller verification."
        )

        priority = "NORMAL"

    else:

        alert = (
            "LOW RISK: No significant security threat detected."
        )

        action = (
            "Continue with normal verification procedures."
        )

        priority = "LOW"

    # ---------------------------------------------------------
    # 4. Return alert information
    # ---------------------------------------------------------

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "priority": priority,
        "alert": alert,
        "action": action,
        "risk_factors": risk_factors
    }


# -------------------------------------------------------------
# CRITICAL-RISK TEST
# -------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------------------------
    # Simulated CRITICAL-risk result
    # ---------------------------------------------------------

    result = generate_alert_action(
        risk_score=100,
        risk_level="CRITICAL",
        risk_factors=[
            "Unknown caller",
            "Urgent request",
            "High-value transaction",
            "First-time or unusual request"
        ]
    )

    # ---------------------------------------------------------
    # Display Alert
    # ---------------------------------------------------------

    print("Alert & Action")
    print("-------------------------")

    print("\nRisk Score:")
    print(f"{result['risk_score']}/100")

    print("\nRisk Level:")
    print(result["risk_level"])

    print("\nPriority:")
    print(result["priority"])

    print("\nAlert:")
    print(result["alert"])

    print("\nRisk Factors:")

    for factor in result["risk_factors"]:
        print(f"- {factor}")

    print("\nRecommended Action:")
    print(result["action"])