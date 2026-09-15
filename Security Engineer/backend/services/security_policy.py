"""
Security Policy and Escalation Engine

Determines the security response after risk analysis.

This module converts risk information into an explicit
security decision suitable for API/dashboard integration.

Possible decisions:

LOW
    ALLOW_WITH_NORMAL_VERIFICATION

MEDIUM
    VERIFY

HIGH
    STEP_UP_AUTHENTICATION

CRITICAL
    BLOCK_AND_ESCALATE
"""


def _validate_score(name: str, value: float):
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")

    if value < 0 or value > 100:
        raise ValueError(f"{name} must be between 0 and 100")


def determine_security_policy(
    risk_score: float,
    risk_level: str,
    caller_known: bool,
    urgent_request: bool,
    high_value_transaction: bool
):
    """
    Determine the appropriate security response.

    Parameters
    ----------
    risk_score : float
        Final contextual risk score from 0-100.

    risk_level : str
        LOW, MEDIUM, HIGH or CRITICAL.

    caller_known : bool
        Whether the caller is recognized.

    urgent_request : bool
        Whether the caller is attempting to create urgency.

    high_value_transaction : bool
        Whether the request involves a high-value transaction.
    """

    _validate_score("risk_score", risk_score)

    if not isinstance(risk_level, str):
        raise TypeError("risk_level must be a string")

    valid_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    }

    risk_level = risk_level.upper()

    if risk_level not in valid_levels:
        raise ValueError(
            "risk_level must be LOW, MEDIUM, HIGH or CRITICAL"
        )

    if not isinstance(caller_known, bool):
        raise TypeError("caller_known must be True or False")

    if not isinstance(urgent_request, bool):
        raise TypeError("urgent_request must be True or False")

    if not isinstance(high_value_transaction, bool):
        raise TypeError(
            "high_value_transaction must be True or False"
        )

    risk_factors = []

    if not caller_known:
        risk_factors.append("Unknown caller")

    if urgent_request:
        risk_factors.append("Urgency detected")

    if high_value_transaction:
        risk_factors.append("High-value transaction")

    # ---------------------------------------------------------
    # CRITICAL
    # ---------------------------------------------------------

    if risk_level == "CRITICAL":

        decision = "BLOCK_AND_ESCALATE"

        priority = "IMMEDIATE"

        actions = [
            "Block transaction authorization",
            "Require independent caller verification",
            "Require multi-factor authentication",
            "Escalate to security team",
            "Create critical security event"
        ]

        alert_message = (
            "Potential AI voice impersonation detected. "
            "Transaction authorization must be blocked."
        )

    # ---------------------------------------------------------
    # HIGH
    # ---------------------------------------------------------

    elif risk_level == "HIGH":

        decision = "STEP_UP_AUTHENTICATION"

        priority = "HIGH"

        actions = [
            "Require additional authentication",
            "Perform independent caller verification",
            "Restrict high-value transaction approval",
            "Create high-priority security event"
        ]

        alert_message = (
            "High-risk voice interaction detected. "
            "Additional authentication is required."
        )

    # ---------------------------------------------------------
    # MEDIUM
    # ---------------------------------------------------------

    elif risk_level == "MEDIUM":

        decision = "VERIFY"

        priority = "NORMAL"

        actions = [
            "Perform additional caller verification",
            "Monitor the interaction",
            "Continue only after verification"
        ]

        alert_message = (
            "Suspicious call characteristics detected. "
            "Additional verification is recommended."
        )

    # ---------------------------------------------------------
    # LOW
    # ---------------------------------------------------------

    else:

        decision = "ALLOW_WITH_NORMAL_VERIFICATION"

        priority = "LOW"

        actions = [
            "Continue normal verification procedures",
            "Monitor normally"
        ]

        alert_message = (
            "No significant security threat detected."
        )

    # ---------------------------------------------------------
    # Dynamic escalation
    # ---------------------------------------------------------

    escalation_required = (
        risk_level == "CRITICAL"
        or (
            risk_level == "HIGH"
            and high_value_transaction
        )
    )

    if escalation_required:
        escalation_reason = (
            "Risk level and transaction context require "
            "security escalation."
        )
    else:
        escalation_reason = (
            "No immediate security escalation required."
        )

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "security_decision": decision,
        "priority": priority,
        "escalation_required": escalation_required,
        "escalation_reason": escalation_reason,
        "risk_factors": risk_factors,
        "actions": actions,
        "alert_message": alert_message
    }


if __name__ == "__main__":

    result = determine_security_policy(
        risk_score=95.5,
        risk_level="CRITICAL",
        caller_known=False,
        urgent_request=True,
        high_value_transaction=True
    )

    print("\nSECURITY POLICY ENGINE")
    print("=" * 45)

    print(f"Risk Score          : {result['risk_score']}")
    print(f"Risk Level          : {result['risk_level']}")
    print(f"Security Decision   : {result['security_decision']}")
    print(f"Priority            : {result['priority']}")
    print(
        f"Escalation Required : "
        f"{result['escalation_required']}"
    )

    print("\nRisk Factors:")

    for factor in result["risk_factors"]:
        print(f"- {factor}")

    print("\nSecurity Actions:")

    for action in result["actions"]:
        print(f"- {action}")

    print("\nAlert:")
    print(result["alert_message"])