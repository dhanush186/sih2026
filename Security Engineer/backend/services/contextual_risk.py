"""
Contextual Risk Analysis Engine

Combines the base AI voice risk score with contextual
information about the caller and requested action.

Pipeline:

AI Voice Analysis
        ↓
Base Risk Engine
        ↓
Contextual Risk Engine
        ↓
Security Policy Engine
        ↓
Risk Monitoring
        ↓
Security Decision Override
        ↓
Alert + Action
        ↓
Audit Log
"""

from backend.services.audit_logger import create_audit_event
from backend.services.risk_engine import calculate_risk
from backend.services.alert_action import generate_alert_action
from backend.services.security_policy import determine_security_policy
from backend.services.risk_monitor import RiskMonitor


# =========================================================
# GLOBAL RISK MONITOR
# =========================================================

risk_monitor = RiskMonitor(
    escalation_threshold=3,
    monitoring_window_minutes=15
)


# =========================================================
# VALIDATION
# =========================================================

def _validate_boolean(name: str, value: bool):
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be True or False"
        )


# =========================================================
# CONTEXTUAL RISK CALCULATION
# =========================================================

def calculate_contextual_risk(
    base_risk_score: float,
    caller_known: bool,
    urgent_request: bool,
    high_value_transaction: bool,
    first_time_request: bool
):
    """
    Adjust the base AI risk score using contextual
    security information.
    """

    # -----------------------------------------------------
    # Validate base risk score
    # -----------------------------------------------------

    if not isinstance(base_risk_score, (int, float)):
        raise TypeError(
            "base_risk_score must be a number"
        )

    if base_risk_score < 0 or base_risk_score > 100:
        raise ValueError(
            "base_risk_score must be between 0 and 100"
        )

    # -----------------------------------------------------
    # Validate boolean inputs
    # -----------------------------------------------------

    _validate_boolean(
        "caller_known",
        caller_known
    )

    _validate_boolean(
        "urgent_request",
        urgent_request
    )

    _validate_boolean(
        "high_value_transaction",
        high_value_transaction
    )

    _validate_boolean(
        "first_time_request",
        first_time_request
    )

    # -----------------------------------------------------
    # Start with base risk
    # -----------------------------------------------------

    contextual_score = base_risk_score
    contextual_bonus = 0

    risk_factors = []

    # -----------------------------------------------------
    # Individual contextual risk factors
    # -----------------------------------------------------

    if not caller_known:
        contextual_score += 10
        contextual_bonus += 10

        risk_factors.append(
            "Unknown caller"
        )

    if urgent_request:
        contextual_score += 10
        contextual_bonus += 10

        risk_factors.append(
            "Urgent request detected"
        )

    if high_value_transaction:
        contextual_score += 10
        contextual_bonus += 10

        risk_factors.append(
            "High-value transaction requested"
        )

    if first_time_request:
        contextual_score += 5
        contextual_bonus += 5

        risk_factors.append(
            "First-time or unusual request"
        )

    # -----------------------------------------------------
    # Context combinations
    # -----------------------------------------------------

    if not caller_known and urgent_request:
        contextual_score += 5
        contextual_bonus += 5

        risk_factors.append(
            "Unknown caller combined with urgent request"
        )

    if urgent_request and high_value_transaction:
        contextual_score += 5
        contextual_bonus += 5

        risk_factors.append(
            "Urgent request combined with high-value transaction"
        )

    if not caller_known and high_value_transaction:
        contextual_score += 5
        contextual_bonus += 5

        risk_factors.append(
            "Unknown caller combined with high-value transaction"
        )

    if urgent_request and first_time_request:
        contextual_score += 3
        contextual_bonus += 3

        risk_factors.append(
            "Urgent first-time request"
        )

    # -----------------------------------------------------
    # Clamp score
    # -----------------------------------------------------

    contextual_score = round(
        max(0, min(100, contextual_score)),
        2
    )

    # -----------------------------------------------------
    # Determine contextual risk level
    # -----------------------------------------------------

    if contextual_score >= 80:
        risk_level = "CRITICAL"

        recommendation = [
            "Do NOT authorize the requested transaction",
            "Perform an independent callback",
            "Require MFA verification",
            "Escalate to security team"
        ]

    elif contextual_score >= 60:
        risk_level = "HIGH"

        recommendation = [
            "Verify the caller independently",
            "Require additional authentication",
            "Do not approve high-value requests without verification"
        ]

    elif contextual_score >= 30:
        risk_level = "MEDIUM"

        recommendation = [
            "Perform additional caller verification",
            "Monitor the request carefully"
        ]

    else:
        risk_level = "LOW"

        recommendation = [
            "Continue normal verification procedures"
        ]

    # -----------------------------------------------------
    # Remove duplicate risk factors
    # -----------------------------------------------------

    risk_factors = list(
        dict.fromkeys(risk_factors)
    )

    # -----------------------------------------------------
    # Return contextual analysis
    # -----------------------------------------------------

    return {
        "base_risk_score": round(
            base_risk_score,
            2
        ),

        "contextual_risk_score": contextual_score,

        "contextual_bonus": contextual_bonus,

        "risk_level": risk_level,

        "risk_factors": risk_factors,

        "recommendation": recommendation
    }


# =========================================================
# COMPLETE CALL ANALYSIS
# =========================================================

def analyze_call(
    ai_probability: float,
    speaker_similarity: float,
    prosody_anomaly: float,
    caller_anomaly: float,
    transaction_risk: float,
    caller_known: bool,
    urgent_request: bool,
    high_value_transaction: bool,
    first_time_request: bool,
    caller_id: str = "UNKNOWN_CALLER",
    session_id: str = "DEFAULT_SESSION"
):
    """
    Complete Risk/Security analysis pipeline.

    Steps:

    1. Base AI voice risk analysis
    2. Contextual risk analysis
    3. Security policy decision
    4. Repeated risk monitoring
    5. Security decision override
    6. Alert/action generation
    7. Audit logging
    """

    # =====================================================
    # STEP 1: BASE RISK ENGINE
    # =====================================================

    base_risk = calculate_risk(
        ai_probability=ai_probability,
        speaker_similarity=speaker_similarity,
        prosody_anomaly=prosody_anomaly,
        caller_anomaly=caller_anomaly,
        transaction_risk=transaction_risk
    )

    # =====================================================
    # STEP 2: CONTEXTUAL RISK ENGINE
    # =====================================================

    contextual_result = calculate_contextual_risk(
        base_risk_score=base_risk["risk_score"],
        caller_known=caller_known,
        urgent_request=urgent_request,
        high_value_transaction=high_value_transaction,
        first_time_request=first_time_request
    )

    # =====================================================
    # STEP 3: SECURITY POLICY ENGINE
    # =====================================================

    security_policy = determine_security_policy(
        risk_score=contextual_result[
            "contextual_risk_score"
        ],

        risk_level=contextual_result[
            "risk_level"
        ],

        caller_known=caller_known,

        urgent_request=urgent_request,

        high_value_transaction=high_value_transaction
    )

    # =====================================================
    # STEP 3.5: REPEATED RISK MONITORING
    # =====================================================

    monitor_event = risk_monitor.record_event(
        risk_score=contextual_result[
            "contextual_risk_score"
        ],

        risk_level=contextual_result[
            "risk_level"
        ],

        security_decision=security_policy[
            "security_decision"
        ],

        caller_id=caller_id,

        session_id=session_id
    )

    monitoring_result = (
        risk_monitor.analyze_monitoring_state(
            caller_id=caller_id,
            session_id=session_id
        )
    )

    # =====================================================
    # STEP 4: SECURITY DECISION OVERRIDE
    # =====================================================

    original_security_decision = (
        security_policy[
            "security_decision"
        ]
    )

    original_priority = (
        security_policy[
            "priority"
        ]
    )

    final_security_decision = (
        original_security_decision
    )

    final_priority = original_priority

    override_applied = False
    override_reason = None

    # -----------------------------------------------------
    # Repeated high-risk activity override
    # -----------------------------------------------------

    if monitoring_result[
        "repeated_risk"
    ][
        "repeated_risk_detected"
    ]:

        final_security_decision = (
            "BLOCK_AND_ESCALATE"
        )

        final_priority = "IMMEDIATE"

        override_applied = True

        override_reason = (
            "Repeated HIGH/CRITICAL risk activity "
            "detected within the monitoring window."
        )

    # =====================================================
    # STEP 5: COMBINE RISK FACTORS
    # =====================================================

    combined_factors = list(
        dict.fromkeys(
            base_risk[
                "risk_factors"
            ]
            + contextual_result[
                "risk_factors"
            ]
            + security_policy[
                "risk_factors"
            ]
        )
    )

    # -----------------------------------------------------
    # Add monitoring factor if override occurs
    # -----------------------------------------------------

    if override_applied:

        combined_factors.append(
            "Repeated high-risk activity detected"
        )

    combined_factors = list(
        dict.fromkeys(combined_factors)
    )

    # =====================================================
    # STEP 6: ALERT + ACTION GENERATION
    # =====================================================

    alert_action = generate_alert_action(
        risk_score=contextual_result[
            "contextual_risk_score"
        ],

        risk_level=contextual_result[
            "risk_level"
        ],

        risk_factors=combined_factors
    )

    # =====================================================
    # STEP 7: AUDIT LOGGING
    # =====================================================

    audit_event = create_audit_event(
        base_risk_score=base_risk[
            "risk_score"
        ],

        contextual_risk_score=contextual_result[
            "contextual_risk_score"
        ],

        confidence=base_risk[
            "confidence"
        ],

        risk_level=contextual_result[
            "risk_level"
        ],

        security_decision=final_security_decision,

        priority=final_priority,

        escalation_required=(
            security_policy[
                "escalation_required"
            ]
            or override_applied
        ),

        risk_factors=combined_factors,

        actions=security_policy[
            "actions"
        ]
    )

    # =====================================================
    # STEP 8: FINAL RESULT
    # =====================================================

    return {

        # -------------------------------------------------
        # Base Risk
        # -------------------------------------------------

        "base_risk": base_risk,

        # -------------------------------------------------
        # Contextual Risk
        # -------------------------------------------------

        "contextual_risk": contextual_result,

        # -------------------------------------------------
        # Original Security Policy
        # -------------------------------------------------

        "security_policy": security_policy,

        # -------------------------------------------------
        # Risk Monitoring
        # -------------------------------------------------

        "risk_monitor": {
            "event": monitor_event,
            "monitoring": monitoring_result
        },

        # -------------------------------------------------
        # Security Decision Override
        # -------------------------------------------------

        "security_override": {
            "override_applied": override_applied,

            "original_decision":
                original_security_decision,

            "original_priority":
                original_priority,

            "final_decision":
                final_security_decision,

            "final_priority":
                final_priority,

            "reason":
                override_reason
        },

        # -------------------------------------------------
        # Alert + Action
        # -------------------------------------------------

        "alert_action": alert_action,

        # -------------------------------------------------
        # Audit Event
        # -------------------------------------------------

        "audit_event": audit_event,

        # -------------------------------------------------
        # Final Security Decision
        # -------------------------------------------------

        "final_security_decision":
            final_security_decision,

        "final_priority":
            final_priority,

        # -------------------------------------------------
        # Combined Risk Factors
        # -------------------------------------------------

        "all_risk_factors":
            combined_factors
    }


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":

    result = analyze_call(

        # AI / Voice signals
        ai_probability=91,
        speaker_similarity=72,
        prosody_anomaly=60,
        caller_anomaly=80,
        transaction_risk=90,

        # Contextual signals
        caller_known=False,
        urgent_request=True,
        high_value_transaction=True,
        first_time_request=True,

        # Monitoring identity
        caller_id="CALLER_A",
        session_id="SESSION_001"
    )

    print("\n")
    print("=" * 70)
    print("          CONTEXTUAL RISK SECURITY ENGINE")
    print("=" * 70)

    # -----------------------------------------------------
    # BASE RISK
    # -----------------------------------------------------

    print("\nBASE RISK")
    print("-" * 70)

    print(
        f"Risk Score        : "
        f"{result['base_risk']['risk_score']}"
    )

    print(
        f"Risk Level        : "
        f"{result['base_risk']['risk_level']}"
    )

    print(
        f"Confidence        : "
        f"{result['base_risk']['confidence']}%"
    )

    print(
        f"Escalation Bonus  : "
        f"{result['base_risk']['escalation_bonus']}"
    )

    # -----------------------------------------------------
    # CONTEXTUAL RISK
    # -----------------------------------------------------

    print("\nCONTEXTUAL RISK")
    print("-" * 70)

    print(
        f"Base Score        : "
        f"{result['contextual_risk']['base_risk_score']}"
    )

    print(
        f"Contextual Score  : "
        f"{result['contextual_risk']['contextual_risk_score']}"
    )

    print(
        f"Contextual Bonus  : "
        f"{result['contextual_risk']['contextual_bonus']}"
    )

    print(
        f"Risk Level        : "
        f"{result['contextual_risk']['risk_level']}"
    )

    print("\nContextual Risk Factors:")

    for factor in result[
        "contextual_risk"
    ]["risk_factors"]:

        print(f"- {factor}")

    # -----------------------------------------------------
    # SECURITY POLICY
    # -----------------------------------------------------

    print("\nSECURITY POLICY")
    print("-" * 70)

    print(
        f"Original Decision : "
        f"{result['security_override']['original_decision']}"
    )

    print(
        f"Original Priority : "
        f"{result['security_override']['original_priority']}"
    )

    print(
        f"Escalation        : "
        f"{result['security_policy']['escalation_required']}"
    )

    print("\nSecurity Actions:")

    for action in result[
        "security_policy"
    ]["actions"]:

        print(f"- {action}")

    # -----------------------------------------------------
    # RISK MONITOR
    # -----------------------------------------------------

    print("\nRISK MONITOR")
    print("-" * 70)

    print(
        f"Caller ID         : "
        f"{result['risk_monitor']['event']['caller_id']}"
    )

    print(
        f"Session ID        : "
        f"{result['risk_monitor']['event']['session_id']}"
    )

    print(
        f"Repeated Risk     : "
        f"{result['risk_monitor']['monitoring']['repeated_risk']['repeated_risk_detected']}"
    )

    print(
        f"High-Risk Count   : "
        f"{result['risk_monitor']['monitoring']['repeated_risk']['high_risk_event_count']}"
    )

    print(
        f"Risk Trend        : "
        f"{result['risk_monitor']['monitoring']['risk_trend']['trend']}"
    )

    print(
        f"Monitoring Status : "
        f"{result['risk_monitor']['monitoring']['monitoring_status']}"
    )

    print(
        f"Security Action   : "
        f"{result['risk_monitor']['monitoring']['security_action']}"
    )

    # -----------------------------------------------------
    # SECURITY OVERRIDE
    # -----------------------------------------------------

    print("\nSECURITY DECISION OVERRIDE")
    print("-" * 70)

    print(
        f"Override Applied  : "
        f"{result['security_override']['override_applied']}"
    )

    print(
        f"Original Decision : "
        f"{result['security_override']['original_decision']}"
    )

    print(
        f"Final Decision    : "
        f"{result['security_override']['final_decision']}"
    )

    print(
        f"Final Priority    : "
        f"{result['security_override']['final_priority']}"
    )

    print(
        f"Override Reason   : "
        f"{result['security_override']['reason']}"
    )

    # -----------------------------------------------------
    # ALERT
    # -----------------------------------------------------

    print("\nALERT + ACTION")
    print("-" * 70)

    print(
        f"Alert Priority    : "
        f"{result['alert_action']['priority']}"
    )

    print(
        f"Alert             : "
        f"{result['alert_action']['alert']}"
    )

    print(
        "Recommended Action:"
    )

    print(
        result['alert_action']['action']
    )

    # -----------------------------------------------------
    # AUDIT EVENT
    # -----------------------------------------------------

    print("\nAUDIT EVENT")
    print("-" * 70)

    print(
        f"Event ID          : "
        f"{result['audit_event']['event_id']}"
    )

    print(
        f"Timestamp         : "
        f"{result['audit_event']['timestamp_utc']}"
    )

    print(
        f"Event Type        : "
        f"{result['audit_event']['event_type']}"
    )

    print(
        f"Audit Risk Level  : "
        f"{result['audit_event']['risk']['risk_level']}"
    )

    print(
        f"Audit Decision    : "
        f"{result['audit_event']['security']['decision']}"
    )

    # -----------------------------------------------------
    # FINAL DECISION
    # -----------------------------------------------------

    print("\nFINAL SECURITY DECISION")
    print("-" * 70)

    print(
        result["final_security_decision"]
    )

    print(
        f"Priority          : "
        f"{result['final_priority']}"
    )

    print("\n")
    print("=" * 70)
    print("       RISK SECURITY ANALYSIS COMPLETED")
    print("=" * 70)