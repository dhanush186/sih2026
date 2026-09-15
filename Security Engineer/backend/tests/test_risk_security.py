"""
Advanced Risk/Security Test Suite

Tests:
- Risk levels
- Security decisions
- Contextual risk
- Repeated-risk escalation
- Security decision override
- Caller/session isolation
- Input validation
- Audit logging
"""

from backend.services.contextual_risk import (
    analyze_call,
    risk_monitor
)


# =========================================================
# TEST 1: LOW RISK
# =========================================================

def test_low_risk():

    result = analyze_call(
        ai_probability=5,
        speaker_similarity=95,
        prosody_anomaly=5,
        caller_anomaly=5,
        transaction_risk=5,

        caller_known=True,
        urgent_request=False,
        high_value_transaction=False,
        first_time_request=False,

        caller_id="TEST_LOW",
        session_id="LOW_SESSION"
    )

    assert (
        result["contextual_risk"]["risk_level"]
        == "LOW"
    )

    assert (
        result["final_security_decision"]
        == "ALLOW_WITH_NORMAL_VERIFICATION"
    )

    print("LOW risk test: PASSED")


# =========================================================
# TEST 2: MEDIUM RISK
# =========================================================

def test_medium_risk():

    result = analyze_call(
        ai_probability=35,
        speaker_similarity=85,
        prosody_anomaly=20,
        caller_anomaly=25,
        transaction_risk=30,

        caller_known=True,
        urgent_request=False,
        high_value_transaction=False,
        first_time_request=True,

        caller_id="TEST_MEDIUM",
        session_id="MEDIUM_SESSION"
    )

    assert (
        result["contextual_risk"]["risk_level"]
        == "MEDIUM"
    )

    assert (
        result["final_security_decision"]
        == "VERIFY"
    )

    print("MEDIUM risk test: PASSED")


# =========================================================
# TEST 3: HIGH RISK
# =========================================================

def test_high_risk():

    result = analyze_call(
        ai_probability=65,
        speaker_similarity=75,
        prosody_anomaly=50,
        caller_anomaly=60,
        transaction_risk=60,

        caller_known=False,
        urgent_request=False,
        high_value_transaction=False,
        first_time_request=False,

        caller_id="TEST_HIGH",
        session_id="HIGH_SESSION"
    )

    assert (
        result["contextual_risk"]["risk_level"]
        == "HIGH"
    )

    assert (
        result["final_security_decision"]
        == "STEP_UP_AUTHENTICATION"
    )

    print("HIGH risk test: PASSED")


# =========================================================
# TEST 4: CRITICAL RISK
# =========================================================

def test_critical_risk():

    result = analyze_call(
        ai_probability=91,
        speaker_similarity=72,
        prosody_anomaly=60,
        caller_anomaly=80,
        transaction_risk=90,

        caller_known=False,
        urgent_request=True,
        high_value_transaction=True,
        first_time_request=True,

        caller_id="TEST_CRITICAL",
        session_id="CRITICAL_SESSION"
    )

    assert (
        result["contextual_risk"]["risk_level"]
        == "CRITICAL"
    )

    assert (
        result["final_security_decision"]
        == "BLOCK_AND_ESCALATE"
    )

    assert (
        result["security_policy"]["priority"]
        == "IMMEDIATE"
    )

    print("CRITICAL risk test: PASSED")


# =========================================================
# TEST 5: REPEATED HIGH-RISK OVERRIDE
# =========================================================

def test_repeated_risk_override():

    caller_id = "REPEATED_TEST_CALLER"
    session_id = "REPEATED_TEST_SESSION"

    for score in [65, 72, 85]:

        if score >= 80:
            level = "CRITICAL"
            decision = "BLOCK_AND_ESCALATE"

        else:
            level = "HIGH"
            decision = "STEP_UP_AUTHENTICATION"

        risk_monitor.record_event(
            risk_score=score,
            risk_level=level,
            security_decision=decision,
            caller_id=caller_id,
            session_id=session_id
        )

    monitoring = (
        risk_monitor.analyze_monitoring_state(
            caller_id=caller_id,
            session_id=session_id
        )
    )

    assert (
        monitoring["repeated_risk"][
            "repeated_risk_detected"
        ]
        is True
    )

    assert (
        monitoring["monitoring_status"]
        == "ESCALATED"
    )

    assert (
        monitoring["security_action"]
        == "ESCALATE_REPEATED_SUSPICIOUS_ACTIVITY"
    )

    print("Repeated risk detection test: PASSED")


# =========================================================
# TEST 6: SECURITY DECISION OVERRIDE
# =========================================================

def test_security_override():

    # Create three suspicious calls
    for i in range(3):

        result = analyze_call(
            ai_probability=65,
            speaker_similarity=75,
            prosody_anomaly=50,
            caller_anomaly=60,
            transaction_risk=60,

            caller_known=False,
            urgent_request=False,
            high_value_transaction=False,
            first_time_request=False,

            caller_id="OVERRIDE_CALLER",
            session_id="OVERRIDE_SESSION"
        )

    assert (
        result["security_override"][
            "override_applied"
        ]
        is True
    )

    assert (
        result["security_override"][
            "original_decision"
        ]
        == "STEP_UP_AUTHENTICATION"
    )

    assert (
        result["security_override"][
            "final_decision"
        ]
        == "BLOCK_AND_ESCALATE"
    )

    assert (
        result["final_security_decision"]
        == "BLOCK_AND_ESCALATE"
    )

    assert (
        result["final_priority"]
        == "IMMEDIATE"
    )

    print("Security decision override test: PASSED")


# =========================================================
# TEST 7: CALLER / SESSION ISOLATION
# =========================================================

def test_caller_session_isolation():

    caller_a = "ISOLATION_CALLER_A"
    session_a = "ISOLATION_SESSION_A"

    caller_b = "ISOLATION_CALLER_B"
    session_b = "ISOLATION_SESSION_B"

    # Two high-risk events for caller A
    for score in [70, 75]:

        risk_monitor.record_event(
            risk_score=score,
            risk_level="HIGH",
            security_decision="STEP_UP_AUTHENTICATION",
            caller_id=caller_a,
            session_id=session_a
        )

    # One high-risk event for caller B
    risk_monitor.record_event(
        risk_score=70,
        risk_level="HIGH",
        security_decision="STEP_UP_AUTHENTICATION",
        caller_id=caller_b,
        session_id=session_b
    )

    result_a = (
        risk_monitor.analyze_monitoring_state(
            caller_id=caller_a,
            session_id=session_a
        )
    )

    result_b = (
        risk_monitor.analyze_monitoring_state(
            caller_id=caller_b,
            session_id=session_b
        )
    )

    assert (
        result_a["repeated_risk"][
            "repeated_risk_detected"
        ]
        is False
    )

    assert (
        result_b["repeated_risk"][
            "repeated_risk_detected"
        ]
        is False
    )

    assert (
        result_a["repeated_risk"][
            "high_risk_event_count"
        ]
        == 2
    )

    assert (
        result_b["repeated_risk"][
            "high_risk_event_count"
        ]
        == 1
    )

    print("Caller/session isolation test: PASSED")


# =========================================================
# TEST 8: INVALID AI PROBABILITY
# =========================================================

def test_invalid_ai_probability():

    try:

        analyze_call(
            ai_probability=150,
            speaker_similarity=80,
            prosody_anomaly=20,
            caller_anomaly=20,
            transaction_risk=20,

            caller_known=True,
            urgent_request=False,
            high_value_transaction=False,
            first_time_request=False
        )

        assert False

    except ValueError:

        pass

    print("Invalid AI probability test: PASSED")


# =========================================================
# TEST 9: INVALID BOOLEAN INPUT
# =========================================================

def test_invalid_boolean_input():

    try:

        analyze_call(
            ai_probability=20,
            speaker_similarity=90,
            prosody_anomaly=10,
            caller_anomaly=10,
            transaction_risk=10,

            caller_known="YES",
            urgent_request=False,
            high_value_transaction=False,
            first_time_request=False
        )

        assert False

    except TypeError:

        pass

    print("Invalid boolean input test: PASSED")


# =========================================================
# TEST 10: AUDIT EVENT
# =========================================================

def test_audit_event():

    result = analyze_call(
        ai_probability=85,
        speaker_similarity=70,
        prosody_anomaly=65,
        caller_anomaly=70,
        transaction_risk=80,

        caller_known=False,
        urgent_request=True,
        high_value_transaction=True,
        first_time_request=True,

        caller_id="AUDIT_TEST_CALLER",
        session_id="AUDIT_TEST_SESSION"
    )

    audit_event = result["audit_event"]

    assert "event_id" in audit_event
    assert "timestamp_utc" in audit_event
    assert "risk" in audit_event
    assert "security" in audit_event

    assert (
        audit_event["risk"]["risk_level"]
        == result["contextual_risk"]["risk_level"]
    )

    assert (
        audit_event["security"]["decision"]
        == result["final_security_decision"]
    )

    print("Audit event verification: PASSED")


# =========================================================
# TEST 11: FINAL DECISION CONSISTENCY
# =========================================================

def test_final_decision_consistency():

    result = analyze_call(
        ai_probability=90,
        speaker_similarity=70,
        prosody_anomaly=70,
        caller_anomaly=75,
        transaction_risk=85,

        caller_known=False,
        urgent_request=True,
        high_value_transaction=True,
        first_time_request=True,

        caller_id="CONSISTENCY_CALLER",
        session_id="CONSISTENCY_SESSION"
    )

    assert (
        result["final_security_decision"]
        == result["security_override"][
            "final_decision"
        ]
    )

    assert (
        result["final_priority"]
        == result["security_override"][
            "final_priority"
        ]
    )

    assert (
        result["audit_event"]["security"]["decision"]
        == result["final_security_decision"]
    )

    print("Final decision consistency test: PASSED")


# =========================================================
# RUN ALL TESTS
# =========================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("        ADVANCED RISK/SECURITY TEST SUITE")
    print("=" * 70)

    test_low_risk()

    test_medium_risk()

    test_high_risk()

    test_critical_risk()

    test_repeated_risk_override()

    test_security_override()

    test_caller_session_isolation()

    test_invalid_ai_probability()

    test_invalid_boolean_input()

    test_audit_event()

    test_final_decision_consistency()

    print("\n")
    print("=" * 70)
    print("        ALL ADVANCED SECURITY TESTS PASSED")
    print("=" * 70)