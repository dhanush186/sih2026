"""
Security Audit Logger

Stores structured security events for traceability and investigation.

Important:
- Raw audio is NOT stored.
- Sensitive audio content is NOT stored.
- Events are stored as JSON Lines (.jsonl).
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


AUDIT_LOG_DIR = Path("backend/logs")
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "security_events.jsonl"

AUDIT_SCHEMA_VERSION = "1.0"


def _utc_timestamp():
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _validate_score(name: str, value: float):
    """Validate that a risk score is between 0 and 100."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")

    if value < 0 or value > 100:
        raise ValueError(f"{name} must be between 0 and 100")


def create_audit_event(
    base_risk_score: float,
    contextual_risk_score: float,
    confidence: float,
    risk_level: str,
    security_decision: str,
    priority: str,
    escalation_required: bool,
    risk_factors: list,
    actions: list
):
    """
    Create and persist a structured security audit event.
    """

    _validate_score("base_risk_score", base_risk_score)
    _validate_score("contextual_risk_score", contextual_risk_score)
    _validate_score("confidence", confidence)

    if not isinstance(risk_level, str):
        raise TypeError("risk_level must be a string")

    if not isinstance(security_decision, str):
        raise TypeError("security_decision must be a string")

    if not isinstance(priority, str):
        raise TypeError("priority must be a string")

    if not isinstance(escalation_required, bool):
        raise TypeError(
            "escalation_required must be True or False"
        )

    if not isinstance(risk_factors, list):
        raise TypeError("risk_factors must be a list")

    if not isinstance(actions, list):
        raise TypeError("actions must be a list")

    event = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "timestamp_utc": _utc_timestamp(),

        "event_type": "VOICE_SECURITY_ANALYSIS",

        "risk": {
            "base_score": round(base_risk_score, 2),
            "contextual_score": round(contextual_risk_score, 2),
            "confidence": round(confidence, 2),
            "risk_level": risk_level.upper()
        },

        "security": {
            "decision": security_decision,
            "priority": priority,
            "escalation_required": escalation_required
        },

        "risk_factors": risk_factors,
        "recommended_actions": actions
    }

    # Create log directory automatically
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Append event as one JSON object per line
    with AUDIT_LOG_FILE.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(event, ensure_ascii=False)
            + "\n"
        )

    return event


def read_audit_events():
    """
    Read all stored audit events.

    Returns:
        list: List of audit event dictionaries.
    """

    if not AUDIT_LOG_FILE.exists():
        return []

    events = []

    with AUDIT_LOG_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Ignore corrupted lines instead of crashing
                continue

    return events


def get_latest_audit_event():
    """Return the most recent audit event."""

    events = read_audit_events()

    if not events:
        return None

    return events[-1]


if __name__ == "__main__":

    event = create_audit_event(
        base_risk_score=95.5,
        contextual_risk_score=100,
        confidence=100,
        risk_level="CRITICAL",
        security_decision="BLOCK_AND_ESCALATE",
        priority="IMMEDIATE",
        escalation_required=True,
        risk_factors=[
            "High AI-generated voice probability",
            "Suspicious caller behavior",
            "High transaction risk",
            "Unknown caller",
            "Urgency detected"
        ],
        actions=[
            "Block transaction authorization",
            "Require independent caller verification",
            "Require multi-factor authentication",
            "Escalate to security team"
        ]
    )

    print("\nSECURITY AUDIT LOGGER")
    print("=" * 50)

    print(f"Event ID            : {event['event_id']}")
    print(f"Timestamp (UTC)     : {event['timestamp_utc']}")
    print(f"Risk Level          : {event['risk']['risk_level']}")
    print(f"Base Risk Score     : {event['risk']['base_score']}")
    print(f"Contextual Score    : {event['risk']['contextual_score']}")
    print(f"Confidence          : {event['risk']['confidence']}")
    print(f"Security Decision   : {event['security']['decision']}")
    print(f"Priority            : {event['security']['priority']}")
    print(
        f"Escalation Required : "
        f"{event['security']['escalation_required']}"
    )

    print("\nAudit event saved to:")
    print(AUDIT_LOG_FILE)