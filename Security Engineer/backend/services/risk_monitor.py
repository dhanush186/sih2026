"""
Advanced Risk Monitoring and Escalation Engine

Tracks repeated suspicious interactions and detects
risk patterns over time.

Features:
- Caller/session-aware monitoring
- Rolling time-window analysis
- Repeated high-risk detection
- Risk trend detection
- Automatic monitoring escalation
- Security escalation recommendations

This module does NOT analyze audio directly.
It monitors risk results produced by the Risk Engine.
"""

from datetime import datetime, timezone, timedelta


class RiskMonitor:
    """
    Advanced security risk monitor.

    The monitor keeps an in-memory history of security
    events during the current application session.

    For the hackathon MVP, this provides lightweight
    real-time monitoring without requiring a database.
    """

    def __init__(
        self,
        escalation_threshold: int = 3,
        monitoring_window_minutes: int = 15
    ):
        # -------------------------------------------------
        # Validate configuration
        # -------------------------------------------------

        if not isinstance(escalation_threshold, int):
            raise TypeError(
                "escalation_threshold must be an integer"
            )

        if escalation_threshold < 1:
            raise ValueError(
                "escalation_threshold must be at least 1"
            )

        if not isinstance(
            monitoring_window_minutes,
            int
        ):
            raise TypeError(
                "monitoring_window_minutes must be an integer"
            )

        if monitoring_window_minutes < 1:
            raise ValueError(
                "monitoring_window_minutes must be at least 1"
            )

        self.escalation_threshold = escalation_threshold

        self.monitoring_window_minutes = (
            monitoring_window_minutes
        )

        # Event history for current application session
        self.events = []

    # =====================================================
    # VALIDATION HELPERS
    # =====================================================

    def _validate_score(
        self,
        risk_score: float
    ):
        if not isinstance(
            risk_score,
            (int, float)
        ):
            raise TypeError(
                "risk_score must be a number"
            )

        if risk_score < 0 or risk_score > 100:
            raise ValueError(
                "risk_score must be between 0 and 100"
            )

    def _validate_text(
        self,
        name: str,
        value: str
    ):
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a string"
            )

        if not value.strip():
            raise ValueError(
                f"{name} cannot be empty"
            )

    # =====================================================
    # RECORD SECURITY EVENT
    # =====================================================

    def record_event(
        self,
        risk_score: float,
        risk_level: str,
        security_decision: str,
        caller_id: str = "UNKNOWN_CALLER",
        session_id: str = "DEFAULT_SESSION"
    ):
        """
        Record a security-risk event.

        caller_id and session_id are optional so that
        the existing contextual_risk.py integration
        remains compatible.
        """

        # -------------------------------------------------
        # Validate inputs
        # -------------------------------------------------

        self._validate_score(risk_score)

        self._validate_text(
            "risk_level",
            risk_level
        )

        self._validate_text(
            "security_decision",
            security_decision
        )

        self._validate_text(
            "caller_id",
            caller_id
        )

        self._validate_text(
            "session_id",
            session_id
        )

        valid_levels = {
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        }

        risk_level = risk_level.upper()

        if risk_level not in valid_levels:
            raise ValueError(
                "risk_level must be "
                "LOW, MEDIUM, HIGH or CRITICAL"
            )

        # -------------------------------------------------
        # Create event
        # -------------------------------------------------

        event = {
            "timestamp_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "risk_score":
                round(risk_score, 2),

            "risk_level":
                risk_level,

            "security_decision":
                security_decision,

            "caller_id":
                caller_id,

            "session_id":
                session_id
        }

        self.events.append(event)

        return event

    # =====================================================
    # GET RECENT EVENTS
    # =====================================================

    def get_recent_events(
        self,
        caller_id=None,
        session_id=None
    ):
        """
        Return events inside the configured
        monitoring time window.

        Optional caller/session filtering is supported.
        """

        current_time = datetime.now(
            timezone.utc
        )

        cutoff_time = (
            current_time
            - timedelta(
                minutes=self.monitoring_window_minutes
            )
        )

        recent_events = []

        for event in self.events:

            event_time = datetime.fromisoformat(
                event["timestamp_utc"]
            )

            # ---------------------------------------------
            # Time-window filtering
            # ---------------------------------------------

            if event_time < cutoff_time:
                continue

            # ---------------------------------------------
            # Caller filtering
            # ---------------------------------------------

            if (
                caller_id is not None
                and event["caller_id"] != caller_id
            ):
                continue

            # ---------------------------------------------
            # Session filtering
            # ---------------------------------------------

            if (
                session_id is not None
                and event["session_id"] != session_id
            ):
                continue

            recent_events.append(event)

        return recent_events

    # =====================================================
    # HIGH-RISK EVENTS
    # =====================================================

    def get_recent_high_risk_events(
        self,
        caller_id=None,
        session_id=None
    ):
        """
        Return recent HIGH and CRITICAL events.
        """

        recent_events = self.get_recent_events(
            caller_id=caller_id,
            session_id=session_id
        )

        return [
            event
            for event in recent_events
            if event["risk_level"]
            in {"HIGH", "CRITICAL"}
        ]

    # =====================================================
    # RISK TREND ANALYSIS
    # =====================================================

    def analyze_risk_trend(
        self,
        caller_id=None,
        session_id=None
    ):
        """
        Detect whether risk is increasing,
        decreasing or stable.
        """

        recent_events = self.get_recent_events(
            caller_id=caller_id,
            session_id=session_id
        )

        # Need at least two events
        if len(recent_events) < 2:

            return {
                "trend": "INSUFFICIENT_DATA",
                "risk_change": 0,
                "message":
                    "Not enough events to determine risk trend."
            }

        scores = [
            event["risk_score"]
            for event in recent_events
        ]

        first_score = scores[0]
        latest_score = scores[-1]

        risk_change = round(
            latest_score - first_score,
            2
        )

        # -------------------------------------------------
        # Increasing risk
        # -------------------------------------------------

        if risk_change >= 10:

            return {
                "trend": "INCREASING",
                "risk_change": risk_change,
                "message":
                    "Risk level is increasing across recent interactions."
            }

        # -------------------------------------------------
        # Decreasing risk
        # -------------------------------------------------

        if risk_change <= -10:

            return {
                "trend": "DECREASING",
                "risk_change": risk_change,
                "message":
                    "Risk level is decreasing across recent interactions."
            }

        # -------------------------------------------------
        # Stable risk
        # -------------------------------------------------

        return {
            "trend": "STABLE",
            "risk_change": risk_change,
            "message":
                "Risk level remains relatively stable."
        }

    # =====================================================
    # REPEATED RISK ANALYSIS
    # =====================================================

    def analyze_repeated_risk(
        self,
        caller_id=None,
        session_id=None
    ):
        """
        Detect repeated HIGH/CRITICAL interactions.

        Escalation occurs when the number of recent
        high-risk events reaches the configured threshold.
        """

        high_risk_events = (
            self.get_recent_high_risk_events(
                caller_id=caller_id,
                session_id=session_id
            )
        )

        high_risk_count = len(
            high_risk_events
        )

        # -------------------------------------------------
        # Escalation threshold reached
        # -------------------------------------------------

        if (
            high_risk_count
            >= self.escalation_threshold
        ):

            return {
                "repeated_risk_detected": True,

                "high_risk_event_count":
                    high_risk_count,

                "monitoring_status":
                    "ESCALATED",

                "security_action":
                    "ESCALATE_REPEATED_SUSPICIOUS_ACTIVITY",

                "message":
                    "Repeated high-risk interactions detected. "
                    "Security escalation is required."
            }

        # -------------------------------------------------
        # Continue monitoring
        # -------------------------------------------------

        return {
            "repeated_risk_detected": False,

            "high_risk_event_count":
                high_risk_count,

            "monitoring_status":
                "MONITORING",

            "security_action":
                "CONTINUE_MONITORING",

            "message":
                "No repeated high-risk activity threshold reached."
        }

    # =====================================================
    # ADVANCED MONITORING ANALYSIS
    # =====================================================

    def analyze_monitoring_state(
        self,
        caller_id=None,
        session_id=None
    ):
        """
        Combine repeated-risk detection and
        risk-trend analysis.
        """

        repeated_risk = (
            self.analyze_repeated_risk(
                caller_id=caller_id,
                session_id=session_id
            )
        )

        trend = (
            self.analyze_risk_trend(
                caller_id=caller_id,
                session_id=session_id
            )
        )

        # -------------------------------------------------
        # Determine monitoring state
        # -------------------------------------------------

        if repeated_risk[
            "repeated_risk_detected"
        ]:

            monitoring_status = "ESCALATED"

            security_action = (
                "ESCALATE_REPEATED_SUSPICIOUS_ACTIVITY"
            )

        elif trend["trend"] == "INCREASING":

            monitoring_status = (
                "RISK_INCREASING"
            )

            security_action = (
                "INCREASE_MONITORING"
            )

        else:

            monitoring_status = (
                repeated_risk[
                    "monitoring_status"
                ]
            )

            security_action = (
                repeated_risk[
                    "security_action"
                ]
            )

        return {
            "monitoring_status":
                monitoring_status,

            "security_action":
                security_action,

            "repeated_risk":
                repeated_risk,

            "risk_trend":
                trend
        }

    # =====================================================
    # MONITORING SUMMARY
    # =====================================================

    def get_monitoring_summary(
        self,
        caller_id=None,
        session_id=None
    ):
        """
        Generate a complete monitoring summary.
        """

        recent_events = self.get_recent_events(
            caller_id=caller_id,
            session_id=session_id
        )

        high_risk_events = (
            self.get_recent_high_risk_events(
                caller_id=caller_id,
                session_id=session_id
            )
        )

        critical_count = sum(
            event["risk_level"] == "CRITICAL"
            for event in recent_events
        )

        average_risk = 0

        if recent_events:

            average_risk = round(
                sum(
                    event["risk_score"]
                    for event in recent_events
                )
                / len(recent_events),
                2
            )

        monitoring_state = (
            self.analyze_monitoring_state(
                caller_id=caller_id,
                session_id=session_id
            )
        )

        return {

            "total_events":
                len(recent_events),

            "high_risk_events":
                len(high_risk_events),

            "critical_events":
                critical_count,

            "average_risk_score":
                average_risk,

            "monitoring_window_minutes":
                self.monitoring_window_minutes,

            "monitoring":
                monitoring_state
        }


# =========================================================
# STANDALONE TEST / DEMO
# =========================================================

if __name__ == "__main__":

    monitor = RiskMonitor(
        escalation_threshold=3,
        monitoring_window_minutes=15
    )

    print("\n")
    print("=" * 70)
    print("        ADVANCED RISK MONITORING ENGINE")
    print("=" * 70)

    # =====================================================
    # CALLER A
    # =====================================================

    print("\nCALLER A")
    print("-" * 70)

    monitor.record_event(
        risk_score=62,
        risk_level="HIGH",
        security_decision="STEP_UP_AUTHENTICATION",
        caller_id="CALLER_A",
        session_id="SESSION_001"
    )

    monitor.record_event(
        risk_score=74,
        risk_level="HIGH",
        security_decision="STEP_UP_AUTHENTICATION",
        caller_id="CALLER_A",
        session_id="SESSION_001"
    )

    monitor.record_event(
        risk_score=91,
        risk_level="CRITICAL",
        security_decision="BLOCK_AND_ESCALATE",
        caller_id="CALLER_A",
        session_id="SESSION_001"
    )

    caller_a_result = (
        monitor.analyze_monitoring_state(
            caller_id="CALLER_A",
            session_id="SESSION_001"
        )
    )

    print(
        f"Monitoring Status : "
        f"{caller_a_result['monitoring_status']}"
    )

    print(
        f"Security Action   : "
        f"{caller_a_result['security_action']}"
    )

    print(
        f"Repeated Risk     : "
        f"{caller_a_result['repeated_risk']['repeated_risk_detected']}"
    )

    print(
        f"High-Risk Count   : "
        f"{caller_a_result['repeated_risk']['high_risk_event_count']}"
    )

    print(
        f"Risk Trend        : "
        f"{caller_a_result['risk_trend']['trend']}"
    )

    # =====================================================
    # CALLER B
    # =====================================================

    print("\nCALLER B")
    print("-" * 70)

    monitor.record_event(
        risk_score=15,
        risk_level="LOW",
        security_decision="ALLOW_WITH_NORMAL_VERIFICATION",
        caller_id="CALLER_B",
        session_id="SESSION_002"
    )

    monitor.record_event(
        risk_score=18,
        risk_level="LOW",
        security_decision="ALLOW_WITH_NORMAL_VERIFICATION",
        caller_id="CALLER_B",
        session_id="SESSION_002"
    )

    caller_b_result = (
        monitor.analyze_monitoring_state(
            caller_id="CALLER_B",
            session_id="SESSION_002"
        )
    )

    print(
        f"Monitoring Status : "
        f"{caller_b_result['monitoring_status']}"
    )

    print(
        f"Security Action   : "
        f"{caller_b_result['security_action']}"
    )

    print(
        f"Repeated Risk     : "
        f"{caller_b_result['repeated_risk']['repeated_risk_detected']}"
    )

    print(
        f"Risk Trend        : "
        f"{caller_b_result['risk_trend']['trend']}"
    )

    # =====================================================
    # GLOBAL SUMMARY
    # =====================================================

    print("\nGLOBAL MONITORING SUMMARY")
    print("-" * 70)

    summary = monitor.get_monitoring_summary()

    print(
        f"Total Events      : "
        f"{summary['total_events']}"
    )

    print(
        f"High-Risk Events  : "
        f"{summary['high_risk_events']}"
    )

    print(
        f"Critical Events   : "
        f"{summary['critical_events']}"
    )

    print(
        f"Average Risk      : "
        f"{summary['average_risk_score']}"
    )

    print(
        f"Monitoring Window : "
        f"{summary['monitoring_window_minutes']} minutes"
    )

    print("\n")
    print("=" * 70)
    print("        ADVANCED MONITORING COMPLETED")
    print("=" * 70)