"""
Basic smoke tests for the API.

Uses an in-memory SQLite DB (via dependency override) so tests don't need a
real MySQL instance. Run with: pytest

Authentication: all protected routes need X-API-Key header. The tests
set a known key via os.environ BEFORE importing the app so that settings
picks it up correctly.
"""

import io
import sys
import os

# Must be set BEFORE importing main/settings so the Settings class picks it up.
TEST_API_KEY = "test-key-for-pytest-only"
os.environ["API_KEY"] = TEST_API_KEY

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from database.connection import Base, get_db

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True, scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


client = TestClient(app)

# Convenience header dict sent with every authenticated request
AUTH = {"X-API-Key": TEST_API_KEY}


# ---------------------------------------------------------------------------
# Public routes — no auth needed
# ---------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Auth guard tests
# ---------------------------------------------------------------------------

def test_missing_api_key_returns_401():
    """Requests without the X-API-Key header must be rejected with 401."""
    resp = client.post("/api/users/", json={"name": "Ghost", "email": "ghost@example.com"})
    assert resp.status_code == 401


def test_wrong_api_key_returns_401():
    """Requests with an incorrect key must also be rejected with 401."""
    resp = client.post(
        "/api/users/",
        json={"name": "Ghost", "email": "ghost2@example.com"},
        headers={"X-API-Key": "totally-wrong-key"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def test_create_and_get_user():
    resp = client.post(
        "/api/users/",
        json={"name": "Alice", "email": "alice@example.com"},
        headers=AUTH,
    )
    assert resp.status_code == 201
    user_id = resp.json()["id"]

    resp = client.get(f"/api/users/{user_id}", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["email"] == "alice@example.com"


def test_duplicate_email_rejected():
    client.post(
        "/api/users/",
        json={"name": "Bob", "email": "bob@example.com"},
        headers=AUTH,
    )
    resp = client.post(
        "/api/users/",
        json={"name": "Bob2", "email": "bob@example.com"},
        headers=AUTH,
    )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Calls + Detection pipeline
# ---------------------------------------------------------------------------

def test_upload_call_and_run_detection_high_risk():
    user_resp = client.post(
        "/api/users/",
        json={"name": "Carol", "email": "carol@example.com"},
        headers=AUTH,
    )
    user_id = user_resp.json()["id"]

    fake_audio = io.BytesIO(b"fake wav bytes")
    resp = client.post(
        "/api/calls/upload",
        data={"user_id": str(user_id), "caller_number": "+911234567890"},
        files={"audio_file": ("ceo_call.wav", fake_audio, "audio/wav")},
        headers=AUTH,
    )
    assert resp.status_code == 201
    call_id = resp.json()["id"]
    assert resp.json()["status"] == "uploaded"

    resp = client.post(
        f"/api/detection/{call_id}",
        json={"deepfake_score": 87, "speaker_score": 91, "prosody_score": 78},
        headers=AUTH,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["verdict"] == "HIGH_RISK"
    assert body["risk_score"] > 70

    alerts_resp = client.get("/api/alerts/", headers=AUTH)
    assert alerts_resp.status_code == 200
    assert any(a["call_id"] == call_id for a in alerts_resp.json())


def test_rejects_bad_file_extension():
    user_resp = client.post(
        "/api/users/",
        json={"name": "Dave", "email": "dave@example.com"},
        headers=AUTH,
    )
    user_id = user_resp.json()["id"]

    bad_file = io.BytesIO(b"not audio")
    resp = client.post(
        "/api/calls/upload",
        data={"user_id": str(user_id)},
        files={"audio_file": ("malware.exe", bad_file, "application/octet-stream")},
        headers=AUTH,
    )
    assert resp.status_code == 400
