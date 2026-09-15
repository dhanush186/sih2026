# Voice Deepfake Detection — Backend (Backend/Database Engineer scope)

FastAPI + MySQL backend for the AI Voice Deepfake Detection & Fraud
Prevention System. This package covers **only** the Backend/Database
Engineer role: REST APIs, database models, file upload handling, auth, and
the risk-scoring persistence layer.

**Not included here** (owned by other team roles): the ML detection model,
the frontend UI. This backend exposes everything those components need to
integrate against.

## Stack
- Python 3.11+
- FastAPI
- SQLAlchemy 2.x
- **MySQL** (PyMySQL driver)
- python-dotenv

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

Edit `.env`:
- Set `DATABASE_URL` to your MySQL connection string.
- Set `API_KEY` (generate one: `python -c "import secrets; print(secrets.token_hex(32))"`).
- Leave `ML_SERVICE_URL` empty until the ML teammate gives you a real endpoint.

Create the database in MySQL:

```sql
CREATE DATABASE voice_security CHARACTER SET utf8mb4;
CREATE USER 'voiceapp'@'localhost' IDENTIFIED BY 'your_password_here';
GRANT ALL PRIVILEGES ON voice_security.* TO 'voiceapp'@'localhost';
FLUSH PRIVILEGES;
```

Then match that user/password in `DATABASE_URL`.

Run the API:

```bash
uvicorn main:app --reload
```

Docs: `http://localhost:8000/docs` (Swagger UI, auto-generated).

Tables are created automatically on startup for the MVP
(`Base.metadata.create_all`). Move to Alembic migrations before this goes
into anything long-lived.

## Run tests

Tests use an in-memory SQLite DB (no MySQL instance required to run them):

```bash
pytest tests/test_api.py -v
```

All 7 tests should pass, covering: health check, auth guard (missing/wrong
key), user creation + duplicate email rejection, full upload → manual
detection → verdict → alert flow, and bad file extension rejection.

## Authentication

Every route except `GET /`, `GET /health`, `GET /api/database/test`, and
`/docs` requires:

```
X-API-Key: <value of API_KEY in .env>
```

Example:
```bash
curl -H "X-API-Key: your_key_here" http://localhost:8000/api/users/
```

## Project layout

```
.
├── main.py                 # FastAPI app, routers, health/db-test endpoints
├── api/
│   ├── auth.py              # X-API-Key dependency
│   ├── users.py             # /api/users
│   ├── calls.py             # /api/calls (incl. audio upload + ML hook)
│   ├── detection.py         # /api/detection (risk engine)
│   └── alerts.py            # /api/alerts
├── database/
│   ├── config.py             # env-driven settings
│   ├── connection.py         # engine/session/Base
│   └── models.py             # User, Call, DetectionResult, Alert
├── schemas/                  # Pydantic request/response models
├── services/
│   ├── risk_engine.py        # score → risk_score → verdict (Risk/Security owns tuning)
│   └── storage.py            # audio upload validation + disk storage
├── tests/
│   ├── test_api.py           # pytest suite
│   └── make_test_wav.py      # generates a dummy .wav for manual testing
├── .env.example
├── .gitignore
└── requirements.txt
```

## API summary

| Method | Path                        | Purpose                                   |
|--------|------------------------------|--------------------------------------------|
| GET    | `/`                          | API info (public)                         |
| GET    | `/health`                    | Health check (public)                     |
| GET    | `/api/database/test`         | Verify MySQL connectivity (public)        |
| POST   | `/api/users/`                | Create user                               |
| GET    | `/api/users/`                | List users                                |
| GET    | `/api/users/{id}`            | Get user                                  |
| DELETE | `/api/users/{id}`            | Delete user                               |
| POST   | `/api/calls/upload`          | Upload audio (multipart: user_id, caller_number, audio_file) |
| GET    | `/api/calls/`                | List calls (optional `?user_id=`)         |
| GET    | `/api/calls/{id}`            | Get call                                  |
| DELETE | `/api/calls/{id}`            | Delete call                               |
| POST   | `/api/detection/{call_id}`   | Submit ML scores → risk engine → stores DetectionResult + Alert |
| GET    | `/api/detection/{call_id}`   | Get detection result                      |
| GET    | `/api/alerts/`               | List alerts (filters: `status_filter`, `severity`) |
| GET    | `/api/alerts/{id}`           | Get alert                                 |
| PATCH  | `/api/alerts/{id}`           | Update alert status (ACTIVE/ACKNOWLEDGED/RESOLVED) |

All routes above except the three marked "(public)" require the `X-API-Key` header.

## Integration contract for the ML teammate

`POST /api/detection/{call_id}` accepts pre-computed scores directly:

```json
{"deepfake_score": 0-100, "speaker_score": 0-100, "prosody_score": 0-100}
```

This is the primary way to exercise the pipeline until a real ML service
exists — submit scores manually and the risk engine + alerting will run
against them.

**Optional automatic mode:** if you (or the ML teammate) stand up a service
and set `ML_SERVICE_URL` in `.env`, every upload will automatically call:

```
POST {ML_SERVICE_URL}/analyze-by-path
form fields: file_path, original_filename
expects JSON back: {"deepfake_score": ..., "speaker_score": ..., "prosody_score": ...}
```

If the ML service's actual contract differs, only `api/calls.py`
(`ANALYZE_PATH` and the request body in `_run_auto_detection`) needs to
change — nothing else in the backend depends on how that service works.

If the ML service is unreachable when this fires, the call is marked
`status: "failed"` and an `ML_SERVICE_UNAVAILABLE` alert is created — this
is visible via `GET /api/calls/{id}` and `GET /api/alerts/`, so a down
service is never silent.

## Risk engine (do not modify without the Risk/Security engineer)

```python
WEIGHTS = {
    "deepfake_score": 0.50,
    "speaker_score":  0.30,
    "prosody_score":  0.20,
}
RISK_THRESHOLD_SUSPICIOUS = 40   # in .env
RISK_THRESHOLD_HIGH       = 70   # in .env
```

These weights and thresholds belong to the Risk/Security engineer's scope —
change them only at their direction.

## Security notes

- Secrets live in `.env` (gitignored), never in code.
- `DEBUG=false` by default — set `true` only for local debugging (it logs full SQL).
- Uploaded files are renamed to a random UUID on disk — the original
  filename is never trusted for the file path.
- File extension and size are validated before anything is written to disk.
- All protected routes require `X-API-Key`.
- This system is a decision-support tool, not proof of caller identity —
  the verdict/recommendation language reflects that.
