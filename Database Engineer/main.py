from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from database.config import settings
from database.connection import Base, engine, SessionLocal
from api import users, calls, detection, alerts
from api.auth import verify_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MVP: create tables directly. Swap for Alembic migrations once the schema stabilizes.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="AI Voice Deepfake Detection API",
    description="Backend for the AI Voice Deepfake Detection & Fraud Prevention System.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All API routers require the X-API-Key header.
# The exempted public routes (GET /, GET /health) are defined below at app level.
_auth = [Depends(verify_api_key)]

app.include_router(users.router, dependencies=_auth)
app.include_router(calls.router, dependencies=_auth)
app.include_router(detection.router, dependencies=_auth)
app.include_router(alerts.router, dependencies=_auth)


@app.get("/")
def root():
    """Public — no auth required."""
    return {
        "service": settings.APP_NAME,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Public — no auth required."""
    return {"status": "healthy", "service": settings.APP_NAME}


@app.get("/api/database/test")
def database_test():
    """Verifies the FastAPI <-> database connection. No auth required (low-sensitivity probe)."""
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return {"database": "connected"}
    except Exception as exc:  # noqa: BLE001
        return {"database": "error", "detail": str(exc)}
    finally:
        db.close()
