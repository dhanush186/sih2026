"""
API Key authentication dependency.

All protected routes require the header:
    X-API-Key: <value of API_KEY in .env>

Exempt routes (no auth required):
    GET  /        — root info
    GET  /health  — health probe
    GET  /docs    — Swagger UI (FastAPI handles this internally)
    GET  /openapi.json — OpenAPI schema (FastAPI handles this internally)
    GET  /api/database/test — internal DB connectivity check (low-sensitivity)

The API key is loaded from settings so it never appears in source code.
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from database.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """
    FastAPI dependency — inject into any router or route that requires authentication.

    Returns None on success; raises HTTP 401 if the key is missing or incorrect.
    """
    if not settings.API_KEY:
        # If no API_KEY is configured (empty string), auth is effectively disabled.
        # This makes local dev easier while the key is being set up, but should
        # never be the case in staging/production.
        return

    if api_key is None or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Add header: X-API-Key: <your key>",
            headers={"WWW-Authenticate": "ApiKey"},
        )
