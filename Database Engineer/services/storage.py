"""
Audio file validation and storage helpers.
"""

import os
from pathlib import Path

from fastapi import UploadFile, HTTPException, status

from database.config import settings
from database.models import generate_upload_filename


def validate_audio_file(file: UploadFile) -> str:
    """Validate extension and return the safe extension (lowercased, with dot)."""
    if not file.filename or "." not in file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File must have a valid extension.")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported audio format '{ext}'. Allowed: {sorted(settings.ALLOWED_AUDIO_EXTENSIONS)}",
        )
    return ext


async def save_audio_file(file: UploadFile) -> tuple[str, str]:
    """
    Stream-validate size, then save the upload to disk under UPLOAD_DIR.
    Returns (stored_filename, stored_path).
    """
    validate_audio_file(file)

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = generate_upload_filename(file.filename)
    stored_path = upload_dir.resolve() / stored_filename  # absolute path

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    size = 0

    with open(stored_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out_file.close()
                os.remove(stored_path)
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"File exceeds max size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
                )
            out_file.write(chunk)

    return stored_filename, str(stored_path)

