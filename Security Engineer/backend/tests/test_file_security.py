from pathlib import Path

from backend.services.file_security import (
    MAX_FILE_SIZE,
    check_file_type,
    check_missing_file,
    check_file_size,
    check_filename,
    check_suspicious_upload,
    validate_upload,
)


def test_valid_file(tmp_path):
    file_path = tmp_path / "sample.wav"
    file_path.write_bytes(b"audio-data")

    result = validate_upload(file_path)

    assert result["safe"] is True
    assert result["recommendation"] == "ALLOW_UPLOAD"
    assert result["failed_checks"] == []


def test_invalid_file_type(tmp_path):
    file_path = tmp_path / "sample.exe"
    file_path.write_bytes(b"malicious-content")

    valid, message = check_file_type(file_path)

    assert valid is False
    assert message == "Invalid file type"


def test_missing_file(tmp_path):
    file_path = tmp_path / "missing.wav"

    valid, message = check_missing_file(file_path)

    assert valid is False
    assert message == "File is missing"


def test_oversized_file(tmp_path):
    file_path = tmp_path / "large.wav"

    with open(file_path, "wb") as file:
        file.seek(MAX_FILE_SIZE + 1)
        file.write(b"\0")

    valid, message = check_file_size(file_path)

    assert valid is False
    assert message == "File exceeds maximum allowed size"


def test_unsafe_filename(tmp_path):
    unsafe_path = tmp_path / "unsafe..wav"
    unsafe_path.write_bytes(b"audio-data")

    valid, message = check_filename(unsafe_path)

    assert valid is False
    assert message == "Unsafe filename"

def test_suspicious_empty_upload(tmp_path):
    file_path = tmp_path / "empty.wav"
    file_path.write_bytes(b"")

    valid, message = check_suspicious_upload(file_path)

    assert valid is False
    assert "empty file" in message


def test_suspicious_unsupported_upload(tmp_path):
    file_path = tmp_path / "payload.exe"
    file_path.write_bytes(b"payload")

    valid, message = check_suspicious_upload(file_path)

    assert valid is False
    assert "unsupported file type" in message


def test_validate_upload_rejects_invalid_file(tmp_path):
    file_path = tmp_path / "payload.exe"
    file_path.write_bytes(b"payload")

    result = validate_upload(file_path)

    assert result["safe"] is False
    assert result["recommendation"] == "REJECT_UPLOAD"
    assert "file_type" in result["failed_checks"]