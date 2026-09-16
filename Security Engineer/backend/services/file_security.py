"""
File Security Validation Module

Member 6 - Risk/Security Engineer

Validates uploaded files before they enter the security/AI pipeline.

Checks:
1. Invalid file types
2. Suspicious uploads
3. Missing files
4. Oversized files
5. Unsafe filenames
"""

from pathlib import Path


# Allowed audio file extensions
ALLOWED_FILE_TYPES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
}

# Maximum upload size: 25 MB
MAX_FILE_SIZE = 25 * 1024 * 1024

# Potentially dangerous filename characters/patterns
UNSAFE_FILENAME_PATTERNS = {
    "..",
    "/",
    "\\",
    ":",
    "*",
    "?",
    '"',
    "<",
    ">",
    "|",
}


def check_file_type(file_path):
    """Check whether the uploaded file has an allowed extension."""

    path = Path(file_path)

    if path.suffix.lower() not in ALLOWED_FILE_TYPES:
        return False, "Invalid file type"

    return True, "File type accepted"


def check_missing_file(file_path):
    """Check whether the uploaded file exists and is a regular file."""

    path = Path(file_path)

    if not path.exists():
        return False, "File is missing"

    if not path.is_file():
        return False, "Uploaded path is not a file"

    return True, "File exists"


def check_file_size(file_path):
    """Check whether the uploaded file is within the maximum size limit."""

    path = Path(file_path)

    if not path.exists() or not path.is_file():
        return False, "File is missing"

    file_size = path.stat().st_size

    if file_size > MAX_FILE_SIZE:
        return False, "File exceeds maximum allowed size"

    return True, "File size accepted"


def check_filename(file_path):
    """Check whether the filename contains unsafe characters or traversal patterns."""

    path = Path(file_path)
    filename = path.name

    for pattern in UNSAFE_FILENAME_PATTERNS:
        if pattern in filename:
            return False, "Unsafe filename"

    return True, "Filename accepted"


def check_suspicious_upload(file_path):
    """
    Perform basic suspicious-upload checks.

    Suspicious conditions:
    - File has no extension
    - File is empty
    - File type is not allowed
    """

    path = Path(file_path)

    if not path.exists() or not path.is_file():
        return False, "Suspicious upload: file is missing"

    if not path.suffix:
        return False, "Suspicious upload: missing file extension"

    if path.stat().st_size == 0:
        return False, "Suspicious upload: empty file"

    valid_type, _ = check_file_type(file_path)

    if not valid_type:
        return False, "Suspicious upload: unsupported file type"

    return True, "Upload appears safe"


def validate_upload(file_path):
    """
    Run all five file-security checks.

    Returns a structured security result.
    """

    checks = {}

    checks["file_type"] = check_file_type(file_path)
    checks["missing_file"] = check_missing_file(file_path)
    checks["file_size"] = check_file_size(file_path)
    checks["filename"] = check_filename(file_path)
    checks["suspicious_upload"] = check_suspicious_upload(file_path)

    failed_checks = [
        name for name, result in checks.items()
        if result[0] is False
    ]

    if failed_checks:
        return {
            "safe": False,
            "failed_checks": failed_checks,
            "checks": checks,
            "recommendation": "REJECT_UPLOAD",
        }

    return {
        "safe": True,
        "failed_checks": [],
        "checks": checks,
        "recommendation": "ALLOW_UPLOAD",
    }