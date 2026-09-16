"""
audio_utils.py
---------------
Small shared helper functions used across the whole module.
Keeping these in one place avoids repeating path/plumbing code
in every other file.
"""

import os

# ---------------------------------------------------------------------
# Project root = the "audio_speaker_engineer" folder itself.
# This makes every path in the project independent of where you
# run the scripts from (VS Code, terminal, another folder, etc.)
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Common folders used by the pipeline
RAW_AUDIO_DIR = os.path.join(PROJECT_ROOT, "audio", "raw")
PROCESSED_AUDIO_DIR = os.path.join(PROJECT_ROOT, "audio", "processed")
MFCC_DIR = os.path.join(PROJECT_ROOT, "features", "mfcc")
SPECTROGRAM_DIR = os.path.join(PROJECT_ROOT, "features", "spectrograms")
PROSODY_DIR = os.path.join(PROJECT_ROOT, "features", "prosody")
SPEAKER_VERIFICATION_DIR = os.path.join(PROJECT_ROOT, "speaker_verification")


def ensure_dir(path: str) -> None:
    """Create a folder if it doesn't already exist (no error if it does)."""
    os.makedirs(path, exist_ok=True)


def get_filename_without_extension(file_path: str) -> str:
    """
    Return just the file name without folder path or extension.
    Example: '/a/b/hello.wav' -> 'hello'
    """
    base_name = os.path.basename(file_path)
    name_without_ext, _ext = os.path.splitext(base_name)
    return name_without_ext


def file_exists(file_path: str) -> bool:
    """Simple wrapper to check if a file exists on disk."""
    return os.path.isfile(file_path)


# Make sure the important output folders exist as soon as this
# module is imported anywhere in the project.
for _folder in (PROCESSED_AUDIO_DIR, MFCC_DIR, SPECTROGRAM_DIR, PROSODY_DIR):
    ensure_dir(_folder)
