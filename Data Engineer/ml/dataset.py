from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"D:\kaggle_cache\datasets\mohammedabdeldayem"
    r"\the-fake-or-real-dataset\versions\2"
    r"\for-original\for-original"
)

SAMPLE_RATE = 16000
DURATION = 2
NUM_SAMPLES = SAMPLE_RATE * DURATION

AUDIO_EXTENSIONS = {".wav", ".mp3"}


# ============================================================
# DATASET CLASS
# ============================================================

class FakeRealDataset(Dataset):

    def __init__(self, split="training"):

        self.files = []
        self.skipped_empty = 0
        self.skipped_unreadable = 0

        split_dir = DATASET_ROOT / split

        real_dir = split_dir / "real"
        fake_dir = split_dir / "fake"

        # ----------------------------------------------------
        # Check directories
        # ----------------------------------------------------

        if not real_dir.exists():
            raise FileNotFoundError(
                f"Real directory not found: {real_dir}"
            )

        if not fake_dir.exists():
            raise FileNotFoundError(
                f"Fake directory not found: {fake_dir}"
            )

        # ----------------------------------------------------
        # Scan REAL files
        # Label: 0
        # ----------------------------------------------------

        self._scan_directory(
            real_dir,
            label=0
        )

        # ----------------------------------------------------
        # Scan FAKE files
        # Label: 1
        # ----------------------------------------------------

        self._scan_directory(
            fake_dir,
            label=1
        )

        # ----------------------------------------------------
        # Dataset statistics
        # ----------------------------------------------------

        real_count = sum(
            1
            for _, label in self.files
            if label == 0
        )

        fake_count = sum(
            1
            for _, label in self.files
            if label == 1
        )

        print()
        print(f"Dataset split: {split}")
        print(f"Real files: {real_count}")
        print(f"Fake files: {fake_count}")
        print(f"Total files: {len(self.files)}")
        print(f"Skipped empty files: {self.skipped_empty}")
        print(
            f"Skipped unreadable files: "
            f"{self.skipped_unreadable}"
        )

    # ========================================================
    # SCAN DIRECTORY
    # ========================================================

    def _scan_directory(self, directory, label):

        for path in directory.iterdir():

            # Ignore directories
            if not path.is_file():
                continue

            # Normalize extension
            extension = path.suffix.lower().strip()

            if extension not in AUDIO_EXTENSIONS:
                continue

            # ------------------------------------------------
            # Skip zero-byte files
            # ------------------------------------------------

            try:
                file_size = path.stat().st_size
            except OSError:
                self.skipped_unreadable += 1
                continue

            if file_size == 0:
                self.skipped_empty += 1
                continue

            # ------------------------------------------------
            # Test whether the audio can actually be decoded
            # ------------------------------------------------

            try:

                waveform, _ = librosa.load(
                    path,
                    sr=SAMPLE_RATE,
                    mono=True,
                    duration=0.25
                )

                if waveform is None:
                    self.skipped_unreadable += 1
                    continue

                if len(waveform) == 0:
                    self.skipped_unreadable += 1
                    continue

            except Exception:
                self.skipped_unreadable += 1
                continue

            # ------------------------------------------------
            # File is valid
            # ------------------------------------------------

            self.files.append(
                (path, label)
            )

    # ========================================================
    # DATASET LENGTH
    # ========================================================

    def __len__(self):
        return len(self.files)

    # ========================================================
    # LOAD ONE AUDIO SAMPLE
    # ========================================================

    def __getitem__(self, index):

        path, label = self.files[index]

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        try:

            waveform, _ = librosa.load(
                path,
                sr=SAMPLE_RATE,
                mono=True
            )

        except Exception as error:

            raise RuntimeError(
                f"Audio file could not be loaded: {path}"
            ) from error

        # ----------------------------------------------------
        # Make audio exactly 2 seconds
        # ----------------------------------------------------

        if len(waveform) < NUM_SAMPLES:

            padding_length = (
                NUM_SAMPLES - len(waveform)
            )

            waveform = np.pad(
                waveform,
                (0, padding_length),
                mode="constant"
            )

        else:

            waveform = waveform[:NUM_SAMPLES]

        # ----------------------------------------------------
        # Convert audio to Mel Spectrogram
        # ----------------------------------------------------

        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
            power=2.0
        )

        # ----------------------------------------------------
        # Convert power spectrogram to decibels
        # ----------------------------------------------------

        mel_spectrogram = librosa.power_to_db(
            mel_spectrogram,
            ref=np.max
        )

        # ----------------------------------------------------
        # Convert to PyTorch tensor
        # ----------------------------------------------------

        mel_tensor = torch.tensor(
            mel_spectrogram,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Normalize Mel spectrogram
        # ----------------------------------------------------

        mean = mel_tensor.mean()
        std = mel_tensor.std()

        mel_tensor = (
            mel_tensor - mean
        ) / (std + 1e-6)

        # ----------------------------------------------------
        # Label
        #
        # 0 = Real
        # 1 = Fake
        # ----------------------------------------------------

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return mel_tensor, label_tensor