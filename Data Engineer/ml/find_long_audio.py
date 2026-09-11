from pathlib import Path
import librosa


DATASET_ROOT = Path(
    r"D:\kaggle_cache\datasets\mohammedabdeldayem"
    r"\the-fake-or-real-dataset\versions\2"
    r"\for-original\for-original"
)

MIN_DURATION = 6.0


for split in ["testing"]:
    for label in ["real", "fake"]:

        folder = DATASET_ROOT / split / label

        for path in folder.iterdir():

            if not path.is_file():
                continue

            if path.suffix.lower().strip() not in {".wav", ".mp3"}:
                continue

            if path.stat().st_size == 0:
                continue

            try:
                duration = librosa.get_duration(
                    path=path
                )

                if duration >= MIN_DURATION:

                    print(
                        f"{label.upper()} | "
                        f"{path.name} | "
                        f"{duration:.2f} seconds"
                    )

                    raise SystemExit

            except Exception:
                continue

print("No suitable audio file found.")