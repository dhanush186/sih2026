from pathlib import Path

import librosa
import torch
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


SAMPLE_RATE = 16000
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = Path("models/speaker_verification")


class SpeakerVerifier:
    def __init__(self):
        print("Loading speaker verification model...")

        self.model = SpeakerRecognition.from_hparams(
            source=MODEL_SOURCE,
            savedir=str(MODEL_DIR),
            local_strategy=LocalStrategy.COPY,
        )

        print("Speaker verification model loaded.")

    def load_audio(self, audio_path):
        """
        Load audio as mono 16 kHz waveform.
        """

        path = Path(audio_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        audio, _ = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True,
        )

        if audio is None or len(audio) == 0:
            raise ValueError(
                f"Audio file is empty: {path}"
            )

        waveform = torch.tensor(
            audio,
            dtype=torch.float32,
        ).unsqueeze(0)

        return waveform

    def compare(self, reference_audio, test_audio):
        """
        Compare two audio files.

        Returns:
            {
                "similarity_score": float,
                "same_speaker": bool
            }
        """

        reference_waveform = self.load_audio(reference_audio)
        test_waveform = self.load_audio(test_audio)

        with torch.no_grad():
            score, prediction = self.model.verify_batch(
                reference_waveform,
                test_waveform,
            )

        similarity_score = float(
            score.squeeze().item()
        )

        same_speaker = bool(
            int(prediction.squeeze().item()) == 1
        )

        return {
            "similarity_score": similarity_score,
            "same_speaker": same_speaker,
        }


if __name__ == "__main__":

    reference = Path(
        r"D:\kaggle_cache\datasets\mohammedabdeldayem\the-fake-or-real-dataset\versions\2\for-original\for-original\testing\real\file1.wav"
    )

    test = Path(
        r"D:\kaggle_cache\datasets\mohammedabdeldayem\the-fake-or-real-dataset\versions\2\for-original\for-original\testing\real\file1000.wav"
    )

    verifier = SpeakerVerifier()

    result = verifier.compare(
        reference,
        test,
    )

    print("\nSPEAKER VERIFICATION")
    print("--------------------")
    print(
        f"Similarity score: "
        f"{result['similarity_score']:.4f}"
    )
    print(
        f"Same speaker: "
        f"{'YES' if result['same_speaker'] else 'NO'}"
    )