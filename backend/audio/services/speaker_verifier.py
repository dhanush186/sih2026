"""
speaker_verifier.py
----------------------
Compares two voice recordings and estimates whether they belong to
the same speaker. This is used to check a suspicious/test voice
against a trusted reference voice.

Two modes are supported:

1. "speechbrain" mode (recommended, more accurate):
   Uses a pretrained SpeechBrain ECAPA-TDNN speaker embedding model.
   Requires installing the extra 'speechbrain' package (see README).

2. "mfcc" mode (fallback, zero extra install):
   Uses the average MFCC vector of each voice as a lightweight
   "embedding" and compares them with cosine similarity. Not as
   accurate as a trained model, but works immediately out of the
   box with no downloads, so the pipeline is always runnable.

The module automatically falls back to "mfcc" mode if SpeechBrain
(or its model download) is not available, so main.py never crashes
just because a teammate hasn't installed the optional dependency.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .mfcc_extractor import extract_mfcc

# Similarity score above this threshold => "same speaker"
DEFAULT_SIMILARITY_THRESHOLD = 0.80

# Lazily-created SpeechBrain model (only loaded if/when needed)
_speechbrain_model = None


def _get_speechbrain_model():
    """
    Try to load the pretrained SpeechBrain speaker embedding model.
    Returns the model, or None if speechbrain isn't installed / the
    pretrained model can't be downloaded (e.g. no internet access).

    To enable this mode, install SpeechBrain first:
        pip install speechbrain

    The first run will download the pretrained model
    ('spkrec-ecapa-voxceleb') into a local 'pretrained_models' folder.
    """
    global _speechbrain_model
    if _speechbrain_model is not None:
        return _speechbrain_model

    try:
        from speechbrain.inference.speaker import EncoderClassifier

        _speechbrain_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
        return _speechbrain_model
    except Exception:
        # speechbrain not installed, or model couldn't be downloaded.
        # We simply fall back to the MFCC method below.
        return None


def _get_mfcc_embedding(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Fallback 'embedding': the time-averaged MFCC vector of the audio.
    Not as powerful as a trained speaker model, but captures enough
    voice-timbre information for a basic same/different comparison.
    """
    mfcc = extract_mfcc(waveform, sr, n_mfcc=20)
    return np.mean(mfcc, axis=1)  # shape: (20,)


def _get_speechbrain_embedding(waveform: np.ndarray, sr: int, model) -> np.ndarray:
    """Compute a speaker embedding using the SpeechBrain model."""
    import torch

    # SpeechBrain expects a torch tensor shaped (batch, samples)
    signal = torch.from_numpy(waveform).float().unsqueeze(0)
    embedding = model.encode_batch(signal)
    return embedding.squeeze().detach().numpy()


def get_speaker_embedding(waveform: np.ndarray, sr: int):
    """
    Compute a speaker embedding for one voice recording, automatically
    using SpeechBrain if available, otherwise the MFCC fallback.

    Returns
    -------
    embedding : np.ndarray
    method_used : str  ("speechbrain" or "mfcc")
    """
    model = _get_speechbrain_model()
    if model is not None:
        embedding = _get_speechbrain_embedding(waveform, sr, model)
        return embedding, "speechbrain"

    embedding = _get_mfcc_embedding(waveform, sr)
    return embedding, "mfcc"


def compare_speakers(reference_waveform: np.ndarray, reference_sr: int,
                      test_waveform: np.ndarray, test_sr: int,
                      threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> dict:
    """
    Compare a reference voice against a test voice.

    Returns
    -------
    dict with:
        'similarity_score' : float between -1 and 1 (cosine similarity)
        'same_speaker'     : bool
        'method_used'      : "speechbrain" or "mfcc"
    """
    ref_embedding, ref_method = get_speaker_embedding(reference_waveform, reference_sr)
    test_embedding, test_method = get_speaker_embedding(test_waveform, test_sr)

    # Reshape to 2D for sklearn's cosine_similarity function
    ref_vec = ref_embedding.reshape(1, -1)
    test_vec = test_embedding.reshape(1, -1)

    similarity = float(cosine_similarity(ref_vec, test_vec)[0][0])
    same_speaker = similarity >= threshold

    return {
        "similarity_score": round(similarity, 4),
        "same_speaker": same_speaker,
        "method_used": ref_method,  # both should match, but report ref's
    }


if __name__ == "__main__":
    # Quick smoke test: compare two synthetic tones (one 'same', one 'different').
    sr = 22050
    t = np.linspace(0, 2, int(sr * 2))

    voice_a = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)
    voice_a_again = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)  # identical
    voice_b = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)  # different pitch

    result_same = compare_speakers(voice_a, sr, voice_a_again, sr)
    result_diff = compare_speakers(voice_a, sr, voice_b, sr)

    print("Same-voice test:", result_same)
    print("Different-voice test:", result_diff)
