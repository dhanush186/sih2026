# Audio/Speaker Engineer Module

Part of: **AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**

This module handles everything on the **audio side** of the pipeline:
loading audio, cleaning it up, extracting MFCC features, generating
Mel spectrograms, analyzing prosody (pitch/energy/rhythm), and
verifying whether two voice samples belong to the same speaker.

The output of this module is designed to plug directly into a
teammate's **CNN/deepfake detection model** and **FastAPI backend**.

---

## 1. Folder Structure

```
audio_speaker_engineer/
│
├── audio/
│   ├── raw/                 # Put input audio files here (.wav, .mp3)
│   └── processed/           # (Reserved for saved cleaned audio, if needed)
│
├── features/
│   ├── mfcc/                 # Saved MFCC .npy feature files
│   ├── spectrograms/         # Saved spectrogram .png images
│   └── prosody/              # (Reserved for saved prosody feature files)
│
├── speaker_verification/     # (Reserved for saved speaker embeddings)
│
├── services/
│   ├── audio_loader.py           # Load audio files (WAV/MP3) into waveforms
│   ├── preprocessing.py          # Resample, mono, normalize, trim silence, chunk
│   ├── mfcc_extractor.py         # MFCC + delta + delta-delta extraction
│   ├── spectrogram_generator.py  # Mel spectrogram generation + image export
│   ├── prosody_analyzer.py       # Pitch, energy, speaking rate, pauses
│   └── speaker_verifier.py       # Compare two voices, return similarity score
│
├── utils/
│   └── audio_utils.py        # Shared paths and small helper functions
│
├── tests/
│   └── test_pipeline.py      # Quick sanity tests (no sample files needed)
│
├── requirements.txt
├── README.md
└── main.py                   # Runs the full pipeline end-to-end
```

---

## 2. Installation

From inside the `audio_speaker_engineer/` folder:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Optional (higher-accuracy speaker verification):**
> Uncomment `speechbrain` and `torch` in `requirements.txt`, then:
> ```bash
> pip install speechbrain torch
> ```
> Without this, `speaker_verifier.py` automatically uses a lightweight
> MFCC-based comparison instead — the project still runs perfectly
> fine, just with slightly lower verification accuracy.

---

## 3. How to Run

1. Put an audio file (`.wav` recommended, `.mp3` also works) into
   `audio/raw/`. For example: `audio/raw/sample.wav`.

2. Run the full pipeline:

```bash
python main.py --audio audio/raw/sample.wav
```

3. To also run **speaker verification** against a trusted reference voice:

```bash
python main.py --audio audio/raw/sample.wav --reference audio/raw/reference.wav
```

### Example output

```
===== AUDIO/SPEAKER ANALYSIS SUMMARY =====
File: audio/raw/sample.wav
Duration: 4.32s | Sample rate: 16000Hz
MFCC: extracted  (shape=(39, 136))
Spectrogram: generated -> features/spectrograms/sample_spectrogram.png
Prosody: extracted
   - pitch_mean: 187.432
   - pitch_std: 24.118
   - pitch_min: 120.500
   - pitch_max: 250.900
   - energy_mean: 0.083
   - energy_std: 0.021
   - speaking_rate: 3.120
   - pause_ratio: 0.180
Speaker similarity: 0.87
Speaker verification: SAME SPEAKER  (method: mfcc)
============================================
```

A JSON version of the same results is also printed — this is the
exact shape of data your teammates' **FastAPI** endpoint would return.

---

## 4. Testing without a real recording

You don't need a sample file to check that everything works — the
test suite generates synthetic audio on the fly:

```bash
python tests/test_pipeline.py
```

or with pytest:

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 5. What Each Feature Means

| Feature | What it captures | Why it matters for deepfake detection |
|---|---|---|
| **MFCC** | Shape of the vocal-tract spectrum over time | Cloned voices often have subtly different spectral "texture" |
| **Delta / Delta-Delta** | How MFCCs change frame-to-frame | Synthetic speech can have unnaturally smooth or jumpy transitions |
| **Mel Spectrogram** | Full time-frequency "image" of the audio | Direct input for CNN-based deepfake classifiers |
| **Pitch (F0)** | Fundamental frequency of the voice | Cloned voices may have flatter or less natural pitch variation |
| **Energy** | Loudness over time | Unnatural loudness patterns can signal synthesis artifacts |
| **Speaking rate** | Approximate pace of speech | Cloned audio sometimes has an unnaturally constant pace |
| **Pause ratio** | Fraction of audio that is silence | Real speech has natural, irregular pause patterns |
| **Speaker similarity** | Cosine similarity between two voice embeddings | Detects impersonation by comparing against a trusted reference voice |

---

## 6. How Teammates Connect to This Module

### A. Connecting to the CNN / Deepfake Detection Model

The teammate building the CNN model needs **spectrograms** (and
optionally MFCCs) as model input. They can import directly:

```python
from services.spectrogram_generator import generate_and_save
from services.mfcc_extractor import extract_mfcc_with_deltas

result = generate_and_save(waveform, sr, "some_file.wav")
model_input = result["mel_spec_db"]   # numpy array -> feed into CNN
```

Saved `.npy` (MFCC) and `.png` (spectrogram) files in
`features/mfcc/` and `features/spectrograms/` can also be loaded
directly with `numpy.load(...)` if they prefer working from disk
instead of in-memory arrays.

### B. Connecting to the FastAPI Backend

The teammate building the API can call `run_pipeline(...)` from
`main.py` inside a FastAPI route and return the result as JSON:

```python
from fastapi import FastAPI, UploadFile
from main import run_pipeline
import shutil

app = FastAPI()

@app.post("/analyze")
async def analyze_audio(file: UploadFile):
    save_path = f"audio/raw/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = run_pipeline(audio_path=save_path)
    return results   # FastAPI automatically converts dict -> JSON
```

That's the entire integration — `run_pipeline()` already returns a
single dictionary containing every feature, the spectrogram image
path, and (optionally) the speaker verification verdict, ready to be
serialized as an API response or passed straight into the CNN model.

---

## 7. Notes

- All output folders (`features/mfcc`, `features/spectrograms`, etc.)
  are created automatically the first time you run the pipeline —
  you don't need to create them manually.
- Default sample rate used internally is **16kHz** (standard for
  speech models). Change `target_sr` in `preprocess_pipeline()` if
  your team's CNN model expects something different.
- `speaker_verifier.py` prints no warnings if SpeechBrain isn't
  installed — it just silently uses the MFCC fallback method, so the
  pipeline never breaks for teammates who haven't installed the
  optional dependency.
