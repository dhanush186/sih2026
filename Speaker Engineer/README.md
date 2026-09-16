# Audio & Speaker Analysis Module

## 🎙️ AI-Powered Real-Time Voice Cloning Impersonation Detection

This module is the **Audio/Speaker Engineer component** of the hackathon project.

It processes an input audio file and extracts important audio and speaker-related information that can be used by the main system to detect possible voice-cloning or impersonation attacks.

---

## 👩‍💻 Module Responsibilities

The Speaker Engineer module handles:

- 🎵 Audio loading and preprocessing
- 🎧 Audio resampling and normalization
- 📊 MFCC feature extraction
- 🌈 Mel spectrogram generation
- 🗣️ Prosody analysis
- 👤 Speaker verification
- 🔗 Providing results for FastAPI/backend integration
- 🧪 Automated pipeline testing

---

## 🔄 Processing Pipeline

```text
Input Audio
     ↓
Audio Loading
     ↓
Preprocessing
     ↓
 ┌───────────────┬────────────────┬─────────────────┐
 ↓               ↓                ↓
MFCC         Spectrogram       Prosody
 ↓               ↓                ↓
 └───────────────┴────────────────┘
                 ↓
        Speaker Verification
                 ↓
          Analysis Results
                 ↓
       FastAPI / Main System

Speaker Engineer/
│
├── audio/
│   ├── raw/
│   │   └── test.wav
│   └── processed/
│
├── features/
│   ├── mfcc/
│   │   └── test_mfcc.npy
│   ├── spectrograms/
│   │   └── test_spectrogram.png
│   └── prosody/
│
├── services/
│   ├── __init__.py
│   ├── audio_loader.py
│   ├── mfcc_extractor.py
│   ├── preprocessing.py
│   ├── prosody_analyzer.py
│   ├── speaker_verifier.py
│   └── spectrogram_generator.py
│
├── speaker_verification/
│   └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
│
├── utils/
│   ├── __init__.py
│   └── audio_utils.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore