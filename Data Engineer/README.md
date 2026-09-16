# ML / Data Engineering Module

## Overview

This module contains the machine-learning pipeline for detecting whether an audio recording is **real human speech** or **AI-generated / deepfake speech**.

The current system uses a pretrained Wav2Vec2 model to extract audio representations and a trained binary classifier to predict the class.

---

## Model Used

### Base Model

**Wav2Vec2**

```text
facebook/wav2vec2-base

Architecture:

Wav2Vec2 embedding (768)
        ↓
Linear(768 → 256)
        ↓
ReLU
        ↓
Dropout
        ↓
Linear(256 → 64)
        ↓
ReLU
        ↓
Dropout
        ↓
Linear(64 → 2)
        ↓
Softmax
        ↓
REAL / FAKE



Trained model stored at :


Data Engineer/models/wav2vec2_deepfake_classifier_full.pth

Input

The prediction pipeline accepts an audio file in:

.wav

Output

The prediction function returns a Python dictionary containing:

{
  "label": "FAKE",
  "confidence": 0.97,
  "real_probability": 0.03,
  "fake_probability": 0.97
}

Prediction Module

The main prediction module is:

Data Engineer/ml/predict.py


load_models()

Loads:

Wav2Vec2 processor
Wav2Vec2 encoder
Trained deepfake classifier
predict_audio(...)

Accepts an audio file and performs inference.

The function:

Loads and preprocesses the audio.
Passes the waveform through Wav2Vec2.
Extracts the audio embedding.
Passes the embedding to the trained classifier.
Applies softmax to obtain probabilities.
Selects the predicted class.
Returns the prediction result.


Prediction Flow:
Input WAV file
       ↓
Audio preprocessing
       ↓
16 kHz mono waveform
       ↓
2-second audio window
       ↓
Wav2Vec2 encoder
       ↓
768-dimensional audio embedding
       ↓
Trained classifier
       ↓
Softmax probabilities
       ↓
REAL / FAKE prediction
       ↓
Confidence + probabilities

Pasted text(9).txt
Document

Absolutely. Since your Git branch is already clean and you only want to add documentation, we can skip the Python setup for now.

Let's create/update:

D:\sih2026\Data Engineer\README.md
Step 1 — Open the README

In VS Code, open:

Data Engineer
└── README.md

Replace its contents with this:

# ML / Data Engineering Module

## Overview

This module contains the machine-learning pipeline for detecting whether an audio recording is real or AI-generated.

The current deepfake detection model uses:

- **Wav2Vec2**: `facebook/wav2vec2-base`
- A trained binary classifier on top of Wav2Vec2 embeddings
- Classes:
  - `0` → REAL
  - `1` → FAKE

The trained classifier is stored in:

```text
Data Engineer/models/wav2vec2_deepfake_classifier_full.pth
Model Architecture

The system first processes the input audio using Wav2Vec2 to extract an audio representation.

The resulting 768-dimensional embedding is passed through a small neural-network classifier:

Audio
  ↓
Wav2Vec2 Base
  ↓
768-dimensional embedding
  ↓
Linear(768 → 256)
  ↓
ReLU + Dropout
  ↓
Linear(256 → 64)
  ↓
ReLU + Dropout
  ↓
Linear(64 → 2)
  ↓
Softmax
  ↓
REAL / FAKE probabilities
Input

The prediction pipeline accepts a .wav audio file.

The current preprocessing:

Load the audio.
Convert it to mono.
Resample to 16 kHz.
Use a 2-second audio window.
Pad shorter audio or truncate longer audio as required.
Pass the processed waveform through Wav2Vec2.
Prediction

The main prediction module is:

Data Engineer/ml/predict.py

The main functions are:

load_models()

Loads:

Wav2Vec2 processor
Wav2Vec2 encoder
Trained deepfake classifier
predict_audio(...)

Accepts an audio file and performs inference.

The prediction result is returned as a dictionary:

{
  "label": "FAKE",
  "confidence": 0.97,
  "real_probability": 0.03,
  "fake_probability": 0.97
}
Output fields
Field	Description
label	Predicted class: REAL or FAKE
confidence	Probability of the predicted class
real_probability	Probability that the audio is real
fake_probability	Probability that the audio is AI-generated
Model Training

The classifier was trained using the Fake-or-Real (FoR) audio dataset.

The final training pipeline uses the available balanced training and validation data and saves the best classifier based on validation performance.

The independent test set is kept separate from the training process.

Validation

The final trained model achieved approximately:

Validation accuracy: 99.66%
Validation F1: 99.66%

On the independent test set at a classification threshold of 0.50:

Accuracy: 85.84%
Fake precision: 96.53%
Fake recall: 75.02%
Fake F1: 84.43%

These results are evaluation measurements on this dataset and should not be interpreted as guaranteed real-world performance.

Test Results

Three audio files should be tested before integration.

Filename	Prediction	Confidence
real_audio.wav	REAL	TBD
ai_generated_audio.wav	FAKE	TBD
another_test_audio.wav	TBD	TBD

Replace the TBD values with the actual results from the three validation tests.

How Prediction Works
Input WAV
   ↓
Audio preprocessing
   ↓
16 kHz waveform
   ↓
2-second window
   ↓
Wav2Vec2
   ↓
Audio embedding
   ↓
Trained classifier
   ↓
Softmax probabilities
   ↓
Prediction + confidence
Limitations

This model has several limitations:

A prediction is a risk signal, not definitive proof that audio is synthetic.
The current pipeline analyzes a 2-second audio window.
False positives and false negatives can occur.
Performance can vary with speakers, accents, languages, microphones, compression, background noise, and unseen AI voice-generation methods.
The reported metrics are based on the available FoR dataset and may not directly represent performance on real-world audio.
Additional aggregation and verification mechanisms may be required when analyzing longer real-time conversations.
Files

Important ML files:

Data Engineer/
├── ml/
│   ├── predict.py
│   ├── wav2vec_model.py
│   ├── train_wav2vec_full.py
│   ├── evaluate_wav2vec.py
│   ├── tune_threshold.py
│   └── ...
│
├── models/
│   └── wav2vec2_deepfake_classifier_full.pth
│
└── README.md
Integration

The ML prediction module is intended to be used by the backend/API layer.

The backend can call the prediction function and use the returned probabilities as an input to the project's broader risk-analysis pipeline.

The ML classifier itself should not be treated as an automatic decision-maker for irreversible actions.


### Step 2 — Save

Press:

```text
Ctrl + S

Then check what changed:

cd D:\sih2026
git status

You should see something like:

modified: Data Engineer/README.md

Paste the git status output here. Then we'll commit and push just this documentation to your Akash branch.

C. Documentation  Ask him to create/update:  Data Engineer/README.md  Explain:  model used input output how prediction works limitations Deliverable to you ML module + 3 test results + README + Git commit + push to Akash branch  

Yes. The documentation deliverable is exactly:

Data Engineer/README.md

It needs to cover the five things your lead requested:

Model used
Input
Output
How prediction works
Limitations

And your final deliverable should be:

ML module
+ 3 test results
+ Data Engineer/README.md
+ Git commit
+ push to Akash

Since you're on the new PC and already on Akash, let's keep this simple.

Do this now

Open:

D:\sih2026\Data Engineer\README.md

Put the following content in it:

# ML / Data Engineering Module

## Model Used

The project uses **Wav2Vec2 (`facebook/wav2vec2-base`)** for audio representation and a trained binary classifier for deepfake voice detection.

The classifier predicts two classes:

- `REAL`
- `FAKE`

The trained model is stored at:

```text
Data Engineer/models/wav2vec2_deepfake_classifier_full.pth
Input

The prediction pipeline accepts a .wav audio file.

The audio is:

Loaded and converted to mono.
Resampled to 16 kHz.
Processed as a 2-second audio window.
Padded or truncated when necessary.
Output

The prediction function returns a dictionary containing:

{
  "label": "FAKE",
  "confidence": 0.97,
  "real_probability": 0.03,
  "fake_probability": 0.97
}
Output fields
Field	Description
label	Predicted class: REAL or FAKE
confidence	Confidence of the predicted class
real_probability	Probability that the audio is real
fake_probability	Probability that the audio is AI-generated
How Prediction Works
Audio file
    ↓
Audio preprocessing
    ↓
16 kHz waveform
    ↓
2-second window
    ↓
Wav2Vec2
    ↓
Audio embedding
    ↓
Trained classifier
    ↓
Softmax probabilities
    ↓
REAL / FAKE prediction

The main prediction module is:

Data Engineer/ml/predict.py

The main functions are:

load_models()

Loads the Wav2Vec2 processor, encoder, and trained classifier.

predict_audio(...)

Takes an audio file and performs inference, returning the prediction and probabilities.

Limitations
The model can produce both false positives and false negatives.
The current prediction pipeline analyzes a 2-second audio window.
Performance may vary across speakers, accents, languages, recording devices, background noise, compression, and unseen voice-generation techniques.
Dataset performance does not guarantee the same performance on real-world audio.
A model prediction is a risk signal, not definitive proof that an audio recording is AI-generated.
Longer real-time conversations may require chunking, aggregation, and additional verification.
Test Results
Filename	Prediction	Confidence
test_real.wav	TBD	TBD
test_ai.wav	TBD	TBD
test_audio_3.wav	TBD	TBD

Replace the TBD values with the actual results from your three audio tests.


### Then save it

Press:

```text
Ctrl + S

Then run:

cd D:\sih2026
git status

Paste that output here. After that we'll commit the README and push it to Akash.

contents to be filled in the readme.md

Yes. Use this as the complete content for Data Engineer/README.md.

# ML / Data Engineering Module

## Overview

This module contains the machine-learning pipeline for detecting whether an audio recording is **real human speech** or **AI-generated / deepfake speech**.

The current system uses a pretrained Wav2Vec2 model to extract audio representations and a trained binary classifier to predict the class.

---

## Model Used

### Base Model

**Wav2Vec2**

```text
facebook/wav2vec2-base

Wav2Vec2 is used as the audio feature extractor.

Classifier

A custom neural-network classifier is trained on top of the Wav2Vec2 embeddings.

Architecture:

Wav2Vec2 embedding (768)
        ↓
Linear(768 → 256)
        ↓
ReLU
        ↓
Dropout
        ↓
Linear(256 → 64)
        ↓
ReLU
        ↓
Dropout
        ↓
Linear(64 → 2)
        ↓
Softmax
        ↓
REAL / FAKE
Trained Model

The trained classifier is stored at:

Data Engineer/models/wav2vec2_deepfake_classifier_full.pth
Input

The prediction pipeline accepts an audio file in:

.wav
Audio preprocessing

The input audio is:

Loaded using the audio processing pipeline.
Converted to mono.
Resampled to 16 kHz.
Limited to a 2-second audio window.
Shorter audio is padded.
Longer audio is truncated to the required window.
Output

The prediction function returns a Python dictionary containing:

{
  "label": "FAKE",
  "confidence": 0.97,
  "real_probability": 0.03,
  "fake_probability": 0.97
}
Output fields
Field	Description
label	Final predicted class: REAL or FAKE
confidence	Probability of the predicted class
real_probability	Probability that the audio is real
fake_probability	Probability that the audio is AI-generated
Prediction Module

The main prediction module is:

Data Engineer/ml/predict.py
load_models()

Loads:

Wav2Vec2 processor
Wav2Vec2 encoder
Trained deepfake classifier
predict_audio(...)

Accepts an audio file and performs inference.

The function:

Loads and preprocesses the audio.
Passes the waveform through Wav2Vec2.
Extracts the audio embedding.
Passes the embedding to the trained classifier.
Applies softmax to obtain probabilities.
Selects the predicted class.
Returns the prediction result.
Prediction Flow
Input WAV file
       ↓
Audio preprocessing
       ↓
16 kHz mono waveform
       ↓
2-second audio window
       ↓
Wav2Vec2 encoder
       ↓
768-dimensional audio embedding
       ↓
Trained classifier
       ↓
Softmax probabilities
       ↓
REAL / FAKE prediction
       ↓
Confidence + probabilities
Model Training

The model was trained using the Fake-or-Real (FoR) audio dataset.

The training pipeline uses balanced real and fake audio samples.

The final full-data training pipeline uses the available training and validation data and saves the best-performing classifier based on validation performance.

The independent test dataset is kept separate from model training.

Evaluation Results

The final trained model achieved the following validation performance:

Metric	Validation Result
Accuracy	99.66%
F1 Score	99.66%

On the independent test set, using a classification threshold of 0.50:

Metric	Test Result
Accuracy	85.84%
Fake Precision	96.53%
Fake Recall	75.02%
Fake F1 Score	84.43%

Limitations

The current ML model has several limitations:

The model can produce both false positives and false negatives.
The current prediction pipeline analyzes a 2-second audio window.
Performance may vary across different speakers and recording conditions.
Performance may vary across different accents and languages.
Background noise, compression, microphones, and audio quality can affect predictions.
The model may perform differently on AI-generation methods that were not represented in the training dataset.
Dataset evaluation results do not guarantee the same performance on real-world audio.
A prediction is a risk signal, not definitive proof that an audio recording is AI-generated.
Longer conversations may require multiple audio chunks and an aggregation/risk-analysis layer instead of relying on a single prediction.

Project Files

Important ML files include:

Data Engineer/
│
├── ml/
│   ├── predict.py
│   ├── wav2vec_model.py
│   ├── train_wav2vec_full.py
│   ├── evaluate_wav2vec.py
│   ├── tune_threshold.py
│   └── ...
│
├── models/
│   └── wav2vec2_deepfake_classifier_full.pth
│
└── README.md

Integration

The ML prediction module is intended to be integrated with the backend/API layer.

The backend can call the prediction function and use the returned probabilities as inputs to the broader real-time risk analysis pipeline.

The ML model provides a classification signal and should be combined with additional verification or risk-analysis mechanisms when used in real-time applications.

### One important part remains

Before you commit, replace this section:

```text
| `________________` | Real | `________` | `____` |
| `________________` | AI-generated | `________` | `____` |
| `________________` | Real / AI-generated | `________` | `____` |