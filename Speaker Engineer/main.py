"""
main.py
--------
Entry point for the Audio/Speaker Engineer module.

Pipeline:
    Audio -> Load -> Preprocess -> MFCC -> Mel Spectrogram
          -> Prosody Analysis -> Speaker Verification (optional)
          -> Combined results dictionary

Run from the project's root folder (audio_speaker_engineer/):

    python main.py --audio audio/raw/sample.wav
    python main.py --audio audio/raw/sample.wav --reference audio/raw/reference.wav

The final `results` dictionary is exactly what a teammate's FastAPI
endpoint would return as JSON, or what the CNN/deepfake model would
consume as input (see README "Connecting to teammates' code").
"""

import argparse
import json

from services.audio_loader import load_audio
from services.preprocessing import preprocess_pipeline, segment_audio
from services.mfcc_extractor import extract_mfcc_with_deltas, save_mfcc_features
from services.spectrogram_generator import generate_and_save
from services.prosody_analyzer import analyze_prosody
from services.speaker_verifier import compare_speakers


def run_pipeline(audio_path: str, reference_path: str = None, save_features: bool = True) -> dict:
    """
    Run the full audio -> features pipeline on a single audio file,
    optionally comparing it against a reference voice.

    Returns a dictionary with every extracted result. This is the
    single object that should be handed off to teammates
    (CNN model input / FastAPI JSON response).
    """
    results = {"audio_path": audio_path}

    # 1. Load
    waveform, sr = load_audio(audio_path)
    results["sample_rate"] = sr
    results["duration_seconds"] = round(len(waveform) / sr, 2)

    # 2. Preprocess (mono, resample, normalize, trim silence)
    clean_waveform, clean_sr = preprocess_pipeline(waveform, sr)

    # 3. MFCC + deltas
    mfcc_result = extract_mfcc_with_deltas(clean_waveform, clean_sr)
    results["mfcc_shape"] = mfcc_result["combined"].shape
    if save_features:
        mfcc_path = save_mfcc_features(mfcc_result["combined"], audio_path)
        results["mfcc_saved_to"] = mfcc_path

    # 4. Mel spectrogram
    spectrogram_result = generate_and_save(clean_waveform, clean_sr, audio_path)
    results["spectrogram_shape"] = spectrogram_result["mel_spec_db"].shape
    results["spectrogram_image"] = spectrogram_result["image_path"]

    # 5. Prosody analysis
    prosody_features = analyze_prosody(clean_waveform, clean_sr)
    results["prosody"] = prosody_features

    # 6. Speaker verification (only if a reference voice was provided)
    if reference_path:
        ref_waveform, ref_sr = load_audio(reference_path)
        ref_clean, ref_clean_sr = preprocess_pipeline(ref_waveform, ref_sr)

        verification = compare_speakers(ref_clean, ref_clean_sr, clean_waveform, clean_sr)
        results["speaker_verification"] = verification
    else:
        results["speaker_verification"] = None

    return results


def print_summary(results: dict) -> None:
    """Print a short, human-friendly summary of the pipeline results."""
    print("\n===== AUDIO/SPEAKER ANALYSIS SUMMARY =====")
    print(f"File: {results['audio_path']}")
    print(f"Duration: {results['duration_seconds']}s | Sample rate: {results['sample_rate']}Hz")
    print(f"MFCC: extracted  (shape={results['mfcc_shape']})")
    print(f"Spectrogram: generated -> {results['spectrogram_image']}")
    print("Prosody: extracted")
    for key, value in results["prosody"].items():
        print(f"   - {key}: {value:.3f}")

    verification = results.get("speaker_verification")
    if verification:
        verdict = "SAME SPEAKER" if verification["same_speaker"] else "DIFFERENT SPEAKER"
        print(f"Speaker similarity: {verification['similarity_score']}")
        print(f"Speaker verification: {verdict}  (method: {verification['method_used']})")
    else:
        print("Speaker verification: skipped (no --reference audio provided)")
    print("============================================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio/Speaker Engineer pipeline: MFCC, spectrogram, prosody, speaker verification."
    )
    parser.add_argument("--audio", required=True, help="Path to the audio file to analyze.")
    parser.add_argument("--reference", required=False, default=None,
                         help="Optional path to a reference/trusted voice sample for speaker verification.")
    parser.add_argument("--no-save", action="store_true",
                         help="Don't save MFCC .npy features to disk (spectrogram image is always saved).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipeline_results = run_pipeline(
        audio_path=args.audio,
        reference_path=args.reference,
        save_features=not args.no_save,
    )

    print_summary(pipeline_results)

    # Also dump the raw results as JSON — this is what would be sent
    # to teammates' FastAPI backend as the response body. (Shapes are
    # tuples, so we convert them to strings to keep it JSON-safe.)
    json_safe_results = dict(pipeline_results)
    json_safe_results["mfcc_shape"] = str(pipeline_results["mfcc_shape"])
    json_safe_results["spectrogram_shape"] = str(pipeline_results["spectrogram_shape"])
    print("Full result as JSON (for FastAPI / teammates):")
    print(json.dumps(json_safe_results, indent=2))
