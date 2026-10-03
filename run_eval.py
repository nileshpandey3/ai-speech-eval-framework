import json
import argparse
from pathlib import Path
from time import perf_counter

from faster_whisper import WhisperModel

from speech_eval.providers import PROVIDERS, create_transcriber
from speech_eval.dataset import fingerprint_dataset
from speech_eval.normalize import NORMALIZATION_VERSION
from speech_eval.score import score_corpus, score_transcript
from speech_eval.transcribe import transcribe_audio
from speech_eval.config import MODEL_SETTINGS, DECODING_SETTINGS
from faster_whisper.utils import download_model


def main(
    manifest_path="data/manifest.jsonl",
    results_path="outputs/results.json",
):
    with Path(manifest_path).open(encoding="utf-8") as file:
        samples = []

        for line in file:
            if line.strip():
                samples.append(json.loads(line))

    # download_model() supports a commit hash and returns the downloaded model directory.
    # It reuses cached files when available
    model_path = download_model(
        MODEL_SETTINGS["name"],
        revision=MODEL_SETTINGS["revision"],
    )

    model = WhisperModel(
        model_path,
        device=MODEL_SETTINGS["device"],
        compute_type=MODEL_SETTINGS["compute_type"],
    )

    references = []
    predictions = []
    clip_results = []

    for sample in samples:
        start = perf_counter()
        prediction = transcribe_audio(model, sample["audio"])
        transcription_seconds = perf_counter() - start

        references.append(sample["reference"])
        predictions.append(prediction)

        result = score_transcript(sample["reference"], prediction)

        clip_results.append(
            {
                "id": sample["id"],
                "slice": sample["slice"],
                "reference": sample["reference"],
                "prediction": prediction,
                "scores": result,
                "transcription_seconds": transcription_seconds,
            }
        )

    total_transcription_seconds = sum(
        clip["transcription_seconds"] for clip in clip_results
    )

    corpus_result = score_corpus(references, predictions)
    evaluation = {
        "model": MODEL_SETTINGS.copy(),
        "decoding": DECODING_SETTINGS.copy(),
        "sample_count": len(clip_results),
        "corpus_scores": corpus_result,
        "clips": clip_results,
        "performance": {
            "total_transcription_seconds": total_transcription_seconds,
        },
        "dataset_fingerprint": fingerprint_dataset(samples),
        "normalization_version": NORMALIZATION_VERSION,
    }

    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2, ensure_ascii=False)

    print("Results saved to:", output_path)
    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate speech transcription accuracy."
    )

    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="whisper",
        help="Transcription provider to evaluate",
    )

    parser.add_argument(
        "--manifest",
        default="data/manifest.jsonl",
        help="Dataset manifest to evaluate",
    )

    parser.add_argument(
        "--output",
        default="outputs/results.json",
        help="Where to save evaluation results",
    )

    args = parser.parse_args()

    main(
        manifest_path=args.manifest,
        results_path=args.output,
    )
