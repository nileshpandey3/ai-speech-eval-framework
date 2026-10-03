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


"""Evaluate speech transcription providers against a dataset manifest."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from speech_eval.dataset import fingerprint_dataset
from speech_eval.normalize import NORMALIZATION_VERSION
from speech_eval.providers import PROVIDERS, create_transcriber
from speech_eval.score import score_corpus, score_transcript


def main(
    manifest_path="data/manifest.jsonl",
    results_path="outputs/results.json",
    provider="whisper",
):
    """Transcribe a dataset, score predictions, and save evaluation evidence."""
    samples = []

    with Path(manifest_path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                samples.append(json.loads(line))

    if not samples:
        raise ValueError("The dataset manifest is empty")

    # Capture the dataset identity before transcription.
    dataset_fingerprint = fingerprint_dataset(samples)

    # Initialize the selected provider once for the whole evaluation.
    transcribe, model_settings, decoding_settings = create_transcriber(provider)

    references = []
    predictions = []
    clip_results = []

    for sample in samples:
        start = perf_counter()
        response = transcribe(sample["audio"])
        transcription_seconds = perf_counter() - start

        prediction = response["text"]

        if not isinstance(prediction, str):
            raise ValueError(
                f"Provider returned a non-string transcript for {sample['id']}"
            )

        references.append(sample["reference"])
        predictions.append(prediction)

        clip_results.append(
            {
                "id": sample["id"],
                "slice": sample["slice"],
                "reference": sample["reference"],
                "prediction": prediction,
                "scores": score_transcript(sample["reference"], prediction),
                "transcription_seconds": transcription_seconds,
                "transcription_metadata": response["metadata"],
            }
        )

    evaluation = {
        "provider": provider,
        "model": model_settings,
        "decoding": decoding_settings,
        "sample_count": len(samples),
        "corpus_scores": score_corpus(references, predictions),
        "clips": clip_results,
        "performance": {
            "total_transcription_seconds": sum(
                clip["transcription_seconds"] for clip in clip_results
            ),
        },
        "dataset_fingerprint": dataset_fingerprint,
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
        provider=args.provider,
    )
