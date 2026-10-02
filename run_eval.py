import json
from pathlib import Path
from time import perf_counter

from faster_whisper import WhisperModel

from speech_eval.dataset import fingerprint_dataset
from speech_eval.normalize import NORMALIZATION_VERSION
from speech_eval.score import score_corpus, score_transcript
from speech_eval.transcribe import transcribe_audio


def main():
    with Path("data/manifest.jsonl").open(encoding="utf-8") as file:
        samples = []

        for line in file:
            if line.strip():
                samples.append(json.loads(line))

    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    references = []
    predictions = []
    clip_results = []

    for sample in samples:
        start = perf_counter()
        prediction = transcribe_audio(model, sample["audio"])
        transcription_seconds = perf_counter()-start

        references.append(sample["reference"])
        predictions.append(prediction)

        result = score_transcript(sample["reference"], prediction)

        clip_results.append({
            "id": sample["id"],
            "slice": sample["slice"],
            "reference": sample["reference"],
            "prediction": prediction,
            "scores": result,
            "transcription_seconds": transcription_seconds,
        })

    total_transcription_seconds = sum(
        clip["transcription_seconds"] for clip in clip_results
    )

    corpus_result = score_corpus(references, predictions)
    evaluation = {
        "model": {
            "name": "tiny.en",
            "device": "cpu",
            "compute_type": "int8",
            "beam_size": 1,
            "temperature": 0,
        },
        "sample_count": len(clip_results),
        "corpus_scores": corpus_result,
        "clips": clip_results,
        "performance": {
            "total_transcription_seconds": total_transcription_seconds,
        },
        "dataset_fingerprint": fingerprint_dataset(samples),
        "normalization_version": NORMALIZATION_VERSION,
    }

    output_path = Path("outputs/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2, ensure_ascii=False)

    print("Results saved to:", output_path)
    return evaluation


if __name__ == "__main__":
    main()
