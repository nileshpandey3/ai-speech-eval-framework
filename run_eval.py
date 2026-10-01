import json
from pathlib import Path

from faster_whisper import WhisperModel

from speech_eval.score import score_transcript
from speech_eval.transcribe import transcribe_audio


def main():
    with Path("data/manifest.jsonl").open(encoding="utf-8") as file:
        samples = []

        for line in file:
            if line.strip():
                samples.append(json.loads(line))

    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    for sample in samples:
        prediction = transcribe_audio(model, sample["audio"])
        result = score_transcript(sample["reference"], prediction)

        print("Clip:", sample["id"])
        print("Reference:", sample["reference"])
        print("Prediction:", prediction)
        print("Scores:", result)
        print()


if __name__ == "__main__":
    main()
