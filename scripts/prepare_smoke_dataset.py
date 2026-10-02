
"""Prepare 12 clips from six speakers for the PR smoke evaluation."""

import json
import shutil
from pathlib import Path


def main():
    source = Path("/Users/nileshpandey/Downloads/LibriSpeech/dev-clean")
    destination = Path("data/clips")
    manifest_path = Path("data/manifest.jsonl")

    destination.mkdir(parents=True, exist_ok=True)

    speakers = sorted(
        path for path in source.iterdir() if path.is_dir()
    )[:6]

    if len(speakers) != 6:
        raise ValueError("Expected at least six speaker directories")

    samples = []

    for speaker in speakers:
        transcript_files = sorted(speaker.rglob("*.trans.txt"))

        if not transcript_files:
            raise ValueError(f"No transcripts for speaker {speaker.name}")

        transcript = transcript_files[0]
        lines = transcript.read_text(encoding="utf-8").splitlines()[:2]

        if len(lines) != 2:
            raise ValueError(f"Expected two clips for speaker {speaker.name}")

        for line in lines:
            clip_id, reference = line.split(" ", 1)
            filename = f"{clip_id}.flac"
            audio_path = destination / filename

            shutil.copyfile(transcript.parent / filename, audio_path)

            samples.append({
                "id": clip_id,
                "audio": audio_path.as_posix(),
                "reference": reference,
                "slice": "clean",
                "speaker_id": speaker.name,
                "source": "LibriSpeech dev-clean",
            })

    with manifest_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample) + "\n")

    print(f"Prepared {len(samples)} clips from {len(speakers)} speakers")


if __name__ == "__main__":
    main()
