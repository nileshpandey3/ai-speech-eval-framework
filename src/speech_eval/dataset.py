import hashlib
import json
from pathlib import Path

import soundfile as sf


def fingerprint_dataset(samples: list[dict]) -> str:
    """
    Identify the exact samples, references, slices, and audio contents.
    The fingerprint changes if you add a clip, edit a reference, change its slice, or replace its audio.
    Reordering the same samples does not change it
    """
    records = []

    for sample in sorted(samples, key=lambda item: item["id"]):
        audio_bytes = Path(sample["audio"]).read_bytes()

        records.append(
            {
                "id": sample["id"],
                "reference": sample["reference"],
                "slice": sample["slice"],
                "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
            }
        )

    content = json.dumps(records, sort_keys=True).encode()

    return hashlib.sha256(content).hexdigest()


def validate_samples(samples: list[dict]) -> None:
    """Reject empty datasets, invalid records, and missing audio files."""
    if not samples:
        raise ValueError("Dataset is empty")

    seen_ids = set()

    for index, sample in enumerate(samples):
        required_fields = {"id", "audio", "reference", "slice"}
        missing_fields = required_fields - sample.keys()

        if missing_fields:
            raise ValueError(f"Sample {index}: missing fields {sorted(missing_fields)}")

        expectations = sample.get("expectations", {})

        try:
            validate_expectations(expectations)
        except ValueError as error:
            raise ValueError(f"Sample {index}: {error}") from error

        is_silence = (
            sample["slice"] == "silence"
            and sample.get("expectations", {}).get("no_transcribed_words") is True
        )

        for field in required_fields:
            value = sample[field]

            if not isinstance(value, str):
                raise ValueError(f"Sample {index}: {field} must be a string")

            # Silence recordings are allowed to have an empty reference.
            if field == "reference" and is_silence:
                continue

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Sample {index}: {field} must be a nonempty string")

        clip_id = sample["id"]

        if clip_id in seen_ids:
            raise ValueError(f"Duplicate clip ID: {clip_id}")

        seen_ids.add(clip_id)

        if not Path(sample["audio"]).is_file():
            raise ValueError(f"{clip_id}: audio file not found")


def validate_audio_files(samples: list[dict]) -> None:
    """Check that each file can be read and contains audio samples."""
    for sample in samples:
        clip_id = sample["id"]

        try:
            audio, _sample_rate = sf.read(
                sample["audio"],
                dtype="float32",
            )
        except (sf.SoundFileError, OSError) as error:
            raise ValueError(f"{clip_id}: audio could not be decoded") from error

        if len(audio) == 0:
            raise ValueError(f"{clip_id}: audio contains no samples")


def validate_expectations(expectations: dict) -> None:
    """Reject unsupported or malformed customer acceptance rules."""
    if not isinstance(expectations, dict):
        raise ValueError("expectations must be a dictionary")

    supported = {
        "required_phrase",
        "accepted_phrases",
        "no_transcribed_words",
    }

    unknown = expectations.keys() - supported
    if unknown:
        raise ValueError(f"Unknown expectations: {sorted(unknown)}")

    if "required_phrase" in expectations:
        phrase = expectations["required_phrase"]

        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("required_phrase must be a nonempty string")

    if "accepted_phrases" in expectations:
        phrases = expectations["accepted_phrases"]

        if not isinstance(phrases, list) or not phrases:
            raise ValueError("accepted_phrases must be a nonempty list")

        for phrase in phrases:
            if not isinstance(phrase, str) or not phrase.strip():
                raise ValueError("Each accepted phrase must be a nonempty string")

    if "no_transcribed_words" in expectations:
        if expectations["no_transcribed_words"] is not True:
            raise ValueError("no_transcribed_words must be true")
