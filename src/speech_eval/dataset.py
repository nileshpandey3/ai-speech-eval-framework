import hashlib
import json
from pathlib import Path


def fingerprint_dataset(samples: list[dict]) -> str:
    """
    Identify the exact samples, references, slices, and audio contents.
    The fingerprint changes if you add a clip, edit a reference, change its slice, or replace its audio.
    Reordering the same samples does not change it
    """
    records = []

    for sample in sorted(samples, key=lambda item: item["id"]):
        audio_bytes = Path(sample["audio"]).read_bytes()

        records.append({
            "id": sample["id"],
            "reference": sample["reference"],
            "slice": sample["slice"],
            "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        })

    content = json.dumps(records, sort_keys=True).encode()

    return hashlib.sha256(content).hexdigest()
