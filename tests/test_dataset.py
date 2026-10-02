from speech_eval.dataset import fingerprint_dataset


def make_sample(audio_path):
    return {
        "id": "clip-001",
        "audio": str(audio_path),
        "reference": "hello world",
        "slice": "clean",
    }


def test_unchanged_dataset_has_same_fingerprint(tmp_path):
    audio = tmp_path / "clip.flac"
    audio.write_bytes(b"original audio")
    samples = [make_sample(audio)]

    assert fingerprint_dataset(samples) == fingerprint_dataset(samples)


def test_changed_reference_changes_fingerprint(tmp_path):
    audio = tmp_path / "clip.flac"
    audio.write_bytes(b"original audio")
    samples = [make_sample(audio)]

    original = fingerprint_dataset(samples)
    samples[0]["reference"] = "goodbye world"

    assert fingerprint_dataset(samples) != original


def test_changed_audio_changes_fingerprint(tmp_path):
    audio = tmp_path / "clip.flac"
    audio.write_bytes(b"original audio")
    samples = [make_sample(audio)]

    original = fingerprint_dataset(samples)
    audio.write_bytes(b"replacement audio")

    assert fingerprint_dataset(samples) != original
    
