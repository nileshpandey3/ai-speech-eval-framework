import soundfile as sf

import pytest

from speech_eval.dataset import (
    fingerprint_dataset,
    validate_samples,
    validate_audio_files,
)


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


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_samples([])


def test_missing_audio_is_rejected(tmp_path):
    sample = make_sample(tmp_path / "missing.flac")

    with pytest.raises(ValueError, match="audio file not found"):
        validate_samples([sample])


def test_duplicate_ids_are_rejected(tmp_path):
    audio = tmp_path / "clip.flac"
    audio.write_bytes(b"placeholder")
    sample = make_sample(audio)

    with pytest.raises(ValueError, match="Duplicate clip ID"):
        validate_samples([sample, sample.copy()])


def test_missing_reference_is_rejected(tmp_path):
    sample = make_sample(tmp_path / "clip.flac")
    del sample["reference"]

    with pytest.raises(ValueError, match="missing fields"):
        validate_samples([sample])


def test_blank_reference_is_rejected(tmp_path):
    sample = make_sample(tmp_path / "clip.flac")
    sample["reference"] = "   "

    with pytest.raises(ValueError, match="reference must be a nonempty string"):
        validate_samples([sample])


def test_valid_audio(tmp_path):
    audio_path = tmp_path / "valid.wav"

    # Write 0.1 seconds of silence at 16,000 samples per second.
    silence = [0.0] * 1600
    sf.write(str(audio_path), silence, samplerate=16000)

    sample = make_sample(audio_path)

    validate_audio_files([sample])


def test_corrupt_audio_is_rejected(tmp_path):
    audio_path = tmp_path / "corrupt.flac"
    audio_path.write_bytes(b"this is not audio")

    with pytest.raises(
        ValueError,
        match="clip-001: audio could not be decoded",
    ):
        validate_audio_files([make_sample(audio_path)])
