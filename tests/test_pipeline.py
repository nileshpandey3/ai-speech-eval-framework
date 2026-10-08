import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import pytest_html.extras as report_extras

from run_eval import main
from speech_eval.compare import compare_evaluations


@pytest.mark.live
def test_live_evaluation_against_baseline(extras):
    candidate = main()

    baseline_path = Path(__file__).resolve().parents[1] / "baselines" / "tiny-en.json"

    with baseline_path.open(encoding="utf-8") as file:
        baseline = json.load(file)

    decision = compare_evaluations(baseline=baseline, candidate=candidate)

    # Attach evidence before asserting so failures remain inspectable.
    extras.append(
        report_extras.text(
            json.dumps(candidate, indent=2),
            name="Candidate evaluation",
        )
    )

    extras.append(
        report_extras.text(
            json.dumps(decision, indent=2),
            name="Regression decision",
        )
    )

    assert candidate["sample_count"] > 0
    assert len(candidate["clips"]) == candidate["sample_count"]
    # GO means “no aggregate WER regression detected on this smoke dataset.”
    # It doesn’t establish production readiness, and an aggregate score can hide one clip getting worse while another improves.
    assert decision["decision"] == "GO", decision["reasons"]


def test_invalid_expectations_block_model_initialization(tmp_path, monkeypatch):
    """Reject malformed evaluation rules before initializing a provider."""
    audio_path = tmp_path / "clip.flac"
    audio_path.write_bytes(b"placeholder")

    sample = {
        "id": "invalid-expectation",
        "audio": str(audio_path),
        "reference": "Your order is AB4821",
        "slice": "customer",
        "expectations": {
            "required_phrase": ["AB4821"],  # Should be a string
        },
    }

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    create_provider = Mock()
    monkeypatch.setattr("run_eval.create_transcriber", create_provider)

    with pytest.raises(
        ValueError,
        match="required_phrase must be a nonempty string",
    ):
        main(
            manifest_path=manifest,
            results_path=tmp_path / "results.json",
        )

    create_provider.assert_not_called()
