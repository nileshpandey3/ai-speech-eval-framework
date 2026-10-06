"""Check Deepgram's smoke evaluation against its reviewed baseline."""

import json
from pathlib import Path

import pytest
import pytest_html.extras as report_extras

from run_eval import main
from speech_eval.compare import compare_evaluations


@pytest.mark.live
@pytest.mark.deepgram_live
def test_deepgram_smoke_against_baseline(extras):
    # Load the reviewed baseline.
    project_root = Path(__file__).resolve().parents[1]
    baseline_path = project_root / "baselines/deepgram-nova3-smoke.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    # Evaluate the current API output.
    candidate = main(
        provider="deepgram",
        results_path="outputs/deepgram-smoke-results.json",
    )
    decision = compare_evaluations(baseline, candidate)

    # Attach evidence before assertions so failures remain inspectable.
    reports = [
        ("Deepgram evaluation", candidate),
        ("Accuracy decision", decision),
    ]

    for name, result in reports:
        extras.append(report_extras.text(json.dumps(result, indent=2), name=name))

    # Check configuration, completeness, and accuracy.
    assert baseline["provider"] == "deepgram"
    assert candidate["provider"] == "deepgram"
    assert candidate["decoding"] == baseline["decoding"]
    assert candidate["sample_count"] == baseline["sample_count"]
    assert len(candidate["clips"]) == candidate["sample_count"]
    assert decision["decision"] == "GO", decision["reasons"]
