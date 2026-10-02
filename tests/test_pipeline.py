import json
from pathlib import Path

import pytest
import pytest_html.extras as report_extras

from run_eval import main
from speech_eval.compare import compare_evaluations


@pytest.mark.live
def test_live_evaluation_produces_results(extras):
    """Run the speech evaluation and attach its results to the report."""
    evaluation = main()

    extras.append(
        report_extras.json(
            evaluation,
            name="Speech evaluation results",
        )
    )

    assert evaluation["sample_count"] > 0
    assert len(evaluation["clips"]) == evaluation["sample_count"]

    """Run the speech model and check for accuracy regressions."""

@pytest.mark.live
def test_live_evaluation_against_baseline(extras):
    candidate = main()

    baseline_path = (
            Path(__file__).resolve().parents[1]
            / "baselines"
            / "tiny-en.json"
    )

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
    assert decision["decision"] == "GO", decision["reasons"]
