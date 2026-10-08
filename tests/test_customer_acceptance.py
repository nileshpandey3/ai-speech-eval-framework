import json

import pytest
import pytest_html.extras as report_extras
from run_eval import main


@pytest.mark.live
@pytest.mark.customer
def test_customer_acceptance(extras):
    """Require every expected customer check to run and pass."""
    result = main(
        manifest_path="data/customer.jsonl",
        results_path="outputs/customer-results.json",
    )
    # attach the evaluation evidence to the customer HTML report
    extras.append(
        report_extras.text(
            json.dumps(result, indent=2),
            name="Customer evaluation evidence",
        )
    )

    expected_checks = {
        "customer-order-001": "accepted_phrase_preserved",
        "customer-negation-001": "required_phrase_preserved",
        "customer-silence-001": "no_transcribed_words",
    }

    clips = {clip["id"]: clip for clip in result["clips"]}

    assert result["sample_count"] == 3
    assert len(result["clips"]) == 3
    assert set(clips) == set(expected_checks)

    for clip_id, check_name in expected_checks.items():
        clip = clips[clip_id]
        checks = clip["customer_checks"]

        assert (
            check_name in checks
        ), f"{clip_id}: expected check {check_name} was skipped"

        assert checks[check_name] is True, (
            f"{clip_id}: {check_name} failed; " f"prediction={clip['prediction']!r}"
        )
