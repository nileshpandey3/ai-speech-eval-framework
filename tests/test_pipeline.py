import pytest
import pytest_html.extras as report_extras

from run_eval import main


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
    
