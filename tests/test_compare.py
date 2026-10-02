import pytest

from speech_eval.compare import compare_evaluations


def make_evaluation(wer: float) -> dict:
    """Create evaluation results with matching comparison metadata."""
    return {
        "dataset_fingerprint": "same-test-dataset",
        "normalization_version": "v1",
        "corpus_scores": {"wer": wer},
    }

@pytest.mark.comparison
class TestCompare:
    @pytest.mark.parametrize(
        "candidate_wer, expected",
        [
            (0.25, "GO"),
            (0.125, "GO"),
            (0.375, "GO"),
            (0.50, "NO_GO"),
        ],
        ids=["unchanged", "improved", "at-limit", "regressed"],
    )
    def test_wer_regression_decision(self, candidate_wer, expected):
        baseline = make_evaluation(0.25)
        candidate = make_evaluation(candidate_wer)

        result = compare_evaluations(
            baseline,
            candidate,
            max_wer_increase=0.125,
        )

        assert result["decision"] == expected


    def test_changed_dataset_blocks_comparison(self):
        baseline = make_evaluation(0.25)
        candidate = make_evaluation(0.0)
        candidate["dataset_fingerprint"] = "different-dataset"

        result = compare_evaluations(baseline, candidate)

        assert result["decision"] == "NO_GO"
        assert "not comparable" in result["reasons"][0]


    def test_missing_metadata_blocks_comparison(self):
        baseline = make_evaluation(0.25)
        candidate = make_evaluation(0.0)
        del candidate["normalization_version"]

        result = compare_evaluations(baseline, candidate)

        assert result["decision"] == "NO_GO"
        assert result["reasons"] == ["Missing normalization_version"]


    def test_nan_wer_blocks_comparison(self):
        baseline = make_evaluation(0.25)
        candidate = make_evaluation(float("nan"))

        result = compare_evaluations(baseline, candidate)

        assert result["decision"] == "NO_GO"
        assert result["reasons"] == ["Invalid WER value"]


    def test_default_gate_blocks_accuracy_regression(self):
        """Reject a candidate whose WER exceeds the baseline."""
        baseline = make_evaluation(0.32)
        candidate = make_evaluation(0.40)

        result = compare_evaluations(
            baseline=baseline,
            candidate=candidate,
        )

        assert result["decision"] == "NO_GO"
        assert result["reasons"]
