"""Check the broader speech dataset against its reviewed baseline."""

import json
from pathlib import Path

import pytest
import pytest_html.extras as report_extras

from run_eval import main
from speech_eval.compare import compare_evaluations


@pytest.mark.live
@pytest.mark.benchmark
def test_benchmark_against_baseline(extras):
    candidate = main(
        manifest_path="data/benchmark.jsonl",
        results_path="outputs/benchmark-results.json",
    )

    baseline_path = (
        Path(__file__).resolve().parents[1] / "baselines" / "tiny-en-benchmark.json"
    )

    with baseline_path.open(encoding="utf-8") as file:
        baseline = json.load(file)

    decision = compare_evaluations(baseline, candidate)

    extras.append(
        report_extras.text(
            json.dumps(candidate, indent=2),
            name="Benchmark evaluation",
        )
    )
    extras.append(
        report_extras.text(
            json.dumps(decision, indent=2),
            name="Benchmark decision",
        )
    )

    assert candidate["sample_count"] == baseline["sample_count"]
    assert len(candidate["clips"]) == candidate["sample_count"]
    assert decision["decision"] == "GO", decision["reasons"]
