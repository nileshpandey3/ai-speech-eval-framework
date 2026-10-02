import math


def compare_evaluations(
    baseline: dict,
    candidate: dict,
    max_wer_increase: float = 0.0,
) -> dict:
    """Decide whether comparable candidate results exceed the WER limit."""
    if not math.isfinite(max_wer_increase) or max_wer_increase < 0:
        raise ValueError("The allowed WER increase must be finite and nonnegative.")

    # Missing or changed comparison metadata blocks the decision.
    for field in ("dataset_fingerprint", "normalization_version"):
        if not baseline.get(field) or not candidate.get(field):
            return {
                "decision": "NO_GO",
                "reasons": [f"Missing {field}"],
            }

        if baseline[field] != candidate[field]:
            return {
                "decision": "NO_GO",
                "reasons": [f"Different {field}; results are not comparable"],
            }

    baseline_wer = baseline["corpus_scores"]["wer"]
    candidate_wer = candidate["corpus_scores"]["wer"]

    for value in (baseline_wer, candidate_wer):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
        ):
            return {
                "decision": "NO_GO",
                "reasons": ["Invalid WER value"],
            }

    allowed_wer = baseline_wer + max_wer_increase
    reasons = []

    if candidate_wer > allowed_wer:
        reasons.append("Corpus WER exceeds the allowed regression limit")

    return {
        "decision": "NO_GO" if reasons else "GO",
        "reasons": reasons,
        "baseline_wer": baseline_wer,
        "candidate_wer": candidate_wer,
        "wer_delta": candidate_wer - baseline_wer,
        "allowed_wer": allowed_wer,
    }
