from speech_eval.normalize import normalize_text


def contains_critical_phrase(prediction: str, expected_phrase: str) -> bool:
    """Check whether the expected phrase appears as consecutive whole words."""
    expected_words = normalize_text(expected_phrase).split()
    predicted_words = normalize_text(prediction).split()

    if not expected_words:
        raise ValueError("Expected phrase must not be blank")

    phrase_length = len(expected_words)

    for start in range(len(predicted_words) - phrase_length + 1):
        words = predicted_words[start : start + phrase_length]

        if words == expected_words:
            return True

    return False


def has_no_transcribed_words(prediction: str) -> bool:
    """Check that a labeled silence clip produced no transcribed words."""
    return normalize_text(prediction).strip() == ""


def evaluate_customer_checks(
    prediction: str,
    expectations: dict,
) -> dict:
    """Evaluate the customer expectations defined for a clip."""
    checks = {}

    if "required_phrase" in expectations:
        checks["required_phrase_preserved"] = contains_critical_phrase(
            prediction,
            expectations["required_phrase"],
        )

    if expectations.get("no_transcribed_words"):
        checks["no_transcribed_words"] = has_no_transcribed_words(prediction)

    if "accepted_phrases" in expectations:
        checks["accepted_phrase_preserved"] = any(
            contains_critical_phrase(prediction, phrase)
            for phrase in expectations["accepted_phrases"]
        )

    return checks
