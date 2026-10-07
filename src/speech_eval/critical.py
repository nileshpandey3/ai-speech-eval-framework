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
