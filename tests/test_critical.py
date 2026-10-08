import pytest

from speech_eval.critical import (
    contains_critical_phrase,
    has_no_transcribed_words,
    evaluate_customer_checks,
)


@pytest.mark.parametrize(
    "prediction, expected_phrase, expected_result",
    [
        ("Your order is AB4821.", "AB4821", True),
        ("Your order is AB4829.", "AB4821", False),
        ("Your order is AB48219.", "AB4821", False),
        ("Please contact customer support.", "customer support", True),
        ("Please contact customer service.", "customer support", False),
        ("", "AB4821", False),
    ],
)
def test_critical_phrase_check(prediction, expected_phrase, expected_result):
    result = contains_critical_phrase(prediction, expected_phrase)

    assert result is expected_result


def test_blank_expected_phrase_is_rejected():
    with pytest.raises(ValueError, match="must not be blank"):
        contains_critical_phrase("Your order is AB4821", "   ")


@pytest.mark.parametrize(
    "prediction, expected_result",
    [
        ("Your order is AB4821.", True),
        ("Your order is AB4829.", False),
        ("Your order is AB48219.", False),
        ("Your order is ready.", False),
    ],
)
def test_customer_order_identifier_is_preserved(prediction, expected_result):
    """Detect changed, extended, or omitted customer order identifiers."""
    result = contains_critical_phrase(prediction, "AB4821")

    assert result is expected_result


@pytest.mark.parametrize(
    "prediction, expected_result",
    [
        ("Please do not cancel my booking.", True),
        ("Please cancel my booking.", False),
        ("Please do cancel my booking.", False),
        ("Please cancel my booking, not my flight.", False),
    ],
)
def test_customer_negation_phrase_is_preserved(prediction, expected_result):
    """Detect loss of the required negative instruction phrase."""
    result = contains_critical_phrase(prediction, "do not cancel")

    assert result is expected_result


@pytest.mark.parametrize(
    "prediction, expected_result",
    [
        ("", True),
        ("   ", True),
        ("Thank you for calling.", False),
        ("Hello.", False),
    ],
)
def test_silence_produces_no_transcribed_words(prediction, expected_result):
    """Flag transcribed words when the recording is labeled as silence."""
    result = has_no_transcribed_words(prediction)

    assert result is expected_result


@pytest.mark.parametrize(
    "prediction, expected_result",
    [
        ("Your order is AB4821.", True),
        ("Your order is A B four eight two one.", True),
        ("Your order is AB4829.", False),
        ("Your order is AB48219.", False),
        ("Your order is CD4821.", False),
        ("Your order is ready.", False),
    ],
)
def test_accepted_order_formats(prediction, expected_result):
    """Accept approved identifier formats and reject incorrect identifiers."""
    expectations = {
        "accepted_phrases": [
            "AB4821",
            "A B four eight two one",
        ]
    }

    checks = evaluate_customer_checks(prediction, expectations)

    assert checks["accepted_phrase_preserved"] is expected_result


@pytest.mark.parametrize(
    "prediction, expected_result",
    [
        ("Your order is AB4821.", True),
        ("Your order is AB 4821.", False),
        ("Your order is A B four eight two one.", False),
        ("Your order is AB4829.", False),
        ("Your order is AB48219.", False),
    ],
)
def test_order_identifier_requires_no_spaces(prediction, expected_result):
    result = contains_critical_phrase(prediction, "AB4821")

    assert result is expected_result
