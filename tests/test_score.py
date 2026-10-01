import pytest

from speech_eval.score import score_transcript


def test_one_substitution():
    """One incorrect word out of four reference words gives WER 0.25."""
    result = score_transcript(
        reference="",
        prediction="",
    )

    assert result["wer"] == pytest.approx(0.25)
    assert result["substitutions"] == 1
    assert result["deletions"] == 0
    assert result["insertions"] == 0
