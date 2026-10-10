import pytest
from src.speech_eval.score import score_transcript, score_corpus


@pytest.mark.transcript_word_scores
class TestScores:

    def test_one_substitution(self):
        """One incorrect word out of four reference words gives WER 0.25."""
        result = score_transcript(
            reference="the cat sat down",
            prediction="the dog sat down",
        )

        assert result["wer"] == pytest.approx(0.25)
        assert result["substitutions"] == 1
        assert result["deletions"] == 0
        assert result["insertions"] == 0

    def test_normalization_before_scoring(self):
        result = score_transcript(
            reference="  THE   CAT\nSAT DOWN  ",
            prediction="the cat sat down",
        )

        assert result["wer"] == 0
        assert result["substitutions"] == 0
        assert result["deletions"] == 0
        assert result["insertions"] == 0

    def test_one_deletion(self):
        """One missing word out of four reference words gives WER 0.25."""
        result = score_transcript(
            reference="the cat sat down",
            prediction="the cat sat",
        )

        assert result["wer"] == pytest.approx(0.25)
        assert result["deletions"] == 1
        assert result["substitutions"] == 0
        assert result["insertions"] == 0

    def test_one_insertion(self):
        """One extra word against four reference words gives WER 0.25."""
        result = score_transcript(
            reference="the cat sat down",
            prediction="the black cat sat down",
        )

        assert result["wer"] == pytest.approx(0.25)
        assert result["insertions"] == 1
        assert result["substitutions"] == 0
        assert result["deletions"] == 0

    def test_corpus_wer_uses_total_reference_words(self):
        """One error across five words gives 0.2, rather than averaging clip WERs."""
        result = score_corpus(
            references=["hello", "the cat sat down"],
            predictions=["goodbye", "the cat sat down"],
        )

        assert result["wer"] == pytest.approx(0.2)
        assert result["substitutions"] == 1
