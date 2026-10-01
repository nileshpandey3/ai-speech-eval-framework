from jiwer import process_words

from speech_eval.normalize import normalize_text


def score_transcript(reference, prediction):

    result = process_words(
        normalize_text(reference),
        normalize_text(prediction))

    return {
        "wer": result.wer,
        "insertions": result.insertions,
        "substitutions": result.substitutions,
        "deletions": result.deletions

    }
