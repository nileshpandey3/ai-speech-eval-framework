from jiwer import process_words


def score_transcript(reference, prediction):

    result = process_words(reference, prediction)

    return {
        "wer": result.wer,
        "insertions": result.insertions,
        "substitutions": result.substitutions,
        "deletions": result.deletions

    }
