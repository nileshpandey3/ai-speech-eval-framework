from jiwer import process_words

from speech_eval.normalize import normalize_text

def score_corpus(references: list[str], predictions: list[str]) -> dict:
    """Calculate corpus(collection) WER using total word edits and reference words."""
    if not references or len(references) != len(predictions):
        raise ValueError("Provide equal, nonempty lists of transcripts.")

    normalized_references = []
    normalized_predictions = []

    for reference in references:
        normalized_references.append(normalize_text(reference))

    for prediction in predictions:
        normalized_predictions.append(normalize_text(prediction))

    result = process_words(
        normalized_references,
        normalized_predictions,
    )

    return {
        "wer": result.wer,
        "insertions": result.insertions,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
    }

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
