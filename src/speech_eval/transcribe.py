from faster_whisper import WhisperModel

from speech_eval.config import DECODING_SETTINGS


def transcribe_audio(model:WhisperModel, audio:str)->str:
    """Transcribe English audio and combine the segment text"""
    segments, _ = model.transcribe(
        audio=audio,
        **DECODING_SETTINGS
    )
    transcript_parts = []
    for segment in segments:
        transcript_parts.append(segment.text.strip())

    # Preserve spaces between transcription segments
    # this can avoid merging "only" and "garment" into "onlygarment"
    return " ".join(transcript_parts)
