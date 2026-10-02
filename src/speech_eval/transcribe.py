from faster_whisper import WhisperModel


def transcribe_audio(model:WhisperModel, audio:str)->str:
    """Transcribe English audio and combine the segment text"""
    segments, _ = model.transcribe(
        audio=audio,
        language='en',
        temperature=0, # Avoid random sampling

        # beam_size=1 means it follows one candidate transcription path at a time.
        # This is faster than exploring several alternatives. A larger value,
        # such as 5, considers more candidate paths and may improve accuracy, but takes more computation.
        beam_size=1,

    )
    transcript_parts = []
    for segment in segments:
        transcript_parts.append(segment.text.strip())

    # Preserve spaces between transcription segments
    # this can avoid merging "only" and "garment" into "onlygarment"
    return " ".join(transcript_parts)
