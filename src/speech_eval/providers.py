"""
Register transcription providers behind a shared result contract
It can be used as an extensible adapter design for batch STT
"""

import os

from speech_eval.config import MODEL_SETTINGS, DECODING_SETTINGS


def create_whisper():
    """Initialize Whisper once and return its transcription adapter."""
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model

    from speech_eval.transcribe import transcribe_audio

    model_path = download_model(
        MODEL_SETTINGS["name"],
        revision=MODEL_SETTINGS["revision"],
    )

    model = WhisperModel(
        model_path,
        device=MODEL_SETTINGS["device"],
        compute_type=MODEL_SETTINGS["compute_type"],
    )

    def transcribe(audio_path):
        return {
            "text": transcribe_audio(model, audio_path),
            "metadata": {},
        }

    return (
        transcribe,
        MODEL_SETTINGS.copy(),
        DECODING_SETTINGS.copy(),
    )


def create_deepgram():
    """Initialize the Deepgram transcription adapter."""
    from speech_eval.deepgram_adapter import transcribe_deepgram

    api_key = os.environ.get("DEEPGRAM_API_KEY")

    if not api_key:
        raise ValueError("Set DEEPGRAM_API_KEY before running Deepgram")

    def transcribe(audio_path):
        return transcribe_deepgram(audio_path, api_key)

    return (
        transcribe,
        {"name": "nova-3", "provider": "deepgram"},
        {
            "language": "en",
            "smart_format": False,
            "punctuate": False,
        },
    )


PROVIDERS = {
    "whisper": create_whisper,
    "deepgram": create_deepgram,
}


def create_transcriber(provider: str):
    """Initialize a registered provider."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown transcription provider: {provider}")

    return PROVIDERS[provider]()
