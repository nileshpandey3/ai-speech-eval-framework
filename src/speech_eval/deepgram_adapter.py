"""Transcribe FLAC audio using Deepgram's pre-recorded API."""

import os
from pathlib import Path

import requests


def transcribe_deepgram(audio_path: str, api_key: str) -> dict:
    """Return transcript text and API metadata; raise on request failure."""
    if not api_key:
        raise ValueError("A Deepgram API key is required")

    with Path(audio_path).open("rb") as audio:
        response = requests.post(
            "https://api.deepgram.com/v1/listen",
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "audio/flac",
            },
            params={
                "model": "nova-3",
                "language": "en",
                "smart_format": "false",
                "punctuate": "false",
            },
            data=audio,
            timeout=(10, 60),
        )

    response.raise_for_status()
    result = response.json()

    try:
        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        metadata = result["metadata"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Malformed Deepgram response") from error

    if not isinstance(transcript, str):
        raise ValueError("Deepgram returned an invalid transcript")

    if not isinstance(metadata, dict):
        raise ValueError("Deepgram returned invalid metadata")

    return {
        "text": transcript,
        "metadata": metadata,
    }
