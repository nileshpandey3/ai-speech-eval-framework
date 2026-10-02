MODEL_SETTINGS = {
    "name": "tiny.en",
    "device": "cpu",
    "compute_type": "int8",
    # python -c "from pathlib import Path; from faster_whisper.utils import download_model; print(Path(download_model('tiny.en', local_files_only=True)).name)"
    "revision": "0d3d19a32d3338f10357c0889762bd8d64bbdeba",
}

DECODING_SETTINGS = {
    "language": "en",
    # beam_size=1 means it follows one candidate transcription path at a time.
    # This is faster than exploring several alternatives. A larger value,
    # such as 5, considers more candidate paths and may improve accuracy, but takes more computation.
    "beam_size": 1,
    "temperature": 0, # Avoid random sampling
}
