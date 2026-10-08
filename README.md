# AI Speech Evaluation Framework

A small, extensible framework for evaluating batch speech-to-text (STT) models and APIs. It combines deterministic framework tests, live transcription evaluations, baseline regression gates, and shareable reports.

Currently supports local Whisper and the Deepgram STT API.

## Testing and Evaluation Layers

| Layer | Scope | Purpose |
|---|---|---|
| Framework tests | Scoring, normalization, dataset fingerprinting, comparison rules | Validate the evaluation machinery |
| Adapter tests | Mocked API responses and failures | Validate provider integration without API costs |
| PR smoke evaluation | 12 clips from six speakers | Give fast feedback on transcription accuracy regressions |
| Scheduled benchmark | 60 clips from 20 speakers | Evaluate a broader clean-speech dataset |
| Optional Deepgram evaluation | Live API transcription | Evaluate the hosted provider against its own reviewed baseline |

## Evaluation Criteria

- **Accuracy:** Corpus word error rate (WER), with insertion, substitution, and deletion counts.
- **Comparability:** Matching dataset fingerprints and normalization versions are required for baseline comparison.
- **Regression gate:** GO when candidate corpus WER is within the configured allowance above the reviewed baseline; otherwise a NO_GO release decision.
- **Performance:** Transcription time is recorded for inspection, but does not currently block the gate and can be used in future to track API/model latency.
- **Evidence:** JSON results and HTML reports expose transcripts, metrics, settings, and comparison decisions.

### Customer Acceptance Criteria

The customer evaluation checks three recorded fixtures:

- **Order identifier:** The transcript must contain order no: `AB4821` as one
  complete token, with no internal spaces. Matching is case-insensitive.
  `AB 4821` and `A B four eight two one` fail this requirement.
- **Negation:** The transcript must preserve the exact phrase
  `do not cancel` under the existing normalization policy.
- **Silence:** The labeled silence recording must produce no
  transcribed words under the existing normalization policy.

Every expected check must execute and pass. A missing or failed check
fails the customer acceptance gate, independently of aggregate WER.

These criteria apply to the three demo fixtures; they do not establish
general identifier accuracy, intent understanding, or silence handling.
Passing suite means the selected tests passed and the evaluated model met the accuracy policy on the selected dataset. It does not establish overall production readiness.

## Scope and Limitations

The current datasets contain clean, read English speech from LibriSpeech. They provide regression signals rather than a comprehensive assessment of model quality.

The framework does not currently evaluate streaming, diarization, noisy conversations, multilingual speech, or production-scale throughput. Aggregate WER can also hide regressions on individual clips.

## Extending to Other STT Providers

Providers implement a shared batch transcription contract:

- Accept an audio file path.
- Return transcript text and provider metadata.
- Expose model and decoding settings.

A new adapter can reuse the existing dataset, normalization, scoring, baseline comparison, and reporting pipeline. Compatibility depends on the provider supporting the required audio and batch transcription behavior.

For larger deployments, the design can expand through representative datasets, slice-specific gates, parallel evaluation workers, controlled performance benchmarks, and centrally stored evaluation artifacts.

## Running Tests

Framework and mocked adapter tests, without live model or API calls:

    python -m pytest -m "not live" -q

Live Whisper smoke evaluation:

    python -m pytest -m "live and not benchmark and not deepgram_live" -q

Broader Whisper benchmark:

    python -m pytest -m "benchmark" -q

Optional paid Deepgram evaluation, requiring `DEEPGRAM_API_KEY`:

    python -m pytest -m "deepgram_live" -q

## Dataset Attribution

Audio and reference transcripts are selected from
[LibriSpeech](https://www.openslr.org/12/)
