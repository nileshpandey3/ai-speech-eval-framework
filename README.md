# AI Speech Evaluation Framework




A Python+Pytest based framework for evaluating **batch speech-to-text accuracy and customer acceptance criteria**. It combines deterministic tests of the evaluation code, live model runs, reviewed baselines, and inspectable evidence in GitHub Actions (CI/CD gates).

The project demonstrates how to give developers fast PR feedback while running a broader evaluation on a schedule. It supports **local Whisper through faster-whisper** and the **Deepgram prerecorded STT API** through a shared adapter interface.

**A passing gate means the selected evaluation criteria passed on the selected fixtures. It does not establish production readiness or overall model quality.**

## Why evaluate more than WER?

A transcript can preserve meaning but still break a downstream workflow. For example, an order number `AB 4821` may be understandable to a person but fail an integration requiring `AB4821` without spaces. Conversely, formatting differences can increase WER even when the required identifier is preserved.

This framework reports aggregate transcription accuracy alongside explicit customer checks, so those signals remain visible separately.

## Evaluation layers

| Layer | Current scope | Execution |
|---|---|---|
| Deterministic framework tests | Scoring, normalization, dataset validation, fingerprints, comparison rules, and customer checks | Every PR and push to `main` |
| Mocked API tests | Deepgram request construction, response parsing, HTTP failures, and timeouts | With framework tests; no API calls |
| Live Whisper smoke gate | 12 LibriSpeech clips from 6 speakers, compared with a stored baseline | After framework tests in PR/main CI |
| Customer acceptance gate | Identifier formatting, negation phrase preservation, and silence across 3 fixtures | PR/main CI; also attempted if the preceding accuracy gate fails |
| Scheduled Whisper benchmark | 60 LibriSpeech clips from 20 speakers, compared with a separate baseline | Daily at 10:17 UTC, or manual dispatch |
| Optional Deepgram evaluation | Live transcription of the smoke dataset against a Deepgram baseline | Explicit local invocation; requires credentials and uses API credits |

The benchmark includes the 12 smoke clips. It is a broader regression dataset, not an independent holdout set. Scheduled runs cover this manifest, not the model's entire operating domain.

## Evaluation criteria

### Transcription accuracy

JiWER calculates per-clip and corpus word error rate:

`WER = (substitutions + deletions + insertions) / reference word count`

Corpus WER uses total word edits and reference words, rather than averaging clip percentages. WER may exceed 1 when there are many insertions. Empty-reference behavior follows JiWER 4; silence also has a dedicated customer check.

Normalization policy `v1` lowercases text, removes commas and periods, and collapses whitespace. It does not equate number words with digits, expand abbreviations, or remove every punctuation mark.

### Baseline regression decision

`compare_evaluations()` requires matching dataset fingerprints and normalization versions, and valid finite, nonnegative WER values.

- `GO`: candidate corpus WER is at most baseline WER plus the allowed increase.
- `NO_GO`: WER exceeds that limit, or the comparison fails the metadata/value checks.
- The current live tests use the default **zero allowed increase**.
- The allowance is an absolute WER difference, not a relative percentage.

The dataset fingerprint includes clip IDs, reference text, slices, and audio bytes. It is independent of manifest ordering. It currently excludes customer expectations.

The comparison helper does not enforce provider or decoding compatibility; the Deepgram live test adds those assertions separately. A `NO_GO` reason must be inspected to distinguish a measured regression from an invalid comparison.

### Customer acceptance

| Fixture | Required behavior                                                                                                                              |
|---|------------------------------------------------------------------------------------------------------------------------------------------------|
| Order identifier | Contains order number `AB4821` as a complete normalized token, without internal spaces. If the result is `AB 4821` , the spoken-out form fail. |
| Negation | Contains the consecutive normalized words `do not cancel`.                                                                                     |
| Silence | Contains no text after the current normalization policy is applied.                                                                            |

The customer test verifies the expected clip IDs and requires every expected check to execute and pass. These checks are separate from the WER regression gate.

The evaluator also supports an explicit `accepted_phrases` list. The current order fixture deliberately uses the stricter `required_phrase` rule.

Phrase checks currently establish phrase presence, not full semantic correctness, absence of contradictory statements, or general intent understanding.

### Performance and evidence

Each completed evaluation saves predictions, references, edit counts, model/decoding settings, dataset identity, provider metadata, customer checks, and transcription timings.

Timing surrounds the transcription adapter call. It excludes provider initialization and dataset validation; API timings include request overhead. Timings are diagnostic and do not currently gate releases or establish streaming latency or throughput.

## Quick start

Run commands from the repository root. CI uses Python 3.14; package metadata declares Python 3.11 or newer.

```bash
git clone https://github.com/nileshpandey3/ai-speech-eval-framework.git
cd ai-speech-eval-framework
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run deterministic tests without loading a model or calling an API:

```bash
python -m pytest -m "not live" -q
```

Evaluate Whisper and write JSON evidence:

```bash
python run_eval.py --provider whisper \
  --manifest data/manifest.jsonl \
  --output outputs/results.json
```

The first Whisper run downloads the pinned model revision. Subsequent runs can reuse its local cache. `run_eval.py` generates evidence; **pytest performs the acceptance and baseline assertions**. Successful script execution alone does not mean a quality gate passed.

## Run gates and generate reports

```bash
mkdir -p outputs

# Smoke accuracy only
python -m pytest -m "live and not benchmark and not deepgram_live and not customer" -q \
  --html=outputs/live-report.html

# Customer acceptance only
python -m pytest -m "customer and not deepgram_live" -q \
  --html=outputs/customer-report.html

# Broader benchmark
python -m pytest -m benchmark -q \
  --html=outputs/benchmark-report.html
```

These commands use pytest-html's external assets. Keep each HTML report with its generated `assets/` directory when sharing. Expand a test result to inspect its evidence links.

CI uploads evaluation artifacts even when a gate fails and retains them for 14 days. The customer report includes external assets; the current CI accuracy and benchmark reports use self-contained HTML, with JSON results also uploaded separately.

Failing assertions produce failing CI checks. Configure the relevant checks as required in GitHub branch protection or rulesets to enforce merge blocking.

### Optional Deepgram run

```bash
python -m pip install -e ".[dev,deepgram]"
# Set DEEPGRAM_API_KEY in your shell or secret manager before running.
python -m pytest -m deepgram_live -q \
  --html=outputs/deepgram-report.html
```

This calls the paid API. The checked-in workflows do not run it. Use explicit markers: bare `pytest` also selects live tests, including Deepgram.

## Code map

| File | Responsibility |
|---|---|
| `run_eval.py` | Load and validate a manifest, initialize a provider, transcribe, score, and save evidence |
| `src/speech_eval/dataset.py` | Validate records, customer expectations, and decodable audio; fingerprint the dataset |
| `src/speech_eval/providers.py` | Register and initialize transcription providers |
| `src/speech_eval/transcribe.py` | Consume Whisper segments and combine transcript text |
| `src/speech_eval/deepgram_adapter.py` | Call and validate Deepgram's prerecorded API |
| `src/speech_eval/normalize.py` | Define the versioned text normalization policy |
| `src/speech_eval/score.py` | Calculate clip and corpus WER |
| `src/speech_eval/critical.py` | Evaluate customer phrase and silence expectations |
| `src/speech_eval/compare.py` | Compare baseline and candidate accuracy |
| `tests/` | Deterministic tests and explicitly marked live evaluations |
| `baselines/` | Stored results used by the live regression gates |

## Extending to another STT provider

Implement a factory that returns a transcription callable, model settings, and decoding settings. The callable accepts an audio path and returns:

```python
{"text": "recognized speech", "metadata": {}}
```

Register the factory in `PROVIDERS`, add mocked contract/error tests, and define the appropriate live baseline and settings checks. The existing manifest validation, scoring, customer checks, and reporting can then be reused.

New adapters must handle their provider's audio formats, credentials, response schema, and errors. The current Deepgram adapter sends FLAC. Streaming and diarization require additional contracts and evaluation logic.
