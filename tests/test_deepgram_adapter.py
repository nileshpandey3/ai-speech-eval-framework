"""Test the Deepgram adapter without network calls or API credentials."""

from unittest.mock import Mock

import pytest
import requests

from speech_eval.deepgram_adapter import transcribe_deepgram


@pytest.mark.deepgram_adapter
class TestDeepGramAdapter:
    def test_missing_api_key_makes_no_request(self, monkeypatch):
        post = Mock()

        monkeypatch.setattr(
            "speech_eval.deepgram_adapter.requests.post",
            post,
        )

        with pytest.raises(ValueError, match="API key"):
            transcribe_deepgram("unused.flac", "")

        post.assert_not_called()

    def test_extracts_transcript_and_metadata(self, tmp_path, monkeypatch):
        audio = tmp_path / "sample.flac"
        audio.write_bytes(b"test-audio")

        metadata = {"request_id": "test-request"}

        response = Mock()
        response.json.return_value = {
            "results": {
                "channels": [
                    {
                        "alternatives": [{"transcript": "hello world"}],
                    }
                ],
            },
            "metadata": metadata,
        }

        def fake_post(url, **kwargs):
            assert url == "https://api.deepgram.com/v1/listen"
            assert kwargs["data"].read() == b"test-audio"
            assert kwargs["headers"]["Authorization"] == "Token fake-key"
            assert kwargs["params"]["model"] == "nova-3"
            assert kwargs["timeout"] == (10, 60)
            return response

        monkeypatch.setattr(
            "speech_eval.deepgram_adapter.requests.post",
            fake_post,
        )

        result = transcribe_deepgram(str(audio), "fake-key")

        assert result == {
            "text": "hello world",
            "metadata": metadata,
        }
        response.raise_for_status.assert_called_once()

    @pytest.mark.parametrize("status", [401, 429, 503])
    def test_http_failure_is_not_scored(self, tmp_path, monkeypatch, status):
        audio = tmp_path / "sample.flac"
        audio.write_bytes(b"test-audio")

        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")

        monkeypatch.setattr(
            "speech_eval.deepgram_adapter.requests.post",
            Mock(return_value=response),
        )

        with pytest.raises(requests.HTTPError):
            transcribe_deepgram(str(audio), "fake-key")

        response.json.assert_not_called()

    def test_missing_transcript_is_rejected(self, tmp_path, monkeypatch):
        audio = tmp_path / "sample.flac"
        audio.write_bytes(b"test-audio")

        response = Mock()
        response.json.return_value = {
            "results": {"channels": []},
            "metadata": {},
        }

        monkeypatch.setattr(
            "speech_eval.deepgram_adapter.requests.post",
            Mock(return_value=response),
        )

        with pytest.raises(ValueError, match="Malformed Deepgram response"):
            transcribe_deepgram(str(audio), "fake-key")

    def test_timeout_propagates(self, tmp_path, monkeypatch):
        audio = tmp_path / "sample.flac"
        audio.write_bytes(b"test-audio")

        monkeypatch.setattr(
            "speech_eval.deepgram_adapter.requests.post",
            Mock(side_effect=requests.Timeout("Request timed out")),
        )

        with pytest.raises(requests.Timeout):
            transcribe_deepgram(str(audio), "fake-key")
