"""Regression tests for Streamlit's HTTP client without starting a UI server."""

import importlib.util
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
API_MODULE_PATH = REPOSITORY_ROOT / "streamlit" / "api.py"
SPEC = importlib.util.spec_from_file_location("streamlit_api_client", API_MODULE_PATH)
api_client = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(api_client)


class DummyResponse:
    def __init__(self, payload, content=b""):
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_send_voice_returns_text_when_no_audio_file(monkeypatch):
    payload = {
        "success": True,
        "message": "Speech processed.",
        "data": {
            "transcription": "recognized words",
            "response": "Assistant reply",
            "audio_file": None,
            "audio_error": "Speech playback is temporarily unavailable.",
        },
    }
    calls = {"get": 0}

    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda *args, **kwargs: DummyResponse(payload),
    )

    def unexpected_get(*args, **kwargs):
        calls["get"] += 1
        raise AssertionError("Audio should not be fetched when audio_file is absent.")

    monkeypatch.setattr(api_client.requests, "get", unexpected_get)

    result = api_client.send_voice("session-1", b"wav-data")

    assert result["transcription"] == "recognized words"
    assert result["response"] == "Assistant reply"
    assert result["audio_bytes"] is None
    assert result["audio_error"]
    assert calls["get"] == 0


def test_upload_document_posts_pdf_to_rag_endpoint(monkeypatch):
    payload = {
        "success": True,
        "message": "Document processed.",
        "data": {
            "duplicate": False,
            "file_name": "sample.pdf",
            "chunks": 3,
            "characters": 1200,
        },
    }
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return DummyResponse(payload)

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = api_client.upload_document("sample.pdf", b"%PDF-test")

    assert captured["url"].endswith("/rag/extract")
    assert captured["files"]["file"][0] == "sample.pdf"
    assert captured["files"]["file"][2] == "application/pdf"
    assert result["chunks"] == 3


@pytest.mark.parametrize(
    ("filename", "payload"),
    [("", b"data"), ("notes.txt", b"data"), ("notes.pdf", b"")],
)
def test_upload_document_rejects_invalid_input(filename, payload):
    with pytest.raises(ValueError):
        api_client.upload_document(filename, payload)
