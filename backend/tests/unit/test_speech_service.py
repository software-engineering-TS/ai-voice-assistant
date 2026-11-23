from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile

import app.services.speech_service as speech_module
from app.services.speech_service import SpeechService


@pytest.fixture
def speech_service(tmp_path, monkeypatch):
    monkeypatch.setattr(
        speech_module.settings, "UPLOAD_FOLDER", str(tmp_path / "uploads")
    )
    monkeypatch.setattr(
        speech_module.settings, "OUTPUT_FOLDER", str(tmp_path / "outputs")
    )
    monkeypatch.setattr(speech_module.settings, "MAX_AUDIO_UPLOAD_SIZE_MB", 1)
    return SpeechService()


def make_upload(payload=b"audio-bytes", filename="test.wav"):
    return UploadFile(filename=filename, file=BytesIO(payload))


def patch_successful_pipeline(monkeypatch, response="Assistant reply", transcription="recognized words"):
    monkeypatch.setattr(
        speech_module.speech_to_text_service,
        "transcribe",
        lambda _path: transcription,
    )
    monkeypatch.setattr(
        speech_module.assistant_service,
        "generate_response",
        AsyncMock(return_value={"response": response, "sources": []}),
    )
    generated = []

    async def fake_tts(text, output_file):
        generated.append((text, output_file))
        return output_file

    monkeypatch.setattr(
        speech_module.text_to_speech_service,
        "generate",
        fake_tts,
    )
    return generated


@pytest.mark.asyncio
async def test_process_speech_pipeline_and_unique_output(speech_service, monkeypatch):
    generated = patch_successful_pipeline(monkeypatch)

    first = await speech_service.process(object(), "session-123", make_upload())
    second = await speech_service.process(object(), "session-123", make_upload())

    assert first["transcription"] == "recognized words"
    assert first["response"] == "Assistant reply"
    assert first["audio_file"].startswith("outputs/")
    assert first["audio_file"].endswith("_response.mp3")
    assert first["audio_file"] != second["audio_file"]
    assert generated[0][0] == "Assistant reply"
    assert generated[0][1] != generated[1][1]
    assert list(Path(speech_module.settings.UPLOAD_FOLDER).iterdir()) == []


@pytest.mark.asyncio
async def test_process_rejects_unsafe_session_id_before_writing(speech_service):
    upload = make_upload()
    with pytest.raises(ValueError, match="session_id"):
        await speech_service.process(object(), "../../outside", upload)

    assert upload.file.closed
    assert not Path(speech_module.settings.UPLOAD_FOLDER).exists()


@pytest.mark.asyncio
async def test_process_rejects_missing_filename(speech_service):
    upload = make_upload(filename="")
    with pytest.raises(ValueError, match="filename"):
        await speech_service.process(object(), "session-1", upload)
    assert upload.file.closed


@pytest.mark.asyncio
async def test_process_rejects_unsupported_audio_extension(speech_service):
    upload = make_upload(filename="payload.exe")
    with pytest.raises(ValueError, match="supported extension"):
        await speech_service.process(object(), "session-1", upload)
    assert upload.file.closed


@pytest.mark.asyncio
async def test_process_enforces_audio_upload_limit(speech_service):
    payload = b"x" * (speech_module.settings.MAX_AUDIO_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="maximum allowed size"):
        await speech_service.process(object(), "session-1", make_upload(payload))

    upload_dir = Path(speech_module.settings.UPLOAD_FOLDER)
    assert list(upload_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_process_rejects_empty_upload(speech_service):
    with pytest.raises(ValueError, match="empty"):
        await speech_service.process(object(), "session-1", make_upload(payload=b""))


@pytest.mark.asyncio
async def test_process_rejects_invalid_transcription(speech_service, monkeypatch):
    patch_successful_pipeline(monkeypatch, transcription=None)
    with pytest.raises(ValueError, match="transcription"):
        await speech_service.process(object(), "session-1", make_upload())

    assert list(Path(speech_module.settings.UPLOAD_FOLDER).iterdir()) == []


@pytest.mark.asyncio
async def test_process_rejects_empty_transcription(speech_service, monkeypatch):
    patch_successful_pipeline(monkeypatch, transcription="   ")
    with pytest.raises(ValueError, match="transcription"):
        await speech_service.process(object(), "session-1", make_upload())


@pytest.mark.asyncio
async def test_process_propagates_assistant_failure_and_cleans_audio(speech_service, monkeypatch):
    monkeypatch.setattr(
        speech_module.speech_to_text_service,
        "transcribe",
        lambda _path: "recognized words",
    )
    monkeypatch.setattr(
        speech_module.assistant_service,
        "generate_response",
        AsyncMock(side_effect=RuntimeError("assistant failed")),
    )

    with pytest.raises(RuntimeError, match="assistant failed"):
        await speech_service.process(object(), "session-1", make_upload())

    assert list(Path(speech_module.settings.UPLOAD_FOLDER).iterdir()) == []


@pytest.mark.asyncio
async def test_process_rejects_invalid_assistant_response(speech_service, monkeypatch):
    monkeypatch.setattr(
        speech_module.speech_to_text_service,
        "transcribe",
        lambda _path: "recognized words",
    )
    monkeypatch.setattr(
        speech_module.assistant_service,
        "generate_response",
        AsyncMock(return_value={"response": None}),
    )

    with pytest.raises(ValueError, match="Assistant returned"):
        await speech_service.process(object(), "session-1", make_upload())


@pytest.mark.asyncio
async def test_process_rejects_empty_assistant_response(speech_service, monkeypatch):
    patch_successful_pipeline(monkeypatch, response="  ")
    with pytest.raises(ValueError, match="Assistant returned"):
        await speech_service.process(object(), "session-1", make_upload())


@pytest.mark.asyncio
async def test_process_propagates_tts_failure_and_cleans_temp_audio(speech_service, monkeypatch):
    patch_successful_pipeline(monkeypatch)

    async def fail_after_partial_output(text, output_file):
        Path(output_file).write_bytes(b"partial mp3")
        raise RuntimeError("tts failed")

    monkeypatch.setattr(
        speech_module.text_to_speech_service,
        "generate",
        fail_after_partial_output,
    )

    with pytest.raises(RuntimeError, match="tts failed"):
        await speech_service.process(object(), "session-1", make_upload())

    assert list(Path(speech_module.settings.UPLOAD_FOLDER).iterdir()) == []
    assert list(Path(speech_module.settings.OUTPUT_FOLDER).iterdir()) == []


@pytest.mark.asyncio
async def test_process_closes_uploaded_file_on_success(speech_service, monkeypatch):
    patch_successful_pipeline(monkeypatch)
    upload = make_upload()
    await speech_service.process(object(), "session-1", upload)
    assert upload.file.closed


@pytest.mark.asyncio
async def test_tts_test_generates_unique_file(speech_service, monkeypatch):
    paths = []

    async def fake_tts(text, output_file):
        paths.append(output_file)
        return output_file

    monkeypatch.setattr(
        speech_module.text_to_speech_service,
        "generate",
        fake_tts,
    )

    first = await speech_service.tts_test()
    second = await speech_service.tts_test()

    assert first["audio"] != second["audio"]
    assert len(paths) == 2

@pytest.mark.asyncio
async def test_process_returns_text_when_tts_provider_fails(speech_service, monkeypatch):
    patch_successful_pipeline(monkeypatch)

    async def unavailable_tts(text, output_file):
        raise RuntimeError("Edge TTS returned HTTP 403")

    monkeypatch.setattr(
        speech_module.text_to_speech_service,
        "generate",
        unavailable_tts,
    )

    result = await speech_service.process(
        object(),
        "session-tts-fallback",
        make_upload(),
    )

    assert result["transcription"] == "recognized words"
    assert result["response"] == "Assistant reply"
    assert result["audio_file"] is None
    assert "temporarily unavailable" in result["audio_error"]
    assert list(Path(speech_module.settings.UPLOAD_FOLDER).iterdir()) == []
    assert list(Path(speech_module.settings.OUTPUT_FOLDER).iterdir()) == []
