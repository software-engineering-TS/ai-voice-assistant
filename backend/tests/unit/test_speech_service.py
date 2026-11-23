from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile

import app.services.speech_service as speech_module
from app.services.speech_service import SpeechService


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(speech_module.settings, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setattr(speech_module.settings, "OUTPUT_FOLDER", str(tmp_path / "outputs"))
    monkeypatch.setattr(speech_module.settings, "MAX_AUDIO_UPLOAD_SIZE_MB", 1)
    return SpeechService()


def make_upload(payload=b"audio-bytes", filename="test.wav"):
    return UploadFile(filename=filename, file=BytesIO(payload))


@pytest.mark.asyncio
async def test_process_speech_pipeline_and_unique_output(service, monkeypatch):
    monkeypatch.setattr(
        speech_module.speech_to_text_service,
        "transcribe",
        lambda path: "recognized words",
    )
    monkeypatch.setattr(
        speech_module.assistant_service,
        "generate_response",
        AsyncMock(return_value={"response": "Assistant reply"}),
    )
    generated = []

    async def fake_tts(text, output_file):
        generated.append(output_file)
        return output_file

    monkeypatch.setattr(
        speech_module.text_to_speech_service,
        "generate",
        fake_tts,
    )

    first = await service.process(object(), "session-123", make_upload())
    second = await service.process(object(), "session-123", make_upload())

    assert first["transcription"] == "recognized words"
    assert first["response"] == "Assistant reply"
    assert first["audio_file"].startswith("outputs/")
    assert first["audio_file"].endswith("_response.mp3")
    assert first["audio_file"] != second["audio_file"]
    assert generated[0] != generated[1]
    assert list(Path(speech_module.settings.UPLOAD_FOLDER).iterdir()) == []


@pytest.mark.asyncio
async def test_process_rejects_unsafe_session_id_before_writing(service):
    with pytest.raises(ValueError, match="Session ID"):
        await service.process(object(), "../../outside", make_upload())

    assert not Path(speech_module.settings.UPLOAD_FOLDER).exists()


@pytest.mark.asyncio
async def test_process_rejects_unsupported_audio_extension(service):
    with pytest.raises(ValueError, match="supported extension"):
        await service.process(object(), "session-1", make_upload(filename="payload.exe"))


@pytest.mark.asyncio
async def test_process_enforces_audio_upload_limit(service):
    payload = b"x" * (speech_module.settings.MAX_AUDIO_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="maximum allowed size"):
        await service.process(object(), "session-1", make_upload(payload))

    upload_dir = Path(speech_module.settings.UPLOAD_FOLDER)
    assert list(upload_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_process_removes_temporary_audio_after_transcription_failure(service, monkeypatch):
    monkeypatch.setattr(
        speech_module.speech_to_text_service,
        "transcribe",
        lambda _path: (_ for _ in ()).throw(RuntimeError("transcription failed")),
    )

    with pytest.raises(RuntimeError, match="transcription failed"):
        await service.process(object(), "session-1", make_upload())

    upload_dir = Path(speech_module.settings.UPLOAD_FOLDER)
    assert list(upload_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_process_rejects_empty_upload(service):
    with pytest.raises(ValueError, match="empty"):
        await service.process(object(), "session-1", make_upload(payload=b""))


@pytest.mark.asyncio
async def test_tts_test_generates_unique_file(service, monkeypatch):
    paths = []

    async def fake_tts(text, output_file):
        paths.append(output_file)
        return output_file

    monkeypatch.setattr(speech_module.text_to_speech_service, "generate", fake_tts)

    first = await service.tts_test()
    second = await service.tts_test()

    assert first["audio"] != second["audio"]
    assert len(paths) == 2