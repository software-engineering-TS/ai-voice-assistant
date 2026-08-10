from unittest.mock import AsyncMock

import pytest

import app.api.routes.speech as speech_module


@pytest.mark.asyncio
async def test_speech_chat_successfully():

    expected = {
        "transcription": "Hello",
        "response": "Hi there",
        "audio_file": "outputs/test.mp3",
    }

    speech_module.speech_service.process = AsyncMock(
        return_value=expected
    )

    db = object()
    file = object()

    result = await speech_module.speech_chat(
        session_id="session123",
        file=file,
        db=db,
    )

    assert result.success is True

    assert (
        result.message
        == "Speech processed successfully."
    )

    assert result.data == expected

    speech_module.speech_service.process.assert_awaited_once_with(
        db=db,
        session_id="session123",
        file=file,
    )


@pytest.mark.asyncio
async def test_tts_test_successfully():

    expected = {
        "audio": "outputs/test.mp3",
    }

    speech_module.speech_service.tts_test = AsyncMock(
        return_value=expected
    )

    result = await speech_module.tts_test()

    assert result.success is True

    assert (
        result.message
        == "TTS test completed successfully."
    )

    assert result.data == expected

    speech_module.speech_service.tts_test.assert_awaited_once_with()