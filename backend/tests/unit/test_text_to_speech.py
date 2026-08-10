from unittest.mock import AsyncMock, MagicMock

import pytest

import app.speech.text_to_speech as tts_module


@pytest.mark.asyncio
async def test_generate_successfully(tmp_path, monkeypatch):

    output_file = tmp_path / "audio" / "test.mp3"

    fake_communicate = MagicMock()

    fake_communicate.save = AsyncMock()

    monkeypatch.setattr(
        tts_module.edge_tts,
        "Communicate",
        MagicMock(
            return_value=fake_communicate
        ),
    )

    service = tts_module.TextToSpeechService()

    result = await service.generate(
        text="Hello world",
        output_file=str(output_file),
    )

    assert result == str(output_file)

    assert output_file.parent.exists()

    tts_module.edge_tts.Communicate.assert_called_once_with(
        text="Hello world",
        voice=tts_module.settings.TTS_VOICE,
    )

    fake_communicate.save.assert_awaited_once_with(
        str(output_file)
    )
    
    
@pytest.mark.asyncio
async def test_generate_propagates_save_error(
    tmp_path,
    monkeypatch,
):

    output_file = tmp_path / "audio" / "test.mp3"

    fake_communicate = MagicMock()

    fake_communicate.save = AsyncMock(
        side_effect=RuntimeError("TTS failed")
    )

    monkeypatch.setattr(
        tts_module.edge_tts,
        "Communicate",
        MagicMock(
            return_value=fake_communicate
        ),
    )

    service = tts_module.TextToSpeechService()

    with pytest.raises(
        RuntimeError,
        match="TTS failed",
    ):
        await service.generate(
            text="Hello",
            output_file=str(output_file),
        )