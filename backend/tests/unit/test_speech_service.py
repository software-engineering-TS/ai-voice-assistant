from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile

from app.services.speech_service import SpeechService


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def speech_service():
    return SpeechService()


@pytest.fixture
def upload_file():
    return UploadFile(
        filename="test_audio.wav",
        file=BytesIO(b"fake audio content"),
    )


# ============================================================
# process()
# ============================================================

@pytest.mark.asyncio
async def test_process_successfully(
    speech_service,
    upload_file,
):
    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ) as mock_mkdir,
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ) as mock_copy,
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            return_value="Hello assistant",
        ) as mock_transcribe,
        patch(
            "app.services.speech_service.assistant_service.generate_response",
            new=AsyncMock(
                return_value={
    "response": "Hello! How can I help?",
    "sources": [],
}
            ),
        ) as mock_assistant,
        patch(
            "app.services.speech_service.text_to_speech_service.generate",
            new=AsyncMock(
                return_value="outputs/test.mp3"
            ),
        ) as mock_tts,
    ):
        mock_open.return_value.__enter__.return_value = MagicMock()

        result = await speech_service.process(
            db=db,
            session_id="test-session",
            file=upload_file,
        )

    mock_mkdir.assert_called_once_with(
    parents=True,
    exist_ok=True,
)

    mock_copy.assert_called_once()

    mock_transcribe.assert_called_once()

    mock_assistant.assert_awaited_once_with(
        db=db,
        session_id="test-session",
        user_message="Hello assistant",
    )

    mock_tts.assert_awaited_once()

    assert result["transcription"] == "Hello assistant"
    assert result["response"] == "Hello! How can I help?"
    assert result["audio_file"].endswith(
        "test-session_response.mp3"
    )


# ============================================================
# process() - Verify TTS output path
# ============================================================

@pytest.mark.asyncio
async def test_process_uses_session_id_for_output_file(
    speech_service,
    upload_file,
):
    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ),
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ),
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            return_value="Test transcription",
        ),
        patch(
            "app.services.speech_service.assistant_service.generate_response",
            new=AsyncMock(
                return_value={
    "response": "Test response",
    "sources": [],
}
            ),
        ),
        patch(
            "app.services.speech_service.text_to_speech_service.generate",
            new=AsyncMock(
                return_value="audio.mp3"
            ),
        ) as mock_tts,
    ):
        mock_open.return_value.__enter__.return_value = MagicMock()

        result = await speech_service.process(
            db=db,
            session_id="session-123",
            file=upload_file,
        )

    mock_tts.assert_awaited_once_with(
        text="Test response",
        output_file="outputs/session-123_response.mp3",
    )

    assert result["audio_file"] == (
        "outputs/session-123_response.mp3"
    )


# ============================================================
# process() - Transcription failure
# ============================================================

@pytest.mark.asyncio
async def test_process_propagates_transcription_error(
    speech_service,
    upload_file,
):
    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ),
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ),
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            side_effect=RuntimeError(
                "Transcription failed"
            ),
        ),
    ):
        mock_open.return_value.__enter__.return_value = MagicMock()

        with pytest.raises(RuntimeError, match="Transcription failed"):
            await speech_service.process(
                db=db,
                session_id="session",
                file=upload_file,
            )


# ============================================================
# process() - Assistant failure
# ============================================================

@pytest.mark.asyncio
async def test_process_propagates_assistant_error(
    speech_service,
    upload_file,
):
    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ),
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ),
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            return_value="Hello",
        ),
        patch(
            "app.services.speech_service.assistant_service.generate_response",
            new=AsyncMock(
                side_effect=RuntimeError(
                    "Assistant failed"
                )
            ),
        ),
    ):
        mock_open.return_value.__enter__.return_value = MagicMock()

        with pytest.raises(RuntimeError, match="Assistant failed"):
            await speech_service.process(
                db=db,
                session_id="session",
                file=upload_file,
            )


# ============================================================
# process() - TTS failure
# ============================================================

@pytest.mark.asyncio
async def test_process_propagates_tts_error(
    speech_service,
    upload_file,
):
    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ),
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ),
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            return_value="Hello",
        ),
        patch(
            "app.services.speech_service.assistant_service.generate_response",
            new=AsyncMock(
                return_value={
    "response": "Hi",
    "sources": [],
}
            ),
        ),
        patch(
            "app.services.speech_service.text_to_speech_service.generate",
            new=AsyncMock(
                side_effect=RuntimeError(
                    "TTS failed"
                )
            ),
        ),
    ):
        mock_open.return_value.__enter__.return_value = MagicMock()

        with pytest.raises(RuntimeError, match="TTS failed"):
            await speech_service.process(
                db=db,
                session_id="session",
                file=upload_file,
            )


# ============================================================
# tts_test()
# ============================================================

@pytest.mark.asyncio
async def test_tts_test_successfully(
    speech_service,
):
    with patch(
        "app.services.speech_service.text_to_speech_service.generate",
        new=AsyncMock(
            return_value="outputs/test.mp3"
        ),
    ) as mock_tts:

        result = await speech_service.tts_test()

    mock_tts.assert_awaited_once_with(
        text=(
            "Hello Tarun. "
            "Welcome to your AI Voice Assistant."
        ),
        output_file="outputs/test.mp3",
    )

    assert result == {
        "audio": "outputs/test.mp3",
    }


# ============================================================
# tts_test() - Failure
# ============================================================

@pytest.mark.asyncio
async def test_tts_test_propagates_error(
    speech_service,
):
    with patch(
        "app.services.speech_service.text_to_speech_service.generate",
        new=AsyncMock(
            side_effect=RuntimeError(
                "TTS generation failed"
            )
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="TTS generation failed",
        ):
            await speech_service.tts_test()
            
@pytest.mark.asyncio
async def test_process_rejects_missing_filename(
    speech_service,
):
    """
    Uploaded audio without a filename should be rejected.
    """

    file = UploadFile(
        filename="",
        file=BytesIO(
            b"audio content"
        ),
    )

    db = MagicMock()

    with pytest.raises(
        ValueError,
        match="Uploaded audio file must have a filename",
    ):
        await speech_service.process(
            db=db,
            session_id="test-session",
            file=file,
        )
        
@pytest.mark.asyncio
async def test_process_rejects_invalid_transcription(
    speech_service,
    upload_file,
):
    """
    A non-string Speech-to-Text result should be rejected.
    """

    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ),
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ),
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            return_value=None,
        ),
    ):

        mock_open.return_value.__enter__.return_value = (
            MagicMock()
        )

        with pytest.raises(
            ValueError,
            match="Speech-to-Text returned an invalid result",
        ):
            await speech_service.process(
                db=db,
                session_id="test-session",
                file=upload_file,
            )
            
@pytest.mark.asyncio
async def test_process_rejects_empty_transcription(
    speech_service,
    upload_file,
):
    """
    Empty Speech-to-Text results should be rejected.
    """

    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ),
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ),
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            return_value="   ",
        ),
    ):
        mock_open.return_value.__enter__.return_value = (
            MagicMock()
        )

        with pytest.raises(
            ValueError,
            match="Speech-to-Text returned empty transcription",
        ):
            await speech_service.process(
                db=db,
                session_id="test-session",
                file=upload_file,
            )
            
@pytest.mark.asyncio
async def test_process_rejects_invalid_assistant_response(
    speech_service,
    upload_file,
):
    """
    A non-string assistant response should be rejected.
    """

    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ),
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ),
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            return_value="Hello assistant",
        ),
        patch(
            "app.services.speech_service.assistant_service.generate_response",
            new=AsyncMock(
                return_value={
                    "response": None,
                    "sources": [],
                }
            ),
        ),
    ):
        mock_open.return_value.__enter__.return_value = (
            MagicMock()
        )

        with pytest.raises(
            ValueError,
            match="Assistant returned an invalid response",
        ):
            await speech_service.process(
                db=db,
                session_id="test-session",
                file=upload_file,
            )
            
            
@pytest.mark.asyncio
async def test_process_rejects_empty_assistant_response(
    speech_service,
    upload_file,
):
    """
    An empty assistant response should be rejected.
    """

    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ),
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ),
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            return_value="Hello assistant",
        ),
        patch(
            "app.services.speech_service.assistant_service.generate_response",
            new=AsyncMock(
                return_value={
                    "response": "   ",
                    "sources": [],
                }
            ),
        ),
    ):
        mock_open.return_value.__enter__.return_value = (
            MagicMock()
        )

        with pytest.raises(
            ValueError,
            match="Assistant returned an empty response",
        ):
            await speech_service.process(
                db=db,
                session_id="test-session",
                file=upload_file,
            )
            
@pytest.mark.asyncio
async def test_process_handles_audio_cleanup_failure(
    speech_service,
    upload_file,
):
    """
    Cleanup failures should be logged without replacing
    the successful speech-processing result.
    """

    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ),
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ),
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            return_value="Hello assistant",
        ),
        patch(
            "app.services.speech_service.assistant_service.generate_response",
            new=AsyncMock(
                return_value={
                    "response": "Hello!",
                    "sources": [],
                }
            ),
        ),
        patch(
            "app.services.speech_service.text_to_speech_service.generate",
            new=AsyncMock(
                return_value="outputs/test.mp3"
            ),
        ),
        patch(
            "app.services.speech_service.Path.exists",
            return_value=True,
        ),
        patch(
            "app.services.speech_service.Path.unlink",
            side_effect=OSError("Permission denied"),
        ),
        patch(
            "app.services.speech_service.app_logger.exception"
        ) as mock_logger_exception,
    ):
        mock_open.return_value.__enter__.return_value = (
            MagicMock()
        )

        result = await speech_service.process(
            db=db,
            session_id="test-session",
            file=upload_file,
        )

        assert result["transcription"] == "Hello assistant"
        assert result["response"] == "Hello!"
        mock_logger_exception.assert_called_once_with(
            mock_logger_exception.call_args.args[0]
        )
        
@pytest.mark.asyncio
async def test_process_removes_temporary_audio_file(
    speech_service,
    upload_file,
):
    """
    Successfully remove the temporary audio file
    after speech processing completes.
    """

    db = MagicMock()

    with (
        patch(
            "app.services.speech_service.Path.mkdir"
        ),
        patch(
            "app.services.speech_service.open",
            create=True,
        ) as mock_open,
        patch(
            "app.services.speech_service.shutil.copyfileobj"
        ),
        patch(
            "app.services.speech_service.speech_to_text_service.transcribe",
            return_value="Hello assistant",
        ),
        patch(
            "app.services.speech_service.assistant_service.generate_response",
            new=AsyncMock(
                return_value={
                    "response": "Hello!",
                    "sources": [],
                },
            ),
        ),
        patch(
            "app.services.speech_service.text_to_speech_service.generate",
            new=AsyncMock(
                return_value="outputs/test.mp3",
            ),
        ),
        patch(
            "app.services.speech_service.Path.exists",
            return_value=True,
        ),
        patch(
            "app.services.speech_service.Path.unlink",
        ) as mock_unlink,
        patch(
            "app.services.speech_service.app_logger.info",
        ) as mock_logger_info,
    ):
        mock_open.return_value.__enter__.return_value = (
            MagicMock()
        )

        result = await speech_service.process(
            db=db,
            session_id="test-session",
            file=upload_file,
        )

        assert result["transcription"] == "Hello assistant"
        assert result["response"] == "Hello!"

        mock_unlink.assert_called_once()

        assert any(
            "Removed temporary audio file"
            in str(call)
            for call in mock_logger_info.call_args_list
        )