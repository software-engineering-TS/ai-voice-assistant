from unittest.mock import MagicMock

from app.speech.speech_to_text import SpeechToTextService


def test_transcribe_returns_combined_text(monkeypatch):
    """
    Whisper segments should be combined into one
    cleaned transcription string.
    """

    service = object.__new__(SpeechToTextService)

    segment_1 = MagicMock()
    segment_1.text = " Hello"

    segment_2 = MagicMock()
    segment_2.text = "world "

    model = MagicMock()
    model.transcribe.return_value = (
        [segment_1, segment_2],
        MagicMock(),
    )

    service.model = model

    result = service.transcribe("test.wav")

    assert result == "Hello world"

    model.transcribe.assert_called_once_with(
        "test.wav"
    )