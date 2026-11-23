"""Lazy Faster-Whisper speech-to-text integration."""

from app.config.logging_config import app_logger
from app.config.settings import settings


class SpeechToTextService:
    def __init__(self, lazy: bool = False):
        self.model = None
        self._lazy = lazy
        if not self._lazy:
            self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """Load Whisper on first actual transcription, not at module import."""
        if self.model is not None:
            return

        from faster_whisper import WhisperModel

        app_logger.info("Loading Whisper model | Model=%s", settings.WHISPER_MODEL)
        self.model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        app_logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        self.ensure_loaded()
        segments, _ = self.model.transcribe(audio_path)
        return " ".join(segment.text for segment in segments).strip()


speech_to_text_service = SpeechToTextService(lazy=True)