from functools import lru_cache

from app.config.settings import settings
from faster_whisper import WhisperModel

from app.config.logging_config import app_logger


class SpeechToTextService:

    def __init__(self):
        app_logger.info("Loading Whisper Model...")

        self.model = WhisperModel(
    settings.WHISPER_MODEL,
    device=settings.WHISPER_DEVICE,
    compute_type=settings.WHISPER_COMPUTE_TYPE,
)

        app_logger.info("Whisper Loaded Successfully.")

    def transcribe(self, audio_path: str) -> str:

        segments, _ = self.model.transcribe(audio_path)

        text = " ".join(segment.text for segment in segments)

        return text.strip()


speech_to_text_service = SpeechToTextService()