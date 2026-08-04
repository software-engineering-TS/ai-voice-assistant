from functools import lru_cache

from faster_whisper import WhisperModel

from app.config.logging_config import app_logger


class SpeechToTextService:

    def __init__(self):
        app_logger.info("Loading Whisper Model...")

        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
        )

        app_logger.info("Whisper Loaded Successfully.")

    def transcribe(self, audio_path: str) -> str:

        segments, _ = self.model.transcribe(audio_path)

        text = " ".join(segment.text for segment in segments)

        return text.strip()


@lru_cache
def get_speech_to_text_service():
    return SpeechToTextService()