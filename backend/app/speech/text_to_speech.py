from pathlib import Path

import edge_tts

from app.config.logging_config import app_logger
from app.config.settings import settings


class TextToSpeechService:

    async def generate(
        self,
        text: str,
        output_file: str,
    ) -> str:

        Path(output_file).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        communicate = edge_tts.Communicate(
            text=text,
            voice=settings.TTS_VOICE,
        )

        await communicate.save(output_file)

        app_logger.info(
            f"Generated speech: {output_file}"
        )

        return output_file


text_to_speech_service = TextToSpeechService()