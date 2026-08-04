from pathlib import Path
import edge_tts

from app.config.logging_config import app_logger


class TextToSpeechService:

    async def generate(
        self,
        text: str,
        output_file: str,
    ):

        Path(output_file).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        communicate = edge_tts.Communicate(
            text=text,
            voice="en-US-AriaNeural",
        )

        await communicate.save(output_file)

        app_logger.info(
            f"Speech saved to {output_file}"
        )

        return output_file


text_to_speech_service = TextToSpeechService()