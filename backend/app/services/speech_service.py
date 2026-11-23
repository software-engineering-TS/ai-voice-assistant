"""Speech upload orchestration with bounded, uniquely named temporary files."""

from pathlib import Path
import time
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.assistant.assistant import assistant_service
from app.config.logging_config import app_logger
from app.config.settings import settings
from app.security import validate_session_id
from app.speech.speech_to_text import speech_to_text_service
from app.speech.text_to_speech import text_to_speech_service


class SpeechService:
    """Handle audio upload, transcription, response generation and TTS."""

    ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}
    UPLOAD_CHUNK_SIZE = 1024 * 1024

    async def process(
        self,
        db: Session,
        session_id: str,
        file: UploadFile,
    ) -> dict:
        session_id = validate_session_id(session_id)

        original_filename = Path(file.filename or "").name
        extension = Path(original_filename).suffix.lower()
        if not original_filename or extension not in self.ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                "Audio filename must use a supported extension: "
                + ", ".join(sorted(self.ALLOWED_AUDIO_EXTENSIONS))
            )

        max_bytes = settings.MAX_AUDIO_UPLOAD_SIZE_MB * 1024 * 1024
        upload_dir = Path(settings.UPLOAD_FOLDER)
        output_dir = Path(settings.OUTPUT_FOLDER)
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        request_id = uuid4().hex
        temporary_filename = f"{request_id}{extension}"
        audio_path = upload_dir / temporary_filename
        output_filename = f"{request_id}_response.mp3"
        output_path = output_dir / output_filename

        app_logger.info(
            f"Audio upload received | Filename={original_filename} | Session={session_id}"
        )

        total_bytes = 0
        try:
            with audio_path.open("wb") as buffer:
                while True:
                    chunk = await file.read(self.UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise ValueError(
                            "Audio upload exceeds the maximum allowed size "
                            f"of {settings.MAX_AUDIO_UPLOAD_SIZE_MB} MB."
                        )
                    buffer.write(chunk)

            if total_bytes == 0:
                raise ValueError("Uploaded audio file is empty.")

            started = time.perf_counter()
            transcription = speech_to_text_service.transcribe(str(audio_path))
            app_logger.info(
                f"Speech-to-Text completed | Time={time.perf_counter() - started:.2f}s"
            )

            if not isinstance(transcription, str) or not transcription.strip():
                raise ValueError("Speech-to-Text returned an empty or invalid transcription.")
            transcription = transcription.strip()

            assistant_response = await assistant_service.generate_response(
                db=db,
                session_id=session_id,
                user_message=transcription,
            )
            response = assistant_response.get("response")
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Assistant returned an empty or invalid response.")
            response = response.strip()

            started = time.perf_counter()
            await text_to_speech_service.generate(
                text=response,
                output_file=str(output_path),
            )
            app_logger.info(
                f"Text-to-Speech completed | Time={time.perf_counter() - started:.2f}s"
            )

            return {
                "transcription": transcription,
                "response": response,
                "audio_file": f"outputs/{output_filename}",
            }
        finally:
            try:
                if audio_path.exists():
                    audio_path.unlink()
            except OSError:
                app_logger.exception(f"Failed to remove temporary audio file {audio_path}")
            await file.close()

    async def tts_test(self) -> dict:
        output_dir = Path(settings.OUTPUT_FOLDER)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"tts-test-{uuid4().hex}.mp3"
        output = await text_to_speech_service.generate(
            text="Hello. Welcome to your AI Voice Assistant.",
            output_file=str(output_path),
        )
        return {"audio": output}


speech_service = SpeechService()