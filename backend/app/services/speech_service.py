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
        """Process one audio request and always close/remove temporary input."""
        audio_path: Path | None = None
        output_path: Path | None = None
        completed = False

        try:
            session_id = validate_session_id(session_id)

            original_name = (file.filename or "").strip()
            if not original_name:
                raise ValueError("Uploaded audio file must have a filename.")

            # The client filename is used only for extension validation/logging.
            # It is never used as a storage path.
            original_filename = Path(original_name).name
            extension = Path(original_filename).suffix.lower()
            if extension not in self.ALLOWED_AUDIO_EXTENSIONS:
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
            audio_path = upload_dir / f"{request_id}{extension}"
            output_filename = f"{request_id}_response.mp3"
            output_path = output_dir / output_filename

            app_logger.info(
                "Audio upload received | "
                f"Filename={original_filename} | Session={session_id}"
            )

            total_bytes = 0
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
                "Speech-to-Text completed | "
                f"Time={time.perf_counter() - started:.2f}s"
            )

            if not isinstance(transcription, str) or not transcription.strip():
                raise ValueError(
                    "Speech-to-Text returned an empty or invalid transcription."
                )
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

            audio_file = None
            audio_error = None
            started = time.perf_counter()
            try:
                await text_to_speech_service.generate(
                    text=response,
                    output_file=str(output_path),
                )
                app_logger.info(
                    "Text-to-Speech completed | "
                    f"Time={time.perf_counter() - started:.2f}s"
                )
                # StaticFiles is mounted at /outputs; return a URL path,
                # never the local filesystem path configured by the operator.
                audio_file = f"outputs/{output_filename}"
            except Exception:
                # Transcription and LLM response are already complete and
                # persisted. An external TTS provider outage must not turn a
                # successful voice conversation into an HTTP 500 response.
                app_logger.exception(
                    "Text-to-Speech unavailable; returning text response."
                )
                audio_error = (
                    "Speech playback is temporarily unavailable. "
                    "Your voice message was transcribed and answered."
                )
                try:
                    if output_path.exists():
                        output_path.unlink()
                except OSError:
                    app_logger.exception(
                        "Failed to remove incomplete TTS output | "
                        f"File={output_path}"
                    )

            completed = True
            return {
                "transcription": transcription,
                "response": response,
                "audio_file": audio_file,
                "audio_error": audio_error,
            }
        finally:
            if not completed and output_path is not None:
                try:
                    if output_path.exists():
                        output_path.unlink()
                except OSError:
                    app_logger.exception(
                        "Failed to remove incomplete audio output | "
                        f"File={output_path}"
                    )

            if audio_path is not None:
                try:
                    if audio_path.exists():
                        audio_path.unlink()
                except OSError:
                    app_logger.exception(
                        "Failed to remove temporary audio file | "
                        f"File={audio_path}"
                    )

            await file.close()

    async def tts_test(self) -> dict:
        output_dir = Path(settings.OUTPUT_FOLDER)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"tts-test-{uuid4().hex}.mp3"
        output_path = output_dir / output_filename
        await text_to_speech_service.generate(
            text="Hello. Welcome to your AI Voice Assistant.",
            output_file=str(output_path),
        )
        return {"audio": f"outputs/{output_filename}"}


speech_service = SpeechService()