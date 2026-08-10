from pathlib import Path
import shutil
import time
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.assistant.assistant import assistant_service
from app.config.logging_config import app_logger
from app.config.settings import settings
from app.speech.speech_to_text import speech_to_text_service
from app.speech.text_to_speech import text_to_speech_service


class SpeechService:

    async def process(
        self,
        db: Session,
        session_id: str,
        file: UploadFile,
    ) -> dict:
        """
        Process an uploaded audio file through the complete
        speech assistant pipeline.

        Flow:
            Audio Upload
                ↓
            Speech-to-Text
                ↓
            Assistant / RAG / LLM
                ↓
            Text-to-Speech
                ↓
            Response

        The uploaded audio file is treated as temporary
        input and removed after processing.
        """

        # --------------------------------------------------------
        # Upload Directory
        # --------------------------------------------------------

        upload_dir = Path("uploads")

        upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Validate Filename
        # --------------------------------------------------------

        original_filename = Path(
            file.filename or ""
        ).name

        if not original_filename:
            raise ValueError(
                "Uploaded audio file must have a filename."
            )

        # --------------------------------------------------------
        # Generate Safe Unique Temporary Filename
        # --------------------------------------------------------

        extension = Path(
            original_filename
        ).suffix.lower()

        temporary_filename = (
            f"{uuid4().hex}{extension}"
        )

        audio_path = (
            upload_dir / temporary_filename
        )

        app_logger.info(
            "Audio upload received | "
            f"Filename={original_filename}"
        )

        try:

            # ----------------------------------------------------
            # Save Uploaded Audio
            # ----------------------------------------------------

            with open(
                audio_path,
                "wb",
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer,
                )

            app_logger.info(
                "Audio file saved temporarily | "
                f"File={temporary_filename}"
            )

            # ----------------------------------------------------
            # Speech To Text
            # ----------------------------------------------------

            start = time.perf_counter()

            transcription = (
                speech_to_text_service.transcribe(
                    audio_path
                )
            )

            app_logger.info(
                "Speech-to-Text completed | "
                f"Time={time.perf_counter() - start:.2f}s"
            )

            # ----------------------------------------------------
            # Validate Transcription
            # ----------------------------------------------------

            if not isinstance(
                transcription,
                str,
            ):

                raise ValueError(
                    "Speech-to-Text returned an invalid result."
                )

            transcription = transcription.strip()

            if not transcription:

                raise ValueError(
                    "Speech-to-Text returned empty transcription."
                )

            # ----------------------------------------------------
            # Assistant
            # ----------------------------------------------------

            start = time.perf_counter()

            assistant_response = (
                await assistant_service.generate_response(
                    db=db,
                    session_id=session_id,
                    user_message=transcription,
                )
            )

            response = assistant_response["response"]

            app_logger.info(
                "Assistant processing completed | "
                f"Time={time.perf_counter() - start:.2f}s"
            )

            # ----------------------------------------------------
            # Validate Assistant Response
            # ----------------------------------------------------

            if not isinstance(
                response,
                str,
            ):

                raise ValueError(
                    "Assistant returned an invalid response."
                )

            response = response.strip()

            if not response:

                raise ValueError(
                    "Assistant returned an empty response."
                )

            # ----------------------------------------------------
            # Text To Speech
            # ----------------------------------------------------

            output_audio = (
                f"{settings.OUTPUT_FOLDER}/"
                f"{session_id}_response.mp3"
            )

            start = time.perf_counter()

            await text_to_speech_service.generate(
                text=response,
                output_file=output_audio,
            )

            app_logger.info(
                "Text-to-Speech completed | "
                f"Time={time.perf_counter() - start:.2f}s"
            )

            # ----------------------------------------------------
            # Final Response
            # ----------------------------------------------------

            return {
                "transcription": transcription,
                "response": response,
                "audio_file": output_audio,
            }

        finally:

            # ----------------------------------------------------
            # Cleanup Temporary Audio
            # ----------------------------------------------------

            if audio_path.exists():

                try:

                    audio_path.unlink()

                    app_logger.info(
                        "Removed temporary audio file | "
                        f"File={temporary_filename}"
                    )

                except OSError:

                    app_logger.exception(
                        "Failed to remove temporary audio file | "
                        f"File={temporary_filename}"
                    )

    async def tts_test(
        self,
    ):

        output = await text_to_speech_service.generate(
            text=(
                "Hello Tarun. "
                "Welcome to your AI Voice Assistant."
            ),
            output_file="outputs/test.mp3",
        )

        return {
            "audio": output,
        }


speech_service = SpeechService()