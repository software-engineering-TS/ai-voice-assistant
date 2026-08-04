from pathlib import Path
import shutil
import time

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.assistant.assistant import assistant_service
from app.config.settings import settings
from app.database.database import get_db
from app.speech.speech_to_text import speech_to_text_service
from app.speech.text_to_speech import text_to_speech_service

from app.config.logging_config import app_logger

router = APIRouter(
    prefix="/speech",
    tags=["Speech"],
)


@router.post("/chat")
async def speech_chat(
    session_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    audio_path = upload_dir / file.filename

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    stt = speech_to_text_service

    start = time.perf_counter()

    text = stt.transcribe(audio_path)

    app_logger.info(
        f"Speech-to-Text took {time.perf_counter() - start:.2f} seconds"
    )

    start = time.perf_counter()

    response = await assistant_service.generate_response(
        db=db,
        session_id=session_id,
        user_message=text,
    )
    app_logger.info(
    f"Assistant took {time.perf_counter() - start:.2f} seconds"
)

    output_audio = (
        f"{settings.OUTPUT_FOLDER}/{session_id}_response.mp3"
    )

    start = time.perf_counter()

    await text_to_speech_service.generate(
    text=response,
    output_file=output_audio,
    )

    app_logger.info(
    f"Text-to-Speech took {time.perf_counter() - start:.2f} seconds"
)

    return {
        "transcription": text,
        "response": response,
        "audio_file": output_audio,
    }

from app.speech.text_to_speech import text_to_speech_service


@router.get("/tts-test")
async def tts_test():

    output = await text_to_speech_service.generate(
        text="Hello Tarun. Welcome to your AI Voice Assistant.",
        output_file="outputs/test.mp3",
    )

    return {
        "audio": output
    }