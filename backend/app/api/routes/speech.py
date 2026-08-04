from pathlib import Path
import shutil

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.assistant.assistant import assistant_service
from app.config.settings import settings
from app.database.database import get_db
from app.speech.speech_to_text import get_speech_to_text_service
from app.speech.text_to_speech import text_to_speech_service

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

    stt = get_speech_to_text_service()

    text = stt.transcribe(str(audio_path))

    response = await assistant_service.generate_response(
        db=db,
        session_id=session_id,
        user_message=text,
    )

    output_audio = (
    f"{settings.OUTPUT_FOLDER}/{session_id}_response.mp3"
)
    await text_to_speech_service.generate(
    text=response,
    output_file=output_audio,
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