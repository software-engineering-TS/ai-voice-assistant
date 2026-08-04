from pathlib import Path
import shutil

from sqlalchemy.orm import Session

from app.assistant.assistant import assistant_service
from app.speech.speech_to_text import get_speech_to_text_service
from app.speech.text_to_speech import text_to_speech_service


class VoiceService:

    async def process_voice(
        self,
        db: Session,
        session_id: str,
        file,
    ):

        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        audio_path = upload_dir / file.filename

        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        stt = get_speech_to_text_service()

        transcription = stt.transcribe(str(audio_path))

        response = await assistant_service.generate_response(
            db=db,
            session_id=session_id,
            user_message=transcription,
        )

        output_audio = Path("outputs")
        output_audio.mkdir(exist_ok=True)

        output_file = output_audio / f"{session_id}.mp3"

        await text_to_speech_service.generate(
            text=response,
            output_file=str(output_file),
        )

        return {
            "transcription": transcription,
            "response": response,
            "audio_file": str(output_file),
        }


voice_service = VoiceService()