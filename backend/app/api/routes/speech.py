from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.response import ApiResponse
from app.services.speech_service import speech_service

router = APIRouter(
    prefix="/speech",
    tags=["Speech"],
)


@router.post(
    "/chat",
    response_model=ApiResponse,
)
async def speech_chat(
    session_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    result = await speech_service.process(
        db=db,
        session_id=session_id,
        file=file,
    )

    return ApiResponse(
        success=True,
        message="Speech processed successfully.",
        data=result,
    )


@router.get(
    "/tts-test",
    response_model=ApiResponse,
)
async def tts_test():

    result = await speech_service.tts_test()

    return ApiResponse(
        success=True,
        message="TTS test completed successfully.",
        data=result,
    )