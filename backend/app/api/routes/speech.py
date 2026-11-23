from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.response import ApiResponse
from app.services.speech_service import speech_service
from app.security import SESSION_ID_PATTERN

router = APIRouter(
    prefix="/speech",
    tags=["Speech"],
)


@router.post(
    "/chat",
    response_model=ApiResponse,
)
async def speech_chat(
    session_id: str = Query(
        ...,
        min_length=1,
        max_length=100,
        pattern=SESSION_ID_PATTERN,
    ),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    try:
        result = await speech_service.process(
            db=db,
            session_id=session_id,
            file=file,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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