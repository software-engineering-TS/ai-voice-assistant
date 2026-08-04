from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.assistant.assistant import assistant_service
from app.database.database import get_db
from app.models.chat import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session=Depends(get_db),):
    response = await assistant_service.generate_response(
        db=db,
        session_id=request.session_id,
        user_message=request.message,
    )
    return ChatResponse(response=response)