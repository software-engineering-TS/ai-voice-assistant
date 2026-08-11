import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.assistant.assistant import assistant_service
from app.config.logging_config import app_logger
from app.database.database import get_db
from app.models.chat import ChatRequest, ChatResponse
from app.models.response import ApiResponse


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# ============================================================
# Normal Chat
# ============================================================

@router.post(
    "",
    response_model=ApiResponse,
    summary="Generate AI response",
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Generate a complete AI response.

    Supports:
    - Conversation history
    - General AI responses
    - RAG document selection
    """

    start_time = time.perf_counter()

    app_logger.info(
        "Chat request received | "
        f"Session={request.session_id}"
    )

    try:

        result = (
            await assistant_service.generate_response(
                db=db,
                session_id=request.session_id,
                user_message=request.message,
                documents=request.documents,
            )
        )

        processing_time = (
            time.perf_counter()
            - start_time
        )

        response = result.get(
            "response",
            "",
        )

        sources = result.get(
            "sources",
            [],
        )

        app_logger.info(
            "Chat completed | "
            f"Session={request.session_id} | "
            f"ProcessingTime={processing_time:.3f}s | "
            f"Sources={len(sources)}"
        )

        return ApiResponse(
            success=True,
            message="Response generated successfully.",
            data=ChatResponse(
                response=response,
                sources=sources,
                processing_time=round(
                    processing_time,
                    3,
                ),
            ),
        )

    except ValueError as exc:

        app_logger.warning(
            "Invalid chat request | "
            f"Session={request.session_id} | "
            f"Error={exc}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:

        app_logger.exception(
            "Chat request failed | "
            f"Session={request.session_id}"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )


# ============================================================
# Streaming Chat
# ============================================================

@router.post(
    "/stream",
    summary="Stream AI response",
)
async def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Stream an AI response using Server-Sent Events.

    The assistant service handles:
    - Conversation history
    - RAG retrieval
    - Prompt construction
    - Ollama streaming
    - Conversation persistence
    """

    app_logger.info(
        "Streaming chat request received | "
        f"Session={request.session_id}"
    )

    async def response_generator():
        """
        Wrap the assistant streaming generator so that
        errors occurring during streaming are logged.
        """
    
        try:
        
            async for token in (
                assistant_service.generate_stream_response(
                    db=db,
                    session_id=request.session_id,
                    user_message=request.message,
                    documents=request.documents,
                )
            ):
    
                if token:
                
                    yield f"data: {token}\n\n"
    
        except ValueError as exc:
        
            app_logger.warning(
                "Invalid streaming request | "
                f"Session={request.session_id} | "
                f"Error={exc}"
            )
    
            yield (
                "data: "
                "Error: "
                f"{str(exc)}"
                "\n\n"
            )
    
        except Exception:
        
            app_logger.exception(
                "Streaming chat failed | "
                f"Session={request.session_id}"
            )
    
            yield (
                "data: "
                "Error: Unable to generate the response."
                "\n\n"
            )

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )