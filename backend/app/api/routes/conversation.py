from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.assistant.conversation_service import (
    conversation_service,
)
from app.config.logging_config import app_logger
from app.database.database import get_db
from app.models.response import ApiResponse


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


# ============================================================
# Get All Sessions
# ============================================================

@router.get(
    "",
    response_model=ApiResponse,
    summary="Get all conversation sessions",
)
async def get_sessions(
    db: Session = Depends(get_db),
) -> ApiResponse:
    """
    Retrieve all conversation sessions.
    """

    app_logger.info(
        "Fetching all conversation sessions."
    )

    try:

        sessions = (
            conversation_service.get_sessions(
                db=db,
            )
        )

        app_logger.info(
            "Conversation sessions retrieved | "
            f"Count={len(sessions)}"
        )

        return ApiResponse(
            success=True,
            message="Sessions retrieved successfully.",
            data={
                "total_sessions": len(sessions),
                "sessions": sessions,
            },
        )

    except Exception:

        app_logger.exception(
            "Failed to fetch conversation sessions."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve sessions.",
        )


# ============================================================
# Conversation Statistics
# ============================================================

@router.get(
    "/statistics",
    response_model=ApiResponse,
    summary="Conversation statistics",
)
async def statistics(
    db: Session = Depends(get_db),
) -> ApiResponse:
    """
    Retrieve conversation statistics.
    """

    app_logger.info(
        "Fetching conversation statistics."
    )

    try:

        statistics_data = (
            conversation_service.get_statistics(
                db=db,
            )
        )

        return ApiResponse(
            success=True,
            message=(
                "Conversation statistics "
                "retrieved successfully."
            ),
            data=statistics_data,
        )

    except Exception:

        app_logger.exception(
            "Failed to fetch conversation statistics."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve statistics.",
        )


# ============================================================
# Delete All Conversations
# ============================================================

@router.delete(
    "",
    response_model=ApiResponse,
    summary="Delete all conversations",
)
async def delete_all(
    db: Session = Depends(get_db),
) -> ApiResponse:
    """
    Delete all conversation messages.
    """

    app_logger.warning(
        "Request received to delete all conversations."
    )

    try:

        deleted = (
            conversation_service.clear_all(
                db=db,
            )
        )

        if not deleted:

            app_logger.info(
                "No conversations existed to delete."
            )

            return ApiResponse(
                success=True,
                message="No conversations found to delete.",
                data={
                    "deleted": False,
                },
            )

        app_logger.warning(
            "All conversations deleted successfully."
        )

        return ApiResponse(
            success=True,
            message=(
                "All conversations deleted "
                "successfully."
            ),
            data={
                "deleted": True,
            },
        )

    except Exception:

        app_logger.exception(
            "Failed to delete all conversations."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to delete conversations.",
        )


# ============================================================
# Get Conversation History
# ============================================================

@router.get(
    "/{session_id}",
    response_model=ApiResponse,
    summary="Get conversation history",
)
async def get_history(
    session_id: str,
    db: Session = Depends(get_db),
) -> ApiResponse:
    """
    Retrieve conversation history for a session.
    """

    session_id = session_id.strip()

    if not session_id:

        raise HTTPException(
            status_code=400,
            detail="Session ID cannot be empty.",
        )

    app_logger.info(
        "Fetching conversation history | "
        f"Session={session_id}"
    )

    try:

        history = (
            conversation_service.get_history(
                db=db,
                session_id=session_id,
            )
        )

        # --------------------------------------------------------
        # Empty list means no conversation exists.
        # --------------------------------------------------------

        if not history:

            app_logger.info(
                "Conversation not found | "
                f"Session={session_id}"
            )

            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        app_logger.info(
            "Conversation history retrieved | "
            f"Session={session_id} | "
            f"Messages={len(history)}"
        )

        return ApiResponse(
            success=True,
            message=(
                "Conversation retrieved "
                "successfully."
            ),
            data={
                "session_id": session_id,
                "messages": history,
            },
        )

    except HTTPException:

        raise

    except Exception:

        app_logger.exception(
            "Failed to retrieve conversation | "
            f"Session={session_id}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve conversation.",
        )


# ============================================================
# Delete Conversation
# ============================================================

@router.delete(
    "/{session_id}",
    response_model=ApiResponse,
    summary="Delete conversation",
)
async def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
) -> ApiResponse:
    """
    Delete a single conversation session.
    """

    session_id = session_id.strip()

    if not session_id:

        raise HTTPException(
            status_code=400,
            detail="Session ID cannot be empty.",
        )

    app_logger.warning(
        "Deleting conversation | "
        f"Session={session_id}"
    )

    try:

        deleted = (
            conversation_service.clear_session(
                db=db,
                session_id=session_id,
            )
        )

        if not deleted:

            app_logger.info(
                "Conversation not found for deletion | "
                f"Session={session_id}"
            )

            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        app_logger.warning(
            "Conversation deleted successfully | "
            f"Session={session_id}"
        )

        return ApiResponse(
            success=True,
            message=(
                "Conversation deleted "
                "successfully."
            ),
            data={
                "session_id": session_id,
                "deleted": True,
            },
        )

    except HTTPException:

        raise

    except Exception:

        app_logger.exception(
            "Failed to delete conversation | "
            f"Session={session_id}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to delete conversation.",
        )