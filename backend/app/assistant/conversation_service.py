from sqlalchemy.orm import Session

from app.config.logging_config import app_logger
from app.database.repository import conversation_repository


class ConversationService:
    """
    Business/service layer for conversation management.

    This layer coordinates conversation operations between
    API routes and the conversation repository.
    """

    # ============================================================
    # Get Conversation History
    # ============================================================

    def get_history(
        self,
        db: Session,
        session_id: str,
    ) -> list[dict]:
        """
        Retrieve conversation history for a session.
        """

        app_logger.info(
            f"Loading conversation history | "
            f"Session={session_id}"
        )

        return conversation_repository.get_messages(
            db=db,
            session_id=session_id,
        )

    # ============================================================
    # Get Sessions
    # ============================================================

    def get_sessions(
        self,
        db: Session,
    ) -> list[dict]:
        """
        Retrieve all conversation sessions.
        """

        app_logger.info(
            "Loading all conversation sessions."
        )

        return conversation_repository.get_sessions(
            db=db,
        )

    # ============================================================
    # Get Statistics
    # ============================================================

    def get_statistics(
        self,
        db: Session,
    ) -> dict:
        """
        Retrieve conversation statistics.
        """

        app_logger.info(
            "Loading conversation statistics."
        )

        return conversation_repository.get_statistics(
            db=db,
        )

    # ============================================================
    # Delete One Session
    # ============================================================

    def clear_session(
        self,
        db: Session,
        session_id: str,
    ) -> int:
        """
        Delete all messages belonging to one session.

        Returns:
            Number of deleted messages.
        """

        app_logger.warning(
            f"Deleting conversation | "
            f"Session={session_id}"
        )

        deleted = conversation_repository.clear_session(
            db=db,
            session_id=session_id,
        )

        app_logger.info(
            f"Conversation deletion completed | "
            f"Session={session_id} | "
            f"Deleted={deleted}"
        )

        return deleted

    # ============================================================
    # Delete All Conversations
    # ============================================================

    def clear_all(
        self,
        db: Session,
    ) -> int:
        """
        Delete all conversation messages.

        Returns:
            Number of deleted messages.
        """

        app_logger.warning(
            "Deleting all conversations."
        )

        deleted = conversation_repository.clear_all(
            db=db,
        )

        app_logger.info(
            f"All conversation deletion completed | "
            f"Deleted={deleted}"
        )

        return deleted


# ============================================================
# Service Instance
# ============================================================

conversation_service = ConversationService()