from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config.logging_config import app_logger
from app.database.models import Conversation


class ConversationRepository:
    """
    Repository responsible for database operations related
    to conversation messages.
    """

    # ============================================================
    # Save Message
    # ============================================================

    def save_message(
        self,
        db: Session,
        session_id: str,
        role: str,
        message: str,
    ) -> Conversation:
        """
        Save a single conversation message.
        """

        app_logger.info(
            f"Saving {role} message | Session={session_id}"
        )

        try:
            conversation = Conversation(
                session_id=session_id,
                role=role,
                message=message,
            )

            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            app_logger.info(
                f"Message saved successfully | "
                f"Session={session_id} | "
                f"MessageID={conversation.id}"
            )

            return conversation

        except SQLAlchemyError:

            db.rollback()

            app_logger.exception(
                f"Failed to save conversation | "
                f"Session={session_id}"
            )

            raise

    # ============================================================
    # Get Messages
    # ============================================================

    def get_messages(
        self,
        db: Session,
        session_id: str,
    ) -> list[dict]:
        """
        Retrieve conversation history for a session.
        """

        app_logger.info(
            f"Loading messages | Session={session_id}"
        )

        try:
            statement = (
                select(Conversation)
                .where(
                    Conversation.session_id == session_id
                )
                .order_by(
                    Conversation.created_at.asc(),
                    Conversation.id.asc(),
                )
            )

            conversations = (
                db.execute(statement)
                .scalars()
                .all()
            )

            app_logger.info(
                f"Retrieved {len(conversations)} messages | "
                f"Session={session_id}"
            )

            return [
                {
                    "role": conversation.role,
                    "content": conversation.message,
                }
                for conversation in conversations
            ]

        except SQLAlchemyError:

            app_logger.exception(
                f"Failed to load conversation | "
                f"Session={session_id}"
            )

            raise

    # ============================================================
    # Get Sessions
    # ============================================================

    def get_sessions(
        self,
        db: Session,
    ) -> list[dict]:
        """
        Retrieve all unique conversation sessions.
        """

        app_logger.info(
            "Loading conversation sessions."
        )

        try:
            statement = (
                select(Conversation.session_id)
                .distinct()
                .order_by(
                    Conversation.session_id.asc()
                )
            )

            sessions = (
                db.execute(statement)
                .scalars()
                .all()
            )

            app_logger.info(
                f"Retrieved {len(sessions)} sessions."
            )

            return [
                {
                    "session_id": session_id,
                }
                for session_id in sessions
            ]

        except SQLAlchemyError:

            app_logger.exception(
                "Failed to load conversation sessions."
            )

            raise

    # ============================================================
    # Statistics
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

        try:
            total_messages = db.scalar(
                select(
                    func.count(Conversation.id)
                )
            ) or 0

            total_sessions = db.scalar(
                select(
                    func.count(
                        func.distinct(
                            Conversation.session_id
                        )
                    )
                )
            ) or 0

            statistics = {
                "messages": total_messages,
                "sessions": total_sessions,
            }

            app_logger.info(
                f"Conversation statistics: {statistics}"
            )

            return statistics

        except SQLAlchemyError:

            app_logger.exception(
                "Failed to load conversation statistics."
            )

            raise

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

        app_logger.info(
            f"Deleting conversation session | "
            f"Session={session_id}"
        )

        try:
            statement = delete(Conversation).where(
                Conversation.session_id == session_id
            )

            result = db.execute(statement)

            deleted = result.rowcount or 0

            db.commit()

            app_logger.info(
                f"Deleted {deleted} messages | "
                f"Session={session_id}"
            )

            return deleted

        except SQLAlchemyError:

            db.rollback()

            app_logger.exception(
                f"Failed to delete session | "
                f"Session={session_id}"
            )

            raise

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
            "Deleting ALL conversation messages."
        )

        try:
            statement = delete(Conversation)

            result = db.execute(statement)

            deleted = result.rowcount or 0

            db.commit()

            app_logger.warning(
                f"Deleted {deleted} conversation messages."
            )

            return deleted

        except SQLAlchemyError:

            db.rollback()

            app_logger.exception(
                "Failed to delete all conversations."
            )

            raise


# ============================================================
# Repository Instance
# ============================================================

conversation_repository = ConversationRepository()