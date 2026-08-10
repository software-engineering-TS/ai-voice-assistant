from sqlalchemy.orm import Session

from app.config.logging_config import app_logger
from app.database.repository import conversation_repository


class MemoryService:
    """
    Service responsible for preparing conversation history
    for the AI assistant.
    """

    # ============================================================
    # Get Conversation History
    # ============================================================

    def get_conversation_history(
        self,
        db: Session,
        session_id: str,
        limit: int = 6,
    ) -> str:
        """
        Retrieve the most recent conversation messages and
        format them for the LLM.

        `limit` represents the maximum number of messages,
        including both user and assistant messages.
        """

        if limit <= 0:
            app_logger.warning(
                f"Invalid memory limit={limit}. "
                f"Returning empty history."
            )

            return ""

        app_logger.info(
            f"Loading conversation memory | "
            f"Session={session_id} | "
            f"Limit={limit}"
        )

        try:

            messages = conversation_repository.get_messages(
                db=db,
                session_id=session_id,
            )

            if not messages:

                app_logger.info(
                    f"No conversation history found | "
                    f"Session={session_id}"
                )

                return ""

            # ----------------------------------------------------
            # Keep only the most recent messages
            # ----------------------------------------------------

            recent_messages = messages[-limit:]

            history = []

            for message in recent_messages:

                role = message.get(
                    "role",
                    "unknown",
                )

                content = message.get(
                    "content",
                    "",
                ).strip()

                if not content:
                    continue

                if role == "user":

                    history.append(
                        f"User: {content}"
                    )

                elif role == "assistant":

                    history.append(
                        f"Assistant: {content}"
                    )

                else:

                    app_logger.warning(
                        f"Unknown conversation role: {role}"
                    )

            conversation_history = "\n".join(
                history
            )

            app_logger.info(
                f"Loaded {len(history)} memory messages | "
                f"Session={session_id}"
            )

            return conversation_history

        except Exception:

            app_logger.exception(
                f"Failed to load conversation memory | "
                f"Session={session_id}"
            )

            raise


# ============================================================
# Service Instance
# ============================================================

memory_service = MemoryService()