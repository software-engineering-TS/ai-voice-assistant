from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


class Conversation(Base):
    """
    Stores conversation messages for each chat session.
    """

    __tablename__ = "conversations"

    # ============================================================
    # Primary Key
    # ============================================================

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ============================================================
    # Session
    # ============================================================

    session_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    # ============================================================
    # Message Role
    # ============================================================

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # ============================================================
    # Message Content
    # ============================================================

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # ============================================================
    # Timestamp
    # ============================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # ============================================================
    # Indexes
    # ============================================================

    __table_args__ = (
        Index(
            "ix_conversations_session_created",
            "session_id",
            "created_at",
        ),
    )

    # ============================================================
    # Representation
    # ============================================================

    def __repr__(self) -> str:
        return (
            "Conversation("
            f"id={self.id}, "
            f"session_id='{self.session_id}', "
            f"role='{self.role}', "
            f"created_at='{self.created_at}'"
            ")"
        )