from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.database.database import Base


class Conversation(Base):

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(String, index=True)

    role = Column(String)

    message = Column(Text)

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    def __repr__(self):
        return (
        f"Conversation("
        f"id={self.id}, "
        f"session_id='{self.session_id}', "
        f"role='{self.role}'"
        f")"
    )