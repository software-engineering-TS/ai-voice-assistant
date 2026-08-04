from sqlalchemy.orm import Session

from app.database.models import Conversation


class ConversationRepository:

    def save_message(
        self,
        db: Session,
        session_id: str,
        role: str,
        message: str,
    ):

        conversation = Conversation(
            session_id=session_id,
            role=role,
            message=message,
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return conversation

    def get_messages(
        self,
        db: Session,
        session_id: str,
    ):

        conversations = (
            db.query(Conversation)
            .filter(
                Conversation.session_id == session_id
            )
            .order_by(
                Conversation.created_at.asc()
            )
            .all()
        )

        return [
            {
                "role": conversation.role,
                "content": conversation.message,
            }
            for conversation in conversations
        ]

    def clear_session(
        self,
        db: Session,
        session_id: str,
    ):

        db.query(Conversation).filter(
            Conversation.session_id == session_id
        ).delete()

        db.commit()


conversation_repository = ConversationRepository()