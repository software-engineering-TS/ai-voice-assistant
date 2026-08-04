from sqlalchemy.orm import Session

from app.config.logging_config import app_logger
from app.database.repository import conversation_repository
from app.llm.ollama_client import ollama_client


class AssistantService:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def generate_response(
        self,
        db: Session,
        session_id: str,
        user_message: str,
    ):

        app_logger.info("Saving user message.")

        conversation_repository.save_message(
            db=db,
            session_id=session_id,
            role="user",
            message=user_message,
        )

        messages = conversation_repository.get_messages(
            db=db,
            session_id=session_id,
        )

        response = await self.llm_client.generate(messages)

        conversation_repository.save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            message=response,
        )

        return response

assistant_service = AssistantService(llm_client=ollama_client) 