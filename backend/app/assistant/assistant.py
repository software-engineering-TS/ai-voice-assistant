from typing import Any, AsyncGenerator

from sqlalchemy.orm import Session

from app.assistant.memory_service import memory_service
from app.config.logging_config import app_logger
from app.config.settings import settings
from app.database.repository import conversation_repository
from app.llm.ollama_client import ollama_client
from app.prompts.prompt_builder import prompt_builder
from app.rag.retrieval_service import retrieval_service
from app.security import validate_session_id


class AssistantService:
    """
    Main orchestration service for the AI assistant.

    Responsibilities:

    1. Load conversation history.
    2. Retrieve relevant RAG context.
    3. Build the LLM prompt.
    4. Generate the LLM response.
    5. Save conversation messages.
    6. Return response and sources separately.
    7. Support normal and streaming responses.
    """

    def __init__(
        self,
        llm_client,
    ) -> None:

        self.llm_client = llm_client

    # ============================================================
    # Validate Input
    # ============================================================

    @staticmethod
    def _validate_input(
        session_id: str,
        user_message: str,
    ) -> tuple[str, str]:

        if not isinstance(session_id, str):

            raise TypeError(
                "session_id must be a string."
            )

        if not isinstance(user_message, str):

            raise TypeError(
                "user_message must be a string."
            )

        session_id = validate_session_id(session_id)
        user_message = user_message.strip()

        if not user_message:

            raise ValueError(
                "user_message cannot be empty."
            )

        return (
            session_id,
            user_message,
        )

    # ============================================================
    # Conversation History
    # ============================================================

    def _get_conversation_history(
        self,
        db: Session,
        session_id: str,
    ) -> str:
        """
        Retrieve recent conversation history.
        """

        app_logger.info(
            "Loading conversation history | "
            f"Session={session_id}"
        )

        try:

            history = (
                memory_service.get_conversation_history(
                    db=db,
                    session_id=session_id,
                    limit=settings.MAX_HISTORY,
                )
            )

        except Exception:

            app_logger.exception(
                "Failed to load conversation history | "
                f"Session={session_id}"
            )

            raise

        app_logger.info(
            "Conversation history loaded | "
            f"Characters={len(history)}"
        )

        return history

    # ============================================================
    # Retrieve RAG Context
    # ============================================================

    def _retrieve_context(
        self,
        query: str,
        documents: list[str] | None,
    ) -> tuple[str, list[str]]:
        """
        Retrieve relevant RAG context.

        documents=None:
            Search across all indexed documents.

        documents=[...]:
            Search only the selected documents.
        """

        app_logger.info(
            "Starting context retrieval."
        )

        try:

            retrieval = (
                retrieval_service.retrieve(
                    query=query,
                    documents=documents,
                )
            )

        except Exception:

            app_logger.exception(
                "Context retrieval failed."
            )

            # Allow the assistant to fall back
            # to general AI mode.
            return (
                "",
                [],
            )

        context = retrieval.get(
            "context",
            "",
        )

        sources = retrieval.get(
            "sources",
            [],
        )

        if not isinstance(
            context,
            str,
        ):

            context = str(
                context
                or ""
            )

        if not isinstance(
            sources,
            list,
        ):

            sources = []

        # Remove invalid / duplicate sources.
        sources = list(
            dict.fromkeys(
                str(source).strip()
                for source in sources
                if source and str(source).strip()
            )
        )

        if context.strip():

            app_logger.info(
                "RAG context retrieved | "
                f"Characters={len(context)} | "
                f"Sources={len(sources)}"
            )

        else:

            app_logger.info(
                "No relevant RAG context retrieved."
            )

        return (
            context,
            sources,
        )

    # ============================================================
    # Build Prompt
    # ============================================================

    def _build_prompt(
        self,
        history: str,
        question: str,
        context: str,
    ) -> str:
        """
        Build the final prompt sent to the LLM.
        """

        app_logger.info(
            "Building assistant prompt."
        )

        prompt = prompt_builder.build_prompt(
            history=history,
            question=question,
            context=context,
        )

        app_logger.info(
            "Assistant prompt built | "
            f"Characters={len(prompt)}"
        )

        return prompt

    # ============================================================
    # Save Conversation Message
    # ============================================================

    @staticmethod
    def _save_message(
        db: Session,
        session_id: str,
        role: str,
        message: str,
    ) -> None:
        """
        Save a conversation message.
        """

        if not message:

            return

        message = message.strip()

        if not message:

            return

        try:

            conversation_repository.save_message(
                db=db,
                session_id=session_id,
                role=role,
                message=message,
            )

        except Exception:

            app_logger.exception(
                "Failed to save conversation message | "
                f"Session={session_id} | "
                f"Role={role}"
            )

            raise

    # ============================================================
    # Build Response
    # ============================================================

    @staticmethod
    def _build_result(
        response: str,
        sources: list[str],
    ) -> dict[str, Any]:
        """
        Build the service response.

        Sources are intentionally returned separately
        instead of being appended to the AI response.
        """

        response = (
            response.strip()
            if response
            else ""
        )

        sources = list(
            dict.fromkeys(
                source.strip()
                for source in sources
                if isinstance(
                    source,
                    str,
                )
                and source.strip()
            )
        )

        return {
            "response": response,
            "sources": sources,
        }

    # ============================================================
    # Generate Normal Response
    # ============================================================

    async def generate_response(
        self,
        db: Session,
        session_id: str,
        user_message: str,
        documents: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a complete assistant response.

        Returns:

        {
            "response": "...",
            "sources": [...]
        }
        """

        (
            session_id,
            user_message,
        ) = self._validate_input(
            session_id=session_id,
            user_message=user_message,
        )

        app_logger.info(
            "Generating assistant response | "
            f"Session={session_id}"
        )

        # --------------------------------------------------------
        # Load previous history BEFORE saving current question
        # --------------------------------------------------------

        conversation_history = (
            self._get_conversation_history(
                db=db,
                session_id=session_id,
            )
        )

        # --------------------------------------------------------
        # Retrieve RAG Context Only When Documents Are Provided
        # --------------------------------------------------------

        if documents:
            app_logger.info(
                "Document context requested | "
                f"Documents={documents}"
            )

            context, sources = (
                self._retrieve_context(
                    query=user_message,
                    documents=documents,
                )
            )

        else:
            app_logger.info(
                "No documents provided | "
                "Using General AI mode."
            )

            context = ""
            sources = []

        # --------------------------------------------------------
        # Build Prompt
        # --------------------------------------------------------

        prompt = self._build_prompt(
            history=conversation_history,
            question=user_message,
            context=context,
        )

        # --------------------------------------------------------
        # Save User Message
        # --------------------------------------------------------

        self._save_message(
            db=db,
            session_id=session_id,
            role="user",
            message=user_message,
        )

        # --------------------------------------------------------
        # Generate LLM Response
        # --------------------------------------------------------

        try:

            app_logger.info(
                "Sending request to Ollama."
            )

            response = (
                await self.llm_client.generate(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                )
            )

        except Exception:

            app_logger.exception(
                "LLM generation failed | "
                f"Session={session_id}"
            )

            response = (
                "Sorry, I couldn't generate "
                "a response at the moment."
            )

        # --------------------------------------------------------
        # Validate Response
        # --------------------------------------------------------

        if not isinstance(
            response,
            str,
        ):

            app_logger.error(
                "LLM returned an invalid response type."
            )

            response = (
                "Sorry, I couldn't generate "
                "a valid response."
            )

        response = response.strip()

        # --------------------------------------------------------
        # Save ONLY the AI response
        # --------------------------------------------------------

        if response:

            self._save_message(
                db=db,
                session_id=session_id,
                role="assistant",
                message=response,
            )

        # --------------------------------------------------------
        # Build Final Result
        # --------------------------------------------------------

        result = self._build_result(
            response=response,
            sources=sources,
        )

        app_logger.info(
            "Assistant response generated successfully | "
            f"Session={session_id} | "
            f"Characters={len(response)} | "
            f"Sources={len(sources)}"
        )

        return result

    # ============================================================
    # Generate Streaming Response
    # ============================================================

    async def generate_stream_response(
        self,
        db: Session,
        session_id: str,
        user_message: str,
        documents: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming assistant response.

        Tokens are yielded as soon as they are received
        from Ollama.

        The complete response is saved to the database
        after streaming finishes.
        """

        (
            session_id,
            user_message,
        ) = self._validate_input(
            session_id=session_id,
            user_message=user_message,
        )

        app_logger.info(
            "Starting streaming response | "
            f"Session={session_id}"
        )

        # --------------------------------------------------------
        # Load History
        # --------------------------------------------------------

        conversation_history = (
            self._get_conversation_history(
                db=db,
                session_id=session_id,
            )
        )

        # --------------------------------------------------------
        # Retrieve RAG Context Only When Documents Are Provided
        # --------------------------------------------------------
        
        if documents:
            app_logger.info(
                "Document context requested | "
                f"Documents={documents}"
            )
        
            context, sources = (
                self._retrieve_context(
                    query=user_message,
                    documents=documents,
                )
            )
        
        else:
            app_logger.info(
                "No documents provided | "
                "Using General AI mode."
            )
        
            context = ""
            sources = []

        # --------------------------------------------------------
        # Build Prompt
        # --------------------------------------------------------

        prompt = self._build_prompt(
            history=conversation_history,
            question=user_message,
            context=context,
        )

        # --------------------------------------------------------
        # Save User Message
        # --------------------------------------------------------

        self._save_message(
            db=db,
            session_id=session_id,
            role="user",
            message=user_message,
        )

        # --------------------------------------------------------
        # Stream LLM Response
        # --------------------------------------------------------

        response_parts: list[str] = []

        try:

            app_logger.info(
                "Starting Ollama streaming."
            )

            async for token in (
                self.llm_client.stream_generate(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                )
            ):

                if not token:

                    continue

                response_parts.append(
                    token
                )

                yield token

        except Exception:

            app_logger.exception(
                "Streaming generation failed | "
                f"Session={session_id}"
            )

            error_message = (
                "Sorry, I couldn't generate "
                "a response at the moment."
            )

            response_parts = [
                error_message
            ]

            yield error_message

        # --------------------------------------------------------
        # Build Complete Response
        # --------------------------------------------------------

        response = "".join(
            response_parts
        ).strip()

        # --------------------------------------------------------
        # Save ONLY the AI response
        # --------------------------------------------------------

        if response:

            self._save_message(
                db=db,
                session_id=session_id,
                role="assistant",
                message=response,
            )

        # --------------------------------------------------------
        # Sources
        # --------------------------------------------------------
        #
        # IMPORTANT:
        # StreamingResponse currently yields text tokens.
        # Therefore we don't append "Sources:" to the response.
        #
        # The non-stream endpoint returns sources separately.
        #
        # If you later want structured streaming events,
        # we can send sources as a final SSE event.
        # --------------------------------------------------------

        app_logger.info(
            "Streaming response completed | "
            f"Session={session_id} | "
            f"Characters={len(response)} | "
            f"Sources={len(sources)}"
        )


# ============================================================
# Shared Assistant Service
# ============================================================

assistant_service = AssistantService(
    llm_client=ollama_client
)