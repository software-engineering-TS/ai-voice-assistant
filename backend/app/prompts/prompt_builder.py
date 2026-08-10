from app.config.logging_config import app_logger
from app.prompts.system_prompts import (
    GENERAL_AI_PROMPT,
    RAG_PROMPT,
)


class PromptBuilder:
    """
    Builds prompts for the AI assistant.

    Supports two modes:

    1. General AI mode
       - Uses conversation history.
       - Uses the LLM's general knowledge.

    2. RAG mode
       - Uses conversation history.
       - Uses retrieved document context.
       - Grounds document-related answers in retrieved context.
    """

    # ============================================================
    # Build Prompt
    # ============================================================

    @staticmethod
    def build_prompt(
        history: str,
        question: str,
        context: str = "",
    ) -> str:
        """
        Build the appropriate prompt based on whether
        retrieved document context is available.

        Parameters:
            history:
                Previous conversation history.

            question:
                Current user question.

            context:
                Retrieved document context.

        Returns:
            Fully constructed prompt string.
        """

        # --------------------------------------------------------
        # Normalize Inputs
        # --------------------------------------------------------

        history = (
            history.strip()
            if history
            else ""
        )

        question = (
            question.strip()
            if question
            else ""
        )

        context = (
            context.strip()
            if context
            else ""
        )

        # --------------------------------------------------------
        # Validate Question
        # --------------------------------------------------------

        if not question:

            raise ValueError(
                "Question cannot be empty."
            )

        # --------------------------------------------------------
        # Default Conversation History
        # --------------------------------------------------------

        if not history:

            history = (
                "No previous conversation."
            )

        # ========================================================
        # RAG MODE
        # ========================================================

        if context:

            app_logger.info(
                "Building RAG prompt."
            )

            return (
                f"{RAG_PROMPT.strip()}\n\n"

                "================ "
                "CONVERSATION HISTORY "
                "================\n\n"

                f"{history}\n\n"

                "================ "
                "RETRIEVED DOCUMENT CONTEXT "
                "================\n\n"

                f"{context}\n\n"

                "================ "
                "CURRENT QUESTION "
                "================\n\n"

                f"{question}\n\n"

                "================ "
                "ASSISTANT ANSWER "
                "================\n\n"
            )

        # ========================================================
        # GENERAL AI MODE
        # ========================================================

        app_logger.info(
            "Building General AI prompt."
        )

        return (
            f"{GENERAL_AI_PROMPT.strip()}\n\n"

            "================ "
            "CONVERSATION HISTORY "
            "================\n\n"

            f"{history}\n\n"

            "================ "
            "CURRENT QUESTION "
            "================\n\n"

            f"{question}\n\n"

            "================ "
            "ASSISTANT ANSWER "
            "================\n\n"
        )


# ============================================================
# Shared Prompt Builder
# ============================================================

prompt_builder = PromptBuilder()