import pytest

from app.prompts.prompt_builder import (
    PromptBuilder,
    prompt_builder,
)
from app.prompts.system_prompts import (
    GENERAL_AI_PROMPT,
    RAG_PROMPT,
)


# ============================================================
# Initialization
# ============================================================


def test_prompt_builder_initializes_successfully():
    builder = PromptBuilder()

    assert isinstance(
        builder,
        PromptBuilder,
    )


def test_shared_prompt_builder_is_initialized():
    assert isinstance(
        prompt_builder,
        PromptBuilder,
    )


# ============================================================
# General AI Prompt
# ============================================================


def test_build_prompt_creates_general_ai_prompt():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="How are you?",
    )

    assert (
        GENERAL_AI_PROMPT.strip()
        in result
    )

    assert "User: Hello" in result
    assert "How are you?" in result


def test_general_ai_prompt_does_not_contain_rag_context_section():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="How are you?",
    )

    assert (
        "RETRIEVED DOCUMENT CONTEXT"
        not in result
    )


def test_general_ai_prompt_contains_conversation_section():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="How are you?",
    )

    assert (
        "CONVERSATION HISTORY"
        in result
    )


def test_general_ai_prompt_contains_question_section():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="How are you?",
    )

    assert (
        "CURRENT QUESTION"
        in result
    )


def test_general_ai_prompt_contains_answer_section():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="How are you?",
    )

    assert (
        "ASSISTANT ANSWER"
        in result
    )


# ============================================================
# RAG Prompt
# ============================================================


def test_build_prompt_creates_rag_prompt():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="What does the document say?",
        context="The document contains information about AI.",
    )

    assert (
        RAG_PROMPT.strip()
        in result
    )

    assert "User: Hello" in result

    assert (
        "What does the document say?"
        in result
    )

    assert (
        "The document contains information about AI."
        in result
    )


def test_rag_prompt_contains_document_context_section():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="What does the document say?",
        context="Document content.",
    )

    assert (
        "RETRIEVED DOCUMENT CONTEXT"
        in result
    )


def test_rag_prompt_contains_conversation_section():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="What does the document say?",
        context="Document content.",
    )

    assert (
        "CONVERSATION HISTORY"
        in result
    )


def test_rag_prompt_contains_question_section():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="What does the document say?",
        context="Document content.",
    )

    assert (
        "CURRENT QUESTION"
        in result
    )


def test_rag_prompt_contains_answer_section():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="What does the document say?",
        context="Document content.",
    )

    assert (
        "ASSISTANT ANSWER"
        in result
    )


# ============================================================
# Input Normalization
# ============================================================


def test_question_whitespace_is_stripped():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="   What is AI?   ",
    )

    assert (
        "What is AI?"
        in result
    )

    assert (
        "   What is AI?   "
        not in result
    )


def test_history_whitespace_is_stripped():
    result = PromptBuilder.build_prompt(
        history="   User: Hello   ",
        question="What is AI?",
    )

    assert "User: Hello" in result

    assert (
        "   User: Hello   "
        not in result
    )


def test_context_whitespace_is_stripped():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="What is AI?",
        context="   Document context   ",
    )

    assert (
        "Document context"
        in result
    )

    assert (
        "   Document context   "
        not in result
    )


# ============================================================
# Empty History
# ============================================================


def test_empty_history_uses_default_history():
    result = PromptBuilder.build_prompt(
        history="",
        question="What is AI?",
    )

    assert (
        "No previous conversation."
        in result
    )


def test_whitespace_history_uses_default_history():
    result = PromptBuilder.build_prompt(
        history="   ",
        question="What is AI?",
    )

    assert (
        "No previous conversation."
        in result
    )


def test_none_history_uses_default_history():
    result = PromptBuilder.build_prompt(
        history=None,
        question="What is AI?",
    )

    assert (
        "No previous conversation."
        in result
    )


# ============================================================
# Empty Context
# ============================================================


def test_empty_context_uses_general_ai_mode():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="What is AI?",
        context="",
    )

    assert (
        GENERAL_AI_PROMPT.strip()
        in result
    )

    assert (
        RAG_PROMPT.strip()
        not in result
    )


def test_whitespace_context_uses_general_ai_mode():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="What is AI?",
        context="   ",
    )

    assert (
        GENERAL_AI_PROMPT.strip()
        in result
    )

    assert (
        "RETRIEVED DOCUMENT CONTEXT"
        not in result
    )


def test_none_context_uses_general_ai_mode():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="What is AI?",
        context=None,
    )

    assert (
        GENERAL_AI_PROMPT.strip()
        in result
    )


# ============================================================
# Empty Question
# ============================================================


def test_empty_question_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        PromptBuilder.build_prompt(
            history="User: Hello",
            question="",
        )


def test_whitespace_question_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        PromptBuilder.build_prompt(
            history="User: Hello",
            question="   ",
        )


def test_none_question_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        PromptBuilder.build_prompt(
            history="User: Hello",
            question=None,
        )


# ============================================================
# RAG Mode Selection
# ============================================================


def test_non_empty_context_selects_rag_mode():
    result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="Explain this.",
        context="Retrieved information.",
    )

    assert (
        RAG_PROMPT.strip()
        in result
    )

    assert (
        GENERAL_AI_PROMPT.strip()
        not in result
    )


def test_context_is_included_only_in_rag_mode():
    rag_result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="Explain this.",
        context="Retrieved information.",
    )

    general_result = PromptBuilder.build_prompt(
        history="User: Hello",
        question="Explain this.",
    )

    assert (
        "Retrieved information."
        in rag_result
    )

    assert (
        "Retrieved information."
        not in general_result
    )


# ============================================================
# Prompt Content Ordering
# ============================================================


def test_general_prompt_sections_are_in_correct_order():
    result = PromptBuilder.build_prompt(
        history="Conversation",
        question="Question",
    )

    system_position = result.index(
        GENERAL_AI_PROMPT.strip()
    )

    history_position = result.index(
        "CONVERSATION HISTORY"
    )

    question_position = result.index(
        "CURRENT QUESTION"
    )

    answer_position = result.index(
        "ASSISTANT ANSWER"
    )

    assert system_position < history_position
    assert history_position < question_position
    assert question_position < answer_position


def test_rag_prompt_sections_are_in_correct_order():
    result = PromptBuilder.build_prompt(
        history="Conversation",
        question="Question",
        context="Context",
    )

    system_position = result.index(
        RAG_PROMPT.strip()
    )

    history_position = result.index(
        "CONVERSATION HISTORY"
    )

    context_position = result.index(
        "RETRIEVED DOCUMENT CONTEXT"
    )

    question_position = result.index(
        "CURRENT QUESTION"
    )

    answer_position = result.index(
        "ASSISTANT ANSWER"
    )

    assert system_position < history_position
    assert history_position < context_position
    assert context_position < question_position
    assert question_position < answer_position


# ============================================================
# Complete Prompt Content
# ============================================================


def test_general_prompt_contains_all_required_content():
    result = PromptBuilder.build_prompt(
        history="User: Tell me about Python.",
        question="What is FastAPI?",
    )

    assert (
        GENERAL_AI_PROMPT.strip()
        in result
    )

    assert (
        "User: Tell me about Python."
        in result
    )

    assert (
        "What is FastAPI?"
        in result
    )

    assert (
        "CONVERSATION HISTORY"
        in result
    )

    assert (
        "CURRENT QUESTION"
        in result
    )

    assert (
        "ASSISTANT ANSWER"
        in result
    )


def test_rag_prompt_contains_all_required_content():
    result = PromptBuilder.build_prompt(
        history="User: Tell me about Python.",
        question="What does the document say?",
        context="FastAPI is a Python web framework.",
    )

    assert (
        RAG_PROMPT.strip()
        in result
    )

    assert (
        "User: Tell me about Python."
        in result
    )

    assert (
        "FastAPI is a Python web framework."
        in result
    )

    assert (
        "What does the document say?"
        in result
    )

    assert (
        "CONVERSATION HISTORY"
        in result
    )

    assert (
        "RETRIEVED DOCUMENT CONTEXT"
        in result
    )

    assert (
        "CURRENT QUESTION"
        in result
    )

    assert (
        "ASSISTANT ANSWER"
        in result
    )