from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

import app.assistant.assistant as assistant_module


# ============================================================
# Fake Database
# ============================================================


class FakeDatabase:
    pass


# ============================================================
# Fake LLM Client
# ============================================================


class FakeLLMClient:
    def __init__(self):
        self.generate = AsyncMock(
            return_value="AI generated response."
        )

        self.stream_tokens = [
            "Hello",
            " ",
            "from",
            " ",
            "AI",
            ".",
        ]

    async def _stream_generate(self, messages):
        for token in self.stream_tokens:
            yield token

    def stream_generate(self, messages):
        return self._stream_generate(messages)


# ============================================================
# Service Factory
# ============================================================


def create_service(monkeypatch):
    llm_client = FakeLLMClient()

    service = (
        assistant_module.AssistantService(
            llm_client=llm_client
        )
    )

    return service, llm_client


# ============================================================
# Validation
# ============================================================


def test_validate_input_accepts_valid_values():
    result = (
        assistant_module.AssistantService._validate_input(
            session_id="session-123",
            user_message="Hello AI",
        )
    )

    assert result == (
        "session-123",
        "Hello AI",
    )


def test_validate_input_strips_session_id():
    result = (
        assistant_module.AssistantService._validate_input(
            session_id="  session-123  ",
            user_message="Hello",
        )
    )

    assert result == (
        "session-123",
        "Hello",
    )


def test_validate_input_strips_user_message():
    result = (
        assistant_module.AssistantService._validate_input(
            session_id="session-123",
            user_message="  Hello AI  ",
        )
    )

    assert result == (
        "session-123",
        "Hello AI",
    )


def test_validate_input_rejects_non_string_session_id():
    with pytest.raises(
        TypeError,
        match="session_id must be a string",
    ):
        assistant_module.AssistantService._validate_input(
            session_id=123,
            user_message="Hello",
        )


def test_validate_input_rejects_non_string_user_message():
    with pytest.raises(
        TypeError,
        match="user_message must be a string",
    ):
        assistant_module.AssistantService._validate_input(
            session_id="session-123",
            user_message=123,
        )


def test_validate_input_rejects_empty_session_id():
    with pytest.raises(
        ValueError,
        match="session_id cannot be empty",
    ):
        assistant_module.AssistantService._validate_input(
            session_id="",
            user_message="Hello",
        )


def test_validate_input_rejects_whitespace_session_id():
    with pytest.raises(
        ValueError,
        match="session_id cannot be empty",
    ):
        assistant_module.AssistantService._validate_input(
            session_id="   ",
            user_message="Hello",
        )


def test_validate_input_rejects_empty_user_message():
    with pytest.raises(
        ValueError,
        match="user_message cannot be empty",
    ):
        assistant_module.AssistantService._validate_input(
            session_id="session-123",
            user_message="",
        )


def test_validate_input_rejects_whitespace_user_message():
    with pytest.raises(
        ValueError,
        match="user_message cannot be empty",
    ):
        assistant_module.AssistantService._validate_input(
            session_id="session-123",
            user_message="   ",
        )


# ============================================================
# Conversation History
# ============================================================


def test_get_conversation_history_calls_memory_service(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    memory_mock = MagicMock()

    memory_mock.get_conversation_history.return_value = (
        "User: Hello"
    )

    monkeypatch.setattr(
        assistant_module,
        "memory_service",
        memory_mock,
    )

    db = FakeDatabase()

    result = service._get_conversation_history(
        db=db,
        session_id="session-123",
    )

    assert result == "User: Hello"

    memory_mock.get_conversation_history.assert_called_once_with(
        db=db,
        session_id="session-123",
        limit=assistant_module.settings.MAX_HISTORY,
    )


def test_get_conversation_history_propagates_failure(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    memory_mock = MagicMock()

    memory_mock.get_conversation_history.side_effect = (
        RuntimeError("Memory failure")
    )

    monkeypatch.setattr(
        assistant_module,
        "memory_service",
        memory_mock,
    )

    with pytest.raises(
        RuntimeError,
        match="Memory failure",
    ):
        service._get_conversation_history(
            db=FakeDatabase(),
            session_id="session-123",
        )


# ============================================================
# RAG Retrieval
# ============================================================


def test_retrieve_context_calls_retrieval_service(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    retrieval_mock = MagicMock()

    retrieval_mock.retrieve.return_value = {
        "context": "Relevant document content.",
        "sources": [
            "document.pdf",
        ],
    }

    monkeypatch.setattr(
        assistant_module,
        "retrieval_service",
        retrieval_mock,
    )

    result = service._retrieve_context(
        query="What is AI?",
        documents=None,
    )

    assert result == (
        "Relevant document content.",
        ["document.pdf"],
    )

    retrieval_mock.retrieve.assert_called_once_with(
        query="What is AI?",
        documents=None,
    )


def test_retrieve_context_passes_selected_documents(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    retrieval_mock = MagicMock()

    retrieval_mock.retrieve.return_value = {
        "context": "Document context.",
        "sources": [
            "file1.pdf",
        ],
    }

    monkeypatch.setattr(
        assistant_module,
        "retrieval_service",
        retrieval_mock,
    )

    documents = [
        "file1.pdf",
        "file2.pdf",
    ]

    service._retrieve_context(
        query="Question",
        documents=documents,
    )

    retrieval_mock.retrieve.assert_called_once_with(
        query="Question",
        documents=documents,
    )


def test_retrieve_context_returns_empty_when_retrieval_fails(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    retrieval_mock = MagicMock()

    retrieval_mock.retrieve.side_effect = (
        RuntimeError("Retrieval failed")
    )

    monkeypatch.setattr(
        assistant_module,
        "retrieval_service",
        retrieval_mock,
    )

    result = service._retrieve_context(
        query="Question",
        documents=None,
    )

    assert result == (
        "",
        [],
    )


def test_retrieve_context_handles_missing_context(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    retrieval_mock = MagicMock()

    retrieval_mock.retrieve.return_value = {
        "sources": [
            "document.pdf",
        ],
    }

    monkeypatch.setattr(
        assistant_module,
        "retrieval_service",
        retrieval_mock,
    )

    result = service._retrieve_context(
        query="Question",
        documents=None,
    )

    assert result == (
        "",
        ["document.pdf"],
    )


def test_retrieve_context_handles_non_string_context(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    retrieval_mock = MagicMock()

    retrieval_mock.retrieve.return_value = {
        "context": 123,
        "sources": [],
    }

    monkeypatch.setattr(
        assistant_module,
        "retrieval_service",
        retrieval_mock,
    )

    result = service._retrieve_context(
        query="Question",
        documents=None,
    )

    assert result == (
        "123",
        [],
    )


def test_retrieve_context_handles_invalid_sources(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    retrieval_mock = MagicMock()

    retrieval_mock.retrieve.return_value = {
        "context": "Context",
        "sources": "invalid",
    }

    monkeypatch.setattr(
        assistant_module,
        "retrieval_service",
        retrieval_mock,
    )

    result = service._retrieve_context(
        query="Question",
        documents=None,
    )

    assert result == (
        "Context",
        [],
    )


def test_retrieve_context_removes_duplicate_sources(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    retrieval_mock = MagicMock()

    retrieval_mock.retrieve.return_value = {
        "context": "Context",
        "sources": [
            "document.pdf",
            "document.pdf",
            "other.pdf",
            "document.pdf",
        ],
    }

    monkeypatch.setattr(
        assistant_module,
        "retrieval_service",
        retrieval_mock,
    )

    result = service._retrieve_context(
        query="Question",
        documents=None,
    )

    assert result == (
        "Context",
        [
            "document.pdf",
            "other.pdf",
        ],
    )


def test_retrieve_context_removes_empty_sources(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    retrieval_mock = MagicMock()

    retrieval_mock.retrieve.return_value = {
        "context": "Context",
        "sources": [
            "",
            "  ",
            "document.pdf",
        ],
    }

    monkeypatch.setattr(
        assistant_module,
        "retrieval_service",
        retrieval_mock,
    )

    result = service._retrieve_context(
        query="Question",
        documents=None,
    )

    assert result == (
        "Context",
        ["document.pdf"],
    )


# ============================================================
# Prompt Building
# ============================================================


def test_build_prompt_calls_prompt_builder(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    prompt_mock = MagicMock()

    prompt_mock.build_prompt.return_value = (
        "Final assistant prompt"
    )

    monkeypatch.setattr(
        assistant_module,
        "prompt_builder",
        prompt_mock,
    )

    result = service._build_prompt(
        history="User: Hello",
        question="What is AI?",
        context="Document context",
    )

    assert result == (
        "Final assistant prompt"
    )

    prompt_mock.build_prompt.assert_called_once_with(
        history="User: Hello",
        question="What is AI?",
        context="Document context",
    )


# ============================================================
# Save Message
# ============================================================


def test_save_message_saves_valid_message(
    monkeypatch,
):
    repository_mock = MagicMock()

    monkeypatch.setattr(
        assistant_module,
        "conversation_repository",
        repository_mock,
    )

    db = FakeDatabase()

    assistant_module.AssistantService._save_message(
        db=db,
        session_id="session-123",
        role="user",
        message="Hello",
    )

    repository_mock.save_message.assert_called_once_with(
        db=db,
        session_id="session-123",
        role="user",
        message="Hello",
    )


def test_save_message_strips_message(
    monkeypatch,
):
    repository_mock = MagicMock()

    monkeypatch.setattr(
        assistant_module,
        "conversation_repository",
        repository_mock,
    )

    assistant_module.AssistantService._save_message(
        db=FakeDatabase(),
        session_id="session-123",
        role="user",
        message="   Hello   ",
    )

    repository_mock.save_message.assert_called_once_with(
        db=repository_mock.save_message.call_args.kwargs["db"],
        session_id="session-123",
        role="user",
        message="Hello",
    )


def test_save_message_ignores_empty_message(
    monkeypatch,
):
    repository_mock = MagicMock()

    monkeypatch.setattr(
        assistant_module,
        "conversation_repository",
        repository_mock,
    )

    assistant_module.AssistantService._save_message(
        db=FakeDatabase(),
        session_id="session-123",
        role="user",
        message="",
    )

    repository_mock.save_message.assert_not_called()


def test_save_message_ignores_whitespace_message(
    monkeypatch,
):
    repository_mock = MagicMock()

    monkeypatch.setattr(
        assistant_module,
        "conversation_repository",
        repository_mock,
    )

    assistant_module.AssistantService._save_message(
        db=FakeDatabase(),
        session_id="session-123",
        role="user",
        message="   ",
    )

    repository_mock.save_message.assert_not_called()


def test_save_message_propagates_repository_failure(
    monkeypatch,
):
    repository_mock = MagicMock()

    repository_mock.save_message.side_effect = (
        RuntimeError("Database failure")
    )

    monkeypatch.setattr(
        assistant_module,
        "conversation_repository",
        repository_mock,
    )

    with pytest.raises(
        RuntimeError,
        match="Database failure",
    ):
        assistant_module.AssistantService._save_message(
            db=FakeDatabase(),
            session_id="session-123",
            role="user",
            message="Hello",
        )


# ============================================================
# Build Result
# ============================================================


def test_build_result_returns_response_and_sources():
    result = (
        assistant_module.AssistantService._build_result(
            response="Hello AI",
            sources=["document.pdf"],
        )
    )

    assert result == {
        "response": "Hello AI",
        "sources": ["document.pdf"],
    }


def test_build_result_strips_response():
    result = (
        assistant_module.AssistantService._build_result(
            response="   Hello AI   ",
            sources=[],
        )
    )

    assert result["response"] == (
        "Hello AI"
    )


def test_build_result_removes_duplicate_sources():
    result = (
        assistant_module.AssistantService._build_result(
            response="Answer",
            sources=[
                "file.pdf",
                "file.pdf",
                "other.pdf",
            ],
        )
    )

    assert result["sources"] == [
        "file.pdf",
        "other.pdf",
    ]


def test_build_result_removes_invalid_sources():
    result = (
        assistant_module.AssistantService._build_result(
            response="Answer",
            sources=[
                "file.pdf",
                "",
                "   ",
                123,
                None,
            ],
        )
    )

    assert result["sources"] == [
        "file.pdf",
    ]


def test_build_result_handles_empty_response():
    result = (
        assistant_module.AssistantService._build_result(
            response="",
            sources=[],
        )
    )

    assert result == {
        "response": "",
        "sources": [],
    }


# ============================================================
# Generate Normal Response
# ============================================================


@pytest.mark.asyncio
async def test_generate_response_successfully(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(
            return_value="User: Previous question"
        ),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        MagicMock(
            return_value=(
                "Relevant context",
                ["document.pdf"],
            )
        ),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(
            return_value="Final prompt",
        ),
    )

    save_mock = MagicMock()

    monkeypatch.setattr(
        service,
        "_save_message",
        save_mock,
    )

    llm_client.generate.return_value = (
        "AI answer"
    )

    result = await service.generate_response(
        db=FakeDatabase(),
        session_id="session-123",
        user_message="Hello",
        documents=["document.pdf"],
    )

    assert result == {
        "response": "AI answer",
        "sources": ["document.pdf"],
    }

    llm_client.generate.assert_awaited_once_with(
        messages=[
            {
                "role": "user",
                "content": "Final prompt",
            }
        ]
    )


def test_generate_response_calls_dependencies_in_order(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    calls = []

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        lambda **kwargs: (
            calls.append("history")
            or "History"
        ),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        lambda **kwargs: (
            calls.append("retrieval")
            or ("Context", ["doc.pdf"])
        ),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        lambda **kwargs: (
            calls.append("prompt")
            or "Prompt"
        ),
    )

    monkeypatch.setattr(
        service,
        "_save_message",
        lambda **kwargs: calls.append(
            f"save:{kwargs['role']}"
        ),
    )

    async def generate(**kwargs):
        calls.append("llm")
        return "Response"

    llm_client.generate = AsyncMock(
        side_effect=generate
    )

    import asyncio

    asyncio.run(
        service.generate_response(
            db=FakeDatabase(),
            session_id="session-123",
            user_message="Hello",
            documents=["doc.pdf"],
        )
    )

    assert calls == [
        "history",
        "retrieval",
        "prompt",
        "save:user",
        "llm",
        "save:assistant",
    ]


@pytest.mark.asyncio
async def test_generate_response_saves_user_message_before_llm(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    calls = []

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(
            return_value=""
        ),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        MagicMock(
            return_value=("", []),
        ),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(
            return_value="Prompt",
        ),
    )

    def save_message(**kwargs):
        calls.append(
            kwargs["role"]
        )

    monkeypatch.setattr(
        service,
        "_save_message",
        save_message,
    )

    async def generate(**kwargs):
        calls.append("llm")
        return "Answer"

    llm_client.generate = AsyncMock(
        side_effect=generate
    )

    await service.generate_response(
        db=FakeDatabase(),
        session_id="session-123",
        user_message="Hello",
    )

    assert calls == [
        "user",
        "llm",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_generate_response_passes_documents_to_retrieval(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(return_value=""),
    )

    retrieval_mock = MagicMock(
        return_value=("", [])
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        retrieval_mock,
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(return_value="Prompt"),
    )

    monkeypatch.setattr(
        service,
        "_save_message",
        MagicMock(),
    )

    await service.generate_response(
        db=FakeDatabase(),
        session_id="session-123",
        user_message="Question",
        documents=[
            "document.pdf",
        ],
    )

    retrieval_mock.assert_called_once_with(
        query="Question",
        documents=["document.pdf"],
    )


@pytest.mark.asyncio
async def test_generate_response_handles_llm_failure(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(return_value=""),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        MagicMock(return_value=("", [])),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(return_value="Prompt"),
    )

    save_mock = MagicMock()

    monkeypatch.setattr(
        service,
        "_save_message",
        save_mock,
    )

    llm_client.generate.side_effect = (
        RuntimeError("Ollama unavailable")
    )

    result = await service.generate_response(
        db=FakeDatabase(),
        session_id="session-123",
        user_message="Hello",
    )

    assert result["response"] == (
        "Sorry, I couldn't generate "
        "a response at the moment."
    )

    assert result["sources"] == []


@pytest.mark.asyncio
async def test_generate_response_handles_invalid_llm_response_type(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(return_value=""),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        MagicMock(return_value=("", [])),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(return_value="Prompt"),
    )

    save_mock = MagicMock()

    monkeypatch.setattr(
        service,
        "_save_message",
        save_mock,
    )

    llm_client.generate.return_value = 12345

    result = await service.generate_response(
        db=FakeDatabase(),
        session_id="session-123",
        user_message="Hello",
    )

    assert result["response"] == (
        "Sorry, I couldn't generate "
        "a valid response."
    )


@pytest.mark.asyncio
async def test_generate_response_strips_llm_response(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(return_value=""),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        MagicMock(return_value=("", [])),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(return_value="Prompt"),
    )

    monkeypatch.setattr(
        service,
        "_save_message",
        MagicMock(),
    )

    llm_client.generate.return_value = (
        "   AI response   "
    )

    result = await service.generate_response(
        db=FakeDatabase(),
        session_id="session-123",
        user_message="Hello",
    )

    assert result["response"] == (
        "AI response"
    )


@pytest.mark.asyncio
async def test_generate_response_rejects_invalid_input(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="session_id cannot be empty",
    ):
        await service.generate_response(
            db=FakeDatabase(),
            session_id="",
            user_message="Hello",
        )


@pytest.mark.asyncio
async def test_generate_response_does_not_call_llm_for_invalid_input(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    with pytest.raises(ValueError):
        await service.generate_response(
            db=FakeDatabase(),
            session_id="session-123",
            user_message="   ",
        )

    llm_client.generate.assert_not_awaited()


# ============================================================
# Streaming Response
# ============================================================


async def collect_stream(generator):
    tokens = []

    async for token in generator:
        tokens.append(token)

    return tokens


@pytest.mark.asyncio
async def test_generate_stream_response_successfully(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(
            return_value="Previous conversation"
        ),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        MagicMock(
            return_value=(
                "Context",
                ["document.pdf"],
            )
        ),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(
            return_value="Prompt",
        ),
    )

    save_mock = MagicMock()

    monkeypatch.setattr(
        service,
        "_save_message",
        save_mock,
    )

    tokens = await collect_stream(
        service.generate_stream_response(
            db=FakeDatabase(),
            session_id="session-123",
            user_message="Hello",
        )
    )

    assert tokens == [
        "Hello",
        " ",
        "from",
        " ",
        "AI",
        ".",
    ]

    save_mock.assert_any_call(
        db=ANY,
        session_id="session-123",
        role="user",
        message="Hello",
    )

    save_mock.assert_any_call(
        db=ANY,
        session_id="session-123",
        role="assistant",
        message="Hello from AI.",
    )


@pytest.mark.asyncio
async def test_stream_response_builds_complete_response(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(return_value=""),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        MagicMock(return_value=("", [])),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(return_value="Prompt"),
    )

    save_mock = MagicMock()

    monkeypatch.setattr(
        service,
        "_save_message",
        save_mock,
    )

    tokens = await collect_stream(
        service.generate_stream_response(
            db=FakeDatabase(),
            session_id="session-123",
            user_message="Hello",
        )
    )

    assert tokens == [
        "Hello",
        " ",
        "from",
        " ",
        "AI",
        ".",
    ]

    save_mock.assert_any_call(
        db=ANY,
        session_id="session-123",
        role="assistant",
        message="Hello from AI.",
    )


@pytest.mark.asyncio
async def test_stream_response_ignores_empty_tokens(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    llm_client.stream_tokens = [
        "Hello",
        "",
        None,
        " ",
        "AI",
    ]

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(return_value=""),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        MagicMock(return_value=("", [])),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(return_value="Prompt"),
    )

    monkeypatch.setattr(
        service,
        "_save_message",
        MagicMock(),
    )

    tokens = await collect_stream(
        service.generate_stream_response(
            db=FakeDatabase(),
            session_id="session-123",
            user_message="Hello",
        )
    )

    assert tokens == [
        "Hello",
        " ",
        "AI",
    ]


@pytest.mark.asyncio
async def test_stream_response_handles_llm_failure(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    async def failing_stream(messages):
        raise RuntimeError(
            "Ollama streaming failure"
        )

        yield

    llm_client = MagicMock()

    llm_client.stream_generate = (
        failing_stream
    )

    service.llm_client = llm_client

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(return_value=""),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        MagicMock(return_value=("", [])),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(return_value="Prompt"),
    )

    save_mock = MagicMock()

    monkeypatch.setattr(
        service,
        "_save_message",
        save_mock,
    )

    tokens = await collect_stream(
        service.generate_stream_response(
            db=FakeDatabase(),
            session_id="session-123",
            user_message="Hello",
        )
    )

    assert tokens == [
        "Sorry, I couldn't generate "
        "a response at the moment."
    ]

    save_mock.assert_any_call(
        db=ANY,
        session_id="session-123",
        role="assistant",
        message=(
            "Sorry, I couldn't generate "
            "a response at the moment."
        ),
    )


@pytest.mark.asyncio
async def test_stream_response_rejects_invalid_session(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="session_id cannot be empty",
    ):
        await collect_stream(
            service.generate_stream_response(
                db=FakeDatabase(),
                session_id="",
                user_message="Hello",
            )
        )


@pytest.mark.asyncio
async def test_stream_response_rejects_invalid_message(
    monkeypatch,
):
    service, _ = create_service(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="user_message cannot be empty",
    ):
        await collect_stream(
            service.generate_stream_response(
                db=FakeDatabase(),
                session_id="session-123",
                user_message="   ",
            )
        )


@pytest.mark.asyncio
async def test_stream_response_passes_prompt_to_llm(
    monkeypatch,
):
    service, llm_client = create_service(
        monkeypatch
    )

    monkeypatch.setattr(
        service,
        "_get_conversation_history",
        MagicMock(return_value="History"),
    )

    monkeypatch.setattr(
        service,
        "_retrieve_context",
        MagicMock(
            return_value=("Context", [])
        ),
    )

    monkeypatch.setattr(
        service,
        "_build_prompt",
        MagicMock(return_value="FINAL PROMPT"),
    )

    monkeypatch.setattr(
        service,
        "_save_message",
        MagicMock(),
    )

    captured = {}

    async def stream(messages):
        captured["messages"] = messages

        yield "Answer"

    llm_client.stream_generate = stream

    tokens = await collect_stream(
        service.generate_stream_response(
            db=FakeDatabase(),
            session_id="session-123",
            user_message="Question",
        )
    )

    assert tokens == ["Answer"]

    assert captured["messages"] == [
        {
            "role": "user",
            "content": "FINAL PROMPT",
        }
    ]


# ============================================================
# Shared Service Instance
# ============================================================


def test_shared_assistant_service_is_initialized():
    assert isinstance(
        assistant_module.assistant_service,
        assistant_module.AssistantService,
    )


def test_assistant_service_uses_ollama_client():
    assert (
        assistant_module.assistant_service.llm_client
        is assistant_module.ollama_client
    )