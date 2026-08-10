import pytest

import app.assistant.memory_service as memory_module


# ============================================================
# Fake Database
# ============================================================


class FakeDatabase:
    """
    Dummy database object for unit tests.
    """

    pass


# ============================================================
# Fake Conversation Repository
# ============================================================


class FakeConversationRepository:
    """
    Fake repository used to isolate MemoryService
    from the real database.
    """

    def __init__(self):
        self.messages = []
        self.calls = []

        self.error = None

    def get_messages(
        self,
        *,
        db,
        session_id,
    ):
        self.calls.append(
            {
                "db": db,
                "session_id": session_id,
            }
        )

        if self.error is not None:
            raise self.error

        return self.messages


# ============================================================
# Service Factory
# ============================================================


def create_service(
    monkeypatch,
):
    repository = (
        FakeConversationRepository()
    )

    monkeypatch.setattr(
        memory_module,
        "conversation_repository",
        repository,
    )

    service = (
        memory_module.MemoryService()
    )

    return service, repository


# ============================================================
# Initialization
# ============================================================


def test_memory_service_initializes_successfully():
    service = (
        memory_module.MemoryService()
    )

    assert isinstance(
        service,
        memory_module.MemoryService,
    )


def test_shared_memory_service_is_initialized():
    assert isinstance(
        memory_module.memory_service,
        memory_module.MemoryService,
    )


# ============================================================
# Invalid Limit
# ============================================================


def test_invalid_zero_limit_returns_empty_history(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
        limit=0,
    )

    assert result == ""

    assert repository.calls == []


def test_invalid_negative_limit_returns_empty_history(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
        limit=-5,
    )

    assert result == ""

    assert repository.calls == []


# ============================================================
# Empty Conversation
# ============================================================


def test_empty_message_list_returns_empty_history(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = []

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == ""


def test_none_message_list_returns_empty_history(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = None

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == ""


# ============================================================
# Repository Interaction
# ============================================================


def test_repository_receives_correct_arguments(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    db = FakeDatabase()

    service.get_conversation_history(
        db=db,
        session_id="session-123",
        limit=6,
    )

    assert repository.calls == [
        {
            "db": db,
            "session_id": "session-123",
        }
    ]


# ============================================================
# User Messages
# ============================================================


def test_user_message_is_formatted_correctly(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == "User: Hello"


# ============================================================
# Assistant Messages
# ============================================================


def test_assistant_message_is_formatted_correctly(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help?",
        }
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert (
        result
        == "Assistant: Hello! How can I help?"
    )


# ============================================================
# Multiple Messages
# ============================================================


def test_multiple_messages_are_formatted_in_order(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hi there.",
        },
        {
            "role": "user",
            "content": "How are you?",
        },
        {
            "role": "assistant",
            "content": "I'm doing well.",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    expected = (
        "User: Hello\n"
        "Assistant: Hi there.\n"
        "User: How are you?\n"
        "Assistant: I'm doing well."
    )

    assert result == expected


# ============================================================
# Limit Behavior
# ============================================================


def test_limit_keeps_only_most_recent_messages(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
            "content": "Message 1",
        },
        {
            "role": "assistant",
            "content": "Message 2",
        },
        {
            "role": "user",
            "content": "Message 3",
        },
        {
            "role": "assistant",
            "content": "Message 4",
        },
        {
            "role": "user",
            "content": "Message 5",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
        limit=2,
    )

    expected = (
        "Assistant: Message 4\n"
        "User: Message 5"
    )

    assert result == expected


def test_limit_keeps_exact_number_of_recent_messages(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
            "content": "Message 1",
        },
        {
            "role": "assistant",
            "content": "Message 2",
        },
        {
            "role": "user",
            "content": "Message 3",
        },
        {
            "role": "assistant",
            "content": "Message 4",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
        limit=3,
    )

    expected = (
        "Assistant: Message 2\n"
        "User: Message 3\n"
        "Assistant: Message 4"
    )

    assert result == expected


def test_limit_larger_than_message_count_keeps_all_messages(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hi",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
        limit=100,
    )

    expected = (
        "User: Hello\n"
        "Assistant: Hi"
    )

    assert result == expected


# ============================================================
# Whitespace Handling
# ============================================================


def test_message_content_is_stripped(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
            "content": "   Hello world   ",
        }
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == "User: Hello world"


def test_whitespace_only_messages_are_skipped(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
            "content": "   ",
        },
        {
            "role": "assistant",
            "content": "Valid response",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == (
        "Assistant: Valid response"
    )


def test_empty_content_messages_are_skipped(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
            "content": "",
        },
        {
            "role": "assistant",
            "content": "Valid response",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == (
        "Assistant: Valid response"
    )


# ============================================================
# Unknown Roles
# ============================================================


def test_unknown_role_is_skipped(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "system",
            "content": "System message",
        },
        {
            "role": "user",
            "content": "Hello",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == "User: Hello"


def test_unknown_role_with_empty_content_is_skipped(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": "Hello",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == "User: Hello"


# ============================================================
# Missing Fields
# ============================================================


def test_missing_role_is_treated_as_unknown(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "content": "Message without role",
        },
        {
            "role": "user",
            "content": "Valid message",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == "User: Valid message"


def test_missing_content_is_skipped(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
        },
        {
            "role": "assistant",
            "content": "Valid response",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == (
        "Assistant: Valid response"
    )


# ============================================================
# Mixed Valid / Invalid Messages
# ============================================================


def test_mixed_messages_are_processed_correctly(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "system",
            "content": "System information",
        },
        {
            "role": "assistant",
            "content": "Hi",
        },
        {
            "role": "user",
            "content": "   ",
        },
        {
            "role": "assistant",
            "content": "How can I help?",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
        limit=10,
    )

    expected = (
        "User: Hello\n"
        "Assistant: Hi\n"
        "Assistant: How can I help?"
    )

    assert result == expected


# ============================================================
# Repository Failure
# ============================================================


def test_repository_failure_is_propagated(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.error = RuntimeError(
        "Database unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="Database unavailable",
    ):
        service.get_conversation_history(
            db=FakeDatabase(),
            session_id="session-123",
        )


# ============================================================
# Default Limit
# ============================================================


def test_default_limit_is_six_messages(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages = [
        {
            "role": "user",
            "content": "Message 1",
        },
        {
            "role": "assistant",
            "content": "Message 2",
        },
        {
            "role": "user",
            "content": "Message 3",
        },
        {
            "role": "assistant",
            "content": "Message 4",
        },
        {
            "role": "user",
            "content": "Message 5",
        },
        {
            "role": "assistant",
            "content": "Message 6",
        },
        {
            "role": "user",
            "content": "Message 7",
        },
    ]

    result = service.get_conversation_history(
        db=FakeDatabase(),
        session_id="session-123",
    )

    expected = (
        "Assistant: Message 2\n"
        "User: Message 3\n"
        "Assistant: Message 4\n"
        "User: Message 5\n"
        "Assistant: Message 6\n"
        "User: Message 7"
    )

    assert result == expected