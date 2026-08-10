import pytest

import app.assistant.conversation_service as conversation_module


# ============================================================
# Fake Database
# ============================================================


class FakeDatabase:
    """
    Dummy database object used only for unit tests.
    """

    pass


# ============================================================
# Fake Repository
# ============================================================


class FakeConversationRepository:
    """
    Fake repository used to verify service-to-repository
    communication.
    """

    def __init__(self):
        self.get_messages_calls = []
        self.get_sessions_calls = []
        self.get_statistics_calls = []
        self.clear_session_calls = []
        self.clear_all_calls = []

        self.messages_result = []
        self.sessions_result = []
        self.statistics_result = {}
        self.clear_session_result = 0
        self.clear_all_result = 0

        self.get_messages_error = None
        self.get_sessions_error = None
        self.get_statistics_error = None
        self.clear_session_error = None
        self.clear_all_error = None

    def get_messages(
        self,
        *,
        db,
        session_id,
    ):
        self.get_messages_calls.append(
            {
                "db": db,
                "session_id": session_id,
            }
        )

        if self.get_messages_error:
            raise self.get_messages_error

        return self.messages_result

    def get_sessions(
        self,
        *,
        db,
    ):
        self.get_sessions_calls.append(
            {
                "db": db,
            }
        )

        if self.get_sessions_error:
            raise self.get_sessions_error

        return self.sessions_result

    def get_statistics(
        self,
        *,
        db,
    ):
        self.get_statistics_calls.append(
            {
                "db": db,
            }
        )

        if self.get_statistics_error:
            raise self.get_statistics_error

        return self.statistics_result

    def clear_session(
        self,
        *,
        db,
        session_id,
    ):
        self.clear_session_calls.append(
            {
                "db": db,
                "session_id": session_id,
            }
        )

        if self.clear_session_error:
            raise self.clear_session_error

        return self.clear_session_result

    def clear_all(
        self,
        *,
        db,
    ):
        self.clear_all_calls.append(
            {
                "db": db,
            }
        )

        if self.clear_all_error:
            raise self.clear_all_error

        return self.clear_all_result


# ============================================================
# Service Factory
# ============================================================


def create_service(monkeypatch):
    repository = FakeConversationRepository()

    monkeypatch.setattr(
        conversation_module,
        "conversation_repository",
        repository,
    )

    service = (
        conversation_module.ConversationService()
    )

    return service, repository


# ============================================================
# Get Conversation History
# ============================================================


def test_get_history_returns_repository_result(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    db = FakeDatabase()

    expected_history = [
        {
            "role": "user",
            "message": "Hello",
        },
        {
            "role": "assistant",
            "message": "Hi there.",
        },
    ]

    repository.messages_result = (
        expected_history
    )

    result = service.get_history(
        db=db,
        session_id="session-123",
    )

    assert result == expected_history


def test_get_history_passes_correct_arguments(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    db = FakeDatabase()

    service.get_history(
        db=db,
        session_id="session-123",
    )

    assert (
        repository.get_messages_calls
        == [
            {
                "db": db,
                "session_id": "session-123",
            }
        ]
    )


def test_get_history_returns_empty_list(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.messages_result = []

    result = service.get_history(
        db=FakeDatabase(),
        session_id="empty-session",
    )

    assert result == []


def test_get_history_propagates_repository_error(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.get_messages_error = (
        RuntimeError("Repository failure")
    )

    with pytest.raises(
        RuntimeError,
        match="Repository failure",
    ):
        service.get_history(
            db=FakeDatabase(),
            session_id="session-123",
        )


# ============================================================
# Get Sessions
# ============================================================


def test_get_sessions_returns_repository_result(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    expected_sessions = [
        {
            "session_id": "session-1",
            "message_count": 5,
        },
        {
            "session_id": "session-2",
            "message_count": 10,
        },
    ]

    repository.sessions_result = (
        expected_sessions
    )

    result = service.get_sessions(
        db=FakeDatabase()
    )

    assert result == expected_sessions


def test_get_sessions_passes_database_to_repository(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    db = FakeDatabase()

    service.get_sessions(
        db=db
    )

    assert (
        repository.get_sessions_calls
        == [
            {
                "db": db,
            }
        ]
    )


def test_get_sessions_returns_empty_list(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.sessions_result = []

    result = service.get_sessions(
        db=FakeDatabase()
    )

    assert result == []


def test_get_sessions_propagates_repository_error(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.get_sessions_error = (
        RuntimeError("Session lookup failed")
    )

    with pytest.raises(
        RuntimeError,
        match="Session lookup failed",
    ):
        service.get_sessions(
            db=FakeDatabase()
        )


# ============================================================
# Get Statistics
# ============================================================


def test_get_statistics_returns_repository_result(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    expected_statistics = {
        "total_sessions": 5,
        "total_messages": 42,
        "user_messages": 20,
        "assistant_messages": 22,
    }

    repository.statistics_result = (
        expected_statistics
    )

    result = service.get_statistics(
        db=FakeDatabase()
    )

    assert result == expected_statistics


def test_get_statistics_passes_database_to_repository(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    db = FakeDatabase()

    service.get_statistics(
        db=db
    )

    assert (
        repository.get_statistics_calls
        == [
            {
                "db": db,
            }
        ]
    )


def test_get_statistics_returns_empty_dictionary(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.statistics_result = {}

    result = service.get_statistics(
        db=FakeDatabase()
    )

    assert result == {}


def test_get_statistics_propagates_repository_error(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.get_statistics_error = (
        RuntimeError("Statistics failure")
    )

    with pytest.raises(
        RuntimeError,
        match="Statistics failure",
    ):
        service.get_statistics(
            db=FakeDatabase()
        )


# ============================================================
# Clear One Session
# ============================================================


def test_clear_session_returns_deleted_count(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.clear_session_result = 7

    result = service.clear_session(
        db=FakeDatabase(),
        session_id="session-123",
    )

    assert result == 7


def test_clear_session_passes_correct_arguments(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    db = FakeDatabase()

    service.clear_session(
        db=db,
        session_id="session-123",
    )

    assert (
        repository.clear_session_calls
        == [
            {
                "db": db,
                "session_id": "session-123",
            }
        ]
    )


def test_clear_session_returns_zero_when_nothing_deleted(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.clear_session_result = 0

    result = service.clear_session(
        db=FakeDatabase(),
        session_id="missing-session",
    )

    assert result == 0


def test_clear_session_propagates_repository_error(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.clear_session_error = (
        RuntimeError("Delete failed")
    )

    with pytest.raises(
        RuntimeError,
        match="Delete failed",
    ):
        service.clear_session(
            db=FakeDatabase(),
            session_id="session-123",
        )


# ============================================================
# Clear All Conversations
# ============================================================


def test_clear_all_returns_deleted_count(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.clear_all_result = 25

    result = service.clear_all(
        db=FakeDatabase()
    )

    assert result == 25


def test_clear_all_passes_database_to_repository(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    db = FakeDatabase()

    service.clear_all(
        db=db
    )

    assert (
        repository.clear_all_calls
        == [
            {
                "db": db,
            }
        ]
    )


def test_clear_all_returns_zero_when_nothing_deleted(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.clear_all_result = 0

    result = service.clear_all(
        db=FakeDatabase()
    )

    assert result == 0


def test_clear_all_propagates_repository_error(
    monkeypatch,
):
    service, repository = create_service(
        monkeypatch
    )

    repository.clear_all_error = (
        RuntimeError("Clear all failed")
    )

    with pytest.raises(
        RuntimeError,
        match="Clear all failed",
    ):
        service.clear_all(
            db=FakeDatabase()
        )


# ============================================================
# Shared Service Instance
# ============================================================


def test_shared_conversation_service_is_initialized():
    assert isinstance(
        conversation_module.conversation_service,
        conversation_module.ConversationService,
    )