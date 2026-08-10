import pytest
from unittest.mock import MagicMock

from sqlalchemy.exc import SQLAlchemyError

from app.database.models import Conversation
from app.database.repository import (
    ConversationRepository,
    conversation_repository,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def repository():
    return ConversationRepository()


@pytest.fixture
def db():
    return MagicMock()


# ============================================================
# Save Message
# ============================================================

def test_save_message_successfully(
    repository,
    db,
):
    conversation = MagicMock()
    conversation.id = 1

    # Conversation() is created inside the repository,
    # so refresh() will populate the mocked object.
    db.refresh.side_effect = (
        lambda obj: setattr(obj, "id", 1)
    )

    result = repository.save_message(
        db=db,
        session_id="session-123",
        role="user",
        message="Hello",
    )

    assert isinstance(
        result,
        Conversation,
    )

    assert result.session_id == "session-123"
    assert result.role == "user"
    assert result.message == "Hello"

    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()


def test_save_message_passes_correct_data(
    repository,
    db,
):
    db.refresh.side_effect = (
        lambda obj: setattr(obj, "id", 1)
    )

    result = repository.save_message(
        db=db,
        session_id="session-456",
        role="assistant",
        message="Hello there.",
    )

    added_object = db.add.call_args.args[0]

    assert isinstance(
        added_object,
        Conversation,
    )

    assert added_object.session_id == "session-456"
    assert added_object.role == "assistant"
    assert added_object.message == "Hello there."


def test_save_message_commits_before_refresh(
    repository,
    db,
):
    db.refresh.side_effect = (
        lambda obj: setattr(obj, "id", 1)
    )

    repository.save_message(
        db=db,
        session_id="session-123",
        role="user",
        message="Test",
    )

    assert db.commit.call_count == 1
    assert db.refresh.call_count == 1

    assert (
        db.commit.call_args_list[0]
        is not db.refresh.call_args_list[0]
    )


def test_save_message_rolls_back_on_database_error(
    repository,
    db,
):
    db.commit.side_effect = SQLAlchemyError(
        "Database failure"
    )

    with pytest.raises(SQLAlchemyError):
        repository.save_message(
            db=db,
            session_id="session-123",
            role="user",
            message="Hello",
        )

    db.rollback.assert_called_once()


def test_save_message_does_not_refresh_after_commit_failure(
    repository,
    db,
):
    db.commit.side_effect = SQLAlchemyError(
        "Commit failed"
    )

    with pytest.raises(SQLAlchemyError):
        repository.save_message(
            db=db,
            session_id="session-123",
            role="user",
            message="Hello",
        )

    db.refresh.assert_not_called()


# ============================================================
# Get Messages
# ============================================================

def test_get_messages_returns_formatted_messages(
    repository,
    db,
):
    first = MagicMock()
    first.role = "user"
    first.message = "Hello"

    second = MagicMock()
    second.role = "assistant"
    second.message = "Hi there"

    db.execute.return_value.scalars.return_value.all.return_value = [
        first,
        second,
    ]

    result = repository.get_messages(
        db=db,
        session_id="session-123",
    )

    assert result == [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hi there",
        },
    ]


def test_get_messages_returns_empty_list(
    repository,
    db,
):
    db.execute.return_value.scalars.return_value.all.return_value = []

    result = repository.get_messages(
        db=db,
        session_id="session-123",
    )

    assert result == []


def test_get_messages_executes_query(
    repository,
    db,
):
    db.execute.return_value.scalars.return_value.all.return_value = []

    repository.get_messages(
        db=db,
        session_id="session-123",
    )

    db.execute.assert_called_once()


def test_get_messages_propagates_database_error(
    repository,
    db,
):
    db.execute.side_effect = SQLAlchemyError(
        "Database failure"
    )

    with pytest.raises(SQLAlchemyError):
        repository.get_messages(
            db=db,
            session_id="session-123",
        )


# ============================================================
# Get Sessions
# ============================================================

def test_get_sessions_returns_unique_sessions(
    repository,
    db,
):
    db.execute.return_value.scalars.return_value.all.return_value = [
        "session-1",
        "session-2",
        "session-3",
    ]

    result = repository.get_sessions(
        db=db,
    )

    assert result == [
        {"session_id": "session-1"},
        {"session_id": "session-2"},
        {"session_id": "session-3"},
    ]


def test_get_sessions_returns_empty_list(
    repository,
    db,
):
    db.execute.return_value.scalars.return_value.all.return_value = []

    result = repository.get_sessions(
        db=db,
    )

    assert result == []


def test_get_sessions_executes_query(
    repository,
    db,
):
    db.execute.return_value.scalars.return_value.all.return_value = []

    repository.get_sessions(
        db=db,
    )

    db.execute.assert_called_once()


def test_get_sessions_propagates_database_error(
    repository,
    db,
):
    db.execute.side_effect = SQLAlchemyError(
        "Database failure"
    )

    with pytest.raises(SQLAlchemyError):
        repository.get_sessions(
            db=db,
        )


# ============================================================
# Statistics
# ============================================================

def test_get_statistics_returns_correct_values(
    repository,
    db,
):
    db.scalar.side_effect = [
        10,
        3,
    ]

    result = repository.get_statistics(
        db=db,
    )

    assert result == {
        "messages": 10,
        "sessions": 3,
    }

    assert db.scalar.call_count == 2


def test_get_statistics_returns_zero_when_counts_are_none(
    repository,
    db,
):
    db.scalar.side_effect = [
        None,
        None,
    ]

    result = repository.get_statistics(
        db=db,
    )

    assert result == {
        "messages": 0,
        "sessions": 0,
    }


def test_get_statistics_propagates_database_error(
    repository,
    db,
):
    db.scalar.side_effect = SQLAlchemyError(
        "Database failure"
    )

    with pytest.raises(SQLAlchemyError):
        repository.get_statistics(
            db=db,
        )


# ============================================================
# Clear Session
# ============================================================

def test_clear_session_returns_deleted_count(
    repository,
    db,
):
    db.execute.return_value.rowcount = 5

    result = repository.clear_session(
        db=db,
        session_id="session-123",
    )

    assert result == 5
    db.execute.assert_called_once()
    db.commit.assert_called_once()


def test_clear_session_returns_zero_when_nothing_deleted(
    repository,
    db,
):
    db.execute.return_value.rowcount = 0

    result = repository.clear_session(
        db=db,
        session_id="session-123",
    )

    assert result == 0
    db.commit.assert_called_once()


def test_clear_session_returns_zero_when_rowcount_is_none(
    repository,
    db,
):
    db.execute.return_value.rowcount = None

    result = repository.clear_session(
        db=db,
        session_id="session-123",
    )

    assert result == 0


def test_clear_session_rolls_back_on_database_error(
    repository,
    db,
):
    db.execute.side_effect = SQLAlchemyError(
        "Delete failed"
    )

    with pytest.raises(SQLAlchemyError):
        repository.clear_session(
            db=db,
            session_id="session-123",
        )

    db.rollback.assert_called_once()
    db.commit.assert_not_called()


# ============================================================
# Clear All
# ============================================================

def test_clear_all_returns_deleted_count(
    repository,
    db,
):
    db.execute.return_value.rowcount = 12

    result = repository.clear_all(
        db=db,
    )

    assert result == 12
    db.execute.assert_called_once()
    db.commit.assert_called_once()


def test_clear_all_returns_zero_when_nothing_deleted(
    repository,
    db,
):
    db.execute.return_value.rowcount = 0

    result = repository.clear_all(
        db=db,
    )

    assert result == 0
    db.commit.assert_called_once()


def test_clear_all_returns_zero_when_rowcount_is_none(
    repository,
    db,
):
    db.execute.return_value.rowcount = None

    result = repository.clear_all(
        db=db,
    )

    assert result == 0


def test_clear_all_rolls_back_on_database_error(
    repository,
    db,
):
    db.execute.side_effect = SQLAlchemyError(
        "Delete failed"
    )

    with pytest.raises(SQLAlchemyError):
        repository.clear_all(
            db=db,
        )

    db.rollback.assert_called_once()
    db.commit.assert_not_called()


# ============================================================
# Shared Repository Instance
# ============================================================

def test_shared_conversation_repository_is_initialized():
    assert conversation_repository is not None

    assert isinstance(
        conversation_repository,
        ConversationRepository,
    )