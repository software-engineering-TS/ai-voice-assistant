import pytest
from fastapi import HTTPException

from app.api.routes.conversation import (
    get_sessions,
    statistics,
    delete_all,
    get_history,
    delete_session,
)


class FakeDatabase:
    pass


# ============================================================
# Get All Sessions
# ============================================================


@pytest.mark.asyncio
async def test_get_sessions_successfully(monkeypatch):
    """
    Successfully returns all conversation sessions.
    """

    sessions = [
        {"session_id": "session-1"},
        {"session_id": "session-2"},
    ]

    def mock_get_sessions(db):
        assert isinstance(db, FakeDatabase)
        return sessions

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_sessions",
        mock_get_sessions,
    )

    result = await get_sessions(
        db=FakeDatabase(),
    )

    assert result.success is True
    assert result.message == (
        "Sessions retrieved successfully."
    )

    assert result.data["total_sessions"] == 2
    assert result.data["sessions"] == sessions


@pytest.mark.asyncio
async def test_get_sessions_returns_empty_list(monkeypatch):
    """
    Returns an empty session list when no sessions exist.
    """

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_sessions",
        lambda db: [],
    )

    result = await get_sessions(
        db=FakeDatabase(),
    )

    assert result.success is True
    assert result.data["total_sessions"] == 0
    assert result.data["sessions"] == []


@pytest.mark.asyncio
async def test_get_sessions_passes_database_to_service(
    monkeypatch,
):
    """
    Database session is passed to the service.
    """

    database = FakeDatabase()
    received = {}

    def mock_get_sessions(db):
        received["db"] = db
        return []

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_sessions",
        mock_get_sessions,
    )

    await get_sessions(db=database)

    assert received["db"] is database


@pytest.mark.asyncio
async def test_get_sessions_handles_service_failure(
    monkeypatch,
):
    """
    Service failures are converted to HTTP 500.
    """

    def mock_get_sessions(db):
        raise RuntimeError("Database failure")

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_sessions",
        mock_get_sessions,
    )

    with pytest.raises(HTTPException) as exc_info:

        await get_sessions(
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == (
        "Unable to retrieve sessions."
    )


# ============================================================
# Statistics
# ============================================================


@pytest.mark.asyncio
async def test_statistics_successfully(monkeypatch):
    """
    Successfully returns conversation statistics.
    """

    statistics_data = {
        "messages": 10,
        "sessions": 3,
    }

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_statistics",
        lambda db: statistics_data,
    )

    result = await statistics(
        db=FakeDatabase(),
    )

    assert result.success is True
    assert result.message == (
        "Conversation statistics retrieved successfully."
    )
    assert result.data == statistics_data


@pytest.mark.asyncio
async def test_statistics_passes_database_to_service(
    monkeypatch,
):
    """
    Database session is passed to statistics service.
    """

    database = FakeDatabase()
    received = {}

    def mock_get_statistics(db):
        received["db"] = db
        return {}

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_statistics",
        mock_get_statistics,
    )

    await statistics(db=database)

    assert received["db"] is database


@pytest.mark.asyncio
async def test_statistics_handles_service_failure(
    monkeypatch,
):
    """
    Statistics service failure returns HTTP 500.
    """

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_statistics",
        lambda db: (_ for _ in ()).throw(
            RuntimeError("Database failure")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:

        await statistics(
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == (
        "Unable to retrieve statistics."
    )


# ============================================================
# Delete All Conversations
# ============================================================


@pytest.mark.asyncio
async def test_delete_all_successfully(monkeypatch):
    """
    Successfully deletes all conversations.
    """

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.clear_all",
        lambda db: 5,
    )

    result = await delete_all(
        db=FakeDatabase(),
    )

    assert result.success is True
    assert result.message == (
        "All conversations deleted successfully."
    )
    assert result.data == {
        "deleted": True,
    }


@pytest.mark.asyncio
async def test_delete_all_returns_not_found_message_when_empty(
    monkeypatch,
):
    """
    Returns a successful response when there is
    nothing to delete.
    """

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.clear_all",
        lambda db: 0,
    )

    result = await delete_all(
        db=FakeDatabase(),
    )

    assert result.success is True
    assert result.message == (
        "No conversations found to delete."
    )
    assert result.data == {
        "deleted": False,
    }


@pytest.mark.asyncio
async def test_delete_all_passes_database_to_service(
    monkeypatch,
):
    """
    Database session is passed to clear_all.
    """

    database = FakeDatabase()
    received = {}

    def mock_clear_all(db):
        received["db"] = db
        return 0

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.clear_all",
        mock_clear_all,
    )

    await delete_all(db=database)

    assert received["db"] is database


@pytest.mark.asyncio
async def test_delete_all_handles_service_failure(
    monkeypatch,
):
    """
    Delete-all service failure returns HTTP 500.
    """

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.clear_all",
        lambda db: (_ for _ in ()).throw(
            RuntimeError("Delete failed")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:

        await delete_all(
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == (
        "Unable to delete conversations."
    )


# ============================================================
# Get Conversation History
# ============================================================


@pytest.mark.asyncio
async def test_get_history_successfully(monkeypatch):
    """
    Successfully retrieves conversation history.
    """

    history = [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hi there",
        },
    ]

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_history",
        lambda db, session_id: history,
    )

    result = await get_history(
        session_id="session-123",
        db=FakeDatabase(),
    )

    assert result.success is True
    assert result.message == (
        "Conversation retrieved successfully."
    )

    assert result.data["session_id"] == "session-123"
    assert result.data["messages"] == history


@pytest.mark.asyncio
async def test_get_history_strips_session_id(
    monkeypatch,
):
    """
    Session ID whitespace is removed.
    """

    received = {}

    def mock_get_history(db, session_id):
        received["session_id"] = session_id
        return [
            {
                "role": "user",
                "content": "Hello",
            }
        ]

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_history",
        mock_get_history,
    )

    result = await get_history(
        session_id="  session-123  ",
        db=FakeDatabase(),
    )

    assert received["session_id"] == "session-123"
    assert result.data["session_id"] == "session-123"


@pytest.mark.asyncio
async def test_get_history_rejects_empty_session_id():
    """
    Empty session IDs return HTTP 400.
    """

    with pytest.raises(HTTPException) as exc_info:

        await get_history(
            session_id="",
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Session ID cannot be empty."
    )


@pytest.mark.asyncio
async def test_get_history_rejects_whitespace_session_id():
    """
    Whitespace-only session IDs return HTTP 400.
    """

    with pytest.raises(HTTPException) as exc_info:

        await get_history(
            session_id="   ",
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Session ID cannot be empty."
    )


@pytest.mark.asyncio
async def test_get_history_returns_404_when_not_found(
    monkeypatch,
):
    """
    Missing conversation history returns HTTP 404.
    """

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_history",
        lambda db, session_id: [],
    )

    with pytest.raises(HTTPException) as exc_info:

        await get_history(
            session_id="session-123",
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == (
        "Conversation not found."
    )


@pytest.mark.asyncio
async def test_get_history_handles_service_failure(
    monkeypatch,
):
    """
    Unexpected service failures return HTTP 500.
    """

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.get_history",
        lambda db, session_id: (_ for _ in ()).throw(
            RuntimeError("Database failure")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:

        await get_history(
            session_id="session-123",
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == (
        "Unable to retrieve conversation."
    )


# ============================================================
# Delete Conversation
# ============================================================


@pytest.mark.asyncio
async def test_delete_session_successfully(monkeypatch):
    """
    Successfully deletes a conversation session.
    """

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.clear_session",
        lambda db, session_id: 3,
    )

    result = await delete_session(
        session_id="session-123",
        db=FakeDatabase(),
    )

    assert result.success is True
    assert result.message == (
        "Conversation deleted successfully."
    )

    assert result.data == {
        "session_id": "session-123",
        "deleted": True,
    }


@pytest.mark.asyncio
async def test_delete_session_strips_session_id(
    monkeypatch,
):
    """
    Session ID whitespace is removed before deletion.
    """

    received = {}

    def mock_clear_session(db, session_id):
        received["session_id"] = session_id
        return 1

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.clear_session",
        mock_clear_session,
    )

    result = await delete_session(
        session_id="  session-123  ",
        db=FakeDatabase(),
    )

    assert received["session_id"] == "session-123"
    assert result.data["session_id"] == "session-123"


@pytest.mark.asyncio
async def test_delete_session_rejects_empty_session_id():
    """
    Empty session IDs return HTTP 400.
    """

    with pytest.raises(HTTPException) as exc_info:

        await delete_session(
            session_id="",
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Session ID cannot be empty."
    )


@pytest.mark.asyncio
async def test_delete_session_rejects_whitespace_session_id():
    """
    Whitespace-only session IDs return HTTP 400.
    """

    with pytest.raises(HTTPException) as exc_info:

        await delete_session(
            session_id="   ",
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Session ID cannot be empty."
    )


@pytest.mark.asyncio
async def test_delete_session_returns_404_when_not_found(
    monkeypatch,
):
    """
    Deleting a nonexistent conversation returns HTTP 404.
    """

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.clear_session",
        lambda db, session_id: 0,
    )

    with pytest.raises(HTTPException) as exc_info:

        await delete_session(
            session_id="session-123",
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == (
        "Conversation not found."
    )


@pytest.mark.asyncio
async def test_delete_session_handles_service_failure(
    monkeypatch,
):
    """
    Unexpected deletion failures return HTTP 500.
    """

    monkeypatch.setattr(
        "app.api.routes.conversation.conversation_service.clear_session",
        lambda db, session_id: (_ for _ in ()).throw(
            RuntimeError("Delete failed")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:

        await delete_session(
            session_id="session-123",
            db=FakeDatabase(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == (
        "Unable to delete conversation."
    )


# ============================================================
# Router Configuration
# ============================================================


def test_conversation_router_has_correct_prefix():
    """
    Router uses the expected URL prefix.
    """

    from app.api.routes.conversation import router

    assert router.prefix == "/conversations"


def test_conversation_router_has_correct_tag():
    """
    Router uses the expected OpenAPI tag.
    """

    from app.api.routes.conversation import router

    assert router.tags == ["Conversations"]