import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.api.routes.chat import (
    chat,
    chat_stream,
    router,
)


# ============================================================
# Helpers
# ============================================================


def make_request(
    session_id="test-session",
    message="Hello",
    documents=None,
):
    """
    Create a lightweight request object.

    The router accesses only these three attributes,
    so a SimpleNamespace keeps these tests isolated
    from Pydantic validation.
    """
    return SimpleNamespace(
        session_id=session_id,
        message=message,
        documents=documents,
    )


async def async_tokens(tokens):
    """
    Async generator used to mock streaming responses.
    """
    for token in tokens:
        yield token


# ============================================================
# Normal Chat
# ============================================================


@pytest.mark.asyncio
async def test_chat_successfully():
    """
    Chat endpoint should return a successful ApiResponse.
    """

    request = make_request()

    assistant_service = __import__(
        "app.api.routes.chat",
        fromlist=["assistant_service"],
    ).assistant_service

    assistant_service.generate_response = AsyncMock(
        return_value={
            "response": "Hello from AI",
            "sources": [
                "test.pdf"
            ],
        }
    )

    result = await chat(
        request=request,
        db="test-db",
    )

    assert result.success is True
    assert result.message == (
        "Response generated successfully."
    )

    assert result.data.response == "Hello from AI"
    assert result.data.sources == [
        "test.pdf"
    ]
    assert result.data.processing_time >= 0

    assistant_service.generate_response.assert_awaited_once_with(
        db="test-db",
        session_id="test-session",
        user_message="Hello",
        documents=None,
    )


@pytest.mark.asyncio
async def test_chat_handles_missing_response_and_sources():
    """
    Missing response/sources should use the router defaults.
    """

    request = make_request()

    assistant_service = __import__(
        "app.api.routes.chat",
        fromlist=["assistant_service"],
    ).assistant_service

    assistant_service.generate_response = AsyncMock(
        return_value={}
    )

    result = await chat(
        request=request,
        db="test-db",
    )

    assert result.success is True
    assert result.data.response == ""
    assert result.data.sources == []
    assert result.data.processing_time >= 0


@pytest.mark.asyncio
async def test_chat_passes_documents_to_assistant():
    """
    Selected documents should be passed to the assistant.
    """

    request = make_request(
        documents=["abc123", "def456"]
    )

    assistant_service = __import__(
        "app.api.routes.chat",
        fromlist=["assistant_service"],
    ).assistant_service

    assistant_service.generate_response = AsyncMock(
        return_value={
            "response": "Document answer",
            "sources": [],
        }
    )

    await chat(
        request=request,
        db="database",
    )

    assistant_service.generate_response.assert_awaited_once_with(
        db="database",
        session_id="test-session",
        user_message="Hello",
        documents=["abc123", "def456"],
    )


@pytest.mark.asyncio
async def test_chat_handles_value_error():
    """
    ValueError from the assistant should become HTTP 400.
    """

    request = make_request()

    assistant_service = __import__(
        "app.api.routes.chat",
        fromlist=["assistant_service"],
    ).assistant_service

    assistant_service.generate_response = AsyncMock(
        side_effect=ValueError("Invalid message")
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat(
            request=request,
            db="test-db",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid message"


@pytest.mark.asyncio
async def test_chat_handles_unexpected_error():
    """
    Unexpected assistant errors should become HTTP 500.
    """

    request = make_request()

    assistant_service = __import__(
        "app.api.routes.chat",
        fromlist=["assistant_service"],
    ).assistant_service

    assistant_service.generate_response = AsyncMock(
        side_effect=RuntimeError("Database unavailable")
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat(
            request=request,
            db="test-db",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == (
        "Internal server error."
    )


# ============================================================
# Streaming Chat
# ============================================================


@pytest.mark.asyncio
async def test_chat_stream_returns_streaming_response():
    """
    Streaming endpoint should return StreamingResponse.
    """

    request = make_request()

    assistant_service = __import__(
        "app.api.routes.chat",
        fromlist=["assistant_service"],
    ).assistant_service

    assistant_service.generate_stream_response = MagicMock(
        return_value=async_tokens(
            ["Hello", " ", "world"]
        )
    )

    response = await chat_stream(
        request=request,
        db="test-db",
    )

    assert isinstance(
        response,
        StreamingResponse,
    )

    assert response.media_type == (
        "text/event-stream"
    )

    assert response.headers["cache-control"] == (
        "no-cache"
    )

    assert response.headers["connection"] == (
        "keep-alive"
    )

    assert response.headers["x-accel-buffering"] == (
        "no"
    )


@pytest.mark.asyncio
async def test_chat_stream_generates_tokens():
    """
    Streaming generator should yield every non-empty token.
    """

    request = make_request()

    assistant_service = __import__(
        "app.api.routes.chat",
        fromlist=["assistant_service"],
    ).assistant_service

    assistant_service.generate_stream_response = MagicMock(
        return_value=async_tokens(
            ["Hello", "", " ", "world"]
        )
    )

    response = await chat_stream(
        request=request,
        db="test-db",
    )

    tokens = []

    async for token in response.body_iterator:
        tokens.append(token)

    assert tokens == [
        "data: Hello\n\n",
        "data:  \n\n",
        "data: world\n\n",
    ]


@pytest.mark.asyncio
async def test_chat_stream_passes_correct_arguments():
    """
    Streaming request should pass all expected arguments.
    """

    request = make_request(
        session_id="session-123",
        message="Explain RAG",
        documents=["doc-1"],
    )

    assistant_service = __import__(
        "app.api.routes.chat",
        fromlist=["assistant_service"],
    ).assistant_service

    assistant_service.generate_stream_response = MagicMock(
        return_value=async_tokens(["answer"])
    )

    response = await chat_stream(
        request=request,
        db="database",
    )

    async for _ in response.body_iterator:
        pass

    assistant_service.generate_stream_response.assert_called_once_with(
        db="database",
        session_id="session-123",
        user_message="Explain RAG",
        documents=["doc-1"],
    )


@pytest.mark.asyncio
async def test_chat_stream_handles_value_error():
    """
    ValueError during streaming should be converted
    into a readable error message.
    """

    request = make_request()

    async def failing_generator():
        raise ValueError("Invalid streaming request")
        yield  # pragma: no cover

    assistant_service = __import__(
        "app.api.routes.chat",
        fromlist=["assistant_service"],
    ).assistant_service

    assistant_service.generate_stream_response = MagicMock(
        return_value=failing_generator()
    )

    response = await chat_stream(
        request=request,
        db="test-db",
    )

    tokens = []

    async for token in response.body_iterator:
        tokens.append(token)

    assert tokens == [
    "data: Error: Invalid streaming request\n\n"
]


@pytest.mark.asyncio
async def test_chat_stream_handles_unexpected_error():
    """
    Unexpected streaming errors should return the
    generic streaming error message.
    """

    request = make_request()

    async def failing_generator():
        raise RuntimeError("Unexpected failure")
        yield  # pragma: no cover

    assistant_service = __import__(
        "app.api.routes.chat",
        fromlist=["assistant_service"],
    ).assistant_service

    assistant_service.generate_stream_response = MagicMock(
        return_value=failing_generator()
    )

    response = await chat_stream(
        request=request,
        db="test-db",
    )

    tokens = []

    async for token in response.body_iterator:
        tokens.append(token)

    assert tokens == [
        "data: Error: Unable to generate the response.\n\n"
    ]


# ============================================================
# Router Configuration
# ============================================================


def test_chat_router_has_correct_prefix():
    assert router.prefix == "/chat"


def test_chat_router_has_correct_tag():
    assert router.tags == ["Chat"]


def test_chat_router_contains_normal_chat_route():
    paths = [
        route.path
        for route in router.routes
        if hasattr(route, "path")
    ]

    assert "/chat" in paths


def test_chat_router_contains_stream_route():
    paths = [
        route.path
        for route in router.routes
        if hasattr(route, "path")
    ]

    assert "/chat/stream" in paths