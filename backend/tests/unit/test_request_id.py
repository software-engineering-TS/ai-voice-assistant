import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.middleware.request_id import request_id_middleware


# ============================================================
# Helpers
# ============================================================

def create_request(
    method="GET",
    path="/test",
):
    request = MagicMock()

    request.method = method

    request.url.path = path

    request.state = MagicMock()

    return request


def create_response(
    status_code=200,
):
    response = MagicMock()

    response.status_code = status_code
    response.headers = {}

    return response


# ============================================================
# Successful Request
# ============================================================

@pytest.mark.asyncio
async def test_request_id_middleware_returns_response(
    monkeypatch,
):
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    result = await request_id_middleware(
        request,
        call_next,
    )

    assert result is response


# ============================================================
# Request ID Generation
# ============================================================

@pytest.mark.asyncio
async def test_request_id_middleware_generates_request_id(
    monkeypatch,
):
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    fixed_uuid = uuid.UUID(
        "12345678-1234-5678-1234-567812345678"
    )

    monkeypatch.setattr(
        "app.middleware.request_id.uuid.uuid4",
        lambda: fixed_uuid,
    )

    await request_id_middleware(
        request,
        call_next,
    )

    assert request.state.request_id == str(
        fixed_uuid
    )


@pytest.mark.asyncio
async def test_request_id_is_valid_uuid(
    monkeypatch,
):
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    await request_id_middleware(
        request,
        call_next,
    )

    request_id = request.state.request_id

    parsed_uuid = uuid.UUID(request_id)

    assert str(parsed_uuid) == request_id


# ============================================================
# Request State
# ============================================================

@pytest.mark.asyncio
async def test_request_id_is_stored_on_request_state():
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    await request_id_middleware(
        request,
        call_next,
    )

    assert hasattr(
        request.state,
        "request_id",
    )

    assert request.state.request_id


# ============================================================
# X-Request-ID Header
# ============================================================

@pytest.mark.asyncio
async def test_response_contains_request_id_header():
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    await request_id_middleware(
        request,
        call_next,
    )

    assert (
        response.headers["X-Request-ID"]
        == request.state.request_id
    )


@pytest.mark.asyncio
async def test_request_id_header_matches_request_state():
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    await request_id_middleware(
        request,
        call_next,
    )

    header_request_id = (
        response.headers["X-Request-ID"]
    )

    state_request_id = (
        request.state.request_id
    )

    assert header_request_id == state_request_id


# ============================================================
# Request Lifecycle
# ============================================================

@pytest.mark.asyncio
async def test_call_next_receives_request():
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    await request_id_middleware(
        request,
        call_next,
    )

    call_next.assert_awaited_once_with(
        request
    )


@pytest.mark.asyncio
async def test_middleware_preserves_response_status_code():
    request = create_request()

    response = create_response(
        status_code=201
    )

    call_next = AsyncMock(
        return_value=response
    )

    result = await request_id_middleware(
        request,
        call_next,
    )

    assert result.status_code == 201


# ============================================================
# Different HTTP Methods
# ============================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    [
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    ],
)
async def test_request_id_middleware_supports_http_methods(
    method,
):
    request = create_request(
        method=method
    )

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    result = await request_id_middleware(
        request,
        call_next,
    )

    assert result is response

    assert request.state.request_id

    assert (
        response.headers["X-Request-ID"]
        == request.state.request_id
    )


# ============================================================
# Different Status Codes
# ============================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code",
    [
        200,
        201,
        400,
        404,
        500,
    ],
)
async def test_request_id_middleware_supports_status_codes(
    status_code,
):
    request = create_request()

    response = create_response(
        status_code=status_code
    )

    call_next = AsyncMock(
        return_value=response
    )

    result = await request_id_middleware(
        request,
        call_next,
    )

    assert result.status_code == status_code

    assert (
        result.headers["X-Request-ID"]
        == request.state.request_id
    )


# ============================================================
# Unique Request IDs
# ============================================================

@pytest.mark.asyncio
async def test_each_request_gets_unique_request_id():
    request_one = create_request()

    request_two = create_request()

    response_one = create_response()

    response_two = create_response()

    call_next_one = AsyncMock(
        return_value=response_one
    )

    call_next_two = AsyncMock(
        return_value=response_two
    )

    await request_id_middleware(
        request_one,
        call_next_one,
    )

    await request_id_middleware(
        request_two,
        call_next_two,
    )

    request_id_one = request_one.state.request_id
    request_id_two = request_two.state.request_id

    assert request_id_one != request_id_two


# ============================================================
# Failure Propagation
# ============================================================

@pytest.mark.asyncio
async def test_call_next_failure_is_propagated():
    request = create_request()

    expected_error = RuntimeError(
        "Request processing failed."
    )

    async def failing_call_next(request):
        raise expected_error

    with pytest.raises(
        RuntimeError,
        match="Request processing failed.",
    ):
        await request_id_middleware(
            request,
            failing_call_next,
        )