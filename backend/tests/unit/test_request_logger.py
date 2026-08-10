import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.middleware.request_logger import log_requests


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

    return request


def create_response(status_code=200):
    response = MagicMock()

    response.status_code = status_code

    return response


# ============================================================
# Basic Behavior
# ============================================================

@pytest.mark.asyncio
async def test_log_requests_returns_response():
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    result = await log_requests(
        request,
        call_next,
    )

    assert result is response


@pytest.mark.asyncio
async def test_log_requests_calls_next_with_request():
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    await log_requests(
        request,
        call_next,
    )

    call_next.assert_awaited_once_with(
        request
    )


# ============================================================
# Incoming Request Logging
# ============================================================

@pytest.mark.asyncio
async def test_log_requests_logs_incoming_request(
    monkeypatch,
):
    request = create_request(
        method="GET",
        path="/health",
    )

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    logger = MagicMock()

    monkeypatch.setattr(
        "app.middleware.request_logger.app_logger",
        logger,
    )

    await log_requests(
        request,
        call_next,
    )

    first_log = logger.info.call_args_list[0]

    assert first_log == call(
        "Incoming Request: GET /health"
    )


# ============================================================
# Completed Request Logging
# ============================================================

@pytest.mark.asyncio
async def test_log_requests_logs_completed_request(
    monkeypatch,
):
    request = create_request(
        method="GET",
        path="/health",
    )

    response = create_response(
        status_code=200,
    )

    call_next = AsyncMock(
        return_value=response
    )

    logger = MagicMock()

    monkeypatch.setattr(
        "app.middleware.request_logger.app_logger",
        logger,
    )

    await log_requests(
        request,
        call_next,
    )

    assert logger.info.call_count == 2

    completed_log = (
        logger.info.call_args_list[1].args[0]
    )

    assert (
        "Completed: GET /health"
        in completed_log
    )

    assert "Status=200" in completed_log
    assert "Time=" in completed_log


@pytest.mark.asyncio
async def test_completed_log_contains_status_code(
    monkeypatch,
):
    request = create_request()

    response = create_response(
        status_code=201,
    )

    call_next = AsyncMock(
        return_value=response
    )

    logger = MagicMock()

    monkeypatch.setattr(
        "app.middleware.request_logger.app_logger",
        logger,
    )

    await log_requests(
        request,
        call_next,
    )

    completed_log = (
        logger.info.call_args_list[1].args[0]
    )

    assert "Status=201" in completed_log


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
async def test_log_requests_supports_http_methods(
    monkeypatch,
    method,
):
    request = create_request(
        method=method,
        path="/test",
    )

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    logger = MagicMock()

    monkeypatch.setattr(
        "app.middleware.request_logger.app_logger",
        logger,
    )

    result = await log_requests(
        request,
        call_next,
    )

    assert result is response

    assert logger.info.call_count == 2

    incoming_log = (
        logger.info.call_args_list[0].args[0]
    )

    completed_log = (
        logger.info.call_args_list[1].args[0]
    )

    assert f"Incoming Request: {method} /test" == (
        incoming_log
    )

    assert f"Completed: {method} /test" in (
        completed_log
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
async def test_log_requests_supports_status_codes(
    monkeypatch,
    status_code,
):
    request = create_request()

    response = create_response(
        status_code=status_code
    )

    call_next = AsyncMock(
        return_value=response
    )

    logger = MagicMock()

    monkeypatch.setattr(
        "app.middleware.request_logger.app_logger",
        logger,
    )

    await log_requests(
        request,
        call_next,
    )

    completed_log = (
        logger.info.call_args_list[1].args[0]
    )

    assert (
        f"Status={status_code}"
        in completed_log
    )


# ============================================================
# Request Path Logging
# ============================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/health",
        "/conversations",
        "/api/test",
        "/users/123",
    ],
)
async def test_log_requests_logs_request_path(
    monkeypatch,
    path,
):
    request = create_request(
        path=path
    )

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    logger = MagicMock()

    monkeypatch.setattr(
        "app.middleware.request_logger.app_logger",
        logger,
    )

    await log_requests(
        request,
        call_next,
    )

    incoming_log = (
        logger.info.call_args_list[0].args[0]
    )

    completed_log = (
        logger.info.call_args_list[1].args[0]
    )

    assert path in incoming_log
    assert path in completed_log


# ============================================================
# Timing
# ============================================================

@pytest.mark.asyncio
async def test_log_requests_includes_request_duration(
    monkeypatch,
):
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    logger = MagicMock()

    monkeypatch.setattr(
        "app.middleware.request_logger.app_logger",
        logger,
    )

    await log_requests(
        request,
        call_next,
    )

    completed_log = (
        logger.info.call_args_list[1].args[0]
    )

    # The implementation formats duration
    # to three decimal places.
    assert "Time=" in completed_log

    time_value = completed_log.split(
        "Time="
    )[1]

    assert time_value.endswith("s")

    numeric_value = time_value[:-1]

    assert len(numeric_value.split(".")[1]) == 3

    assert float(numeric_value) >= 0


# ============================================================
# Logger Call Order
# ============================================================

@pytest.mark.asyncio
async def test_incoming_log_occurs_before_completed_log(
    monkeypatch,
):
    request = create_request()

    response = create_response()

    call_next = AsyncMock(
        return_value=response
    )

    logger = MagicMock()

    monkeypatch.setattr(
        "app.middleware.request_logger.app_logger",
        logger,
    )

    await log_requests(
        request,
        call_next,
    )

    assert logger.info.call_count == 2

    first_message = (
        logger.info.call_args_list[0].args[0]
    )

    second_message = (
        logger.info.call_args_list[1].args[0]
    )

    assert first_message.startswith(
        "Incoming Request:"
    )

    assert second_message.startswith(
        "Completed:"
    )


# ============================================================
# Downstream Failure
# ============================================================

@pytest.mark.asyncio
async def test_log_requests_propagates_call_next_failure(
    monkeypatch,
):
    request = create_request()

    expected_error = RuntimeError(
        "Request processing failed."
    )

    async def failing_call_next(request):
        raise expected_error

    logger = MagicMock()

    monkeypatch.setattr(
        "app.middleware.request_logger.app_logger",
        logger,
    )

    with pytest.raises(
        RuntimeError,
        match="Request processing failed.",
    ):
        await log_requests(
            request,
            failing_call_next,
        )


@pytest.mark.asyncio
async def test_log_requests_logs_incoming_request_before_failure(
    monkeypatch,
):
    request = create_request(
        method="POST",
        path="/assistant",
    )

    async def failing_call_next(request):
        raise RuntimeError("LLM failed")

    logger = MagicMock()

    monkeypatch.setattr(
        "app.middleware.request_logger.app_logger",
        logger,
    )

    with pytest.raises(RuntimeError):
        await log_requests(
            request,
            failing_call_next,
        )

    assert logger.info.call_count == 1

    incoming_log = (
        logger.info.call_args_list[0].args[0]
    )

    assert incoming_log == (
        "Incoming Request: POST /assistant"
    )