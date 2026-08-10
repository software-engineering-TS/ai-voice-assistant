import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.exceptions.custom_exceptions import LLMConnectionError, LLMResponseError
from app.llm.ollama_client import OllamaClient


def make_client(monkeypatch, base_url="http://localhost:11434",
                model="llama3.2:3b", timeout=180, max_retries=3):
    settings = MagicMock()
    settings.OLLAMA_BASE_URL = base_url
    settings.OLLAMA_MODEL = model
    settings.REQUEST_TIMEOUT = timeout
    settings.MAX_RETRIES = max_retries
    monkeypatch.setattr("app.llm.ollama_client.settings", settings)
    return OllamaClient()


def make_response(status_code=200, json_data=None, text="", is_error=None):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.is_error = status_code >= 400 if is_error is None else is_error
    response.text = text
    if json_data is not None:
        response.json.return_value = json_data
    else:
        response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
    return response


class AsyncStreamContextManager:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


def make_stream_response(lines, status_code=200):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.is_error = status_code >= 400

    async def aiter_lines():
        for line in lines:
            yield line

    response.aiter_lines = aiter_lines
    response.text = ""
    response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
    return response


# Initialization

def test_client_initializes_successfully(monkeypatch):
    client = make_client(monkeypatch)
    assert client.base_url == "http://localhost:11434"
    assert client.model == "llama3.2:3b"
    assert client.timeout == 180
    assert client.max_retries == 3
    assert client.client is not None


def test_client_strips_configuration(monkeypatch):
    client = make_client(
        monkeypatch,
        base_url="  http://localhost:11434///  ",
        model="  llama3.2:3b  ",
    )
    assert client.base_url == "http://localhost:11434"
    assert client.model == "llama3.2:3b"


def test_client_enforces_minimum_timeout_and_retries(monkeypatch):
    client = make_client(monkeypatch, timeout=0, max_retries=0)
    assert client.timeout == 1
    assert client.max_retries == 1


def test_client_rejects_empty_base_url(monkeypatch):
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL cannot be empty"):
        make_client(monkeypatch, base_url="   ")


def test_client_rejects_empty_model(monkeypatch):
    with pytest.raises(ValueError, match="OLLAMA_MODEL cannot be empty"):
        make_client(monkeypatch, model="   ")


def test_client_has_expected_headers(monkeypatch):
    client = make_client(monkeypatch)
    assert client.client.headers["Accept"] == "application/json"
    assert client.client.headers["Content-Type"] == "application/json"
    assert client.client.headers["User-Agent"] == "AI-Assistant/1.0"


# Lifecycle

@pytest.mark.asyncio
async def test_close_closes_client(monkeypatch):
    client = make_client(monkeypatch)
    client.client.aclose = AsyncMock()
    await client.close()
    client.client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_does_nothing_if_already_closed(monkeypatch):
    client = make_client(monkeypatch)

    client.client.aclose = AsyncMock()

    monkeypatch.setattr(
        type(client.client),
        "is_closed",
        property(lambda self: True),
    )

    await client.close()

    client.client.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_propagates_error(monkeypatch):
    client = make_client(monkeypatch)
    client.client.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
    with pytest.raises(RuntimeError, match="close failed"):
        await client.close()


# Message validation

def test_validate_messages_rejects_empty_list():
    with pytest.raises(LLMResponseError, match="No messages were provided"):
        OllamaClient._validate_messages([])


def test_validate_messages_rejects_none():
    with pytest.raises(LLMResponseError, match="No messages were provided"):
        OllamaClient._validate_messages(None)


def test_validate_messages_rejects_non_list():
    with pytest.raises(LLMResponseError, match="messages must be a list"):
        OllamaClient._validate_messages("hello")


def test_validate_messages_rejects_non_dict():
    with pytest.raises(LLMResponseError, match="Invalid message at index 0"):
        OllamaClient._validate_messages(["invalid"])


@pytest.mark.parametrize("message", [
    {"content": "hello"},
    {"role": "", "content": "hello"},
    {"role": "   ", "content": "hello"},
    {"role": 123, "content": "hello"},
])
def test_validate_messages_rejects_invalid_role(message):
    with pytest.raises(LLMResponseError, match="Invalid message role at index 0"):
        OllamaClient._validate_messages([message])


@pytest.mark.parametrize("message", [
    {"role": "user"},
    {"role": "user", "content": ""},
    {"role": "user", "content": "   "},
    {"role": "user", "content": 123},
    {"role": "user", "content": None},
])
def test_validate_messages_rejects_invalid_content(message):
    with pytest.raises(LLMResponseError, match="Invalid message content at index 0"):
        OllamaClient._validate_messages([message])


def test_validate_messages_accepts_valid_messages():
    OllamaClient._validate_messages([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ])


# Payload

def test_build_payload_without_system_prompt(monkeypatch):
    client = make_client(monkeypatch)
    messages = [{"role": "user", "content": "Hello"}]
    assert client._build_payload(messages, False, None) == {
        "model": "llama3.2:3b",
        "messages": messages,
        "stream": False,
    }


def test_build_payload_strips_system_prompt(monkeypatch):
    client = make_client(monkeypatch)
    payload = client._build_payload([], True, "  Be helpful.  ")
    assert payload["system"] == "Be helpful."
    assert payload["stream"] is True


def test_build_payload_ignores_blank_system_prompt(monkeypatch):
    client = make_client(monkeypatch)
    payload = client._build_payload([], False, "   ")
    assert "system" not in payload


# Parsing and extraction

def test_parse_json_success():
    response = make_response(json_data={"message": {"content": "Hello"}})
    assert OllamaClient._parse_json(response) == {
        "message": {"content": "Hello"}
    }


def test_parse_json_rejects_invalid_json():
    with pytest.raises(LLMResponseError, match="invalid JSON"):
        OllamaClient._parse_json(make_response())


def test_parse_json_rejects_non_dict():
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = ["invalid"]
    with pytest.raises(LLMResponseError, match="invalid response"):
        OllamaClient._parse_json(response)


def test_extract_content_strips_content():
    assert OllamaClient._extract_content({
        "message": {"content": "  Hello  "}
    }) == "Hello"


@pytest.mark.parametrize("data, message", [
    ({}, "valid message"),
    ({"message": None}, "valid message"),
    ({"message": []}, "valid message"),
    ({"message": {"content": 123}}, "invalid message content"),
    ({"message": {"content": ""}}, "empty response"),
    ({"message": {"content": "   "}}, "empty response"),
])
def test_extract_content_rejects_invalid_data(data, message):
    with pytest.raises(LLMResponseError, match=message):
        OllamaClient._extract_content(data)


def test_extract_error_detail_prefers_json():
    response = make_response(
        json_data={"error": "Model not found"},
        text="fallback",
    )
    assert OllamaClient._extract_error_detail(response) == "Model not found"


def test_extract_error_detail_uses_text_fallback():
    response = make_response(text="Something went wrong")
    assert OllamaClient._extract_error_detail(response) == "Something went wrong"


def test_extract_error_detail_truncates_text():
    response = make_response(text="x" * 1000)
    assert len(OllamaClient._extract_error_detail(response)) == 500


def test_is_retryable_status():
    assert OllamaClient._is_retryable_status(429) is True
    assert OllamaClient._is_retryable_status(500) is True
    assert OllamaClient._is_retryable_status(503) is True
    assert OllamaClient._is_retryable_status(400) is False
    assert OllamaClient._is_retryable_status(404) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt, delay", [(1, 1), (2, 2), (3, 4), (4, 8), (5, 8)])
async def test_retry_backoff(monkeypatch, attempt, delay):
    client = make_client(monkeypatch)
    sleep = AsyncMock()
    monkeypatch.setattr("app.llm.ollama_client.asyncio.sleep", sleep)
    await client._wait_before_retry(attempt)
    sleep.assert_awaited_once_with(delay)


# Generate

@pytest.mark.asyncio
async def test_generate_success(monkeypatch):
    client = make_client(monkeypatch)
    response = make_response(
        json_data={"message": {"content": "  Hello from Ollama  "}}
    )
    client.client.post = AsyncMock(return_value=response)

    result = await client.generate([
        {"role": "user", "content": "Hello"}
    ])

    assert result == "Hello from Ollama"
    client.client.post.assert_awaited_once_with(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )


@pytest.mark.asyncio
async def test_generate_includes_system_prompt(monkeypatch):
    client = make_client(monkeypatch)
    client.client.post = AsyncMock(return_value=make_response(
        json_data={"message": {"content": "Answer"}}
    ))

    await client.generate(
        [{"role": "user", "content": "Hello"}],
        system_prompt="  Be concise.  ",
    )

    payload = client.client.post.await_args.kwargs["json"]
    assert payload["system"] == "Be concise."


@pytest.mark.asyncio
async def test_generate_rejects_invalid_messages_before_http(monkeypatch):
    client = make_client(monkeypatch)
    client.client.post = AsyncMock()

    with pytest.raises(LLMResponseError, match="No messages were provided"):
        await client.generate([])

    client.client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_non_retryable_http_error(monkeypatch):
    client = make_client(monkeypatch)
    client.client.post = AsyncMock(return_value=make_response(
        status_code=400,
        json_data={"error": "Bad request"},
    ))

    with pytest.raises(LLMResponseError, match=r"HTTP 400.*Bad request"):
        await client.generate([
            {"role": "user", "content": "Hello"}
        ])

    client.client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_retries_429(monkeypatch):
    client = make_client(monkeypatch, max_retries=3)
    client.client.post = AsyncMock(side_effect=[
        make_response(429, {"error": "Too many requests"}),
        make_response(200, {"message": {"content": "Success"}}),
    ])
    wait = AsyncMock()
    monkeypatch.setattr(client, "_wait_before_retry", wait)

    result = await client.generate([
        {"role": "user", "content": "Hello"}
    ])

    assert result == "Success"
    assert client.client.post.await_count == 2
    wait.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_generate_retries_5xx(monkeypatch):
    client = make_client(monkeypatch, max_retries=3)
    client.client.post = AsyncMock(side_effect=[
        make_response(503, {"error": "Unavailable"}),
        make_response(200, {"message": {"content": "Recovered"}}),
    ])
    wait = AsyncMock()
    monkeypatch.setattr(client, "_wait_before_retry", wait)

    assert await client.generate([
        {"role": "user", "content": "Hello"}
    ]) == "Recovered"
    assert client.client.post.await_count == 2
    wait.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_generate_exhausts_retryable_errors(monkeypatch):
    client = make_client(monkeypatch, max_retries=3)
    client.client.post = AsyncMock(return_value=make_response(
        503, {"error": "Unavailable"}
    ))
    wait = AsyncMock()
    monkeypatch.setattr(client, "_wait_before_retry", wait)

    with pytest.raises(LLMResponseError, match=r"HTTP 503.*Unavailable"):
        await client.generate([
            {"role": "user", "content": "Hello"}
        ])

    assert client.client.post.await_count == 3
    assert wait.await_count == 2


@pytest.mark.asyncio
async def test_generate_connection_error_retries(monkeypatch):
    client = make_client(monkeypatch, max_retries=3)
    client.client.post = AsyncMock(
        side_effect=httpx.ConnectError("connection failed")
    )
    wait = AsyncMock()
    monkeypatch.setattr(client, "_wait_before_retry", wait)

    with pytest.raises(LLMConnectionError, match="Unable to connect to Ollama"):
        await client.generate([
            {"role": "user", "content": "Hello"}
        ])

    assert client.client.post.await_count == 3
    assert wait.await_count == 2


@pytest.mark.asyncio
async def test_generate_timeout_retries(monkeypatch):
    client = make_client(monkeypatch, max_retries=2)
    client.client.post = AsyncMock(
        side_effect=httpx.ReadTimeout("timed out")
    )
    wait = AsyncMock()
    monkeypatch.setattr(client, "_wait_before_retry", wait)

    with pytest.raises(LLMConnectionError, match="timed out"):
        await client.generate([
            {"role": "user", "content": "Hello"}
        ])

    assert client.client.post.await_count == 2
    assert wait.await_count == 1


@pytest.mark.asyncio
async def test_generate_http_error_becomes_connection_error(monkeypatch):
    client = make_client(monkeypatch)
    client.client.post = AsyncMock(
        side_effect=httpx.HTTPError("network failure")
    )

    with pytest.raises(LLMConnectionError, match="Communication with Ollama failed"):
        await client.generate([
            {"role": "user", "content": "Hello"}
        ])


@pytest.mark.asyncio
async def test_generate_unexpected_error_becomes_response_error(monkeypatch):
    client = make_client(monkeypatch)
    client.client.post = AsyncMock(
        side_effect=RuntimeError("unexpected")
    )

    with pytest.raises(LLMResponseError, match="Unexpected LLM error"):
        await client.generate([
            {"role": "user", "content": "Hello"}
        ])


@pytest.mark.asyncio
async def test_generate_invalid_json(monkeypatch):
    client = make_client(monkeypatch)
    client.client.post = AsyncMock(return_value=make_response())

    with pytest.raises(LLMResponseError, match="invalid JSON"):
        await client.generate([
            {"role": "user", "content": "Hello"}
        ])


@pytest.mark.asyncio
async def test_generate_empty_response(monkeypatch):
    client = make_client(monkeypatch)
    client.client.post = AsyncMock(return_value=make_response(
        json_data={"message": {"content": "   "}}
    ))

    with pytest.raises(LLMResponseError, match="empty response"):
        await client.generate([
            {"role": "user", "content": "Hello"}
        ])


# Streaming

@pytest.mark.asyncio
async def test_stream_generate_success(monkeypatch):
    client = make_client(monkeypatch)
    response = make_stream_response([
        json.dumps({"message": {"content": "Hello "}}),
        json.dumps({"message": {"content": "world"}}),
        json.dumps({"done": True}),
    ])
    client.client.stream = MagicMock(
        return_value=AsyncStreamContextManager(response)
    )

    tokens = [
        token async for token in client.stream_generate([
            {"role": "user", "content": "Hello"}
        ])
    ]

    assert tokens == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_stream_generate_passes_stream_true_and_system_prompt(monkeypatch):
    client = make_client(monkeypatch)
    response = make_stream_response([
        json.dumps({"message": {"content": "Answer"}}),
        json.dumps({"done": True}),
    ])
    client.client.stream = MagicMock(
        return_value=AsyncStreamContextManager(response)
    )

    tokens = [
        token async for token in client.stream_generate(
            [{"role": "user", "content": "Hello"}],
            system_prompt="Be helpful",
        )
    ]

    assert tokens == ["Answer"]
    payload = client.client.stream.call_args.kwargs["json"]
    assert payload["stream"] is True
    assert payload["system"] == "Be helpful"


@pytest.mark.asyncio
async def test_stream_generate_ignores_empty_lines_and_invalid_json(monkeypatch):
    client = make_client(monkeypatch)
    response = make_stream_response([
        "",
        "not-json",
        json.dumps({"message": {"content": ""}}),
        json.dumps({"message": {"content": "Answer"}}),
        json.dumps({"done": True}),
    ])
    client.client.stream = MagicMock(
        return_value=AsyncStreamContextManager(response)
    )

    tokens = [
        token async for token in client.stream_generate([
            {"role": "user", "content": "Hello"}
        ])
    ]

    assert tokens == ["Answer"]


@pytest.mark.asyncio
async def test_stream_generate_allows_missing_done_event(monkeypatch):
    client = make_client(monkeypatch)
    response = make_stream_response([
        json.dumps({"message": {"content": "Answer"}}),
    ])
    client.client.stream = MagicMock(
        return_value=AsyncStreamContextManager(response)
    )

    tokens = [
        token async for token in client.stream_generate([
            {"role": "user", "content": "Hello"}
        ])
    ]

    assert tokens == ["Answer"]


@pytest.mark.asyncio
async def test_stream_generate_rejects_empty_stream(monkeypatch):
    client = make_client(monkeypatch)
    response = make_stream_response([json.dumps({"done": True})])
    client.client.stream = MagicMock(
        return_value=AsyncStreamContextManager(response)
    )

    with pytest.raises(LLMResponseError, match="empty streaming response"):
        await anext(client.stream_generate([
            {"role": "user", "content": "Hello"}
        ]))


@pytest.mark.asyncio
async def test_stream_generate_http_error(monkeypatch):
    client = make_client(monkeypatch)

    response = httpx.Response(
        500,
        json={"error": "Server error"},
    )

    client.client.stream = MagicMock(
        return_value=AsyncStreamContextManager(response)
    )

    with pytest.raises(
        LLMResponseError,
        match=r"HTTP 500.*Server error",
    ):
        await anext(
            client.stream_generate([
                {"role": "user", "content": "Hello"}
            ])
        )


@pytest.mark.asyncio
async def test_stream_generate_connection_error(monkeypatch):
    client = make_client(monkeypatch)

    class FailingStream:
        async def __aenter__(self):
            raise httpx.ConnectError("connection failed")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    client.client.stream = MagicMock(return_value=FailingStream())

    with pytest.raises(LLMConnectionError, match="Unable to connect to Ollama"):
        await anext(client.stream_generate([
            {"role": "user", "content": "Hello"}
        ]))


@pytest.mark.asyncio
async def test_stream_generate_timeout_error(monkeypatch):
    client = make_client(monkeypatch)

    class FailingStream:
        async def __aenter__(self):
            raise httpx.ReadTimeout("timed out")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    client.client.stream = MagicMock(return_value=FailingStream())

    with pytest.raises(LLMConnectionError, match="streaming request timed out"):
        await anext(client.stream_generate([
            {"role": "user", "content": "Hello"}
        ]))


@pytest.mark.asyncio
async def test_stream_generate_httpx_error(monkeypatch):
    client = make_client(monkeypatch)

    class FailingStream:
        async def __aenter__(self):
            raise httpx.HTTPError("network failure")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    client.client.stream = MagicMock(return_value=FailingStream())

    with pytest.raises(LLMConnectionError, match="Communication with Ollama failed"):
        await anext(client.stream_generate([
            {"role": "user", "content": "Hello"}
        ]))


@pytest.mark.asyncio
async def test_stream_generate_unexpected_error(monkeypatch):
    client = make_client(monkeypatch)

    class FailingStream:
        async def __aenter__(self):
            raise RuntimeError("unexpected")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    client.client.stream = MagicMock(return_value=FailingStream())

    with pytest.raises(LLMResponseError, match="Streaming generation failed"):
        await anext(client.stream_generate([
            {"role": "user", "content": "Hello"}
        ]))


@pytest.mark.asyncio
async def test_stream_generate_rejects_invalid_messages(monkeypatch):
    client = make_client(monkeypatch)
    client.client.stream = MagicMock()

    with pytest.raises(LLMResponseError, match="No messages were provided"):
        await anext(client.stream_generate([]))

    client.client.stream.assert_not_called()


# Shared instance

def test_shared_ollama_client_is_initialized():
    from app.llm.ollama_client import ollama_client
    assert isinstance(ollama_client, OllamaClient)
    
@pytest.mark.asyncio
async def test_generate_reraises_llm_connection_error(
    monkeypatch,
):
    client = make_client(
        monkeypatch,
        max_retries=1,
    )

    async def raise_connection_error(*args, **kwargs):
        raise LLMConnectionError(
            "Existing connection failure"
        )

    monkeypatch.setattr(
        client,
        "_wait_before_retry",
        raise_connection_error,
    )

    request = AsyncMock(
        side_effect=LLMConnectionError(
            "Existing connection failure"
        )
    )

    monkeypatch.setattr(
        client.client,
        "post",
        request,
    )

    with pytest.raises(
        LLMConnectionError,
        match="Existing connection failure",
    ):
        await client.generate(
            [{"role": "user", "content": "Hello"}]
        )
        
@pytest.mark.asyncio
async def test_stream_generate_reraises_llm_connection_error(
    monkeypatch,
):
    client = make_client(monkeypatch)

    async def raise_connection_error(*args, **kwargs):
        raise LLMConnectionError(
            "Streaming connection failure"
        )

    monkeypatch.setattr(
        client.client,
        "stream",
        raise_connection_error,
    )

    with pytest.raises(
        LLMConnectionError,
        match="Streaming connection failure",
    ):
        async for _ in client.stream_generate(
            [{"role": "user", "content": "Hello"}]
        ):
            pass

@pytest.mark.asyncio
async def test_stream_generate_reraises_llm_connection_error(
    monkeypatch,
):
    client = make_client(monkeypatch)

    class FailingStream:
        async def __aenter__(self):
            raise LLMConnectionError(
                "Streaming connection failure"
            )

        async def __aexit__(
            self,
            exc_type,
            exc,
            tb,
        ):
            return False

    monkeypatch.setattr(
        client.client,
        "stream",
        lambda *args, **kwargs: FailingStream(),
    )

    with pytest.raises(
        LLMConnectionError,
        match="Streaming connection failure",
    ):
        async for _ in client.stream_generate(
            [{"role": "user", "content": "Hello"}]
        ):
            pass
        
def test_extract_error_detail_returns_empty_when_text_fails():
    response = MagicMock(
        spec=httpx.Response
    )

    response.json.side_effect = ValueError(
        "Invalid JSON"
    )

    type(response).text = property(
        lambda self: (
            (_ for _ in ()).throw(
                RuntimeError(
                    "Unable to access response text"
                )
            )
        )
    )

    assert (
        OllamaClient._extract_error_detail(
            response
        )
        == ""
    )
    
@pytest.mark.asyncio
async def test_generate_reraises_llm_connection_error(
    monkeypatch,
):
    client = make_client(
        monkeypatch,
        max_retries=1,
    )

    client.client.post = AsyncMock(
        side_effect=LLMConnectionError(
            "Existing connection failure"
        )
    )

    with pytest.raises(
        LLMConnectionError,
        match="Existing connection failure",
    ):
        await client.generate(
            [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        )
        
def test_generate_raises_when_retry_loop_does_not_execute(
    monkeypatch,
):
    client = make_client(
        monkeypatch,
        max_retries=1,
    )

    client.max_retries = 0

    with pytest.raises(
        LLMResponseError,
        match="LLM generation failed",
    ):
        import asyncio

        asyncio.run(
            client.generate(
                [
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ]
            )
        )
        
@pytest.mark.asyncio
async def test_generate_reraises_llm_connection_error(
    monkeypatch,
):
    client = make_client(
        monkeypatch,
        max_retries=1,
    )

    client.client.post = AsyncMock(
        side_effect=LLMConnectionError(
            "Existing connection failure"
        )
    )

    with pytest.raises(
        LLMConnectionError,
        match="Existing connection failure",
    ):
        await client.generate(
            [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        )
        