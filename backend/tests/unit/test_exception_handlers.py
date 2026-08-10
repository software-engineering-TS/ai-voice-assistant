import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.exceptions.custom_exceptions import (
    AssistantException,
    LLMConnectionError,
    LLMResponseError,
)
from app.exceptions.handlers import register_exception_handlers


# ============================================================
# Test App
# ============================================================

@pytest.fixture
def app():
    app = FastAPI()

    register_exception_handlers(app)

    @app.get("/llm-connection-error")
    async def llm_connection_error():
        raise LLMConnectionError(
            "Unable to connect to Ollama."
        )

    @app.get("/llm-response-error")
    async def llm_response_error():
        raise LLMResponseError(
            "Ollama returned an invalid response."
        )

    @app.get("/assistant-error")
    async def assistant_error():
        raise AssistantException(
            "Something went wrong."
        )

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ============================================================
# Exception Class Tests
# ============================================================

def test_assistant_exception_is_base_exception():
    exception = AssistantException(
        "Base assistant error"
    )

    assert isinstance(exception, Exception)
    assert str(exception) == "Base assistant error"


def test_llm_connection_error_inherits_from_assistant_exception():
    exception = LLMConnectionError(
        "Connection failed"
    )

    assert isinstance(
        exception,
        AssistantException,
    )

    assert isinstance(
        exception,
        Exception,
    )

    assert str(exception) == "Connection failed"


def test_llm_response_error_inherits_from_assistant_exception():
    exception = LLMResponseError(
        "Invalid response"
    )

    assert isinstance(
        exception,
        AssistantException,
    )

    assert isinstance(
        exception,
        Exception,
    )

    assert str(exception) == "Invalid response"


# ============================================================
# LLM Connection Handler
# ============================================================

def test_llm_connection_error_returns_503(client):
    response = client.get(
        "/llm-connection-error"
    )

    assert response.status_code == 503


def test_llm_connection_error_returns_error_status(client):
    response = client.get(
        "/llm-connection-error"
    )

    assert response.json()["status"] == "error"


def test_llm_connection_error_returns_exception_message(client):
    response = client.get(
        "/llm-connection-error"
    )

    assert response.json()["message"] == (
        "Unable to connect to Ollama."
    )


def test_llm_connection_error_returns_expected_structure(client):
    response = client.get(
        "/llm-connection-error"
    )

    assert response.json() == {
        "status": "error",
        "message": "Unable to connect to Ollama.",
    }


# ============================================================
# LLM Response Handler
# ============================================================

def test_llm_response_error_returns_500(client):
    response = client.get(
        "/llm-response-error"
    )

    assert response.status_code == 500


def test_llm_response_error_returns_error_status(client):
    response = client.get(
        "/llm-response-error"
    )

    assert response.json()["status"] == "error"


def test_llm_response_error_returns_exception_message(client):
    response = client.get(
        "/llm-response-error"
    )

    assert response.json()["message"] == (
        "Ollama returned an invalid response."
    )


def test_llm_response_error_returns_expected_structure(client):
    response = client.get(
        "/llm-response-error"
    )

    assert response.json() == {
        "status": "error",
        "message": (
            "Ollama returned an invalid response."
        ),
    }


# ============================================================
# Assistant Exception Handler
# ============================================================

def test_assistant_exception_returns_500(client):
    response = client.get(
        "/assistant-error"
    )

    assert response.status_code == 500


def test_assistant_exception_returns_error_status(client):
    response = client.get(
        "/assistant-error"
    )

    assert response.json()["status"] == "error"


def test_assistant_exception_returns_exception_message(client):
    response = client.get(
        "/assistant-error"
    )

    assert response.json()["message"] == (
        "Something went wrong."
    )


def test_assistant_exception_returns_expected_structure(client):
    response = client.get(
        "/assistant-error"
    )

    assert response.json() == {
        "status": "error",
        "message": "Something went wrong.",
    }


# ============================================================
# Handler Registration
# ============================================================

def test_register_exception_handlers_registers_all_handlers():
    app = FastAPI()

    register_exception_handlers(app)

    handlers = app.exception_handlers

    assert LLMConnectionError in handlers
    assert LLMResponseError in handlers
    assert AssistantException in handlers