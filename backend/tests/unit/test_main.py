import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.mark.asyncio
async def test_lifespan_startup_with_documents(monkeypatch):

    init_called = False

    def mock_init_db():
        nonlocal init_called
        init_called = True

    async def mock_close():
        pass

    monkeypatch.setattr(
        main,
        "init_db",
        mock_init_db,
    )

    monkeypatch.setattr(
        main.vector_store,
        "list_documents",
        lambda: [
            {
                "file_name": "test.pdf",
                "chunks": 5,
            }
        ],
    )

    monkeypatch.setattr(
        main.ollama_client,
        "close",
        mock_close,
    )

    async with main.lifespan(main.app):

        assert init_called is True


@pytest.mark.asyncio
async def test_lifespan_startup_without_documents(monkeypatch):

    monkeypatch.setattr(
        main,
        "init_db",
        lambda: None,
    )

    monkeypatch.setattr(
        main.vector_store,
        "list_documents",
        lambda: [],
    )

    async def mock_close():
        pass

    monkeypatch.setattr(
        main.ollama_client,
        "close",
        mock_close,
    )

    async with main.lifespan(main.app):
        pass


@pytest.mark.asyncio
async def test_lifespan_startup_failure(monkeypatch):

    def mock_init_db():
        raise RuntimeError("Database initialization failed")

    monkeypatch.setattr(
        main,
        "init_db",
        mock_init_db,
    )

    with pytest.raises(
        RuntimeError,
        match="Database initialization failed",
    ):
        async with main.lifespan(main.app):
            pass


@pytest.mark.asyncio
async def test_lifespan_shutdown_closes_ollama(monkeypatch):

    monkeypatch.setattr(
        main,
        "init_db",
        lambda: None,
    )

    monkeypatch.setattr(
        main.vector_store,
        "list_documents",
        lambda: [],
    )

    closed = False

    async def mock_close():
        nonlocal closed
        closed = True

    monkeypatch.setattr(
        main.ollama_client,
        "close",
        mock_close,
    )

    async with main.lifespan(main.app):
        pass

    assert closed is True


@pytest.mark.asyncio
async def test_lifespan_shutdown_failure_is_handled(monkeypatch):

    monkeypatch.setattr(
        main,
        "init_db",
        lambda: None,
    )

    monkeypatch.setattr(
        main.vector_store,
        "list_documents",
        lambda: [],
    )

    async def mock_close():
        raise RuntimeError("Ollama close failed")

    monkeypatch.setattr(
        main.ollama_client,
        "close",
        mock_close,
    )

    # Shutdown exception is caught by lifespan.
    async with main.lifespan(main.app):
        pass


def test_home_endpoint():

    with TestClient(main.app) as client:

        response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["application"] == main.settings.APP_NAME
    assert data["version"] == main.settings.APP_VERSION
    assert data["status"] == "Running"
    assert data["swagger"] == "/docs"
    assert data["redoc"] == "/redoc"
    assert data["health"] == "/health"


def test_application_metadata():

    assert main.app.title == main.settings.APP_NAME
    assert main.app.version == main.settings.APP_VERSION


def test_outputs_route_exists():
    routes = [
        route.path
        for route in main.app.routes
        if hasattr(route, "path")
    ]

    assert "/outputs" in routes


def test_root_route_exists():
    routes = [
        route.path
        for route in main.app.routes
        if hasattr(route, "path")
    ]

    assert "/" in routes

def test_optional_bearer_auth_protects_generated_files(monkeypatch):
    monkeypatch.setattr(main.settings, "API_ACCESS_TOKEN", "test-secret")
    client = TestClient(main.app, raise_server_exceptions=False)

    unauthorized = client.get(
        "/outputs/not-a-real-file.mp3",
        headers={"Origin": "http://localhost:8501"},
    )
    assert unauthorized.status_code == 401
    assert unauthorized.headers.get("access-control-allow-origin") == "http://localhost:8501"

    authorized = client.get(
        "/outputs/not-a-real-file.mp3",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert authorized.status_code == 404


def test_public_home_endpoint_remains_available_with_token_enabled(monkeypatch):
    monkeypatch.setattr(main.settings, "API_ACCESS_TOKEN", "test-secret")
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.get("/")
    assert response.status_code == 200