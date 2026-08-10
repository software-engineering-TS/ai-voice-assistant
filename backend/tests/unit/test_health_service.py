import pytest

import app.services.health_service as health_module


# ============================================================
# Fake Database
# ============================================================


class FakeDatabaseSession:
    def __init__(
        self,
        should_fail=False,
        close_should_fail=False,
    ):
        self.should_fail = should_fail
        self.close_should_fail = close_should_fail
        self.execute_calls = []
        self.closed = False

    def execute(self, query):
        self.execute_calls.append(query)

        if self.should_fail:
            raise RuntimeError(
                "Database unavailable"
            )

        return True

    def close(self):
        if self.close_should_fail:
            raise RuntimeError(
                "Database close failed"
            )

        self.closed = True


# ============================================================
# Fake Ollama Client
# ============================================================


class FakeResponse:
    def __init__(
        self,
        should_fail=False,
    ):
        self.should_fail = should_fail
        self.raise_for_status_called = False

    def raise_for_status(self):
        self.raise_for_status_called = True

        if self.should_fail:
            raise RuntimeError(
                "Ollama returned an error"
            )


class FakeHTTPClient:
    def __init__(
        self,
        is_closed=False,
        should_fail=False,
        response_should_fail=False,
    ):
        self.is_closed = is_closed
        self.should_fail = should_fail
        self.response_should_fail = (
            response_should_fail
        )
        self.get_calls = []

    async def get(self, url):
        self.get_calls.append(url)

        if self.should_fail:
            raise RuntimeError(
                "Ollama connection failed"
            )

        return FakeResponse(
            should_fail=self.response_should_fail
        )


class FakeOllamaClient:
    def __init__(
        self,
        is_closed=False,
        should_fail=False,
        response_should_fail=False,
        model="llama3.2:3b",
        base_url="http://localhost:11434",
    ):
        self.client = FakeHTTPClient(
            is_closed=is_closed,
            should_fail=should_fail,
            response_should_fail=response_should_fail,
        )

        self.model = model
        self.base_url = base_url


# ============================================================
# Fake Vector Store
# ============================================================


class FakeCollection:
    def __init__(
        self,
        count=5,
        should_fail=False,
    ):
        self.count_value = count
        self.should_fail = should_fail

    def count(self):
        if self.should_fail:
            raise RuntimeError(
                "ChromaDB unavailable"
            )

        return self.count_value


class FakeVectorStore:
    def __init__(
        self,
        count=5,
        statistics=None,
        count_should_fail=False,
        statistics_should_fail=False,
    ):
        self.collection = FakeCollection(
            count=count,
            should_fail=count_should_fail,
        )

        self.statistics = (
            statistics
            if statistics is not None
            else {
                "documents": 2,
                "chunks": 5,
            }
        )

        self.statistics_should_fail = (
            statistics_should_fail
        )

        self.statistics_calls = 0

    def get_statistics(self):
        self.statistics_calls += 1

        if self.statistics_should_fail:
            raise RuntimeError(
                "Statistics unavailable"
            )

        return self.statistics


# ============================================================
# Fake Embedding Service
# ============================================================


class FakeEmbeddingService:
    def __init__(
        self,
        model=object(),
        embedding_dimension=384,
    ):
        self.model = model
        self.embedding_dimension = (
            embedding_dimension
        )


# ============================================================
# Fake Reranker Service
# ============================================================


class FakeRerankerService:
    def __init__(
        self,
        model=object(),
    ):
        self.model = model


# ============================================================
# Service Factory
# ============================================================


def create_service(
    monkeypatch,
    *,
    database=None,
    ollama=None,
    vector_store=None,
    embedding_service=None,
    reranker_service=None,
):
    if database is None:
        database = FakeDatabaseSession()

    if ollama is None:
        ollama = FakeOllamaClient()

    if vector_store is None:
        vector_store = FakeVectorStore()

    if embedding_service is None:
        embedding_service = FakeEmbeddingService()

    if reranker_service is None:
        reranker_service = FakeRerankerService()

    monkeypatch.setattr(
        health_module,
        "SessionLocal",
        lambda: database,
    )

    monkeypatch.setattr(
        health_module,
        "ollama_client",
        ollama,
    )

    monkeypatch.setattr(
        health_module,
        "vector_store",
        vector_store,
    )

    monkeypatch.setattr(
        health_module,
        "embedding_service",
        embedding_service,
    )

    monkeypatch.setattr(
        health_module,
        "reranker_service",
        reranker_service,
    )

    return health_module.HealthService()


# ============================================================
# Healthy System
# ============================================================


@pytest.mark.asyncio
async def test_health_check_returns_healthy_when_all_dependencies_are_healthy(
    monkeypatch,
):
    service = create_service(
        monkeypatch
    )

    result = await service.check()

    assert result["status"] == "healthy"

    assert result["components"] == {
        "api": "healthy",
        "database": "healthy",
        "ollama": "healthy",
        "chromadb": "healthy",
        "embedding_model": "healthy",
        "reranker": "healthy",
    }


@pytest.mark.asyncio
async def test_health_check_returns_vector_statistics(
    monkeypatch,
):
    vector_store = FakeVectorStore(
        statistics={
            "documents": 10,
            "chunks": 250,
        }
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    result = await service.check()

    assert result["statistics"] == {
        "documents": 10,
        "chunks": 250,
    }


@pytest.mark.asyncio
async def test_health_check_returns_ollama_model(
    monkeypatch,
):
    ollama = FakeOllamaClient(
        model="llama3.2:3b"
    )

    service = create_service(
        monkeypatch,
        ollama=ollama,
    )

    result = await service.check()

    assert (
        result["ollama_model"]
        == "llama3.2:3b"
    )


# ============================================================
# Database
# ============================================================


@pytest.mark.asyncio
async def test_database_health_check_executes_select_one(
    monkeypatch,
):
    database = FakeDatabaseSession()

    service = create_service(
        monkeypatch,
        database=database,
    )

    await service.check()

    assert len(
        database.execute_calls
    ) == 1

    assert (
        str(
            database.execute_calls[0]
        ).strip()
        == "SELECT 1"
    )


@pytest.mark.asyncio
async def test_database_session_is_closed(
    monkeypatch,
):
    database = FakeDatabaseSession()

    service = create_service(
        monkeypatch,
        database=database,
    )

    await service.check()

    assert database.closed is True


@pytest.mark.asyncio
async def test_database_failure_marks_database_unhealthy(
    monkeypatch,
):
    database = FakeDatabaseSession(
        should_fail=True
    )

    service = create_service(
        monkeypatch,
        database=database,
    )

    result = await service.check()

    assert (
        result["components"]["database"]
        == "unhealthy"
    )

    assert result["status"] == "degraded"


@pytest.mark.asyncio
async def test_database_session_is_closed_after_database_failure(
    monkeypatch,
):
    database = FakeDatabaseSession(
        should_fail=True
    )

    service = create_service(
        monkeypatch,
        database=database,
    )

    await service.check()

    assert database.closed is True


@pytest.mark.asyncio
async def test_database_close_failure_does_not_fail_health_check(
    monkeypatch,
):
    database = FakeDatabaseSession(
        close_should_fail=True
    )

    service = create_service(
        monkeypatch,
        database=database,
    )

    result = await service.check()

    assert result["components"]["database"] == (
        "healthy"
    )


# ============================================================
# Ollama
# ============================================================


@pytest.mark.asyncio
async def test_ollama_health_check_calls_tags_endpoint(
    monkeypatch,
):
    ollama = FakeOllamaClient(
        base_url="http://localhost:11434"
    )

    service = create_service(
        monkeypatch,
        ollama=ollama,
    )

    await service.check()

    assert ollama.client.get_calls == [
        "http://localhost:11434/api/tags"
    ]


@pytest.mark.asyncio
async def test_ollama_closed_client_marks_unhealthy(
    monkeypatch,
):
    ollama = FakeOllamaClient(
        is_closed=True
    )

    service = create_service(
        monkeypatch,
        ollama=ollama,
    )

    result = await service.check()

    assert (
        result["components"]["ollama"]
        == "unhealthy"
    )

    assert (
        ollama.client.get_calls == []
    )


@pytest.mark.asyncio
async def test_ollama_connection_failure_marks_unhealthy(
    monkeypatch,
):
    ollama = FakeOllamaClient(
        should_fail=True
    )

    service = create_service(
        monkeypatch,
        ollama=ollama,
    )

    result = await service.check()

    assert (
        result["components"]["ollama"]
        == "unhealthy"
    )


@pytest.mark.asyncio
async def test_ollama_http_error_marks_unhealthy(
    monkeypatch,
):
    ollama = FakeOllamaClient(
        response_should_fail=True
    )

    service = create_service(
        monkeypatch,
        ollama=ollama,
    )

    result = await service.check()

    assert (
        result["components"]["ollama"]
        == "unhealthy"
    )


# ============================================================
# ChromaDB
# ============================================================


@pytest.mark.asyncio
async def test_chromadb_health_check_reads_collection_count(
    monkeypatch,
):
    vector_store = FakeVectorStore(
        count=123
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    result = await service.check()

    assert result["components"]["chromadb"] == (
        "healthy"
    )


@pytest.mark.asyncio
async def test_chromadb_failure_marks_unhealthy(
    monkeypatch,
):
    vector_store = FakeVectorStore(
        count_should_fail=True
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    result = await service.check()

    assert (
        result["components"]["chromadb"]
        == "unhealthy"
    )

    assert result["status"] == "degraded"


# ============================================================
# Embedding Model
# ============================================================


@pytest.mark.asyncio
async def test_embedding_model_is_healthy_when_initialized(
    monkeypatch,
):
    embedding_service = FakeEmbeddingService(
        model=object(),
        embedding_dimension=384,
    )

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
    )

    result = await service.check()

    assert (
        result["components"]["embedding_model"]
        == "healthy"
    )


@pytest.mark.asyncio
async def test_missing_embedding_model_marks_unhealthy(
    monkeypatch,
):
    embedding_service = FakeEmbeddingService(
        model=None,
        embedding_dimension=384,
    )

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
    )

    result = await service.check()

    assert (
        result["components"]["embedding_model"]
        == "unhealthy"
    )


@pytest.mark.asyncio
async def test_missing_embedding_dimension_marks_unhealthy(
    monkeypatch,
):
    embedding_service = FakeEmbeddingService(
        model=object(),
        embedding_dimension=None,
    )

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
    )

    result = await service.check()

    assert (
        result["components"]["embedding_model"]
        == "unhealthy"
    )


# ============================================================
# Reranker
# ============================================================


@pytest.mark.asyncio
async def test_reranker_is_healthy_when_initialized(
    monkeypatch,
):
    reranker_service = FakeRerankerService(
        model=object()
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = await service.check()

    assert (
        result["components"]["reranker"]
        == "healthy"
    )


@pytest.mark.asyncio
async def test_missing_reranker_model_marks_unhealthy(
    monkeypatch,
):
    reranker_service = FakeRerankerService(
        model=None
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = await service.check()

    assert (
        result["components"]["reranker"]
        == "unhealthy"
    )


# ============================================================
# Overall Status
# ============================================================


@pytest.mark.asyncio
async def test_overall_status_is_degraded_when_database_is_unhealthy(
    monkeypatch,
):
    database = FakeDatabaseSession(
        should_fail=True
    )

    service = create_service(
        monkeypatch,
        database=database,
    )

    result = await service.check()

    assert result["status"] == "degraded"


@pytest.mark.asyncio
async def test_overall_status_is_degraded_when_ollama_is_unhealthy(
    monkeypatch,
):
    ollama = FakeOllamaClient(
        is_closed=True
    )

    service = create_service(
        monkeypatch,
        ollama=ollama,
    )

    result = await service.check()

    assert result["status"] == "degraded"


@pytest.mark.asyncio
async def test_overall_status_is_degraded_when_chromadb_is_unhealthy(
    monkeypatch,
):
    vector_store = FakeVectorStore(
        count_should_fail=True
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    result = await service.check()

    assert result["status"] == "degraded"


@pytest.mark.asyncio
async def test_overall_status_is_degraded_when_embedding_model_is_unhealthy(
    monkeypatch,
):
    embedding_service = FakeEmbeddingService(
        model=None
    )

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
    )

    result = await service.check()

    assert result["status"] == "degraded"


@pytest.mark.asyncio
async def test_overall_status_is_degraded_when_reranker_is_unhealthy(
    monkeypatch,
):
    reranker_service = FakeRerankerService(
        model=None
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = await service.check()

    assert result["status"] == "degraded"


# ============================================================
# Statistics Failure
# ============================================================


@pytest.mark.asyncio
async def test_statistics_failure_returns_default_statistics(
    monkeypatch,
):
    vector_store = FakeVectorStore(
        statistics_should_fail=True
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    result = await service.check()

    assert result["statistics"] == {
        "documents": 0,
        "chunks": 0,
    }


@pytest.mark.asyncio
async def test_statistics_failure_does_not_make_system_degraded(
    monkeypatch,
):
    vector_store = FakeVectorStore(
        statistics_should_fail=True
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    result = await service.check()

    assert result["status"] == "healthy"


# ============================================================
# Multiple Failures
# ============================================================


@pytest.mark.asyncio
async def test_multiple_unhealthy_components_are_reported(
    monkeypatch,
):
    database = FakeDatabaseSession(
        should_fail=True
    )

    ollama = FakeOllamaClient(
        is_closed=True
    )

    vector_store = FakeVectorStore(
        count_should_fail=True
    )

    embedding_service = FakeEmbeddingService(
        model=None
    )

    reranker_service = FakeRerankerService(
        model=None
    )

    service = create_service(
        monkeypatch,
        database=database,
        ollama=ollama,
        vector_store=vector_store,
        embedding_service=embedding_service,
        reranker_service=reranker_service,
    )

    result = await service.check()

    assert result["status"] == "degraded"

    assert result["components"] == {
        "api": "healthy",
        "database": "unhealthy",
        "ollama": "unhealthy",
        "chromadb": "unhealthy",
        "embedding_model": "unhealthy",
        "reranker": "unhealthy",
    }


# ============================================================
# Result Structure
# ============================================================


@pytest.mark.asyncio
async def test_health_check_returns_expected_top_level_structure(
    monkeypatch,
):
    service = create_service(
        monkeypatch
    )

    result = await service.check()

    assert set(result.keys()) == {
        "status",
        "components",
        "statistics",
        "ollama_model",
    }


@pytest.mark.asyncio
async def test_health_check_contains_all_component_statuses(
    monkeypatch,
):
    service = create_service(
        monkeypatch
    )

    result = await service.check()

    expected_components = {
        "api",
        "database",
        "ollama",
        "chromadb",
        "embedding_model",
        "reranker",
    }

    assert set(
        result["components"].keys()
    ) == expected_components


@pytest.mark.asyncio
async def test_health_check_component_values_are_valid(
    monkeypatch,
):
    service = create_service(
        monkeypatch
    )

    result = await service.check()

    for component_status in (
        result["components"].values()
    ):
        assert component_status in {
            "healthy",
            "unhealthy",
        }


@pytest.mark.asyncio
async def test_health_check_overall_status_is_valid(
    monkeypatch,
):
    service = create_service(
        monkeypatch
    )

    result = await service.check()

    assert result["status"] in {
        "healthy",
        "degraded",
    }