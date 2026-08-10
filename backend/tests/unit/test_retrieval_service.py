import pytest

import app.rag.retrieval_service as retrieval_module


# ============================================================
# Fake Dependencies
# ============================================================


class FakeEmbeddingService:
    """
    Fake embedding service for unit tests.
    """

    def __init__(self, embeddings=None):
        self.embeddings = (
            embeddings
            if embeddings is not None
            else [[0.1, 0.2, 0.3]]
        )

        self.calls = []

    def generate_embeddings(self, chunks):
        self.calls.append(chunks)

        return self.embeddings


class FakeVectorStore:
    """
    Fake vector store for unit tests.
    """

    def __init__(self, results=None):
        self.results = (
            results
            if results is not None
            else {
                "documents": [
                    [
                        "Test document content."
                    ]
                ],
                "metadatas": [
                    [
                        {
                            "source": "test.pdf",
                            "file_type": "pdf",
                            "chunk": 1,
                        }
                    ]
                ],
                "distances": [
                    [
                        0.1
                    ]
                ],
            }
        )

        self.calls = []

    def search(
        self,
        query_embedding,
        top_k,
        documents=None,
    ):
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "documents": documents,
            }
        )

        return self.results


class FakeRerankerService:
    """
    Fake reranker for unit tests.
    """

    def __init__(self, results=None):
        self.results = (
            results
            if results is not None
            else [
                {
                    "chunk": "Test document content.",
                    "metadata": {
                        "source": "test.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.9,
                }
            ]
        )

        self.calls = []

    def rerank(
        self,
        query,
        documents,
        top_k,
    ):
        self.calls.append(
            {
                "query": query,
                "documents": documents,
                "top_k": top_k,
            }
        )

        return self.results


# ============================================================
# Service Factory
# ============================================================


def create_service(
    monkeypatch,
    embedding_service=None,
    vector_store=None,
    reranker_service=None,
):
    """
    Create RetrievalService with mocked dependencies.
    """

    if embedding_service is None:
        embedding_service = FakeEmbeddingService()

    if vector_store is None:
        vector_store = FakeVectorStore()

    if reranker_service is None:
        reranker_service = FakeRerankerService()

    monkeypatch.setattr(
        retrieval_module,
        "embedding_service",
        embedding_service,
    )

    monkeypatch.setattr(
        retrieval_module,
        "vector_store",
        vector_store,
    )

    monkeypatch.setattr(
        retrieval_module,
        "reranker_service",
        reranker_service,
    )

    return retrieval_module.RetrievalService()


# ============================================================
# Empty Result
# ============================================================


def test_empty_result_structure():
    """
    Empty retrieval results should always have the
    same response structure.
    """

    result = (
        retrieval_module.RetrievalService
        ._empty_result()
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }


# ============================================================
# Query Validation
# ============================================================


def test_retrieve_rejects_non_string_query(
    monkeypatch,
):
    """
    Query must be a string.
    """

    service = create_service(
        monkeypatch
    )

    with pytest.raises(
        TypeError,
        match="query must be a string.",
    ):

        service.retrieve(
            query=123
        )


def test_retrieve_returns_empty_for_empty_query(
    monkeypatch,
):
    """
    Empty queries should return an empty result.
    """

    service = create_service(
        monkeypatch
    )

    result = service.retrieve(
        query=""
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }


def test_retrieve_returns_empty_for_whitespace_query(
    monkeypatch,
):
    """
    Whitespace-only queries should return an empty result.
    """

    service = create_service(
        monkeypatch
    )

    result = service.retrieve(
        query="   "
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }


# ============================================================
# Document Normalization
# ============================================================


def test_normalize_documents_successfully():
    """
    Document names should be stripped and deduplicated.
    """

    result = (
        retrieval_module.RetrievalService
        ._normalize_documents(
            [
                " resume.pdf ",
                "resume.pdf",
                "project.pdf",
                " ",
                "",
            ]
        )
    )

    assert result == [
        "resume.pdf",
        "project.pdf",
    ]


def test_normalize_documents_ignores_non_string_values():
    """
    Non-string document names should be ignored.
    """

    result = (
        retrieval_module.RetrievalService
        ._normalize_documents(
            [
                "resume.pdf",
                None,
                123,
                "project.pdf",
            ]
        )
    )

    assert result == [
        "resume.pdf",
        "project.pdf",
    ]


def test_normalize_documents_returns_empty_for_none():
    """
    None should produce an empty document selection.
    """

    result = (
        retrieval_module.RetrievalService
        ._normalize_documents(
            None
        )
    )

    assert result == []


def test_normalize_documents_returns_empty_for_empty_list():
    """
    Empty lists should produce an empty document selection.
    """

    result = (
        retrieval_module.RetrievalService
        ._normalize_documents(
            []
        )
    )

    assert result == []


# ============================================================
# Basic Retrieval Pipeline
# ============================================================


def test_retrieve_successfully(
    monkeypatch,
):
    """
    A valid query should execute the complete
    retrieval pipeline.
    """

    embedding_service = (
        FakeEmbeddingService(
            embeddings=[
                [0.1, 0.2, 0.3]
            ]
        )
    )

    vector_store = FakeVectorStore()

    reranker_service = (
        FakeRerankerService()
    )

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
        vector_store=vector_store,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="What is this document about?"
    )

    assert result["context"]

    assert result["sources"] == [
        "test.pdf (Chunk 1)"
    ]

    assert len(
        result["documents"]
    ) == 1


def test_query_embedding_is_generated(
    monkeypatch,
):
    """
    The user query should be passed to the
    embedding service.
    """

    embedding_service = (
        FakeEmbeddingService()
    )

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
    )

    service.retrieve(
        query="Explain Python."
    )

    assert embedding_service.calls == [
        ["Explain Python."]
    ]


def test_vector_search_receives_query_embedding(
    monkeypatch,
):
    """
    The generated query embedding should be passed
    to the vector store.
    """

    embedding_service = (
        FakeEmbeddingService(
            embeddings=[
                [1.0, 2.0, 3.0]
            ]
        )
    )

    vector_store = FakeVectorStore()

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    service.retrieve(
        query="test"
    )

    assert (
        vector_store.calls[0][
            "query_embedding"
        ]
        == [1.0, 2.0, 3.0]
    )


def test_document_selection_is_passed_to_vector_store(
    monkeypatch,
):
    """
    Selected documents should be forwarded to
    vector search.
    """

    vector_store = FakeVectorStore()

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    service.retrieve(
        query="test",
        documents=[
            "resume.pdf",
            "project.pdf",
        ],
    )

    assert (
        vector_store.calls[0]["documents"]
        == [
            "resume.pdf",
            "project.pdf",
        ]
    )


def test_no_document_selection_searches_all_documents(
    monkeypatch,
):
    """
    None or empty document selection should cause
    vector search across all documents.
    """

    vector_store = FakeVectorStore()

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    service.retrieve(
        query="test",
        documents=None,
    )

    assert (
        vector_store.calls[0]["documents"]
        is None
    )


# ============================================================
# Embedding Handling
# ============================================================


def test_empty_embedding_returns_empty_result(
    monkeypatch,
):
    """
    If the embedding service returns no embeddings,
    retrieval should stop.
    """

    embedding_service = (
        FakeEmbeddingService(
            embeddings=[]
        )
    )

    vector_store = FakeVectorStore()

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = service.retrieve(
        query="test"
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }

    assert vector_store.calls == []


def test_empty_query_embedding_returns_empty_result(
    monkeypatch,
):
    """
    An empty generated embedding should stop retrieval.
    """

    embedding_service = (
        FakeEmbeddingService(
            embeddings=[
                []
            ]
        )
    )

    vector_store = FakeVectorStore()

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = service.retrieve(
        query="test"
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }

    assert vector_store.calls == []


def test_embedding_failure_propagates(
    monkeypatch,
):
    """
    Embedding failures should not be silently swallowed.
    """

    class FailingEmbeddingService:

        def generate_embeddings(
            self,
            chunks,
        ):
            raise RuntimeError(
                "Embedding failed."
            )

    service = create_service(
        monkeypatch,
        embedding_service=(
            FailingEmbeddingService()
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding failed.",
    ):

        service.retrieve(
            query="test"
        )


# ============================================================
# Vector Search Handling
# ============================================================


def test_empty_vector_search_returns_empty_result(
    monkeypatch,
):
    """
    No vector search results should stop the pipeline.
    """

    vector_store = FakeVectorStore(
        results={
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
    )

    reranker_service = (
        FakeRerankerService()
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }

    assert reranker_service.calls == []


def test_empty_chunks_returns_empty_result(
    monkeypatch,
):
    """
    Empty chunk arrays should stop retrieval.
    """

    vector_store = FakeVectorStore(
        results={
            "documents": [
                []
            ],
            "metadatas": [
                []
            ],
            "distances": [
                []
            ],
        }
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    result = service.retrieve(
        query="test"
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }


def test_incomplete_vector_results_are_handled(
    monkeypatch,
):
    """
    ChromaDB result arrays with different lengths should
    only process the aligned portion.
    """

    vector_store = FakeVectorStore(
        results={
            "documents": [
                [
                    "First",
                    "Second",
                    "Third",
                ]
            ],
            "metadatas": [
                [
                    {
                        "source": "a.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    {
                        "source": "a.pdf",
                        "file_type": "pdf",
                        "chunk": 2,
                    },
                ]
            ],
            "distances": [
                [
                    0.1,
                    0.2,
                    0.3,
                ]
            ],
        }
    )

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "First",
                    "metadata": {
                        "source": "a.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.9,
                },
                {
                    "chunk": "Second",
                    "metadata": {
                        "source": "a.pdf",
                        "file_type": "pdf",
                        "chunk": 2,
                    },
                    "distance": 0.2,
                    "rerank_score": 0.8,
                },
            ]
        )
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert len(
        result["documents"]
    ) == 2


def test_invalid_metadata_is_skipped(
    monkeypatch,
):
    """
    Results with non-dictionary metadata should be ignored.
    """

    vector_store = FakeVectorStore(
        results={
            "documents": [
                [
                    "Valid content",
                    "Invalid metadata content",
                ]
            ],
            "metadatas": [
                [
                    {
                        "source": "valid.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "invalid metadata",
                ]
            ],
            "distances": [
                [
                    0.1,
                    0.2,
                ]
            ],
        }
    )

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "Valid content",
                    "metadata": {
                        "source": "valid.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.9,
                }
            ]
        )
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert len(
        result["documents"]
    ) == 1


def test_invalid_distance_is_skipped(
    monkeypatch,
):
    """
    Invalid vector distances should be ignored.
    """

    vector_store = FakeVectorStore(
        results={
            "documents": [
                [
                    "Invalid distance",
                    "Valid content",
                ]
            ],
            "metadatas": [
                [
                    {
                        "source": "bad.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    {
                        "source": "good.pdf",
                        "file_type": "pdf",
                        "chunk": 2,
                    },
                ]
            ],
            "distances": [
                [
                    "not-a-number",
                    0.2,
                ]
            ],
        }
    )

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "Valid content",
                    "metadata": {
                        "source": "good.pdf",
                        "file_type": "pdf",
                        "chunk": 2,
                    },
                    "distance": 0.2,
                    "rerank_score": 0.8,
                }
            ]
        )
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert len(
        result["documents"]
    ) == 1


def test_vector_search_failure_propagates(
    monkeypatch,
):
    """
    Vector store failures should propagate.
    """

    class FailingVectorStore:

        def search(
            self,
            query_embedding,
            top_k,
            documents=None,
        ):
            raise RuntimeError(
                "Vector search failed."
            )

    service = create_service(
        monkeypatch,
        vector_store=FailingVectorStore(),
    )

    with pytest.raises(
        RuntimeError,
        match="Vector search failed.",
    ):

        service.retrieve(
            query="test"
        )


# ============================================================
# Reranking
# ============================================================


def test_reranker_receives_retrieved_documents(
    monkeypatch,
):
    """
    Valid vector results should be passed to the reranker.
    """

    reranker_service = (
        FakeRerankerService()
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    service.retrieve(
        query="test query"
    )

    assert len(
        reranker_service.calls
    ) == 1

    call = reranker_service.calls[0]

    assert call["query"] == "test query"

    assert call["top_k"] == (
        retrieval_module.settings.RAG_TOP_K
    )

    assert len(
        call["documents"]
    ) == 1


def test_empty_reranker_result_returns_empty_result(
    monkeypatch,
):
    """
    If reranking returns no documents, retrieval should
    return an empty result.
    """

    reranker_service = (
        FakeRerankerService(
            results=[]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }


def test_reranker_failure_propagates(
    monkeypatch,
):
    """
    Reranker failures should propagate.
    """

    class FailingReranker:

        def rerank(
            self,
            query,
            documents,
            top_k,
        ):
            raise RuntimeError(
                "Reranker failed."
            )

    service = create_service(
        monkeypatch,
        reranker_service=(
            FailingReranker()
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Reranker failed.",
    ):

        service.retrieve(
            query="test"
        )


# ============================================================
# Relevance Filtering
# ============================================================


def test_documents_below_threshold_are_removed(
    monkeypatch,
):
    """
    Documents below the configured rerank threshold
    should be rejected.
    """

    monkeypatch.setattr(
        retrieval_module.settings,
        "RERANK_SCORE_THRESHOLD",
        0.7,
    )

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "High score",
                    "metadata": {
                        "source": "high.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.9,
                },
                {
                    "chunk": "Low score",
                    "metadata": {
                        "source": "low.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.2,
                    "rerank_score": 0.5,
                },
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert len(
        result["documents"]
    ) == 1

    assert (
        result["documents"][0]["source"]
        == "high.pdf"
    )


def test_documents_equal_to_threshold_are_kept(
    monkeypatch,
):
    """
    Scores equal to the threshold should be accepted.
    """

    monkeypatch.setattr(
        retrieval_module.settings,
        "RERANK_SCORE_THRESHOLD",
        0.7,
    )

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "Threshold document",
                    "metadata": {
                        "source": "threshold.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.7,
                }
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert len(
        result["documents"]
    ) == 1


def test_invalid_rerank_score_is_skipped(
    monkeypatch,
):
    """
    Documents with invalid rerank scores should be skipped.
    """

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "Invalid score",
                    "metadata": {
                        "source": "invalid.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": "invalid",
                },
                {
                    "chunk": "Valid score",
                    "metadata": {
                        "source": "valid.pdf",
                        "file_type": "pdf",
                        "chunk": 2,
                    },
                    "distance": 0.2,
                    "rerank_score": 0.8,
                },
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert len(
        result["documents"]
    ) == 1

    assert (
        result["documents"][0]["source"]
        == "valid.pdf"
    )


def test_no_documents_above_threshold_returns_empty(
    monkeypatch,
):
    """
    If every document fails the threshold, retrieval
    should return an empty result.
    """

    monkeypatch.setattr(
        retrieval_module.settings,
        "RERANK_SCORE_THRESHOLD",
        0.9,
    )

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "Low score",
                    "metadata": {
                        "source": "low.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.5,
                }
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }


# ============================================================
# Duplicate Removal
# ============================================================


def test_duplicate_chunks_are_removed(
    monkeypatch,
):
    """
    Same source + chunk number should only appear once.
    """

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "Duplicate content one",
                    "metadata": {
                        "source": "resume.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.9,
                },
                {
                    "chunk": "Duplicate content two",
                    "metadata": {
                        "source": "resume.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.2,
                    "rerank_score": 0.8,
                },
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert len(
        result["documents"]
    ) == 1


def test_same_chunk_number_from_different_sources_is_kept(
    monkeypatch,
):
    """
    Same chunk number from different files should not
    be treated as a duplicate.
    """

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "Resume chunk",
                    "metadata": {
                        "source": "resume.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.9,
                },
                {
                    "chunk": "Project chunk",
                    "metadata": {
                        "source": "project.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.2,
                    "rerank_score": 0.8,
                },
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert len(
        result["documents"]
    ) == 2


# ============================================================
# Top-K
# ============================================================


def test_final_top_k_is_respected(
    monkeypatch,
):
    """
    Final results should not exceed RAG_TOP_K.
    """

    monkeypatch.setattr(
        retrieval_module.settings,
        "RAG_TOP_K",
        2,
    )

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "One",
                    "metadata": {
                        "source": "a.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.9,
                },
                {
                    "chunk": "Two",
                    "metadata": {
                        "source": "b.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.2,
                    "rerank_score": 0.8,
                },
                {
                    "chunk": "Three",
                    "metadata": {
                        "source": "c.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.3,
                    "rerank_score": 0.7,
                },
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert len(
        result["documents"]
    ) == 2


# ============================================================
# Context and Sources
# ============================================================


def test_context_contains_document_information(
    monkeypatch,
):
    """
    Final context should contain source, file type,
    chunk number and content.
    """

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "Python is a programming language.",
                    "metadata": {
                        "source": "python.pdf",
                        "file_type": "pdf",
                        "chunk": 3,
                    },
                    "distance": 0.123456,
                    "rerank_score": 0.987654,
                }
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="What is Python?"
    )

    context = result["context"]

    assert "Document: python.pdf" in context
    assert "File Type: pdf" in context
    assert "Chunk: 3" in context
    assert (
        "Python is a programming language."
        in context
    )


def test_sources_are_generated_correctly(
    monkeypatch,
):
    """
    Sources should contain filename and chunk number.
    """

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "First",
                    "metadata": {
                        "source": "resume.pdf",
                        "file_type": "pdf",
                        "chunk": 2,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.9,
                },
                {
                    "chunk": "Second",
                    "metadata": {
                        "source": "project.pdf",
                        "file_type": "pdf",
                        "chunk": 4,
                    },
                    "distance": 0.2,
                    "rerank_score": 0.8,
                },
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert result["sources"] == [
        "resume.pdf (Chunk 2)",
        "project.pdf (Chunk 4)",
    ]


def test_duplicate_sources_are_not_repeated(
    monkeypatch,
):
    """
    Identical source labels should only appear once.
    """

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "First",
                    "metadata": {
                        "source": "resume.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.9,
                },
                {
                    "chunk": "Second",
                    "metadata": {
                        "source": "resume.pdf",
                        "file_type": "pdf",
                        "chunk": 2,
                    },
                    "distance": 0.2,
                    "rerank_score": 0.8,
                },
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    # These are different chunk labels, therefore both
    # are expected to be present.
    assert result["sources"] == [
        "resume.pdf (Chunk 1)",
        "resume.pdf (Chunk 2)",
    ]


def test_search_result_metadata_is_returned(
    monkeypatch,
):
    """
    Final documents should contain source, file type,
    chunk, distance and rerank score.
    """

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "Test content",
                    "metadata": {
                        "source": "test.pdf",
                        "file_type": "pdf",
                        "chunk": 5,
                    },
                    "distance": 0.123456,
                    "rerank_score": 0.987654,
                }
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert result["documents"] == [
        {
            "source": "test.pdf",
            "file_type": "pdf",
            "chunk": 5,
            "distance": 0.1235,
            "rerank_score": 0.9877,
        }
    ]


def test_empty_content_returns_empty_result(
    monkeypatch,
):
    """
    If all final chunks have empty content, retrieval
    should return an empty result.
    """

    reranker_service = (
        FakeRerankerService(
            results=[
                {
                    "chunk": "   ",
                    "metadata": {
                        "source": "empty.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    },
                    "distance": 0.1,
                    "rerank_score": 0.9,
                }
            ]
        )
    )

    service = create_service(
        monkeypatch,
        reranker_service=reranker_service,
    )

    result = service.retrieve(
        query="test"
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }
    
def test_retrieve_returns_empty_when_vector_results_are_incomplete(
    monkeypatch,
):
    """
    Incomplete ChromaDB results should return
    the standard empty retrieval structure.
    """

    vector_store = FakeVectorStore(
        results={
            "documents": [
                [
                    "Test document"
                ]
            ],
            "metadatas": [
                []
            ],
            "distances": [
                []
            ],
        }
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    result = service.retrieve(
        query="test query"
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }
    
def test_retrieve_returns_empty_when_retrieved_document_is_empty(
    monkeypatch,
):
    """
    Empty chunks should be ignored, and if no valid
    documents remain, the standard empty result is returned.
    """

    vector_store = FakeVectorStore(
        results={
            "documents": [
                [
                    ""
                ]
            ],
            "metadatas": [
                [
                    {
                        "source": "test.pdf",
                        "file_type": "pdf",
                        "chunk": 1,
                    }
                ]
            ],
            "distances": [
                [
                    0.1
                ]
            ],
        }
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    result = service.retrieve(
        query="test query"
    )

    assert result == {
        "context": "",
        "sources": [],
        "documents": [],
    }