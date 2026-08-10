import pytest

import app.rag.reranker_service as reranker_module


class FakeCrossEncoder:
    """
    Fake CrossEncoder used for unit testing.
    """

    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs):
        """
        Return deterministic scores based on
        document position.
        """

        scores = []

        for index, _pair in enumerate(pairs):
            scores.append(
                float(index + 1)
            )

        return scores


def create_service(monkeypatch):
    """
    Create RerankerService with a fake model.
    """

    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        FakeCrossEncoder,
    )

    return reranker_module.RerankerService()


def test_rerank_successfully(monkeypatch):
    """
    Valid documents should be reranked and returned.
    """

    service = create_service(monkeypatch)

    documents = [
        {
            "chunk": "First document",
            "metadata": {
                "source": "first.txt",
                "chunk": 1,
            },
        },
        {
            "chunk": "Second document",
            "metadata": {
                "source": "second.txt",
                "chunk": 2,
            },
        },
    ]

    result = service.rerank(
        query="test query",
        documents=documents,
        top_k=2,
    )

    assert len(result) == 2

    assert all(
        "rerank_score" in document
        for document in result
    )


def test_rerank_returns_empty_for_empty_query(
    monkeypatch,
):
    """
    Empty queries should return no results.
    """

    service = create_service(monkeypatch)

    documents = [
        {
            "chunk": "Test document",
        }
    ]

    result = service.rerank(
        query="",
        documents=documents,
        top_k=5,
    )

    assert result == []


def test_rerank_returns_empty_for_whitespace_query(
    monkeypatch,
):
    """
    Whitespace-only queries should return no results.
    """

    service = create_service(monkeypatch)

    result = service.rerank(
        query="   ",
        documents=[
            {
                "chunk": "Test document",
            }
        ],
        top_k=5,
    )

    assert result == []


def test_rerank_returns_empty_for_no_documents(
    monkeypatch,
):
    """
    No documents should return an empty list.
    """

    service = create_service(monkeypatch)

    result = service.rerank(
        query="test query",
        documents=[],
        top_k=5,
    )

    assert result == []


def test_rerank_returns_empty_for_invalid_top_k(
    monkeypatch,
):
    """
    top_k <= 0 should return an empty list.
    """

    service = create_service(monkeypatch)

    documents = [
        {
            "chunk": "Test document",
        }
    ]

    result = service.rerank(
        query="test query",
        documents=documents,
        top_k=0,
    )

    assert result == []


def test_rerank_skips_invalid_chunks(
    monkeypatch,
):
    """
    Documents without valid text chunks should
    be skipped.
    """

    service = create_service(monkeypatch)

    documents = [
        {
            "chunk": "Valid document",
        },
        {
            "chunk": "",
        },
        {
            "chunk": None,
        },
        {
            "chunk": "Another valid document",
        },
    ]

    result = service.rerank(
        query="test query",
        documents=documents,
        top_k=5,
    )

    assert len(result) == 2

    assert all(
        document["chunk"]
        for document in result
    )


def test_rerank_adds_scores(
    monkeypatch,
):
    """
    Each returned document should contain
    a rerank_score.
    """

    service = create_service(monkeypatch)

    documents = [
        {
            "chunk": "Document one",
        },
        {
            "chunk": "Document two",
        },
    ]

    result = service.rerank(
        query="test query",
        documents=documents,
        top_k=5,
    )

    assert all(
        isinstance(
            document["rerank_score"],
            float,
        )
        for document in result
    )


def test_rerank_sorts_by_score_descending(
    monkeypatch,
):
    """
    Results should be sorted from highest score
    to lowest score.
    """

    class SortingModel:

        def predict(self, pairs):
            return [
                0.2,
                0.9,
                0.5,
            ]

    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        lambda model_name: SortingModel(),
    )

    service = reranker_module.RerankerService()

    documents = [
        {
            "chunk": "Document one",
        },
        {
            "chunk": "Document two",
        },
        {
            "chunk": "Document three",
        },
    ]

    result = service.rerank(
        query="test query",
        documents=documents,
        top_k=5,
    )

    scores = [
        document["rerank_score"]
        for document in result
    ]

    assert scores == [
        0.9,
        0.5,
        0.2,
    ]


def test_rerank_respects_top_k(
    monkeypatch,
):
    """
    Only top_k documents should be returned.
    """

    class ScoreModel:

        def predict(self, pairs):
            return [
                0.1,
                0.9,
                0.5,
                0.8,
            ]

    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        lambda model_name: ScoreModel(),
    )

    service = reranker_module.RerankerService()

    documents = [
        {"chunk": "Document one"},
        {"chunk": "Document two"},
        {"chunk": "Document three"},
        {"chunk": "Document four"},
    ]

    result = service.rerank(
        query="test query",
        documents=documents,
        top_k=2,
    )

    assert len(result) == 2

    assert [
        document["rerank_score"]
        for document in result
    ] == [
        0.9,
        0.8,
    ]


def test_rerank_does_not_mutate_original_documents(
    monkeypatch,
):
    """
    Reranking should operate on copies of the
    original document dictionaries.
    """

    service = create_service(monkeypatch)

    documents = [
        {
            "chunk": "Original document",
            "metadata": {
                "source": "test.txt",
            },
        }
    ]

    result = service.rerank(
        query="test query",
        documents=documents,
        top_k=5,
    )

    assert "rerank_score" not in documents[0]

    assert "rerank_score" in result[0]


def test_rerank_prediction_failure_propagates(
    monkeypatch,
):
    """
    CrossEncoder prediction failures should not
    be silently swallowed.
    """

    class FailingModel:

        def predict(self, pairs):
            raise RuntimeError(
                "Reranker prediction failed."
            )

    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        lambda model_name: FailingModel(),
    )

    service = reranker_module.RerankerService()

    with pytest.raises(RuntimeError):

        service.rerank(
            query="test query",
            documents=[
                {
                    "chunk": "Test document",
                }
            ],
            top_k=5,
        )
        
def test_reranker_rejects_missing_model(monkeypatch):
    monkeypatch.setattr(
        reranker_module.settings,
        "RERANKER_MODEL",
        "",
    )

    with pytest.raises(
        ValueError,
        match="RERANKER_MODEL is not configured.",
    ):
        reranker_module.RerankerService()
        
def test_reranker_model_initialization_failure(monkeypatch):
    class FailingCrossEncoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Model loading failed.")

    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        FailingCrossEncoder,
    )

    with pytest.raises(
        RuntimeError,
        match="Model loading failed.",
    ):
        reranker_module.RerankerService()
        
def test_rerank_rejects_non_string_query(monkeypatch):
    service = create_service(monkeypatch)

    result = service.rerank(
        query=None,
        documents=[
            {"chunk": "Test document"}
        ],
    )

    assert result == []
    
def test_rerank_rejects_invalid_top_k(monkeypatch):
    service = create_service(monkeypatch)

    result = service.rerank(
        query="test",
        documents=[
            {"chunk": "Test document"}
        ],
        top_k="invalid",
    )

    assert result == []
    
@pytest.mark.parametrize("top_k", [0, -1])
def test_rerank_rejects_non_positive_top_k(
    monkeypatch,
    top_k,
):
    service = create_service(monkeypatch)

    result = service.rerank(
        query="test",
        documents=[
            {"chunk": "Test document"}
        ],
        top_k=top_k,
    )

    assert result == []
    
def test_rerank_handles_none_scores(monkeypatch):
    service = create_service(monkeypatch)

    service.model.predict = lambda pairs: None

    result = service.rerank(
        query="test",
        documents=[
            {"chunk": "Test document"}
        ],
    )

    assert result == []
    
def test_rerank_rejects_score_count_mismatch(monkeypatch):
    service = create_service(monkeypatch)

    service.model.predict = lambda pairs: [0.5, 0.8]

    with pytest.raises(
        RuntimeError,
        match="invalid number of scores",
    ):
        service.rerank(
            query="test",
            documents=[
                {"chunk": "Only one document"}
            ],
        )
        
def test_rerank_skips_invalid_score(monkeypatch):
    service = create_service(monkeypatch)

    service.model.predict = lambda pairs: [
        "invalid"
    ]

    result = service.rerank(
        query="test",
        documents=[
            {"chunk": "Test document"}
        ],
    )

    assert result == []
    
def test_rerank_keeps_valid_scores_and_skips_invalid(
    monkeypatch,
):
    service = create_service(monkeypatch)

    service.model.predict = lambda pairs: [
        "invalid",
        0.9,
    ]

    result = service.rerank(
        query="test",
        documents=[
            {"chunk": "Bad score document"},
            {"chunk": "Good score document"},
        ],
    )

    assert len(result) == 1
    assert result[0]["chunk"] == "Good score document"
    assert result[0]["rerank_score"] == 0.9
    
def test_rerank_skips_non_dict_documents(monkeypatch):
    service = create_service(monkeypatch)

    result = service.rerank(
        query="test",
        documents=[
            None,
            "invalid document",
            123,
            {"chunk": "Valid document"},
        ],
    )

    assert len(result) == 1
    assert result[0]["chunk"] == "Valid document"
    
def test_rerank_skips_whitespace_chunks(monkeypatch):
    service = create_service(monkeypatch)

    result = service.rerank(
        query="test",
        documents=[
            {"chunk": "   "},
            {"chunk": "\n\t"},
        ],
    )

    assert result == []