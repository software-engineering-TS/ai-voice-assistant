import numpy as np
import pytest

import app.rag.embedding_service as embedding_module


class FakeEmbeddingModel:
    """
    Fake SentenceTransformer used for unit testing.

    This prevents the real embedding model from loading.
    """

    def __init__(self, dimension=3):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(
        self,
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        return np.array(
            [
                [0.1, 0.2, 0.3]
                for _ in chunks
            ]
        )


def create_service(monkeypatch, dimension=3):
    """
    Create EmbeddingService with a fake model.
    """

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: FakeEmbeddingModel(
            dimension=dimension
        ),
    )

    return embedding_module.EmbeddingService()


def test_generate_embeddings_successfully(
    monkeypatch,
):
    """
    Valid chunks should produce embeddings.
    """

    service = create_service(
        monkeypatch,
        dimension=3,
    )

    chunks = [
        "First document chunk.",
        "Second document chunk.",
    ]

    result = service.generate_embeddings(
        chunks
    )

    assert isinstance(result, list)
    assert len(result) == 2

    assert result[0] == [
        0.1,
        0.2,
        0.3,
    ]

    assert result[1] == [
        0.1,
        0.2,
        0.3,
    ]


def test_generate_embeddings_returns_empty_for_empty_input(
    monkeypatch,
):
    """
    Empty input should return an empty list.
    """

    service = create_service(
        monkeypatch
    )

    result = service.generate_embeddings([])

    assert result == []


def test_generate_embeddings_rejects_non_string_chunks(
    monkeypatch,
):
    """
    Non-string chunks should raise ValueError.
    """

    service = create_service(
        monkeypatch
    )

    chunks = [
        "Valid chunk.",
        None,
        123,
        "Another valid chunk.",
    ]

    with pytest.raises(
        ValueError,
        match="Chunk 1 must be a string.",
    ):

        service.generate_embeddings(
            chunks
        )


def test_generate_embeddings_rejects_empty_chunks(
    monkeypatch,
):
    """
    Empty or whitespace-only chunks should
    raise ValueError.
    """

    service = create_service(
        monkeypatch
    )

    chunks = [
        "Valid chunk.",
        "   ",
        "\n\n",
        "Another valid chunk.",
    ]

    with pytest.raises(
        ValueError,
        match="Chunk 1 is empty.",
    ):

        service.generate_embeddings(
            chunks
        )


def test_generate_embeddings_returns_python_lists(
    monkeypatch,
):
    """
    Numpy vectors should be converted into
    normal Python lists.
    """

    service = create_service(
        monkeypatch
    )

    result = service.generate_embeddings(
        ["Test chunk."]
    )

    assert isinstance(result, list)
    assert isinstance(result[0], list)

    assert all(
        isinstance(value, float)
        for value in result[0]
    )


def test_embedding_dimension_is_detected(
    monkeypatch,
):
    """
    EmbeddingService should determine the model
    embedding dimension during initialization.
    """

    service = create_service(
        monkeypatch,
        dimension=3,
    )

    assert service.embedding_dimension == 3


def test_embedding_dimension_mismatch_raises_error(
    monkeypatch,
):
    """
    A vector with the wrong dimension should
    raise ValueError.
    """

    class WrongDimensionModel:

        def get_sentence_embedding_dimension(
            self,
        ):
            return 3

        def encode(
            self,
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        ):
            return np.array(
                [
                    [0.1, 0.2]
                    for _ in chunks
                ]
            )

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: WrongDimensionModel(),
    )

    service = embedding_module.EmbeddingService()

    with pytest.raises(ValueError):

        service.generate_embeddings(
            ["Test chunk."]
        )


def test_embedding_model_failure_propagates(
    monkeypatch,
):
    """
    Model prediction failures should not be silently
    swallowed by the embedding service.
    """

    class FailingModel:

        def get_sentence_embedding_dimension(
            self,
        ):
            return 3

        def encode(
            self,
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        ):
            raise RuntimeError(
                "Embedding model failed."
            )

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: FailingModel(),
    )

    service = embedding_module.EmbeddingService()

    with pytest.raises(RuntimeError):

        service.generate_embeddings(
            ["Test chunk."]
        )
        
def test_embedding_service_rejects_missing_model(monkeypatch):
    monkeypatch.setattr(
        embedding_module.settings,
        "EMBEDDING_MODEL",
        "",
    )

    with pytest.raises(
        ValueError,
        match="EMBEDDING_MODEL is not configured.",
    ):
        embedding_module.EmbeddingService()
        
def test_embedding_service_model_initialization_failure(
    monkeypatch,
):
    class FailingModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Model loading failed.")

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        FailingModel,
    )

    with pytest.raises(RuntimeError, match="Model loading failed."):
        embedding_module.EmbeddingService()
        
def test_embedding_dimension_uses_get_embedding_dimension(
    monkeypatch,
):
    class ModelWithEmbeddingDimension:
        def get_embedding_dimension(self):
            return 768

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: ModelWithEmbeddingDimension(),
    )

    service = embedding_module.EmbeddingService()

    assert service.embedding_dimension == 768
    
def test_embedding_dimension_detection_failure(
    monkeypatch,
):
    class BrokenDimensionModel:
        def get_embedding_dimension(self):
            raise RuntimeError(
                "Dimension detection failed."
            )

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: BrokenDimensionModel(),
    )

    with pytest.raises(
        RuntimeError,
        match="Dimension detection failed.",
    ):
        embedding_module.EmbeddingService()
        
def test_embedding_service_rejects_invalid_dimension(
    monkeypatch,
):
    class InvalidDimensionModel:
        def get_embedding_dimension(self):
            return 0

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: InvalidDimensionModel(),
    )

    with pytest.raises(
        ValueError,
        match="Embedding model returned an invalid embedding dimension.",
    ):
        embedding_module.EmbeddingService()
        
def test_generate_embeddings_rejects_none_result(
    monkeypatch,
):
    class NoneResultModel:
        def get_sentence_embedding_dimension(self):
            return 3

        def encode(
            self,
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        ):
            return None

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: NoneResultModel(),
    )

    service = embedding_module.EmbeddingService()

    with pytest.raises(
        RuntimeError,
        match="Embedding model returned no result.",
    ):
        service.generate_embeddings(["Test chunk."])
        
def test_generate_embeddings_rejects_invalid_output_type(
    monkeypatch,
):
    class InvalidOutputModel:
        def get_sentence_embedding_dimension(self):
            return 3

        def encode(
            self,
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        ):
            return object()

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: InvalidOutputModel(),
    )

    service = embedding_module.EmbeddingService()

    with pytest.raises(
        RuntimeError,
        match="Invalid embedding model output.",
    ):
        service.generate_embeddings(["Test chunk."])
        
def test_generate_embeddings_rejects_empty_result(
    monkeypatch,
):
    class EmptyResultModel:
        def get_sentence_embedding_dimension(self):
            return 3

        def encode(
            self,
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        ):
            return np.array([])

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: EmptyResultModel(),
    )

    service = embedding_module.EmbeddingService()

    with pytest.raises(
        RuntimeError,
        match="Embedding model returned no vectors.",
    ):
        service.generate_embeddings(["Test chunk."])
        
def test_generate_embeddings_rejects_count_mismatch(
    monkeypatch,
):
    class CountMismatchModel:
        def get_sentence_embedding_dimension(self):
            return 3

        def encode(
            self,
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        ):
            return np.array([
                [0.1, 0.2, 0.3]
            ])

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: CountMismatchModel(),
    )

    service = embedding_module.EmbeddingService()

    with pytest.raises(
        RuntimeError,
        match="Embedding count does not match",
    ):
        service.generate_embeddings(
            [
                "First chunk",
                "Second chunk",
            ]
        )
        
def test_generate_embeddings_rejects_non_list_vector(
    monkeypatch,
):
    class InvalidVectorModel:
        def get_sentence_embedding_dimension(self):
            return 3

        def encode(
            self,
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        ):
            class FakeResult:
                def tolist(self):
                    return (
                        "invalid-vector",
                    )

            return FakeResult()

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        lambda model_name: InvalidVectorModel(),
    )

    service = embedding_module.EmbeddingService()

    with pytest.raises(
        RuntimeError,
        match="Invalid embedding returned",
    ):
        service.generate_embeddings(["Test chunk."])