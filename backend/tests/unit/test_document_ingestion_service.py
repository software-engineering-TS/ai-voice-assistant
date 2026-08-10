import io
from pathlib import Path

from app.config.settings import settings
from app.rag.metadata_builder import metadata_builder
from app.rag.vector_store import vector_store
from app.rag.loaders.document_loader import document_loader
from app.rag.utils.file_hash import file_hash
from app.rag.text_splitter import text_splitter_service
import pytest
from fastapi import UploadFile

import app.services.document_ingestion_service as ingestion_module


# ============================================================
# Fake Dependencies
# ============================================================


class FakeFileHash:
    """
    Fake SHA-256 service.
    """

    def __init__(self, value="test-hash-123"):
        self.value = value
        self.calls = []

    def sha256(self, file_path):
        self.calls.append(file_path)
        return self.value


class FakeVectorStore:
    """
    Fake vector store.
    """

    def __init__(self, exists=False):
        self.exists = exists
        self.document_exists_calls = []
        self.add_documents_calls = []

    def document_exists(self, file_hash):
        self.document_exists_calls.append(file_hash)
        return self.exists

    def add_documents(
        self,
        chunks,
        embeddings,
        metadata,
    ):
        self.add_documents_calls.append(
            {
                "chunks": chunks,
                "embeddings": embeddings,
                "metadata": metadata,
            }
        )


class FakeDocumentLoader:
    """
    Fake document loader.
    """

    def __init__(self, text="This is test document content."):
        self.text = text
        self.calls = []

    def extract_text(self, file_path):
        self.calls.append(file_path)
        return self.text


class FakeTextSplitter:
    """
    Fake text splitter.
    """

    def __init__(self, chunks=None):
        self.chunks = (
            chunks
            if chunks is not None
            else [
                "First chunk.",
                "Second chunk.",
            ]
        )

        self.calls = []

    def split(self, text):
        self.calls.append(text)
        return self.chunks


class FakeMetadataBuilder:
    """
    Fake metadata builder.
    """

    def __init__(self, metadata=None):
        self.metadata = metadata
        self.calls = []

    def build(
        self,
        file_path,
        total_chunks,
        file_hash,
    ):
        self.calls.append(
            {
                "file_path": file_path,
                "total_chunks": total_chunks,
                "file_hash": file_hash,
            }
        )

        if self.metadata is not None:
            return self.metadata

        return [
            {
                "source": Path(file_path).name,
                "file_type": "txt",
                "chunk": index + 1,
                "file_hash": file_hash,
            }
            for index in range(total_chunks)
        ]


class FakeEmbeddingService:
    """
    Fake embedding service.
    """

    def __init__(self, embeddings=None):
        self.embeddings = (
            embeddings
            if embeddings is not None
            else [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )

        self.calls = []

    def generate_embeddings(self, chunks):
        self.calls.append(chunks)
        return self.embeddings


# ============================================================
# UploadFile Factory
# ============================================================


def create_upload_file(
    filename="test.txt",
    content=b"Test document content.",
):
    return UploadFile(
        file=io.BytesIO(content),
        filename=filename,
    )


# ============================================================
# Service Factory
# ============================================================


def create_service(
    monkeypatch,
    *,
    max_upload_size_mb=10,
    file_hash=None,
    vector_store=None,
    document_loader=None,
    text_splitter=None,
    metadata_builder=None,
    embedding_service=None,
):
    """
    Create the ingestion service with mocked dependencies.
    """

    monkeypatch.setattr(
        ingestion_module.settings,
        "MAX_UPLOAD_SIZE_MB",
        max_upload_size_mb,
    )

    if file_hash is None:
        file_hash = FakeFileHash()

    if vector_store is None:
        vector_store = FakeVectorStore()

    if document_loader is None:
        document_loader = FakeDocumentLoader()

    if text_splitter is None:
        text_splitter = FakeTextSplitter()

    if metadata_builder is None:
        metadata_builder = FakeMetadataBuilder()

    if embedding_service is None:
        embedding_service = FakeEmbeddingService()

    monkeypatch.setattr(
        ingestion_module,
        "file_hash",
        file_hash,
    )

    monkeypatch.setattr(
        ingestion_module,
        "vector_store",
        vector_store,
    )

    monkeypatch.setattr(
        ingestion_module,
        "document_loader",
        document_loader,
    )

    monkeypatch.setattr(
        ingestion_module,
        "text_splitter_service",
        text_splitter,
    )

    monkeypatch.setattr(
        ingestion_module,
        "metadata_builder",
        metadata_builder,
    )

    monkeypatch.setattr(
        ingestion_module,
        "embedding_service",
        embedding_service,
    )

    return ingestion_module.DocumentIngestionService()


# ============================================================
# Initialization
# ============================================================


def test_service_initializes_successfully(
    monkeypatch,
):
    """
    Service should initialize when the configured
    maximum upload size is valid.
    """

    service = create_service(
        monkeypatch,
        max_upload_size_mb=10,
    )

    assert service.max_file_size == (
        10 * 1024 * 1024
    )


def test_service_rejects_invalid_max_upload_size(
    monkeypatch,
):
    """
    A non-positive maximum upload size should fail
    during service initialization.
    """

    monkeypatch.setattr(
        ingestion_module.settings,
        "MAX_UPLOAD_SIZE_MB",
        0,
    )

    with pytest.raises(
        ValueError,
        match="MAX_UPLOAD_SIZE_MB must be greater than zero.",
    ):
        ingestion_module.DocumentIngestionService()


# ============================================================
# Upload Validation
# ============================================================


@pytest.mark.asyncio
async def test_ingest_rejects_none_file(
    monkeypatch,
):
    service = create_service(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="No document was provided.",
    ):
        await service.ingest(None)


@pytest.mark.asyncio
async def test_ingest_rejects_missing_filename(
    monkeypatch,
):
    service = create_service(
        monkeypatch
    )

    file = create_upload_file(
        filename=""
    )

    with pytest.raises(
        ValueError,
        match="Uploaded file must have a filename.",
    ):
        await service.ingest(file)


@pytest.mark.asyncio
async def test_ingest_rejects_whitespace_filename(
    monkeypatch,
):
    service = create_service(
        monkeypatch
    )

    file = create_upload_file(
        filename="   "
    )

    with pytest.raises(
        ValueError,
        match="Uploaded file must have a filename.",
    ):
        await service.ingest(file)


@pytest.mark.asyncio
async def test_ingest_rejects_file_without_extension(
    monkeypatch,
):
    service = create_service(
        monkeypatch
    )

    file = create_upload_file(
        filename="document"
    )

    with pytest.raises(
        ValueError,
        match="Uploaded file must have a file extension.",
    ):
        await service.ingest(file)


@pytest.mark.asyncio
async def test_ingest_rejects_unsupported_file_type(
    monkeypatch,
):
    service = create_service(
        monkeypatch
    )

    file = create_upload_file(
        filename="document.exe"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported file type: .exe",
    ):
        await service.ingest(file)


@pytest.mark.asyncio
async def test_ingest_accepts_supported_extensions(
    monkeypatch,
    tmp_path,
):
    """
    Supported extensions should pass validation and
    reach the processing pipeline.
    """

    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    for extension in [
        ".pdf",
        ".docx",
        ".pptx",
        ".txt",
        ".csv",
        ".xlsx",
        ".xls",
    ]:

        service = create_service(
            monkeypatch
        )

        file = create_upload_file(
            filename=f"document{extension}"
        )

        result = await service.ingest(
            file
        )

        assert result["duplicate"] is False
        assert (
            result["file_type"]
            == extension.lstrip(".")
        )


# ============================================================
# Path Traversal
# ============================================================


@pytest.mark.asyncio
async def test_ingest_sanitizes_path_traversal_filename(
    monkeypatch,
    tmp_path,
):
    """
    Uploaded filenames should be reduced to their safe
    basename before being stored.
    """

    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    service = create_service(
        monkeypatch
    )

    file = create_upload_file(
        filename="../../secret.txt"
    )

    result = await service.ingest(
        file
    )

    assert result["file_name"] == "secret.txt"

    assert (
        tmp_path / "secret.txt"
    ).exists()


# ============================================================
# Empty Upload
# ============================================================


@pytest.mark.asyncio
async def test_ingest_rejects_empty_file(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    service = create_service(
        monkeypatch
    )

    file = create_upload_file(
        filename="empty.txt",
        content=b"",
    )

    with pytest.raises(
        ValueError,
        match="Uploaded file is empty.",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "empty.txt"
    ).exists()


# ============================================================
# File Size
# ============================================================


@pytest.mark.asyncio
async def test_ingest_rejects_file_exceeding_max_size(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    service = create_service(
        monkeypatch,
        max_upload_size_mb=1,
    )

    content = b"x" * (
        1024 * 1024 + 1
    )

    file = create_upload_file(
        filename="large.txt",
        content=content,
    )

    with pytest.raises(
        ValueError,
        match="Uploaded file exceeds",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "large.txt"
    ).exists()


# ============================================================
# Successful Ingestion
# ============================================================


@pytest.mark.asyncio
async def test_ingest_successfully(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    monkeypatch.setattr(
        ingestion_module.settings,
        "EMBEDDING_MODEL",
        "test-embedding-model",
    )

    service = create_service(
        monkeypatch
    )

    file = create_upload_file(
        filename="resume.txt",
        content=b"Python backend developer.",
    )

    result = await service.ingest(
        file
    )

    assert result["duplicate"] is False
    assert result["file_name"] == "resume.txt"
    assert result["file_type"] == "txt"
    assert result["file_size"] == len(
        b"Python backend developer."
    )
    assert result["characters"] > 0
    assert result["chunks"] == 2
    assert result["preview"]
    assert (
        result["embedding_model"]
        == "test-embedding-model"
    )
    assert (
        "processing_time"
        in result
    )


@pytest.mark.asyncio
async def test_successful_ingestion_calls_pipeline_in_order(
    monkeypatch,
    tmp_path,
):
    """
    Verify that every major ingestion component is called.
    """

    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    file_hash_service = FakeFileHash(
        value="hash123"
    )

    document_loader = FakeDocumentLoader(
        text="Original extracted text."
    )

    text_splitter = FakeTextSplitter(
        chunks=[
            "Chunk one",
            "Chunk two",
        ]
    )

    metadata_builder = FakeMetadataBuilder()

    embedding_service = FakeEmbeddingService(
        embeddings=[
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    vector_store = FakeVectorStore()

    service = create_service(
        monkeypatch,
        file_hash=file_hash_service,
        document_loader=document_loader,
        text_splitter=text_splitter,
        metadata_builder=metadata_builder,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    file = create_upload_file(
        filename="resume.txt"
    )

    result = await service.ingest(
        file
    )

    assert result["duplicate"] is False

    assert len(
        file_hash_service.calls
    ) == 1

    assert len(
        document_loader.calls
    ) == 1

    assert text_splitter.calls == [
        "Original extracted text."
    ]

    assert len(
        metadata_builder.calls
    ) == 1

    assert embedding_service.calls == [
        [
            "Chunk one",
            "Chunk two",
        ]
    ]

    assert len(
        vector_store.add_documents_calls
    ) == 1


# ============================================================
# Duplicate Detection
# ============================================================


@pytest.mark.asyncio
async def test_ingest_detects_duplicate_document(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    vector_store = FakeVectorStore(
        exists=True
    )

    service = create_service(
        monkeypatch,
        vector_store=vector_store,
    )

    file = create_upload_file(
        filename="duplicate.txt"
    )

    result = await service.ingest(
        file
    )

    assert result["duplicate"] is True
    assert (
        result["file_name"]
        == "duplicate.txt"
    )
    assert (
        result["file_type"]
        == "txt"
    )
    assert (
        result["message"]
        == "Document already indexed."
    )

    assert not (
        tmp_path / "duplicate.txt"
    ).exists()

    assert (
        vector_store.document_exists_calls
        == ["test-hash-123"]
    )


@pytest.mark.asyncio
async def test_duplicate_document_does_not_run_pipeline(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    document_loader = FakeDocumentLoader()
    text_splitter = FakeTextSplitter()
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(
        exists=True
    )

    service = create_service(
        monkeypatch,
        document_loader=document_loader,
        text_splitter=text_splitter,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    file = create_upload_file(
        filename="duplicate.txt"
    )

    await service.ingest(
        file
    )

    assert (
        document_loader.calls == []
    )

    assert (
        text_splitter.calls == []
    )

    assert (
        embedding_service.calls == []
    )

    assert (
        vector_store.add_documents_calls
        == []
    )


# ============================================================
# Text Cleaning
# ============================================================


def test_clean_text_rejects_non_string():
    result = (
        ingestion_module.DocumentIngestionService
        ._clean_text(
            None
        )
    )

    assert result == ""


def test_clean_text_normalizes_line_endings():
    result = (
        ingestion_module.DocumentIngestionService
        ._clean_text(
            "Hello\r\nWorld\rTest"
        )
    )

    assert result == (
        "Hello\nWorld\nTest"
    )


def test_clean_text_collapses_spaces_and_tabs():
    result = (
        ingestion_module.DocumentIngestionService
        ._clean_text(
            "Hello    world\t\tPython"
        )
    )

    assert result == (
        "Hello world Python"
    )


def test_clean_text_preserves_paragraph_boundaries():
    result = (
        ingestion_module.DocumentIngestionService
        ._clean_text(
            "First paragraph.\n\n\n\nSecond paragraph."
        )
    )

    assert result == (
        "First paragraph.\n\n"
        "Second paragraph."
    )


def test_clean_text_strips_outer_whitespace():
    result = (
        ingestion_module.DocumentIngestionService
        ._clean_text(
            "   Hello world   "
        )
    )

    assert result == "Hello world"


# ============================================================
# Empty Extracted Text
# ============================================================


@pytest.mark.asyncio
async def test_ingest_rejects_no_usable_extracted_text(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    document_loader = FakeDocumentLoader(
        text="   \n\n   "
    )

    service = create_service(
        monkeypatch,
        document_loader=document_loader,
    )

    file = create_upload_file(
        filename="empty-text.txt"
    )

    with pytest.raises(
        ValueError,
        match="No usable text could be extracted",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "empty-text.txt"
    ).exists()


# ============================================================
# Empty Chunks
# ============================================================


@pytest.mark.asyncio
async def test_ingest_rejects_empty_chunks(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    text_splitter = FakeTextSplitter(
        chunks=[]
    )

    service = create_service(
        monkeypatch,
        text_splitter=text_splitter,
    )

    file = create_upload_file(
        filename="no-chunks.txt"
    )

    with pytest.raises(
        ValueError,
        match="No usable chunks were generated",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "no-chunks.txt"
    ).exists()


# ============================================================
# Metadata Validation
# ============================================================


@pytest.mark.asyncio
async def test_ingest_rejects_metadata_count_mismatch(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    metadata_builder = FakeMetadataBuilder(
        metadata=[
            {
                "source": "test.txt",
                "file_type": "txt",
                "chunk": 1,
                "file_hash": "hash",
            }
        ]
    )

    service = create_service(
        monkeypatch,
        metadata_builder=metadata_builder,
    )

    file = create_upload_file(
        filename="metadata-mismatch.txt"
    )

    with pytest.raises(
        ValueError,
        match="Metadata count does not match",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "metadata-mismatch.txt"
    ).exists()


# ============================================================
# Embedding Validation
# ============================================================


@pytest.mark.asyncio
async def test_ingest_rejects_embedding_count_mismatch(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    embedding_service = FakeEmbeddingService(
        embeddings=[
            [0.1, 0.2, 0.3]
        ]
    )

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
    )

    file = create_upload_file(
        filename="embedding-mismatch.txt"
    )

    with pytest.raises(
        ValueError,
        match="Embedding count does not match",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "embedding-mismatch.txt"
    ).exists()


@pytest.mark.asyncio
async def test_ingest_rejects_inconsistent_embedding_dimensions(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    embedding_service = FakeEmbeddingService(
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5],
        ]
    )

    service = create_service(
        monkeypatch,
        embedding_service=embedding_service,
    )

    file = create_upload_file(
        filename="invalid-embeddings.txt"
    )

    with pytest.raises(
        ValueError,
        match="Embedding dimensions are inconsistent",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "invalid-embeddings.txt"
    ).exists()


# ============================================================
# Failure Cleanup
# ============================================================


@pytest.mark.asyncio
async def test_hash_failure_removes_uploaded_file(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    class FailingHash:

        def sha256(self, file_path):
            raise RuntimeError(
                "Hash failure"
            )

    service = create_service(
        monkeypatch,
        file_hash=FailingHash(),
    )

    file = create_upload_file(
        filename="hash-failure.txt"
    )

    with pytest.raises(
        RuntimeError,
        match="Hash failure",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "hash-failure.txt"
    ).exists()


@pytest.mark.asyncio
async def test_extraction_failure_removes_uploaded_file(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    class FailingLoader:

        def extract_text(self, file_path):
            raise RuntimeError(
                "Extraction failure"
            )

    service = create_service(
        monkeypatch,
        document_loader=FailingLoader(),
    )

    file = create_upload_file(
        filename="extraction-failure.txt"
    )

    with pytest.raises(
        RuntimeError,
        match="Extraction failure",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "extraction-failure.txt"
    ).exists()


@pytest.mark.asyncio
async def test_embedding_failure_removes_uploaded_file(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    class FailingEmbeddingService:

        def generate_embeddings(self, chunks):
            raise RuntimeError(
                "Embedding failure"
            )

    service = create_service(
        monkeypatch,
        embedding_service=(
            FailingEmbeddingService()
        ),
    )

    file = create_upload_file(
        filename="embedding-failure.txt"
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding failure",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "embedding-failure.txt"
    ).exists()


@pytest.mark.asyncio
async def test_vector_store_failure_removes_uploaded_file(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ingestion_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
        raising=False,
    )

    class FailingVectorStore:

        def document_exists(self, file_hash):
            return False

        def add_documents(
            self,
            chunks,
            embeddings,
            metadata,
        ):
            raise RuntimeError(
                "Vector store failure"
            )

    service = create_service(
        monkeypatch,
        vector_store=FailingVectorStore(),
    )

    file = create_upload_file(
        filename="vector-failure.txt"
    )

    with pytest.raises(
        RuntimeError,
        match="Vector store failure",
    ):
        await service.ingest(file)

    assert not (
        tmp_path / "vector-failure.txt"
    ).exists()


# ============================================================
# File Removal Helper
# ============================================================


def test_remove_file_safely_removes_existing_file(
    tmp_path,
):
    file_path = (
        tmp_path / "test.txt"
    )

    file_path.write_text(
        "test"
    )

    assert file_path.exists()

    (
        ingestion_module.DocumentIngestionService
        ._remove_file(
            file_path
        )
    )

    assert not file_path.exists()


def test_remove_file_ignores_missing_file(
    tmp_path,
):
    file_path = (
        tmp_path / "missing.txt"
    )

    (
        ingestion_module.DocumentIngestionService
        ._remove_file(
            file_path
        )
    )

    assert not file_path.exists()


def test_remove_file_handles_unlink_failure(
    monkeypatch,
    tmp_path,
):
    file_path = (
        tmp_path / "test.txt"
    )

    file_path.write_text(
        "test"
    )

    def failing_unlink(
        self,
    ):
        raise OSError(
            "Unable to remove"
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        failing_unlink,
    )

    # Should not raise.
    (
        ingestion_module.DocumentIngestionService
        ._remove_file(
            file_path
        )
    )
    
@pytest.mark.asyncio
async def test_ingest_rejects_invalid_filename():
    from io import BytesIO

    file = UploadFile(
        file=BytesIO(b"test"),
        filename="..",
    )

    with pytest.raises(ValueError):
        await ingestion_module.document_ingestion_service.ingest(file)
        
@pytest.mark.asyncio
async def test_ingest_handles_file_save_failure(monkeypatch):
    class FailingUploadFile:
        filename = "test.txt"

        async def read(self, size):
            raise RuntimeError("Read failed")

        async def close(self):
            pass

    file = FailingUploadFile()

    with pytest.raises(
        RuntimeError,
        match="Read failed",
    ):
        await ingestion_module.document_ingestion_service.ingest(file)
        
@pytest.mark.asyncio
async def test_ingest_handles_duplicate_check_failure(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
    )

    monkeypatch.setattr(
        file_hash,
        "sha256",
        lambda path: "fake-hash",
    )

    def fail_duplicate_check(hash_value):
        raise RuntimeError(
            "Duplicate check failed"
        )

    monkeypatch.setattr(
        vector_store,
        "document_exists",
        fail_duplicate_check,
    )

    file = UploadFile(
        file=io.BytesIO(b"test content"),
        filename="test.txt",
    )

    with pytest.raises(
        RuntimeError,
        match="Duplicate check failed",
    ):
        await ingestion_module.document_ingestion_service.ingest(file)
        
@pytest.mark.asyncio
async def test_ingest_handles_text_splitting_failure(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
    )

    monkeypatch.setattr(
        file_hash,
        "sha256",
        lambda path: "fake-hash",
    )

    monkeypatch.setattr(
        vector_store,
        "document_exists",
        lambda hash_value: False,
    )

    monkeypatch.setattr(
        document_loader,
        "extract_text",
        lambda path: "Some valid document text",
    )

    def fail_split(text):
        raise RuntimeError(
            "Splitting failed"
        )

    monkeypatch.setattr(
        text_splitter_service,
        "split",
        fail_split,
    )

    file = UploadFile(
        file=io.BytesIO(b"test content"),
        filename="test.txt",
    )

    with pytest.raises(
        RuntimeError,
        match="Splitting failed",
    ):
        await ingestion_module.document_ingestion_service.ingest(file)
        
@pytest.mark.asyncio
async def test_ingest_handles_metadata_generation_failure(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
    )

    monkeypatch.setattr(
        file_hash,
        "sha256",
        lambda path: "fake-hash",
    )

    monkeypatch.setattr(
        vector_store,
        "document_exists",
        lambda hash_value: False,
    )

    monkeypatch.setattr(
        document_loader,
        "extract_text",
        lambda path: "Some valid document text",
    )

    monkeypatch.setattr(
        text_splitter_service,
        "split",
        lambda text: ["chunk one"],
    )

    def fail_metadata(**kwargs):
        raise RuntimeError(
            "Metadata generation failed"
        )

    monkeypatch.setattr(
        metadata_builder,
        "build",
        fail_metadata,
    )

    file = UploadFile(
        file=io.BytesIO(b"test content"),
        filename="test.txt",
    )

    with pytest.raises(
        RuntimeError,
        match="Metadata generation failed",
    ):
        await ingestion_module.document_ingestion_service.ingest(file)
        
@pytest.mark.asyncio
async def test_ingest_rejects_invalid_filename_after_sanitization(
    monkeypatch,
):
    original_path = ingestion_module.Path

    class FakePath:
        def __init__(self, value):
            self.value = value

        @property
        def name(self):
            return ""

    call_count = 0

    def fake_path(value):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return FakePath(value)

        return original_path(value)

    monkeypatch.setattr(
        ingestion_module,
        "Path",
        fake_path,
    )

    file = UploadFile(
        file=io.BytesIO(b"test content"),
        filename="valid.txt",
    )

    with pytest.raises(
        ValueError,
        match="Invalid filename",
    ):
        await ingestion_module.document_ingestion_service.ingest(
            file
        )