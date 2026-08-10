import pytest
from fastapi import HTTPException, UploadFile
from io import BytesIO
from unittest.mock import AsyncMock, patch

from app.api.routes.rag import extract_document, router


# ============================================================
# Helpers
# ============================================================

def create_upload_file(filename="test.pdf"):
    return UploadFile(
        filename=filename,
        file=BytesIO(b"test document content"),
    )


# ============================================================
# Successful Ingestion
# ============================================================

@pytest.mark.asyncio
async def test_extract_document_successfully():
    file = create_upload_file("test.pdf")

    result_data = {
        "file_hash": "abc123",
        "filename": "test.pdf",
        "chunks": 5,
        "duplicate": False,
    }

    with patch(
        "app.api.routes.rag.document_ingestion_service.ingest",
        new=AsyncMock(return_value=result_data),
    ) as mock_ingest:

        response = await extract_document(file)

    mock_ingest.assert_awaited_once_with(file=file)

    assert response.success is True
    assert response.message == "Document processed successfully."
    assert response.data == result_data


# ============================================================
# Duplicate Document
# ============================================================

@pytest.mark.asyncio
async def test_extract_document_returns_duplicate():
    file = create_upload_file("existing.pdf")

    result_data = {
        "file_hash": "duplicate123",
        "filename": "existing.pdf",
        "chunks": 10,
        "duplicate": True,
    }

    with patch(
        "app.api.routes.rag.document_ingestion_service.ingest",
        new=AsyncMock(return_value=result_data),
    ):

        response = await extract_document(file)

    assert response.success is True
    assert response.message == "Document already indexed."
    assert response.data == result_data


# ============================================================
# Missing Filename
# ============================================================

@pytest.mark.asyncio
async def test_extract_document_rejects_missing_filename():
    file = create_upload_file("")
    file.filename = None

    with pytest.raises(HTTPException) as exc_info:
        await extract_document(file)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Uploaded file must have a filename."
    )


# ============================================================
# Whitespace Filename
# ============================================================

@pytest.mark.asyncio
async def test_extract_document_rejects_whitespace_filename():
    file = create_upload_file("   ")

    with pytest.raises(HTTPException) as exc_info:
        await extract_document(file)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Uploaded file must have a filename."
    )


# ============================================================
# ValueError
# ============================================================

@pytest.mark.asyncio
async def test_extract_document_handles_value_error():
    file = create_upload_file("invalid.xyz")

    with patch(
        "app.api.routes.rag.document_ingestion_service.ingest",
        new=AsyncMock(
            side_effect=ValueError(
                "Unsupported document type."
            )
        ),
    ):

        with pytest.raises(HTTPException) as exc_info:
            await extract_document(file)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Unsupported document type."
    )


# ============================================================
# FileNotFoundError
# ============================================================

@pytest.mark.asyncio
async def test_extract_document_handles_file_not_found():
    file = create_upload_file("missing.pdf")

    with patch(
        "app.api.routes.rag.document_ingestion_service.ingest",
        new=AsyncMock(
            side_effect=FileNotFoundError(
                "Uploaded file could not be accessed."
            )
        ),
    ):

        with pytest.raises(HTTPException) as exc_info:
            await extract_document(file)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Uploaded file could not be accessed."
    )


# ============================================================
# Unexpected Exception
# ============================================================

@pytest.mark.asyncio
async def test_extract_document_handles_unexpected_error():
    file = create_upload_file("test.pdf")

    with patch(
        "app.api.routes.rag.document_ingestion_service.ingest",
        new=AsyncMock(
            side_effect=RuntimeError(
                "Unexpected ingestion failure."
            )
        ),
    ):

        with pytest.raises(HTTPException) as exc_info:
            await extract_document(file)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == (
        "Unable to process the document."
    )


# ============================================================
# Router Configuration
# ============================================================

def test_rag_router_has_correct_prefix():
    assert router.prefix == "/rag"


def test_rag_router_has_correct_tag():
    assert "RAG" in router.tags

