import pytest
from fastapi import HTTPException

from app.api.routes.document import (
    delete_document,
    get_document,
    list_documents,
    router,
)
from app.models.response import ApiResponse

# ============================================================
# List Documents
# ============================================================


@pytest.mark.asyncio
async def test_list_documents_successfully(monkeypatch):
    documents = [
        {
            "file_name": "document1.pdf",
            "file_hash": "hash1",
            "chunks": 5,
        },
        {
            "file_name": "document2.txt",
            "file_hash": "hash2",
            "chunks": 3,
        },
    ]

    monkeypatch.setattr(
        "app.api.routes.document.document_service.list_documents",
        lambda: documents,
    )

    response = await list_documents()

    assert isinstance(response, ApiResponse)
    assert response.success is True
    assert response.message == "Documents retrieved successfully."
    assert response.data["total_documents"] == 2
    assert response.data["documents"] == documents


@pytest.mark.asyncio
async def test_list_documents_returns_empty_list(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.document.document_service.list_documents",
        lambda: [],
    )

    response = await list_documents()

    assert response.success is True
    assert response.data["total_documents"] == 0
    assert response.data["documents"] == []


@pytest.mark.asyncio
async def test_list_documents_handles_service_failure(monkeypatch):
    def raise_error():
        raise RuntimeError("Database failure")

    monkeypatch.setattr(
        "app.api.routes.document.document_service.list_documents",
        raise_error,
    )

    with pytest.raises(HTTPException) as exc_info:
        await list_documents()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unable to retrieve documents."


# ============================================================
# Get Document
# ============================================================


@pytest.mark.asyncio
async def test_get_document_successfully(monkeypatch):
    document = {
        "file_name": "resume.pdf",
        "file_hash": "abc123",
        "chunks": 10,
    }

    received_hash = None

    def mock_get_document(file_hash):
        nonlocal received_hash
        received_hash = file_hash
        return document

    monkeypatch.setattr(
        "app.api.routes.document.document_service.get_document",
        mock_get_document,
    )

    response = await get_document("abc123")

    assert isinstance(response, ApiResponse)
    assert response.success is True
    assert response.message == "Document retrieved successfully."
    assert response.data == document
    assert received_hash == "abc123"


@pytest.mark.asyncio
async def test_get_document_strips_file_hash(monkeypatch):
    received_hash = None

    def mock_get_document(file_hash):
        nonlocal received_hash
        received_hash = file_hash

        return {
            "file_hash": file_hash,
        }

    monkeypatch.setattr(
        "app.api.routes.document.document_service.get_document",
        mock_get_document,
    )

    response = await get_document("   abc123   ")

    assert response.success is True
    assert received_hash == "abc123"


@pytest.mark.asyncio
async def test_get_document_rejects_empty_hash():
    with pytest.raises(HTTPException) as exc_info:
        await get_document("")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Document hash cannot be empty."


@pytest.mark.asyncio
async def test_get_document_rejects_whitespace_hash():
    with pytest.raises(HTTPException) as exc_info:
        await get_document("     ")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Document hash cannot be empty."


@pytest.mark.asyncio
async def test_get_document_returns_404_when_not_found(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.document.document_service.get_document",
        lambda file_hash: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_document("missing_hash")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Document not found."


@pytest.mark.asyncio
async def test_get_document_handles_invalid_hash(monkeypatch):
    def raise_value_error(file_hash):
        raise ValueError("Invalid document hash.")

    monkeypatch.setattr(
        "app.api.routes.document.document_service.get_document",
        raise_value_error,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_document("invalid_hash")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid document hash."


@pytest.mark.asyncio
async def test_get_document_handles_service_failure(monkeypatch):
    def raise_error(file_hash):
        raise RuntimeError("Unexpected failure")

    monkeypatch.setattr(
        "app.api.routes.document.document_service.get_document",
        raise_error,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_document("abc123")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unable to retrieve document."


# ============================================================
# Delete Document
# ============================================================


@pytest.mark.asyncio
async def test_delete_document_successfully(monkeypatch):
    result = {
        "file_hash": "abc123",
        "deleted": True,
    }

    received_hash = None

    def mock_delete_document(file_hash):
        nonlocal received_hash
        received_hash = file_hash
        return result

    monkeypatch.setattr(
        "app.api.routes.document.document_service.delete_document",
        mock_delete_document,
    )

    response = await delete_document("abc123")

    assert isinstance(response, ApiResponse)
    assert response.success is True
    assert response.message == "Document deleted successfully."
    assert response.data == result
    assert received_hash == "abc123"


@pytest.mark.asyncio
async def test_delete_document_strips_file_hash(monkeypatch):
    received_hash = None

    def mock_delete_document(file_hash):
        nonlocal received_hash
        received_hash = file_hash

        return {
            "file_hash": file_hash,
            "deleted": True,
        }

    monkeypatch.setattr(
        "app.api.routes.document.document_service.delete_document",
        mock_delete_document,
    )

    response = await delete_document("   abc123   ")

    assert response.success is True
    assert received_hash == "abc123"


@pytest.mark.asyncio
async def test_delete_document_rejects_empty_hash():
    with pytest.raises(HTTPException) as exc_info:
        await delete_document("")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Document hash cannot be empty."


@pytest.mark.asyncio
async def test_delete_document_rejects_whitespace_hash():
    with pytest.raises(HTTPException) as exc_info:
        await delete_document("     ")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Document hash cannot be empty."


@pytest.mark.asyncio
async def test_delete_document_returns_404_when_not_found(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.document.document_service.delete_document",
        lambda file_hash: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_document("missing_hash")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Document not found."


@pytest.mark.asyncio
async def test_delete_document_handles_invalid_hash(monkeypatch):
    def raise_value_error(file_hash):
        raise ValueError("Invalid document hash.")

    monkeypatch.setattr(
        "app.api.routes.document.document_service.delete_document",
        raise_value_error,
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_document("invalid_hash")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid document hash."


@pytest.mark.asyncio
async def test_delete_document_handles_service_failure(monkeypatch):
    def raise_error(file_hash):
        raise RuntimeError("Unexpected failure")

    monkeypatch.setattr(
        "app.api.routes.document.document_service.delete_document",
        raise_error,
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_document("abc123")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unable to delete document."


# ============================================================
# Router Configuration
# ============================================================


def test_document_router_has_correct_prefix():
    assert router.prefix == "/documents"


def test_document_router_has_correct_tag():
    assert "Documents" in router.tags