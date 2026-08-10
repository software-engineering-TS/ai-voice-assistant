from unittest.mock import MagicMock

import app.documents.document_service as document_module


def test_list_documents():
    expected = [
        {
            "file_hash": "abc123",
            "file_name": "test.pdf",
        }
    ]

    document_module.vector_store.list_documents = MagicMock(
        return_value=expected
    )

    service = document_module.DocumentService()

    result = service.list_documents()

    assert result == expected

    document_module.vector_store.list_documents.assert_called_once_with()


def test_get_document():
    expected = {
        "file_hash": "abc123",
        "file_name": "test.pdf",
    }

    document_module.vector_store.get_document = MagicMock(
        return_value=expected
    )

    service = document_module.DocumentService()

    result = service.get_document("abc123")

    assert result == expected

    document_module.vector_store.get_document.assert_called_once_with(
        "abc123"
    )


def test_get_document_not_found():
    document_module.vector_store.get_document = MagicMock(
        return_value=None
    )

    service = document_module.DocumentService()

    result = service.get_document("missing")

    assert result is None

    document_module.vector_store.get_document.assert_called_once_with(
        "missing"
    )


def test_delete_document():
    expected = {
        "deleted": True,
        "file_hash": "abc123",
    }

    document_module.vector_store.delete_document = MagicMock(
        return_value=expected
    )

    service = document_module.DocumentService()

    result = service.delete_document("abc123")

    assert result == expected

    document_module.vector_store.delete_document.assert_called_once_with(
        "abc123"
    )


def test_delete_document_not_found():
    document_module.vector_store.delete_document = MagicMock(
        return_value=None
    )

    service = document_module.DocumentService()

    result = service.delete_document("missing")

    assert result is None

    document_module.vector_store.delete_document.assert_called_once_with(
        "missing"
    )