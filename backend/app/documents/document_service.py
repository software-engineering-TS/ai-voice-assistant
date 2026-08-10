from app.config.logging_config import app_logger
from app.rag.vector_store import vector_store


class DocumentService:
    """
    Service layer for indexed document operations.

    file_hash is used as the unique document identity.
    """

    # ============================================================
    # List Documents
    # ============================================================

    def list_documents(self) -> list[dict]:
        """
        Return all indexed documents.
        """

        app_logger.info(
            "Loading indexed documents."
        )

        return vector_store.list_documents()

    # ============================================================
    # Get Document
    # ============================================================

    def get_document(
        self,
        file_hash: str,
    ) -> dict | None:
        """
        Return details of a specific indexed document.

        Parameters:
            file_hash:
                SHA-256 hash identifying the document.
        """

        app_logger.info(
            "Loading document | "
            f"Hash={file_hash}"
        )

        return vector_store.get_document(
            file_hash
        )

    # ============================================================
    # Delete Document
    # ============================================================

    def delete_document(
        self,
        file_hash: str,
    ) -> dict | None:
        """
        Delete an indexed document and its uploaded file.

        Parameters:
            file_hash:
                SHA-256 hash identifying the document.
        """

        app_logger.info(
            "Deleting document | "
            f"Hash={file_hash}"
        )

        return vector_store.delete_document(
            file_hash
        )


# ============================================================
# Shared Document Service
# ============================================================

document_service = DocumentService()