from fastapi import APIRouter, HTTPException

from app.config.logging_config import app_logger
from app.documents.document_service import document_service
from app.models.response import ApiResponse


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# ============================================================
# List Documents
# ============================================================

@router.get(
    "",
    summary="List all indexed documents",
    response_model=ApiResponse,
)
async def list_documents() -> ApiResponse:
    """
    Return all indexed documents.
    """

    app_logger.info(
        "Listing all indexed documents."
    )

    try:

        documents = (
            document_service.list_documents()
        )

        return ApiResponse(
            success=True,
            message=(
                "Documents retrieved successfully."
            ),
            data={
                "total_documents": len(
                    documents
                ),
                "documents": documents,
            },
        )

    except Exception:

        app_logger.exception(
            "Failed to list documents."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve documents.",
        )


# ============================================================
# Get Document
# ============================================================

@router.get(
    "/{file_hash}",
    summary="Get document details",
    response_model=ApiResponse,
)
async def get_document(
    file_hash: str,
) -> ApiResponse:
    """
    Return details for a specific indexed document.

    The SHA-256 file hash is used as the unique
    document identifier.
    """

    file_hash = file_hash.strip()

    if not file_hash:

        raise HTTPException(
            status_code=400,
            detail="Document hash cannot be empty.",
        )

    app_logger.info(
        "Fetching document | "
        f"Hash={file_hash}"
    )

    try:

        document = (
            document_service.get_document(
                file_hash
            )
        )

    except ValueError as exc:

        app_logger.warning(
            "Invalid document hash | "
            f"Hash={file_hash} | "
            f"Error={exc}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:

        app_logger.exception(
            "Failed to fetch document | "
            f"Hash={file_hash}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve document.",
        )

    if document is None:

        app_logger.warning(
            "Document not found | "
            f"Hash={file_hash}"
        )

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return ApiResponse(
        success=True,
        message=(
            "Document retrieved successfully."
        ),
        data=document,
    )


# ============================================================
# Delete Document
# ============================================================

@router.delete(
    "/{file_hash}",
    summary="Delete document",
    response_model=ApiResponse,
)
async def delete_document(
    file_hash: str,
) -> ApiResponse:
    """
    Delete an indexed document and its uploaded file.

    The SHA-256 file hash is used as the unique
    document identifier.
    """

    file_hash = file_hash.strip()

    if not file_hash:

        raise HTTPException(
            status_code=400,
            detail="Document hash cannot be empty.",
        )

    app_logger.warning(
        "Deleting document | "
        f"Hash={file_hash}"
    )

    try:

        result = (
            document_service.delete_document(
                file_hash
            )
        )

    except ValueError as exc:

        app_logger.warning(
            "Invalid document hash | "
            f"Hash={file_hash} | "
            f"Error={exc}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:

        app_logger.exception(
            "Failed to delete document | "
            f"Hash={file_hash}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to delete document.",
        )

    if result is None:

        app_logger.warning(
            "Document not found for deletion | "
            f"Hash={file_hash}"
        )

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    app_logger.info(
        "Document deleted successfully | "
        f"Hash={file_hash}"
    )

    return ApiResponse(
        success=True,
        message=(
            "Document deleted successfully."
        ),
        data=result,
    )