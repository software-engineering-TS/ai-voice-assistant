from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config.logging_config import app_logger
from app.models.response import ApiResponse
from app.services.document_ingestion_service import (
    document_ingestion_service,
)


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


# ============================================================
# Upload and Index Document
# ============================================================

@router.post(
    "/extract",
    response_model=ApiResponse,
    summary="Upload and index document",
)
async def extract_document(
    file: UploadFile = File(...),
) -> ApiResponse:
    """
    Upload a supported document and index it for RAG.

    Supported document types are handled by the
    DocumentLoader service.
    """

    filename = file.filename or ""

    app_logger.info(
        "RAG document upload received | "
        f"Filename={filename}"
    )

    if not filename.strip():

        app_logger.warning(
            "RAG upload rejected: filename is missing."
        )

        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    try:

        result = (
            await document_ingestion_service.ingest(
                file=file,
            )
        )

    except ValueError as exc:

        app_logger.warning(
            "Document ingestion validation failed | "
            f"Filename={filename} | "
            f"Error={exc}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        app_logger.exception(
            "Uploaded document could not be accessed | "
            f"Filename={filename}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:

        app_logger.exception(
            "Document ingestion failed | "
            f"Filename={filename}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process the document.",
        )

    app_logger.info(
        "RAG document processed successfully | "
        f"Filename={filename}"
    )

    # --------------------------------------------------------
    # Duplicate Document
    # --------------------------------------------------------

    if result.get("duplicate"):

        return ApiResponse(
            success=True,
            message="Document already indexed.",
            data=result,
        )

    # --------------------------------------------------------
    # Successful Ingestion
    # --------------------------------------------------------

    return ApiResponse(
        success=True,
        message="Document processed successfully.",
        data=result,
    )