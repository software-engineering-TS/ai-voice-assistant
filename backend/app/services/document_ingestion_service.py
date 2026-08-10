import re
import time
from pathlib import Path

from fastapi import UploadFile

from app.config.logging_config import app_logger
from app.config.settings import settings
from app.rag.embedding_service import embedding_service
from app.rag.loaders.document_loader import document_loader
from app.rag.metadata_builder import metadata_builder
from app.rag.text_splitter import text_splitter_service
from app.rag.utils.file_hash import file_hash
from app.rag.vector_store import vector_store


class DocumentIngestionService:
    """
    Handles the complete document ingestion pipeline.

    Pipeline:

        Upload
          ↓
        Validate
          ↓
        Save File
          ↓
        SHA-256 Hash
          ↓
        Duplicate Detection
          ↓
        Text Extraction
          ↓
        Text Cleaning
          ↓
        Chunking
          ↓
        Metadata
          ↓
        Embeddings
          ↓
        ChromaDB
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".txt",
        ".csv",
        ".xlsx",
        ".xls",
    }

    UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB

    def __init__(self) -> None:

        self.max_file_size = (
            settings.MAX_UPLOAD_SIZE_MB
            * 1024
            * 1024
        )

        if self.max_file_size <= 0:
            raise ValueError(
                "MAX_UPLOAD_SIZE_MB must be greater than zero."
            )

        app_logger.info(
            "Document ingestion service initialized."
        )

    # ============================================================
    # Ingest Document
    # ============================================================

    async def ingest(
        self,
        file: UploadFile,
    ) -> dict:
        """
        Validate, process, embed, and index an uploaded document.
        """

        start_time = time.perf_counter()

        # --------------------------------------------------------
        # Validate Upload
        # --------------------------------------------------------

        if file is None:
            raise ValueError(
                "No document was provided."
            )

        original_filename = (
            file.filename or ""
        ).strip()

        if not original_filename:
            raise ValueError(
                "Uploaded file must have a filename."
            )

        # Prevent path traversal.
        safe_filename = Path(
            original_filename
        ).name

        if not safe_filename:
            raise ValueError(
                "Invalid filename."
            )

        extension = (
            Path(safe_filename)
            .suffix
            .lower()
        )

        if not extension:
            raise ValueError(
                "Uploaded file must have a file extension."
            )

        if extension not in self.SUPPORTED_EXTENSIONS:

            app_logger.warning(
                "Unsupported document type | "
                f"File={safe_filename} | "
                f"Extension={extension}"
            )

            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        app_logger.info(
            "Document ingestion started | "
            f"File={safe_filename} | "
            f"Type={extension}"
        )

        # --------------------------------------------------------
        # Prepare Upload Directory
        # --------------------------------------------------------

        upload_directory = Path(
            settings.UPLOAD_FOLDER
        )

        upload_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path = (
            upload_directory
            / safe_filename
        )

        # --------------------------------------------------------
        # Save Uploaded File
        # --------------------------------------------------------

        total_bytes = 0

        try:

            with file_path.open(
                "wb"
            ) as buffer:

                while True:

                    chunk = await file.read(
                        self.UPLOAD_CHUNK_SIZE
                    )

                    if not chunk:
                        break

                    total_bytes += len(
                        chunk
                    )

                    if (
                        total_bytes
                        > self.max_file_size
                    ):

                        raise ValueError(
                            "Uploaded file exceeds "
                            f"the maximum allowed size of "
                            f"{settings.MAX_UPLOAD_SIZE_MB} MB."
                        )

                    buffer.write(
                        chunk
                    )

        except ValueError:

            self._remove_file(
                file_path
            )

            raise

        except Exception:

            self._remove_file(
                file_path
            )

            app_logger.exception(
                "Failed to save uploaded document | "
                f"File={safe_filename}"
            )

            raise

        finally:

            await file.close()

        if total_bytes == 0:

            self._remove_file(
                file_path
            )

            raise ValueError(
                "Uploaded file is empty."
            )

        app_logger.info(
            "Document uploaded successfully | "
            f"File={safe_filename} | "
            f"Size={total_bytes} bytes"
        )

        # --------------------------------------------------------
        # Calculate SHA-256
        # --------------------------------------------------------

        try:

            hash_value = file_hash.sha256(
                file_path
            )

        except Exception:

            self._remove_file(
                file_path
            )

            app_logger.exception(
                "Failed to calculate file hash | "
                f"File={safe_filename}"
            )

            raise

        app_logger.info(
            "Document hash generated | "
            f"File={safe_filename}"
        )

        # --------------------------------------------------------
        # Duplicate Detection
        # --------------------------------------------------------

        try:

            duplicate = (
                vector_store.document_exists(
                    hash_value
                )
            )

        except Exception:

            self._remove_file(
                file_path
            )

            app_logger.exception(
                "Document duplicate check failed."
            )

            raise

        if duplicate:

            app_logger.warning(
                "Duplicate document detected | "
                f"File={safe_filename}"
            )

            self._remove_file(
                file_path
            )

            processing_time = (
                time.perf_counter()
                - start_time
            )

            return {
                "duplicate": True,
                "file_name": safe_filename,
                "file_type": extension.lstrip("."),
                "message": (
                    "Document already indexed."
                ),
                "processing_time": round(
                    processing_time,
                    3,
                ),
            }

        # --------------------------------------------------------
        # Extract Text
        # --------------------------------------------------------

        try:

            text = (
                document_loader.extract_text(
                    file_path
                )
            )

        except Exception:

            self._remove_file(
                file_path
            )

            app_logger.exception(
                "Document extraction failed | "
                f"File={safe_filename}"
            )

            raise

        # --------------------------------------------------------
        # Clean Text
        # --------------------------------------------------------

        text = self._clean_text(
            text
        )

        if not text:

            self._remove_file(
                file_path
            )

            app_logger.warning(
                "No usable text extracted | "
                f"File={safe_filename}"
            )

            raise ValueError(
                "No usable text could be extracted "
                "from the document."
            )

        app_logger.info(
            "Text extraction completed | "
            f"Characters={len(text)}"
        )

        # --------------------------------------------------------
        # Split Text
        # --------------------------------------------------------

        try:

            chunks = (
                text_splitter_service.split(
                    text
                )
            )

        except Exception:

            self._remove_file(
                file_path
            )

            app_logger.exception(
                "Text splitting failed | "
                f"File={safe_filename}"
            )

            raise

        if not chunks:

            self._remove_file(
                file_path
            )

            raise ValueError(
                "No usable chunks were generated "
                "from the document."
            )

        app_logger.info(
            "Document split successfully | "
            f"Chunks={len(chunks)}"
        )

        # --------------------------------------------------------
        # Build Metadata
        # --------------------------------------------------------

        try:

            metadata = (
                metadata_builder.build(
                    file_path=str(file_path),
                    total_chunks=len(chunks),
                    file_hash=hash_value,
                )
            )

        except Exception:

            self._remove_file(
                file_path
            )

            app_logger.exception(
                "Metadata generation failed | "
                f"File={safe_filename}"
            )

            raise

        if len(metadata) != len(chunks):

            self._remove_file(
                file_path
            )

            raise ValueError(
                "Metadata count does not match "
                "chunk count."
            )

        # --------------------------------------------------------
        # Generate Embeddings
        # --------------------------------------------------------

        try:

            embeddings = (
                embedding_service.generate_embeddings(
                    chunks
                )
            )

        except Exception:

            self._remove_file(
                file_path
            )

            app_logger.exception(
                "Embedding generation failed | "
                f"File={safe_filename}"
            )

            raise

        if len(embeddings) != len(chunks):

            self._remove_file(
                file_path
            )

            raise ValueError(
                "Embedding count does not match "
                "chunk count."
            )

        # --------------------------------------------------------
        # Validate Embedding Dimensions
        # --------------------------------------------------------

        if embeddings:

            embedding_dimension = (
                len(embeddings[0])
            )

            invalid_embeddings = any(
                len(embedding)
                != embedding_dimension
                for embedding in embeddings
            )

            if invalid_embeddings:

                self._remove_file(
                    file_path
                )

                raise ValueError(
                    "Embedding dimensions are inconsistent."
                )

        # --------------------------------------------------------
        # Store in ChromaDB
        # --------------------------------------------------------

        try:

            vector_store.add_documents(
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata,
            )

        except Exception:

            self._remove_file(
                file_path
            )

            app_logger.exception(
                "Failed to index document | "
                f"File={safe_filename}"
            )

            raise

        # --------------------------------------------------------
        # Completed
        # --------------------------------------------------------

        processing_time = (
            time.perf_counter()
            - start_time
        )

        app_logger.info(
            "Document indexed successfully | "
            f"File={safe_filename} | "
            f"Chunks={len(chunks)} | "
            f"Characters={len(text)} | "
            f"Time={processing_time:.2f}s"
        )

        return {
            "duplicate": False,
            "file_name": safe_filename,
            "file_type": extension.lstrip("."),
            "file_size": total_bytes,
            "characters": len(text),
            "chunks": len(chunks),
            "preview": text[:500],
            "embedding_model": (
                settings.EMBEDDING_MODEL
            ),
            "processing_time": round(
                processing_time,
                3,
            ),
        }

    # ============================================================
    # Clean Text
    # ============================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """
        Normalize extracted text while preserving
        paragraph boundaries.
        """

        if not isinstance(
            text,
            str,
        ):
            return ""

        # Normalize line endings.
        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # Collapse spaces and tabs.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Preserve paragraph boundaries.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ============================================================
    # Remove File
    # ============================================================

    @staticmethod
    def _remove_file(
        file_path: Path,
    ) -> None:
        """
        Safely remove a local uploaded file.
        """

        if not file_path.exists():
            return

        try:

            file_path.unlink()

            app_logger.info(
                "Removed uploaded file | "
                f"File={file_path.name}"
            )

        except OSError:

            app_logger.exception(
                "Failed to remove uploaded file | "
                f"File={file_path.name}"
            )


# ============================================================
# Shared Service
# ============================================================

document_ingestion_service = (
    DocumentIngestionService()
)