from pathlib import Path
from typing import Any

import chromadb

from app.config.logging_config import app_logger
from app.config.settings import settings


class VectorStore:
    """
    Persistent ChromaDB vector store.

    Responsibilities:
    - Store document chunks and embeddings
    - Search document chunks
    - Filter searches by document name
    - Detect duplicate documents using SHA-256
    - List indexed documents
    - Retrieve document details using file hash
    - Delete documents using file hash
    - Delete all documents
    - Provide vector-store statistics
    """

    COLLECTION_NAME = "documents"

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self) -> None:

        app_logger.info(
            "Initializing ChromaDB."
        )

        if not settings.CHROMA_DB_PATH:

            raise ValueError(
                "CHROMA_DB_PATH is not configured."
            )

        try:

            Path(
                settings.CHROMA_DB_PATH
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            self.client = (
                chromadb.PersistentClient(
                    path=settings.CHROMA_DB_PATH
                )
            )

            self.collection = (
                self.client.get_or_create_collection(
                    name=self.COLLECTION_NAME
                )
            )

        except Exception:

            app_logger.exception(
                "Failed to initialize ChromaDB."
            )

            raise

        app_logger.info(
            "ChromaDB initialized successfully | "
            f"Collection={self.COLLECTION_NAME}"
        )

    # ============================================================
    # Add Documents
    # ============================================================

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        """
        Add document chunks, embeddings, and metadata.

        Chroma IDs are generated from:

            file_hash + chunk_number

        This guarantees that different files with the same
        filename can coexist safely.
        """

        if not chunks:

            raise ValueError(
                "No chunks provided for indexing."
            )

        if len(chunks) != len(embeddings):

            raise ValueError(
                "Chunks and embeddings count do not match."
            )

        if len(chunks) != len(metadata):

            raise ValueError(
                "Chunks and metadata count do not match."
            )

        ids: list[str] = []

        for index, item_metadata in enumerate(
            metadata
        ):

            if not item_metadata:

                raise ValueError(
                    f"Metadata missing for chunk {index}."
                )

            source = str(
                item_metadata.get(
                    "source",
                    "",
                )
            ).strip()

            if not source:

                raise ValueError(
                    f"Metadata source missing for chunk {index}."
                )

            file_hash = str(
                item_metadata.get(
                    "file_hash",
                    "",
                )
            ).strip()

            if not file_hash:

                raise ValueError(
                    f"Metadata file_hash missing "
                    f"for chunk {index}."
                )

            chunk_number = item_metadata.get(
                "chunk",
                index + 1,
            )

            try:

                chunk_number = int(
                    chunk_number
                )

            except (TypeError, ValueError):

                raise ValueError(
                    f"Invalid chunk number for chunk {index}."
                )

            if chunk_number <= 0:

                raise ValueError(
                    "Chunk number must be greater than zero."
                )

            document_id = (
                f"{file_hash}_{chunk_number}"
            )

            ids.append(
                document_id
            )

        try:

            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata,
            )

        except Exception:

            app_logger.exception(
                "Failed to add documents to ChromaDB."
            )

            raise

        app_logger.info(
            "Documents stored successfully | "
            f"Chunks={len(chunks)}"
        )

    # ============================================================
    # Search
    # ============================================================

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        documents: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Search ChromaDB using an embedding.

        If document names are supplied, restrict the search
        to those document sources.
        """

        if not query_embedding:

            raise ValueError(
                "Query embedding cannot be empty."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        collection_count = (
            self.collection.count()
        )

        if collection_count == 0:

            app_logger.info(
                "ChromaDB collection is empty."
            )

            return self._empty_search_result()

        search_limit = min(
            top_k,
            collection_count,
        )

        query: dict[str, Any] = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": search_limit,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        # --------------------------------------------------------
        # Document Filtering
        # --------------------------------------------------------

        if documents is not None:

            valid_documents = [
                document.strip()
                for document in documents
                if isinstance(document, str)
                and document.strip()
            ]

            valid_documents = list(
                dict.fromkeys(
                    valid_documents
                )
            )

            if not valid_documents:

                app_logger.info(
                    "Document filter supplied but "
                    "contained no valid document names."
                )

                return self._empty_search_result()

            app_logger.info(
                "Filtering vector search | "
                f"Documents={valid_documents}"
            )

            if len(valid_documents) == 1:

                query["where"] = {
                    "source": valid_documents[0]
                }

            else:

                query["where"] = {
                    "$or": [
                        {
                            "source": document
                        }
                        for document in valid_documents
                    ]
                }

        # --------------------------------------------------------
        # Execute Search
        # --------------------------------------------------------

        try:

            results = (
                self.collection.query(
                    **query
                )
            )

        except Exception:

            app_logger.exception(
                "ChromaDB vector search failed."
            )

            raise

        documents_result = (
            results.get(
                "documents"
            )
            or []
        )

        if not documents_result:

            app_logger.info(
                "ChromaDB returned no documents."
            )

            return self._empty_search_result()

        result_count = len(
            documents_result[0]
        )

        app_logger.info(
            "ChromaDB search completed | "
            f"Results={result_count}"
        )

        return results

    # ============================================================
    # List Documents
    # ============================================================

    def list_documents(
        self,
    ) -> list[dict]:
        """
        Return a summary of all indexed documents.

        Each unique file_hash represents one document.
        """

        try:

            data = self.collection.get(
                include=[
                    "metadatas"
                ]
            )

        except Exception:

            app_logger.exception(
                "Failed to list ChromaDB documents."
            )

            raise

        metadatas = (
            data.get(
                "metadatas"
            )
            or []
        )

        documents: dict[
            str,
            dict[str, Any],
        ] = {}

        for metadata in metadatas:

            if not metadata:
                continue

            source = str(
                metadata.get(
                    "source",
                    "",
                )
            ).strip()

            file_hash = str(
                metadata.get(
                    "file_hash",
                    "",
                )
            ).strip()

            if not source or not file_hash:
                continue

            if file_hash not in documents:

                documents[file_hash] = {
                    "file_name": source,
                    "file_type": metadata.get(
                        "file_type",
                        "unknown",
                    ),
                    "file_hash": file_hash,
                    "chunks": 0,
                }

            documents[file_hash]["chunks"] += 1

        return sorted(
            documents.values(),
            key=lambda item: (
                item["file_name"].lower(),
                item["file_hash"],
            ),
        )

    # ============================================================
    # Get Document
    # ============================================================

    def get_document(
        self,
        file_hash: str,
    ) -> dict | None:
        """
        Retrieve document details using its SHA-256 hash.
        """

        if not isinstance(
            file_hash,
            str,
        ):

            return None

        file_hash = file_hash.strip()

        if not file_hash:

            return None

        app_logger.info(
            "Fetching document by hash | "
            f"Hash={file_hash}"
        )

        try:

            results = (
                self.collection.get(
                    where={
                        "file_hash": file_hash
                    },
                    include=[
                        "documents",
                        "metadatas",
                    ],
                )
            )

        except Exception:

            app_logger.exception(
                "Failed to fetch document | "
                f"Hash={file_hash}"
            )

            raise

        ids = (
            results.get(
                "ids"
            )
            or []
        )

        if not ids:

            app_logger.info(
                "Document not found | "
                f"Hash={file_hash}"
            )

            return None

        metadatas = (
            results.get(
                "metadatas"
            )
            or []
        )

        documents = (
            results.get(
                "documents"
            )
            or []
        )

        chunk_details: list[
            dict[str, Any]
        ] = []

        for metadata, document in zip(
            metadatas,
            documents,
        ):

            metadata = (
                metadata
                or {}
            )

            document = (
                document
                or ""
            )

            chunk_details.append(
                {
                    "chunk": metadata.get(
                        "chunk",
                        0,
                    ),
                    "characters": len(
                        document
                    ),
                    "content": document,
                }
            )

        file_name = (
            metadatas[0].get(
                "source",
                "unknown",
            )
            if metadatas
            else "unknown"
        )

        file_type = (
            metadatas[0].get(
                "file_type",
                "unknown",
            )
            if metadatas
            else "unknown"
        )

        chunk_details.sort(
            key=lambda item: item["chunk"]
        )

        return {
            "file_name": file_name,
            "file_type": file_type,
            "file_hash": file_hash,
            "total_chunks": len(
                chunk_details
            ),
            "total_characters": sum(
                item["characters"]
                for item in chunk_details
            ),
            "chunks": chunk_details,
        }

    # ============================================================
    # Delete Document
    # ============================================================

    def delete_document(
        self,
        file_hash: str,
    ) -> dict | None:
        """
        Delete a document using its SHA-256 hash.

        The vector data is deleted first.

        The physical uploaded file is then removed using
        the filename stored in the document metadata.
        """

        if not isinstance(
            file_hash,
            str,
        ):

            raise ValueError(
                "File hash must be a string."
            )

        file_hash = file_hash.strip()

        if not file_hash:

            raise ValueError(
                "File hash cannot be empty."
            )

        app_logger.warning(
            "Deleting document by hash | "
            f"Hash={file_hash}"
        )

        try:

            result = (
                self.collection.get(
                    where={
                        "file_hash": file_hash
                    },
                    include=[
                        "metadatas"
                    ],
                )
            )

            ids = (
                result.get(
                    "ids"
                )
                or []
            )

            metadatas = (
                result.get(
                    "metadatas"
                )
                or []
            )

            if not ids:

                app_logger.warning(
                    "Document not found | "
                    f"Hash={file_hash}"
                )

                return None

            deleted_chunks = len(
                ids
            )

            file_name = None

            if metadatas:

                file_name = (
                    metadatas[0].get(
                        "source"
                    )
                )

            self.collection.delete(
                ids=ids
            )

        except Exception:

            app_logger.exception(
                "Failed to delete document "
                "from ChromaDB | "
                f"Hash={file_hash}"
            )

            raise

        # --------------------------------------------------------
        # Delete Physical Upload
        # --------------------------------------------------------

        deleted_file = False

        if file_name:

            upload_path = (
                Path(
                    settings.UPLOAD_FOLDER
                )
                / str(file_name)
            )

            if upload_path.is_file():

                try:

                    upload_path.unlink()

                    deleted_file = True

                    app_logger.info(
                        "Deleted uploaded file | "
                        f"File={file_name}"
                    )

                except OSError:

                    app_logger.exception(
                        "Failed to delete uploaded file | "
                        f"File={file_name}"
                    )

        app_logger.info(
            "Document deleted successfully | "
            f"File={file_name} | "
            f"Hash={file_hash} | "
            f"Chunks={deleted_chunks}"
        )

        return {
            "file_name": file_name,
            "file_hash": file_hash,
            "deleted_chunks": deleted_chunks,
            "deleted_file": deleted_file,
        }

    # ============================================================
    # Duplicate Detection
    # ============================================================

    def document_exists(
        self,
        file_hash: str,
    ) -> bool:
        """
        Check whether a document with the supplied
        SHA-256 hash already exists.
        """

        if not isinstance(
            file_hash,
            str,
        ):

            return False

        file_hash = file_hash.strip()

        if not file_hash:

            return False

        try:

            results = (
                self.collection.get(
                    where={
                        "file_hash": file_hash
                    },
                    limit=1,
                )
            )

        except Exception:

            app_logger.exception(
                "Failed to check document hash."
            )

            raise

        ids = (
            results.get(
                "ids"
            )
            or []
        )

        exists = bool(
            ids
        )

        if exists:

            app_logger.warning(
                "Duplicate document detected | "
                f"Hash={file_hash}"
            )

        else:

            app_logger.info(
                "Document hash not found."
            )

        return exists

    # ============================================================
    # Statistics
    # ============================================================

    def get_statistics(
        self,
    ) -> dict:
        """
        Return vector-store statistics.
        """

        documents = (
            self.list_documents()
        )

        total_chunks = sum(
            document.get(
                "chunks",
                0,
            )
            for document in documents
        )

        return {
            "documents": len(
                documents
            ),
            "chunks": total_chunks,
        }

    # ============================================================
    # Delete All Documents
    # ============================================================

    def delete_all_documents(
        self,
    ) -> dict:
        """
        Delete every indexed document
        and uploaded file.
        """

        app_logger.warning(
            "Deleting all indexed documents."
        )

        documents = (
            self.list_documents()
        )

        deleted_documents = len(
            documents
        )

        deleted_chunks = sum(
            document.get(
                "chunks",
                0,
            )
            for document in documents
        )

        # --------------------------------------------------------
        # Delete ChromaDB Vectors
        # --------------------------------------------------------

        try:

            result = (
                self.collection.get()
            )

            all_ids = (
                result.get(
                    "ids"
                )
                or []
            )

            if all_ids:

                self.collection.delete(
                    ids=all_ids
                )

        except Exception:

            app_logger.exception(
                "Failed to delete all ChromaDB documents."
            )

            raise

        # --------------------------------------------------------
        # Delete Uploaded Files
        # --------------------------------------------------------

        uploads_folder = Path(
            settings.UPLOAD_FOLDER
        )

        if uploads_folder.is_dir():

            for path in uploads_folder.iterdir():

                if not path.is_file():
                    continue

                try:

                    path.unlink()

                except OSError:

                    app_logger.exception(
                        "Failed to delete uploaded file | "
                        f"File={path.name}"
                    )

        app_logger.warning(
            "Deleted all documents | "
            f"Documents={deleted_documents} | "
            f"Chunks={deleted_chunks}"
        )

        return {
            "deleted_documents": (
                deleted_documents
            ),
            "deleted_chunks": (
                deleted_chunks
            ),
        }

    # ============================================================
    # Database Info
    # ============================================================

    def get_database_info(
        self,
    ) -> dict:
        """
        Return vector-store information.
        """

        return {
            "collection": (
                self.collection.name
            ),
            "statistics": (
                self.get_statistics()
            ),
        }

    # ============================================================
    # Show Collection
    # ============================================================

    def show_collection(
        self,
    ) -> dict:
        """
        Return basic collection information.
        """

        count = (
            self.collection.count()
        )

        info = {
            "collection": (
                self.collection.name
            ),
            "count": count,
        }

        app_logger.info(
            "ChromaDB collection | "
            f"Name={self.collection.name} | "
            f"Count={count}"
        )

        return info

    # ============================================================
    # Empty Search Result
    # ============================================================

    @staticmethod
    def _empty_search_result() -> dict:
        """
        Return a consistent empty search result.
        """

        return {
            "documents": [
                []
            ],
            "metadatas": [
                []
            ],
            "distances": [
                []
            ],
        }


# ============================================================
# Shared Vector Store
# ============================================================

vector_store = VectorStore()