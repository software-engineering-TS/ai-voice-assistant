from typing import Any

from app.config.logging_config import app_logger
from app.config.settings import settings
from app.rag.embedding_service import embedding_service
from app.rag.reranker_service import reranker_service
from app.rag.vector_store import vector_store


class RetrievalService:
    """
    Service responsible for retrieving relevant document context.

    Retrieval pipeline:

        User Query
             ↓
        Query Embedding
             ↓
        ChromaDB Vector Search
             ↓
        Cross Encoder Reranking
             ↓
        Relevance Filtering
             ↓
        Duplicate Removal
             ↓
        Top-K Selection
             ↓
        Context + Sources
    """

    # ============================================================
    # Retrieve
    # ============================================================

    def retrieve(
        self,
        query: str,
        documents: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve relevant document context.

        Parameters:
            query:
                User's search query.

            documents:
                Optional list of document names.

                If provided:
                    Search only those documents.

                If None:
                    Search across all indexed documents.

        Returns:
            Dictionary containing:

            {
                "context": str,
                "sources": list[str],
                "documents": list[dict]
            }
        """

        # --------------------------------------------------------
        # Validate Query
        # --------------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        query = query.strip()

        if not query:

            app_logger.warning(
                "Empty retrieval query received."
            )

            return self._empty_result()

        # --------------------------------------------------------
        # Normalize Document Selection
        # --------------------------------------------------------

        selected_documents = (
            self._normalize_documents(
                documents
            )
        )

        if selected_documents:

            app_logger.info(
                "Starting RAG retrieval | "
                f"Documents={selected_documents}"
            )

        else:

            app_logger.info(
                "Starting RAG retrieval across "
                "all indexed documents."
            )

        # ========================================================
        # Query Embedding
        # ========================================================

        app_logger.info(
            "Generating query embedding."
        )

        try:

            embeddings = (
                embedding_service.generate_embeddings(
                    [query]
                )
            )

        except Exception:

            app_logger.exception(
                "Failed to generate query embedding."
            )

            raise

        if not embeddings:

            app_logger.warning(
                "Embedding service returned no embedding."
            )

            return self._empty_result()

        query_embedding = embeddings[0]

        if not query_embedding:

            app_logger.warning(
                "Generated query embedding is empty."
            )

            return self._empty_result()

        app_logger.info(
            "Query embedding generated successfully."
        )

        # ========================================================
        # Vector Search
        # ========================================================

        app_logger.info(
            "Searching vector store."
        )

        try:

            results = vector_store.search(
                query_embedding=query_embedding,
                top_k=settings.RAG_SEARCH_TOP_K,
                documents=(
                    selected_documents
                    if selected_documents
                    else None
                ),
            )

        except Exception:

            app_logger.exception(
                "Vector search failed."
            )

            raise

        # ========================================================
        # Extract ChromaDB Results
        # ========================================================

        chunks_result = (
            results.get("documents")
            or []
        )

        metadata_result = (
            results.get("metadatas")
            or []
        )

        distance_result = (
            results.get("distances")
            or []
        )

        if not chunks_result:

            app_logger.info(
                "No documents retrieved from vector search."
            )

            return self._empty_result()

        chunks = chunks_result[0]

        metadata = (
            metadata_result[0]
            if metadata_result
            else []
        )

        distances = (
            distance_result[0]
            if distance_result
            else []
        )

        if not chunks:

            app_logger.info(
                "Vector search returned no chunks."
            )

            return self._empty_result()

        # ========================================================
        # Validate Result Alignment
        # ========================================================

        result_count = min(
            len(chunks),
            len(metadata),
            len(distances),
        )

        if result_count == 0:

            app_logger.warning(
                "Vector search returned incomplete results."
            )

            return self._empty_result()

        if not (
            len(chunks)
            == len(metadata)
            == len(distances)
        ):

            app_logger.warning(
                "ChromaDB result arrays have different "
                "lengths | "
                f"Chunks={len(chunks)} | "
                f"Metadata={len(metadata)} | "
                f"Distances={len(distances)}"
            )

        # ========================================================
        # Build Retrieval Documents
        # ========================================================

        retrieved_documents: list[
            dict[str, Any]
        ] = []

        for index in range(
            result_count
        ):

            chunk = chunks[index]
            meta = metadata[index]
            distance = distances[index]

            if not chunk or not meta:
                continue

            if not isinstance(
                meta,
                dict,
            ):
                app_logger.warning(
                    f"Invalid metadata for result {index}."
                )

                continue

            try:

                distance_value = float(
                    distance
                )

            except (
                TypeError,
                ValueError,
            ):

                app_logger.warning(
                    "Invalid vector distance | "
                    f"Result={index}"
                )

                continue

            retrieved_documents.append(
                {
                    "chunk": str(
                        chunk
                    ),
                    "metadata": meta,
                    "distance": round(
                        distance_value,
                        4,
                    ),
                }
            )

        if not retrieved_documents:

            app_logger.info(
                "No valid retrieved documents available."
            )

            return self._empty_result()

        app_logger.info(
            "Vector search completed | "
            f"Retrieved={len(retrieved_documents)}"
        )

        # ========================================================
        # Cross Encoder Reranking
        # ========================================================

        app_logger.info(
            "Starting document reranking."
        )

        try:

            ranked_documents = (
                reranker_service.rerank(
                    query=query,
                    documents=retrieved_documents,
                    top_k=settings.RAG_TOP_K,
                )
            )

        except Exception:

            app_logger.exception(
                "Document reranking failed."
            )

            raise

        if not ranked_documents:

            app_logger.info(
                "Reranker returned no documents."
            )

            return self._empty_result()

        app_logger.info(
            "Reranking completed | "
            f"Candidates={len(ranked_documents)}"
        )

        # ========================================================
        # Relevance Filtering
        # ========================================================

        threshold = float(
            settings.RERANK_SCORE_THRESHOLD
        )

        app_logger.info(
            "Applying rerank relevance threshold | "
            f"Threshold={threshold}"
        )

        selected_results: list[
            dict[str, Any]
        ] = []

        for document in ranked_documents:

            try:

                score = float(
                    document.get(
                        "rerank_score",
                        0.0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                app_logger.warning(
                    "Invalid rerank score detected."
                )

                continue

            if score < threshold:

                app_logger.debug(
                    "Document rejected by "
                    "rerank threshold | "
                    f"Score={score:.4f} | "
                    f"Threshold={threshold}"
                )

                continue

            document[
                "rerank_score"
            ] = score

            selected_results.append(
                document
            )

        if not selected_results:

            app_logger.info(
                "No documents passed the "
                "rerank relevance threshold."
            )

            return self._empty_result()

        app_logger.info(
            "Relevance filtering completed | "
            f"Selected={len(selected_results)}"
        )

        # ========================================================
        # Remove Duplicate Chunks
        # ========================================================

        seen_chunks: set[
            tuple[str, Any]
        ] = set()

        unique_documents: list[
            dict[str, Any]
        ] = []

        for document in selected_results:

            metadata = document.get(
                "metadata",
                {},
            )

            source = str(
                metadata.get(
                    "source",
                    "unknown",
                )
            ).strip()

            chunk_number = metadata.get(
                "chunk",
                "unknown",
            )

            chunk_key = (
                source,
                chunk_number,
            )

            if chunk_key in seen_chunks:
                continue

            seen_chunks.add(
                chunk_key
            )

            unique_documents.append(
                document
            )

        # ========================================================
        # Final Top-K
        # ========================================================

        final_top_k = max(
            int(settings.RAG_TOP_K),
            1,
        )

        unique_documents = (
            unique_documents[
                :final_top_k
            ]
        )

        final_top_k = max(
            int(settings.RAG_TOP_K),
            1,
        )
        
        unique_documents = (
            unique_documents[
                :final_top_k
            ]
        )
        
        # ========================================================
        # Build Context
        # ========================================================
        
        context_parts: list[str] = []
        sources: list[str] = []

        # ========================================================
        # Build Context
        # ========================================================

        context_parts: list[str] = []
        sources: list[str] = []
        search_results: list[
            dict[str, Any]
        ] = []

        for document in unique_documents:

            metadata = document.get(
                "metadata",
                {},
            )

            source = str(
                metadata.get(
                    "source",
                    "unknown",
                )
            ).strip()

            file_type = str(
                metadata.get(
                    "file_type",
                    "unknown",
                )
            ).strip()

            chunk_number = metadata.get(
                "chunk",
                "unknown",
            )

            content = str(
                document.get(
                    "chunk",
                    "",
                )
            ).strip()

            distance = float(
                document.get(
                    "distance",
                    0.0,
                )
            )

            rerank_score = float(
                document.get(
                    "rerank_score",
                    0.0,
                )
            )

            if not content:
                continue

            # ----------------------------------------------------
            # Context
            # ----------------------------------------------------

            context_parts.append(
                (
                    f"Document: {source}\n"
                    f"File Type: {file_type}\n"
                    f"Chunk: {chunk_number}\n"
                    f"Content:\n"
                    f"{content}"
                )
            )

            # ----------------------------------------------------
            # Sources
            # ----------------------------------------------------

            source_label = (
                f"{source} "
                f"(Chunk {chunk_number})"
            )

            if source_label not in sources:

                sources.append(
                    source_label
                )

            # ----------------------------------------------------
            # Search Metadata
            # ----------------------------------------------------

            search_results.append(
                {
                    "source": source,
                    "file_type": file_type,
                    "chunk": chunk_number,
                    "distance": round(
                        distance,
                        4,
                    ),
                    "rerank_score": round(
                        rerank_score,
                        4,
                    ),
                }
            )

        # ========================================================
        # Final Context
        # ========================================================

        context = "\n\n".join(
            context_parts
        ).strip()

        if not context:

            app_logger.info(
                "No usable context generated."
            )

            return self._empty_result()

        # ========================================================
        # Logging
        # ========================================================

        app_logger.info(
            "RAG retrieval completed | "
            f"SelectedChunks="
            f"{len(search_results)} | "
            f"Sources={len(sources)} | "
            f"ContextCharacters="
            f"{len(context)}"
        )

        for index, result in enumerate(
            search_results,
            start=1,
        ):

            app_logger.info(
                f"RAG result {index} | "
                f"Source={result['source']} | "
                f"Chunk={result['chunk']} | "
                f"Distance={result['distance']} | "
                f"RerankScore="
                f"{result['rerank_score']}"
            )

        return {
            "context": context,
            "sources": sources,
            "documents": search_results,
        }

    # ============================================================
    # Normalize Documents
    # ============================================================

    @staticmethod
    def _normalize_documents(
        documents: list[str] | None,
    ) -> list[str]:
        """
        Clean and deduplicate document names.
        """

        if not documents:
            return []

        normalized: list[str] = []

        for document in documents:

            if not isinstance(
                document,
                str,
            ):
                continue

            document = document.strip()

            if not document:
                continue

            if document not in normalized:

                normalized.append(
                    document
                )

        return normalized

    # ============================================================
    # Empty Result
    # ============================================================

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """
        Return a consistent empty retrieval response.
        """

        return {
            "context": "",
            "sources": [],
            "documents": [],
        }


# ============================================================
# Shared Retrieval Service
# ============================================================

retrieval_service = RetrievalService()