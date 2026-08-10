from typing import Any

from sentence_transformers import CrossEncoder

from app.config.logging_config import app_logger
from app.config.settings import settings


class RerankerService:
    """
    Cross-encoder based document reranking service.

    Takes the initial vector-search results and scores each
    query/document pair for semantic relevance.
    """

    def __init__(self) -> None:

        model_name = settings.RERANKER_MODEL

        if not model_name:
            raise ValueError(
                "RERANKER_MODEL is not configured."
            )

        app_logger.info(
            f"Loading reranker model | Model={model_name}"
        )

        try:

            self.model = CrossEncoder(
                model_name
            )

        except Exception:

            app_logger.exception(
                "Failed to load reranker model."
            )

            raise

        app_logger.info(
            "Reranker model loaded successfully."
        )

    # ============================================================
    # Rerank Documents
    # ============================================================

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Rerank retrieved documents using the cross encoder.

        Parameters:
            query:
                User's search/query text.

            documents:
                Documents returned by vector search.

            top_k:
                Maximum number of ranked documents to return.

        Returns:
            Documents sorted by descending reranker score.
        """

        # --------------------------------------------------------
        # Validate Query
        # --------------------------------------------------------

        if not isinstance(query, str):

            app_logger.warning(
                "Invalid query type received by reranker."
            )

            return []

        query = query.strip()

        if not query:

            app_logger.warning(
                "Empty query received by reranker."
            )

            return []

        # --------------------------------------------------------
        # Validate Documents
        # --------------------------------------------------------

        if not documents:

            app_logger.info(
                "No documents provided for reranking."
            )

            return []

        # --------------------------------------------------------
        # Validate Top-K
        # --------------------------------------------------------

        try:

            top_k = int(top_k)

        except (
            TypeError,
            ValueError,
        ):

            app_logger.warning(
                f"Invalid top_k={top_k}."
            )

            return []

        if top_k <= 0:

            app_logger.warning(
                f"Invalid top_k={top_k}."
            )

            return []

        app_logger.info(
            "Reranking documents | "
            f"Candidates={len(documents)} | "
            f"TopK={top_k}"
        )

        # --------------------------------------------------------
        # Build Query / Document Pairs
        # --------------------------------------------------------

        pairs: list[
            tuple[str, str]
        ] = []

        valid_documents: list[
            dict[str, Any]
        ] = []

        for document in documents:

            if not isinstance(
                document,
                dict,
            ):
                continue

            chunk = document.get(
                "chunk",
                "",
            )

            if not isinstance(
                chunk,
                str,
            ):
                continue

            chunk = chunk.strip()

            if not chunk:
                continue

            pairs.append(
                (
                    query,
                    chunk,
                )
            )

            # Copy so this service doesn't mutate
            # the original retrieval result.
            valid_documents.append(
                dict(document)
            )

        if not pairs:

            app_logger.warning(
                "No valid document chunks available "
                "for reranking."
            )

            return []

        # --------------------------------------------------------
        # Generate Scores
        # --------------------------------------------------------

        try:

            scores = self.model.predict(
                pairs
            )

        except Exception:

            app_logger.exception(
                "Reranker prediction failed."
            )

            raise

        # --------------------------------------------------------
        # Validate Scores
        # --------------------------------------------------------

        if scores is None:

            app_logger.error(
                "Reranker returned no scores."
            )

            return []

        if len(scores) != len(
            valid_documents
        ):

            app_logger.error(
                "Reranker score count does not "
                "match document count | "
                f"Scores={len(scores)} | "
                f"Documents={len(valid_documents)}"
            )

            raise RuntimeError(
                "Reranker returned an invalid "
                "number of scores."
            )

        # --------------------------------------------------------
        # Attach Scores
        # --------------------------------------------------------

        ranked_documents: list[
            dict[str, Any]
        ] = []

        for document, score in zip(
            valid_documents,
            scores,
        ):

            try:

                rerank_score = float(
                    score
                )

            except (
                TypeError,
                ValueError,
            ):

                app_logger.warning(
                    "Invalid reranker score encountered."
                )

                continue

            document[
                "rerank_score"
            ] = rerank_score

            ranked_documents.append(
                document
            )

        if not ranked_documents:

            app_logger.info(
                "No valid reranker scores generated."
            )

            return []

        # --------------------------------------------------------
        # Sort by Relevance
        # --------------------------------------------------------

        ranked_documents.sort(
            key=lambda document:
                document["rerank_score"],
            reverse=True,
        )

        # --------------------------------------------------------
        # Top-K
        # --------------------------------------------------------

        selected_documents = (
            ranked_documents[:top_k]
        )

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------

        app_logger.info(
            "Reranking completed | "
            f"Valid={len(ranked_documents)} | "
            f"Returned={len(selected_documents)}"
        )

        for index, document in enumerate(
            selected_documents,
            start=1,
        ):

            metadata = document.get(
                "metadata",
                {},
            )

            app_logger.debug(
                f"Rerank result {index} | "
                f"Source={metadata.get('source', 'unknown')} | "
                f"Chunk={metadata.get('chunk', 'unknown')} | "
                f"Score="
                f"{document['rerank_score']:.4f}"
            )

        return selected_documents


# ============================================================
# Shared Service Instance
# ============================================================

reranker_service = RerankerService()