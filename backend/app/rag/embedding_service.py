from typing import Any

from sentence_transformers import SentenceTransformer

from app.config.logging_config import app_logger
from app.config.settings import settings


class EmbeddingService:
    """
    Service responsible for generating vector embeddings.

    The same embedding model MUST be used for:

    - Document indexing
    - User query embedding

    Otherwise, vectors stored in ChromaDB and query vectors
    will not be compatible.
    """

    def __init__(self) -> None:

        self.model_name = (
            settings.EMBEDDING_MODEL
        )

        if not self.model_name:
            raise ValueError(
                "EMBEDDING_MODEL is not configured."
            )

        app_logger.info(
            f"Loading embedding model | "
            f"Model={self.model_name}"
        )

        try:

            self.model = SentenceTransformer(
                self.model_name
            )

        except Exception:

            app_logger.exception(
                "Failed to load embedding model."
            )

            raise

        # --------------------------------------------------------
        # Determine Embedding Dimension
        # --------------------------------------------------------

        try:
            if hasattr(self.model, "get_embedding_dimension"):
                self.embedding_dimension = self.model.get_embedding_dimension()
            else:
                self.embedding_dimension = (
                    self.model.get_sentence_embedding_dimension()
                )
        
        except Exception:
            app_logger.exception(
                "Failed to determine embedding dimension."
            )
            raise

        if not self.embedding_dimension:

            raise ValueError(
                "Embedding model returned an invalid "
                "embedding dimension."
            )

        app_logger.info(
            f"Embedding model loaded successfully | "
            f"Dimension={self.embedding_dimension}"
        )

    # ============================================================
    # Generate Embeddings
    # ============================================================

    def generate_embeddings(
        self,
        chunks: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of text chunks.

        Every input chunk must be a non-empty string.
        """

        if not chunks:

            app_logger.warning(
                "No text chunks provided for embedding."
            )

            return []

        # --------------------------------------------------------
        # Validate Input
        # --------------------------------------------------------

        cleaned_chunks: list[str] = []

        for index, chunk in enumerate(
            chunks
        ):

            if not isinstance(
                chunk,
                str,
            ):

                raise ValueError(
                    f"Chunk {index} must be a string."
                )

            cleaned_chunk = chunk.strip()

            if not cleaned_chunk:

                raise ValueError(
                    f"Chunk {index} is empty."
                )

            cleaned_chunks.append(
                cleaned_chunk
            )

        app_logger.info(
            f"Generating embeddings | "
            f"Chunks={len(cleaned_chunks)}"
        )

        # --------------------------------------------------------
        # Generate Embeddings
        # --------------------------------------------------------

        try:

            embeddings = self.model.encode(
                cleaned_chunks,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        except Exception:

            app_logger.exception(
                "Embedding generation failed."
            )

            raise

        # --------------------------------------------------------
        # Validate Model Output
        # --------------------------------------------------------

        if embeddings is None:

            raise RuntimeError(
                "Embedding model returned no result."
            )

        try:

            result = embeddings.tolist()

        except AttributeError as exc:

            app_logger.exception(
                "Embedding model returned an "
                "unexpected result type."
            )

            raise RuntimeError(
                "Invalid embedding model output."
            ) from exc

        if not result:

            raise RuntimeError(
                "Embedding model returned no vectors."
            )

        # --------------------------------------------------------
        # Validate Count
        # --------------------------------------------------------

        if len(result) != len(
            cleaned_chunks
        ):

            app_logger.error(
                "Embedding count mismatch | "
                f"Input={len(cleaned_chunks)} | "
                f"Output={len(result)}"
            )

            raise RuntimeError(
                "Embedding count does not match "
                "input chunk count."
            )

        # --------------------------------------------------------
        # Validate Dimensions
        # --------------------------------------------------------

        for index, vector in enumerate(
            result
        ):

            if not isinstance(
                vector,
                list,
            ):

                raise RuntimeError(
                    f"Invalid embedding returned "
                    f"for chunk {index}."
                )

            if len(vector) != (
                self.embedding_dimension
            ):

                app_logger.error(
                    "Embedding dimension mismatch | "
                    f"Chunk={index} | "
                    f"Expected="
                    f"{self.embedding_dimension} | "
                    f"Actual={len(vector)}"
                )

                raise ValueError(
                    "Embedding dimension mismatch."
                )

        # --------------------------------------------------------
        # Final Logging
        # --------------------------------------------------------

        app_logger.info(
            f"Embeddings generated successfully | "
            f"Count={len(result)} | "
            f"Dimension={self.embedding_dimension}"
        )

        return result


# ============================================================
# Shared Service Instance
# ============================================================

embedding_service = EmbeddingService()