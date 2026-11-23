from sqlalchemy import text

from app.config.logging_config import app_logger
from app.database.database import SessionLocal
from app.llm.ollama_client import ollama_client
from app.rag.embedding_service import embedding_service
from app.rag.reranker_service import reranker_service
from app.rag.vector_store import vector_store


class HealthService:
    """
    Service responsible for checking the health of
    application dependencies.
    """

    # ============================================================
    # Health Check
    # ============================================================

    async def check(self) -> dict:
        """
        Run health checks for the application's dependencies.

        Checks:
        - API
        - Database
        - Ollama
        - ChromaDB
        - Embedding model
        - Reranker
        """

        app_logger.info(
            "Running application health checks."
        )

        status = {
            "api": "healthy",
            "database": "healthy",
            "ollama": "healthy",
            "chromadb": "healthy",
            "embedding_model": "healthy",
            "reranker": "healthy",
        }

        # ========================================================
        # Database
        # ========================================================

        db = None

        try:

            db = SessionLocal()

            db.execute(
                text("SELECT 1")
            )

            app_logger.debug(
                "Database health check passed."
            )

        except Exception:

            status["database"] = "unhealthy"

            app_logger.exception(
                "Database health check failed."
            )

        finally:

            if db is not None:

                try:

                    db.close()

                except Exception:

                    app_logger.exception(
                        "Failed to close health-check "
                        "database session."
                    )

        # ========================================================
        # Ollama
        # ========================================================

        try:

            if ollama_client.client.is_closed:

                status["ollama"] = "unhealthy"

                app_logger.warning(
                    "Ollama health check failed: "
                    "HTTP client is closed."
                )

            else:

                # Check that the Ollama service is reachable
                # without generating an actual LLM response.
                response = await ollama_client.client.get(
                    f"{ollama_client.base_url}/api/tags"
                )

                response.raise_for_status()

                app_logger.debug(
                    "Ollama health check passed."
                )

        except Exception:

            status["ollama"] = "unhealthy"

            app_logger.exception(
                "Ollama health check failed."
            )

        # ========================================================
        # ChromaDB
        # ========================================================

        try:

            count = (
                vector_store.collection.count()
            )

            app_logger.debug(
                "ChromaDB health check passed | "
                f"Chunks={count}"
            )

        except Exception:

            status["chromadb"] = "unhealthy"

            app_logger.exception(
                "ChromaDB health check failed."
            )

        # ========================================================
        # Embedding Model
        # ========================================================

        try:

            if hasattr(embedding_service, "ensure_loaded"):
                embedding_service.ensure_loaded()

            if embedding_service.model is None:

                raise RuntimeError(
                    "Embedding model is not initialized."
                )

            if (
                embedding_service.embedding_dimension
                is None
            ):

                raise RuntimeError(
                    "Embedding dimension is unavailable."
                )

            app_logger.debug(
                "Embedding model health check passed | "
                f"Dimension="
                f"{embedding_service.embedding_dimension}"
            )

        except Exception:

            status["embedding_model"] = "unhealthy"

            app_logger.exception(
                "Embedding model health check failed."
            )

        # ========================================================
        # Reranker
        # ========================================================

        try:

            if hasattr(reranker_service, "ensure_loaded"):
                reranker_service.ensure_loaded()

            if reranker_service.model is None:

                raise RuntimeError(
                    "Reranker model is not initialized."
                )

            app_logger.debug(
                "Reranker model health check passed."
            )

        except Exception:

            status["reranker"] = "unhealthy"

            app_logger.exception(
                "Reranker model health check failed."
            )

        # ========================================================
        # Overall Status
        # ========================================================

        overall_status = (
            "healthy"
            if all(
                value == "healthy"
                for value in status.values()
            )
            else "degraded"
        )

        # ========================================================
        # Statistics
        # ========================================================

        try:

            statistics = (
                vector_store.get_statistics()
            )

        except Exception:

            app_logger.exception(
                "Failed to load vector-store statistics."
            )

            statistics = {
                "documents": 0,
                "chunks": 0,
            }

        # ========================================================
        # Final Result
        # ========================================================

        result = {
            "status": overall_status,
            "components": status,
            "statistics": statistics,
            "ollama_model": ollama_client.model,
        }

        app_logger.info(
            "Health checks completed | "
            f"Status={overall_status}"
        )

        return result


# ============================================================
# Shared Health Service
# ============================================================

health_service = HealthService()