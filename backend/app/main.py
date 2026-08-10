# ============================================================
# Standard Library
# ============================================================

from contextlib import asynccontextmanager
from pathlib import Path


# ============================================================
# Third Party
# ============================================================

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


# ============================================================
# Application
# ============================================================

from app.api.router import api_router
from app.config.logging_config import app_logger
from app.config.settings import settings
from app.database.init_db import init_db
from app.exceptions.handlers import register_exception_handlers
from app.llm.ollama_client import ollama_client
from app.middleware.request_id import request_id_middleware
from app.middleware.request_logger import log_requests
from app.rag.vector_store import vector_store


# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Startup:
        - Initialize database.
        - Validate/access vector store.
        - Log indexed document information.

    Shutdown:
        - Close Ollama HTTP client.
    """

    app_logger.info("=" * 70)
    app_logger.info(
        "Starting AI Voice Assistant Backend"
    )
    app_logger.info("=" * 70)

    # --------------------------------------------------------
    # Startup
    # --------------------------------------------------------

    try:

        # ----------------------------------------------------
        # Database
        # ----------------------------------------------------

        app_logger.info(
            "Initializing database..."
        )

        init_db()

        app_logger.info(
            "Database initialized successfully."
        )

        # ----------------------------------------------------
        # Vector Store
        # ----------------------------------------------------

        app_logger.info(
            "Checking indexed documents..."
        )

        documents = (
            vector_store.list_documents()
        )

        app_logger.info(
            "Indexed documents | "
            f"Count={len(documents)}"
        )

        if documents:

            for document in documents:

                app_logger.info(
                    "Indexed document | "
                    f"File={document.get('file_name', 'unknown')} | "
                    f"Chunks={document.get('chunks', 0)}"
                )

        else:

            app_logger.info(
                "No documents currently indexed."
            )

        app_logger.info(
            "Application startup completed successfully."
        )

    except Exception:

        app_logger.exception(
            "Application startup failed."
        )

        raise

    # --------------------------------------------------------
    # Application Running
    # --------------------------------------------------------

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    app_logger.info(
        "Shutting down application..."
    )

    try:

        await ollama_client.close()

        app_logger.info(
            "Ollama client closed successfully."
        )

    except Exception:

        app_logger.exception(
            "Failed to close Ollama client."
        )

    app_logger.info(
        "Application shutdown completed."
    )

    app_logger.info("=" * 70)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready AI Voice Assistant "
        "with RAG, conversation memory, "
        "speech-to-text, text-to-speech, "
        "and general AI capabilities."
    ),
    lifespan=lifespan,
)


# ============================================================
# Middleware
# ============================================================

app.middleware(
    "http"
)(request_id_middleware)

app.middleware(
    "http"
)(log_requests)


# ============================================================
# Exception Handlers
# ============================================================

register_exception_handlers(
    app
)


# ============================================================
# API Routers
# ============================================================

app.include_router(
    api_router
)


# ============================================================
# Output Directory
# ============================================================

output_directory = Path(
    settings.OUTPUT_FOLDER
)

output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

app.mount(
    "/outputs",
    StaticFiles(
        directory=str(
            output_directory
        )
    ),
    name="outputs",
)


# ============================================================
# Home Endpoint
# ============================================================

@app.get(
    "/",
    tags=["Home"],
    summary="Application Information",
)
async def home() -> dict:

    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "Running",
        "swagger": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }