# ============================================================
# Standard Library
# ============================================================

from contextlib import asynccontextmanager
from pathlib import Path


# ============================================================
# Third Party
# ============================================================

import secrets

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
        "Local-first AI Voice Assistant with RAG, conversation memory, "
        "speech-to-text and text-to-speech. Remote deployment requires "
        "an API access token and HTTPS at the hosting layer."
    ),
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in settings.ALLOWED_ORIGINS.split(",")
    if origin.strip()
]


@app.middleware("http")
async def optional_bearer_auth(request: Request, call_next):
    """Protect API and generated files when an access token is configured.

    Local development remains frictionless when API_ACCESS_TOKEN is empty.
    CORS is not authentication; this check applies to every non-public route.
    """
    token = settings.API_ACCESS_TOKEN.strip()
    public_paths = {"/", "/health", "/health/", "/docs", "/redoc", "/openapi.json"}
    if token and request.url.path not in public_paths and request.method != "OPTIONS":
        authorization = request.headers.get("Authorization", "")
        expected = f"Bearer {token}"
        if not secrets.compare_digest(authorization, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid API access token."},
                headers={"WWW-Authenticate": "Bearer"},
            )
    return await call_next(request)


# ============================================================
# Middleware
# ============================================================

app.middleware(
    "http"
)(request_id_middleware)

app.middleware(
    "http"
)(log_requests)


# Keep CORS outermost so browser clients can read both successful and
# authentication-error responses. CORS still does not replace the token check.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


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