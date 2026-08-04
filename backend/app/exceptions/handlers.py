from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config.logging_config import app_logger
from app.exceptions.custom_exceptions import (
    AssistantException,
    LLMConnectionError,
    LLMResponseError,
)


def register_exception_handlers(app: FastAPI):
    """
    Register all application exception handlers.
    """

    @app.exception_handler(LLMConnectionError)
    async def llm_connection_handler(
        request: Request,
        exc: LLMConnectionError,
    ):
        app_logger.error(str(exc))

        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": str(exc),
            },
        )

    @app.exception_handler(LLMResponseError)
    async def llm_response_handler(
        request: Request,
        exc: LLMResponseError,
    ):
        app_logger.error(str(exc))

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(exc),
            },
        )

    @app.exception_handler(AssistantException)
    async def assistant_exception_handler(
        request: Request,
        exc: AssistantException,
    ):
        app_logger.error(str(exc))

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(exc),
            },
        )