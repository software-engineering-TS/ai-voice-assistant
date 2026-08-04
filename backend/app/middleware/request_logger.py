import time

from fastapi import Request

from app.config.logging_config import app_logger


async def log_requests(request: Request, call_next):
    """
    Logs every incoming request and outgoing response.
    """

    start_time = time.perf_counter()

    app_logger.info(
        f"Incoming Request: {request.method} {request.url.path}"
    )

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    app_logger.info(
        f"Completed: {request.method} {request.url.path} "
        f"Status={response.status_code} "
        f"Time={duration:.3f}s"
    )

    return response