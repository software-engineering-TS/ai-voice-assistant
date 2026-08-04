import time
import uuid

from fastapi import Request

from app.config.logging_config import app_logger


async def request_id_middleware(request: Request, call_next):
    """
    Middleware to generate a unique Request ID
    and log request lifecycle.
    """

    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start = time.perf_counter()

    app_logger.info(
        f"[{request_id}] Incoming {request.method} {request.url.path}"
    )

    response = await call_next(request)

    duration = time.perf_counter() - start

    response.headers["X-Request-ID"] = request_id

    app_logger.info(
        f"[{request_id}] Completed "
        f"{response.status_code} "
        f"in {duration:.3f}s"
    )

    return response