from pathlib import Path

from loguru import logger

from app.config.settings import settings


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.remove()

logger.add(
    LOG_DIR / "application.log",
    level=settings.LOG_LEVEL,
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level} | "
        "{module}:{function}:{line} | "
        "{message}"
    ),
)

logger.add(
    sink=lambda message: print(message, end=""),
    level=settings.LOG_LEVEL,
)

app_logger = logger