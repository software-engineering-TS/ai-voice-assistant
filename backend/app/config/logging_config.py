from pathlib import Path
import sys

from loguru import logger

from app.config.settings import settings


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Remove Default Loguru Handler
# ============================================================

logger.remove()


# ============================================================
# File Logging
# ============================================================

logger.add(
    LOG_DIR / "application.log",
    level=settings.LOG_LEVEL.upper(),
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=False,
    encoding="utf-8",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level:<8} | "
        "{module}:{function}:{line} | "
        "{message}"
    ),
)


# ============================================================
# Console Logging
# ============================================================

logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL.upper(),
    enqueue=True,
    backtrace=True,
    diagnose=False,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "{module}:{function}:{line} | "
        "<level>{message}</level>"
    ),
)


# ============================================================
# Application Logger
# ============================================================

app_logger = logger