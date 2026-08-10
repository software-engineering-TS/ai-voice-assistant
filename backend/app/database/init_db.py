from app.config.logging_config import app_logger
from app.database.database import Base, engine

# Import models so SQLAlchemy registers all ORM models
from app.database import models  # noqa: F401


def init_db() -> None:
    """
    Initialize database tables.

    Creates tables that do not already exist.
    Existing tables are not modified.
    """

    try:
        Base.metadata.create_all(bind=engine)

        app_logger.info(
            "Database tables initialized successfully."
        )

    except Exception:

        app_logger.exception(
            "Database initialization failed."
        )

        raise