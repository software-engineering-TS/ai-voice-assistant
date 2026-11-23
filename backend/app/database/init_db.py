from pathlib import Path

from sqlalchemy.engine import make_url

from app.config.logging_config import app_logger
from app.config.settings import settings
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
        database_url = make_url(settings.DATABASE_URL)
        if database_url.get_backend_name() == "sqlite" and database_url.database:
            database_path = Path(database_url.database)
            if str(database_path) != ":memory:":
                database_path.parent.mkdir(parents=True, exist_ok=True)

        Base.metadata.create_all(bind=engine)

        app_logger.info(
            "Database tables initialized successfully."
        )

    except Exception:

        app_logger.exception(
            "Database initialization failed."
        )

        raise