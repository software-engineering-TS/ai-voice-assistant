from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from fastapi import HTTPException

from app.config.logging_config import app_logger
from app.config.settings import settings


# ============================================================
# DATABASE URL
# ============================================================

DATABASE_URL = settings.DATABASE_URL


# ============================================================
# BASE MODEL
# ============================================================

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.
    """
    pass


# ============================================================
# ENGINE CONFIGURATION
# ============================================================

engine_kwargs = {
    "echo": False,
    "pool_pre_ping": True,
}


# ------------------------------------------------------------
# SQLite Configuration
# ------------------------------------------------------------

if DATABASE_URL.startswith("sqlite"):

    engine_kwargs["connect_args"] = {
        "check_same_thread": False,
    }


# ------------------------------------------------------------
# PostgreSQL / Other Production Databases
# ------------------------------------------------------------

else:

    engine_kwargs.update(
        {
            "pool_size": 10,
            "max_overflow": 20,
            "pool_recycle": 1800,
        }
    )


# ============================================================
# DATABASE ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    **engine_kwargs,
)


# ============================================================
# SESSION FACTORY
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ============================================================
# DATABASE DEPENDENCY
# ============================================================

def get_db() -> Generator[Session, None, None]:
    """
    Create a database session for a request.

    The session is automatically closed after
    the request is completed.
    """

    db = SessionLocal()

    try:
        yield db

    except HTTPException:
        raise

    except Exception:

        db.rollback()

        app_logger.exception(
            "Database session failed. Transaction rolled back."
        )

        raise

    finally:
        db.close()