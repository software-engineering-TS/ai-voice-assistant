import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException

from app.database.database import (
    Base,
    SessionLocal,
    engine,
    get_db,
)


# ============================================================
# Base
# ============================================================

def test_base_initializes_successfully():
    assert Base is not None


def test_base_is_declarative_base():
    assert hasattr(
        Base,
        "metadata",
    )


# ============================================================
# Engine
# ============================================================

def test_engine_initializes_successfully():
    assert engine is not None


def test_engine_has_expected_url():
    assert engine.url is not None


def test_engine_has_pre_ping_enabled():
    assert (
        engine.pool._pre_ping
        is True
    )


# ============================================================
# Session Factory
# ============================================================

def test_session_local_initializes_successfully():
    assert SessionLocal is not None


def test_session_local_creates_session():
    db = SessionLocal()

    try:
        assert db is not None
    finally:
        db.close()


# ============================================================
# get_db
# ============================================================

def test_get_db_yields_database_session(
    monkeypatch,
):
    mock_db = MagicMock()

    monkeypatch.setattr(
        "app.database.database.SessionLocal",
        MagicMock(
            return_value=mock_db
        ),
    )

    generator = get_db()

    result = next(generator)

    assert result is mock_db

    mock_db.rollback.assert_not_called()
    mock_db.close.assert_not_called()

    generator.close()

    mock_db.close.assert_called_once()


def test_get_db_closes_session_after_success(
    monkeypatch,
):
    mock_db = MagicMock()

    monkeypatch.setattr(
        "app.database.database.SessionLocal",
        MagicMock(
            return_value=mock_db
        ),
    )

    generator = get_db()

    next(generator)

    with pytest.raises(StopIteration):
        next(generator)

    mock_db.close.assert_called_once()
    mock_db.rollback.assert_not_called()


def test_get_db_rolls_back_on_exception(
    monkeypatch,
):
    mock_db = MagicMock()

    monkeypatch.setattr(
        "app.database.database.SessionLocal",
        MagicMock(
            return_value=mock_db
        ),
    )

    generator = get_db()

    next(generator)

    test_exception = ValueError(
        "Test database failure"
    )

    with pytest.raises(ValueError, match="Test database failure"):
        generator.throw(test_exception)

    mock_db.rollback.assert_called_once()
    mock_db.close.assert_called_once()


def test_get_db_reraises_original_exception(
    monkeypatch,
):
    mock_db = MagicMock()

    monkeypatch.setattr(
        "app.database.database.SessionLocal",
        MagicMock(
            return_value=mock_db
        ),
    )

    generator = get_db()

    next(generator)

    original_exception = RuntimeError(
        "Original failure"
    )

    with pytest.raises(
        RuntimeError,
        match="Original failure",
    ):
        generator.throw(
            original_exception
        )

    mock_db.rollback.assert_called_once()
    mock_db.close.assert_called_once()


def test_get_db_closes_session_after_failure(
    monkeypatch,
):
    mock_db = MagicMock()

    monkeypatch.setattr(
        "app.database.database.SessionLocal",
        MagicMock(
            return_value=mock_db
        ),
    )

    generator = get_db()

    next(generator)

    with pytest.raises(Exception):
        generator.throw(
            RuntimeError(
                "Database error"
            )
        )

    mock_db.close.assert_called_once()


def test_get_db_does_not_rollback_on_success(
    monkeypatch,
):
    mock_db = MagicMock()

    monkeypatch.setattr(
        "app.database.database.SessionLocal",
        MagicMock(
            return_value=mock_db
        ),
    )

    generator = get_db()

    next(generator)

    with pytest.raises(StopIteration):
        next(generator)

    mock_db.rollback.assert_not_called()
    mock_db.close.assert_called_once()
    
def test_init_db_handles_initialization_failure(monkeypatch):
    """
    Database initialization errors should be logged
    and re-raised.
    """

    import app.database.init_db as init_db_module

    def fail_create_all(*args, **kwargs):
        raise RuntimeError(
            "Database initialization failed"
        )

    monkeypatch.setattr(
        init_db_module.Base.metadata,
        "create_all",
        fail_create_all,
    )

    with pytest.raises(
        RuntimeError,
        match="Database initialization failed",
    ):
        init_db_module.init_db()
        
def test_init_db_initializes_successfully(monkeypatch):
    """
    Database initialization should create tables
    and log successful initialization.
    """

    import app.database.init_db as init_db_module

    create_all_called = False

    def mock_create_all(*args, **kwargs):
        nonlocal create_all_called
        create_all_called = True

    monkeypatch.setattr(
        init_db_module.Base.metadata,
        "create_all",
        mock_create_all,
    )

    init_db_module.init_db()

    assert create_all_called is True
    
def test_get_db_reraises_http_exception(
    monkeypatch,
):
    """
    HTTPException should be re-raised without
    rolling back the database session.
    """

    mock_db = MagicMock()

    monkeypatch.setattr(
        "app.database.database.SessionLocal",
        MagicMock(
            return_value=mock_db
        ),
    )

    generator = get_db()

    next(generator)

    with pytest.raises(
        HTTPException,
        match="Test HTTP failure",
    ):
        generator.throw(
            HTTPException(
                status_code=400,
                detail="Test HTTP failure",
            )
        )

    mock_db.rollback.assert_not_called()
    mock_db.close.assert_called_once()
    
def test_database_configures_production_engine(
    monkeypatch,
):
    """
    Non-SQLite database URLs should configure
    production connection-pool settings.
    """

    import importlib
    import sqlalchemy
    import app.database.database as database_module

    captured_kwargs = {}

    def mock_create_engine(
        url,
        **kwargs,
    ):
        captured_kwargs.update(kwargs)

        return MagicMock()

    # Patch SQLAlchemy itself because database.py
    # imports create_engine during module reload.
    monkeypatch.setattr(
        sqlalchemy,
        "create_engine",
        mock_create_engine,
    )

    monkeypatch.setattr(
        database_module.settings,
        "DATABASE_URL",
        "postgresql://user:password@localhost/testdb",
    )

    importlib.reload(
        database_module
    )

    assert captured_kwargs["pool_size"] == 10
    assert captured_kwargs["max_overflow"] == 20
    assert captured_kwargs["pool_recycle"] == 1800