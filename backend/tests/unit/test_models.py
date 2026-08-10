import re
from datetime import datetime, timezone

from sqlalchemy import inspect

from app.database.models import Conversation


# ============================================================
# Model Initialization
# ============================================================

def test_conversation_model_initializes_successfully():
    conversation = Conversation(
        session_id="session-123",
        role="user",
        message="Hello",
    )

    assert conversation is not None


# ============================================================
# Table Configuration
# ============================================================

def test_conversation_has_correct_table_name():
    assert Conversation.__tablename__ == "conversations"


def test_conversation_has_expected_columns():
    mapper = inspect(Conversation)

    columns = {
        column.key
        for column in mapper.columns
    }

    assert columns == {
        "id",
        "session_id",
        "role",
        "message",
        "created_at",
    }


# ============================================================
# Primary Key
# ============================================================

def test_id_is_primary_key():
    mapper = inspect(Conversation)

    primary_keys = {
        column.key
        for column in mapper.primary_key
    }

    assert primary_keys == {"id"}


def test_id_is_autoincrement():
    column = Conversation.__table__.c.id

    assert column.autoincrement is True


# ============================================================
# Session ID
# ============================================================

def test_session_id_is_required():
    column = Conversation.__table__.c.session_id

    assert column.nullable is False


def test_session_id_has_expected_length():
    column = Conversation.__table__.c.session_id

    assert column.type.length == 100


def test_session_id_is_indexed():
    column = Conversation.__table__.c.session_id

    assert column.index is True


# ============================================================
# Role
# ============================================================

def test_role_is_required():
    column = Conversation.__table__.c.role

    assert column.nullable is False


def test_role_has_expected_length():
    column = Conversation.__table__.c.role

    assert column.type.length == 20


# ============================================================
# Message
# ============================================================

def test_message_is_required():
    column = Conversation.__table__.c.message

    assert column.nullable is False


def test_message_uses_text_type():
    column = Conversation.__table__.c.message

    assert column.type.__class__.__name__ == "Text"


# ============================================================
# Created At
# ============================================================

def test_created_at_is_required():
    column = Conversation.__table__.c.created_at

    assert column.nullable is False


def test_created_at_uses_timezone_aware_datetime():
    column = Conversation.__table__.c.created_at

    assert column.type.timezone is True


def test_created_at_has_default():
    column = Conversation.__table__.c.created_at

    assert column.default is not None


def test_created_at_default_produces_utc_datetime():
    column = (
        Conversation.__table__
        .c.created_at
    )

    default = column.default

    assert default is not None
    assert callable(default.arg)

    value = default.arg(None)

    assert isinstance(
        value,
        datetime,
    )

    assert value.tzinfo is not None

    assert (
        value.utcoffset()
        == timezone.utc.utcoffset(value)
    )


# ============================================================
# Composite Index
# ============================================================

def test_conversation_has_session_created_index():
    indexes = {
        index.name
        for index in Conversation.__table__.indexes
    }

    assert (
        "ix_conversations_session_created"
        in indexes
    )


def test_session_created_index_contains_correct_columns():
    index = next(
        index
        for index in Conversation.__table__.indexes
        if index.name
        == "ix_conversations_session_created"
    )

    assert [
        column.name
        for column in index.columns
    ] == [
        "session_id",
        "created_at",
    ]


# ============================================================
# Representation
# ============================================================

def test_conversation_repr_contains_expected_information():
    created_at = datetime(
        2026,
        8,
        8,
        12,
        30,
        tzinfo=timezone.utc,
    )

    conversation = Conversation(
        id=42,
        session_id="session-123",
        role="user",
        message="Hello",
        created_at=created_at,
    )

    result = repr(conversation)

    assert "Conversation(" in result
    assert "id=42" in result
    assert "session_id='session-123'" in result
    assert "role='user'" in result
    assert "created_at=" in result


def test_conversation_repr_returns_string():
    conversation = Conversation(
        session_id="session-123",
        role="assistant",
        message="Hello",
    )

    result = repr(conversation)

    assert isinstance(result, str)


# ============================================================
# Object Field Assignment
# ============================================================

def test_conversation_stores_session_id():
    conversation = Conversation(
        session_id="session-abc",
        role="user",
        message="Hello",
    )

    assert conversation.session_id == "session-abc"


def test_conversation_stores_role():
    conversation = Conversation(
        session_id="session-abc",
        role="assistant",
        message="Hello",
    )

    assert conversation.role == "assistant"


def test_conversation_stores_message():
    conversation = Conversation(
        session_id="session-abc",
        role="user",
        message="Test message",
    )

    assert conversation.message == "Test message"


# ============================================================
# Model Metadata
# ============================================================

def test_conversation_model_is_mapped():
    mapper = inspect(Conversation)

    assert mapper.class_ is Conversation


def test_conversation_table_is_registered():
    assert (
        Conversation.__table__
        in Conversation.metadata.tables.values()
    )