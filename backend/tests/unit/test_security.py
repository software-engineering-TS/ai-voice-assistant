import pytest

from app.security import validate_session_id
from app.models.chat import ChatRequest


@pytest.mark.parametrize("session_id", ["session-123", "abc_123", "Tarun1", "x"])
def test_validate_session_id_accepts_safe_ids(session_id):
    assert validate_session_id(session_id) == session_id


@pytest.mark.parametrize(
    "session_id",
    ["", "   ", "../outside", "a/b", "a\\b", "white space", "x" * 101, "\nname"],
)
def test_validate_session_id_rejects_unsafe_ids(session_id):
    with pytest.raises(ValueError):
        validate_session_id(session_id)


def test_chat_request_rejects_path_like_session_id():
    with pytest.raises(ValueError):
        ChatRequest(session_id="../outside", message="hello")


def test_chat_request_accepts_valid_session_id():
    request = ChatRequest(session_id="session-123", message="hello")
    assert request.session_id == "session-123"