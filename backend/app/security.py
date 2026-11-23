"""Small, dependency-light validation helpers for public API inputs."""

import re

SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$"
_SESSION_ID_RE = re.compile(SESSION_ID_PATTERN)


def validate_session_id(value: str) -> str:
    """Validate and return a safe conversation identifier.

    Session IDs are database identifiers, never filesystem paths. This
    validation rejects separators and control characters before a value
    can be passed to services or stored.
    """
    if not isinstance(value, str):
        raise ValueError("session_id must be a string.")

    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("session_id cannot contain control characters.")

    normalized = value.strip()
    if not normalized:
        raise ValueError("session_id cannot be empty.")

    if not _SESSION_ID_RE.fullmatch(normalized):
        raise ValueError(
            "session_id must be 1-100 characters and contain only "
            "letters, numbers, hyphens or underscores."
        )
    return normalized
